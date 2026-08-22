import { useEffect, useRef, useState } from "react";

// 2026-08-21 추가: "새로운 뉴스 기사 검증하기" 기능 — URL을 입력하면 agent/api/server.py
// (AWS GPU 서버에서만 실행 가능, VDB/리랭커가 GPU 필요)에 1~8단계 파이프라인 실행을
// 맡기고, 완료되면 그 결과가 반영된 JSON을 다시 불러오도록 부모(App)에 알려준다.
//
// 서버가 몇 분씩 걸릴 수 있어서(HCX 호출 여러 번 + GPU 단계) 요청-응답을 기다리지 않고
// job_id를 받아 폴링하는 비동기 방식으로 간다 — 사용자는 모달을 닫고 다른 걸 봐도 되고,
// 처리가 끝나면 알림만 받는다.
//
// 2026-08-22 추가: URL 하나씩 넣고 매번 기다리는 게 느리다는 피드백 — 입력칸을 여러 개
// 두고 한 번에 여러 URL을 제출할 수 있게 했다. 서버(agent/api/server.py)는 job을 큐에
// 넣고 워커 스레드 1개가 순서대로 처리하므로(GPU 모델/DB 커넥션 동시 접근 위험 방지),
// 여기서도 여러 개를 "동시에 처리 중"으로 보여주지 않고 각 job의 실제 큐/처리 상태를
// 그대로 보여준다 — 프론트가 병렬성을 흉내내지 않는다.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

type JobStatus = "queued" | "fetching" | "processing" | "done" | "failed";

interface JobEntry {
  url: string;
  jobId?: string;
  status: JobStatus;
  // 백엔드(agent/api/server.py)가 snake_case로 응답한다(article_title/claim_count) —
  // 프론트에서 camelCase(articleTitle/claimCount)로 잘못 읽으면 항상 undefined가 되어
  // "주장 0건 처리됨"으로 오표시되는 버그가 있었다(2026-08-20 실제 확인).
  article_title?: string;
  claim_count?: number;
  error?: string;
}

const POLL_INTERVAL_MS = 4000;
const ADD_BATCH_SIZE = 5;
const INITIAL_INPUT_COUNT = 5;

interface VerifyNewArticleButtonProps {
  // job이 "done"으로 끝나면 호출 — App이 export된 JSON을 다시 fetch하도록 알림.
  onVerificationDone: () => void;
}

export function VerifyNewArticleButton({ onVerificationDone }: VerifyNewArticleButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [urlInputs, setUrlInputs] = useState<string[]>(Array(INITIAL_INPUT_COUNT).fill(""));
  const [jobs, setJobs] = useState<JobEntry[]>([]);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // 폴링 콜백이 클로저에 갇힌 옛 jobs를 보고 판단하지 않도록 최신 상태를 ref로도 들고 있는다.
  const jobsRef = useRef<JobEntry[]>([]);
  const notifiedDoneRef = useRef<Set<string>>(new Set());

  useEffect(() => {
    return () => {
      if (pollTimerRef.current) clearInterval(pollTimerRef.current);
    };
  }, []);

  const stopPolling = () => {
    if (pollTimerRef.current) {
      clearInterval(pollTimerRef.current);
      pollTimerRef.current = null;
    }
  };

  const updateJob = (jobId: string, patch: Partial<JobEntry>) => {
    setJobs((prev) => {
      const next = prev.map((j) => (j.jobId === jobId ? { ...j, ...patch } : j));
      jobsRef.current = next;
      return next;
    });
  };

  const pollAll = () => {
    pollTimerRef.current = setInterval(async () => {
      const inFlight = jobsRef.current.filter(
        (j) => j.jobId && j.status !== "done" && j.status !== "failed",
      );
      if (inFlight.length === 0) {
        stopPolling();
        return;
      }
      await Promise.all(
        inFlight.map(async (job) => {
          try {
            const res = await fetch(`${API_BASE_URL}/api/verify/${job.jobId}`, {
              headers: { "X-API-Key": API_KEY },
            });
            if (!res.ok) throw new Error(`상태 조회 실패 (${res.status})`);
            const data = (await res.json()) as Omit<JobEntry, "url" | "jobId">;
            updateJob(job.jobId!, data);
            // done인 job이 나올 때마다 바로 목록을 갱신한다 — 전부 끝날 때까지 기다리지
            // 않고 완료되는 대로 화면에 반영되게(여러 개를 한꺼번에 넣었을 때 체감 대기가
            // 줄어듦). 같은 job을 두 번 알리지 않도록 jobId 기준으로 막는다.
            if (data.status === "done" && !notifiedDoneRef.current.has(job.jobId!)) {
              notifiedDoneRef.current.add(job.jobId!);
              onVerificationDone();
            }
          } catch (e) {
            updateJob(job.jobId!, {
              status: "failed",
              error: e instanceof Error ? e.message : "상태 조회 중 오류",
            });
          }
        }),
      );
    }, POLL_INTERVAL_MS);
  };

  const handleAddInputs = () => {
    setUrlInputs((prev) => [...prev, ...Array(ADD_BATCH_SIZE).fill("")]);
  };

  const handleRemoveInput = (index: number) => {
    setUrlInputs((prev) => prev.filter((_, i) => i !== index));
  };

  const handleInputChange = (index: number, value: string) => {
    setUrlInputs((prev) => prev.map((v, i) => (i === index ? value : v)));
  };

  const handleSubmit = async () => {
    const validUrls = urlInputs.map((u) => u.trim()).filter(Boolean);
    if (validUrls.length === 0) return;

    const initialJobs: JobEntry[] = validUrls.map((url) => ({ url, status: "queued" }));
    setJobs(initialJobs);
    jobsRef.current = initialJobs;
    notifiedDoneRef.current = new Set();

    // 제출 자체는 여러 URL을 순서대로 등록만 한다(서버가 큐에 넣기만 하고 바로 job_id를
    // 돌려주므로 빠름) — 실제 처리 순서/속도는 서버의 워커 스레드 1개가 결정한다.
    for (const url of validUrls) {
      try {
        const res = await fetch(`${API_BASE_URL}/api/verify`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
          body: JSON.stringify({ url }),
        });
        if (!res.ok) {
          const detail = await res.json().catch(() => ({}));
          throw new Error(detail.detail ?? `요청 실패 (${res.status})`);
        }
        const data = (await res.json()) as { job_id: string };
        setJobs((prev) => {
          const next = prev.map((j) => (j.url === url && !j.jobId ? { ...j, jobId: data.job_id } : j));
          jobsRef.current = next;
          return next;
        });
      } catch (e) {
        setJobs((prev) => {
          const next = prev.map((j) =>
            j.url === url && !j.jobId
              ? {
                  ...j,
                  status: "failed" as const,
                  error:
                    e instanceof Error
                      ? `${e.message} — 검증 서버(AWS GPU)가 켜져있고 SSH 터널이 연결돼 있는지 확인하세요.`
                      : "요청 중 오류가 발생했습니다.",
                }
              : j,
          );
          jobsRef.current = next;
          return next;
        });
      }
    }

    pollAll();
  };

  const handleClose = () => {
    // 진행 중인 job은 서버에서 계속 돌아간다(닫아도 취소 안 됨) — 그냥 모달만 닫고,
    // 폴링은 끝날 때까지 백그라운드에서 계속돼 완료되면 목록이 자동 갱신된다.
    setIsOpen(false);
  };

  const resetAndClose = () => {
    stopPolling();
    setJobs([]);
    jobsRef.current = [];
    setUrlInputs(Array(INITIAL_INPUT_COUNT).fill(""));
    setIsOpen(false);
  };

  const isSubmitting = jobs.length > 0;
  const isAllTerminal =
    jobs.length > 0 && jobs.every((j) => j.status === "done" || j.status === "failed");
  const doneCount = jobs.filter((j) => j.status === "done").length;
  const failedCount = jobs.filter((j) => j.status === "failed").length;
  const hasAnyBusy = jobs.some((j) => j.status === "queued" || j.status === "fetching" || j.status === "processing");

  const statusLabel: Record<JobStatus, string> = {
    queued: "대기 중",
    fetching: "원문 수집 중",
    processing: "1~8단계 처리 중",
    done: "완료",
    failed: "실패",
  };

  return (
    <>
      {/* 고정 버튼 — 스크롤해도 항상 같은 위치(화면 우하단)에 떠 있음 */}
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 left-1/2 z-20 flex -translate-x-1/2 items-center gap-2 rounded-full bg-gradient-to-r from-indigo-600 to-fuchsia-600 px-5 py-3 text-sm font-semibold text-white shadow-lg shadow-indigo-500/30 transition hover:shadow-xl hover:shadow-indigo-500/40 active:scale-95"
      >
        <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none">
          <path
            d="M12 5v14M5 12h14"
            stroke="currentColor"
            strokeWidth={2.5}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
        새로운 뉴스 기사 검증하기
        {hasAnyBusy && (
          <span className="ml-1 h-2 w-2 animate-pulse rounded-full bg-white" aria-label="처리 중" />
        )}
      </button>

      {isOpen && (
        <div
          className="fixed inset-0 z-30 flex items-center justify-center bg-black/40 px-4"
          onClick={handleClose}
        >
          <div
            className="flex h-[560px] w-[440px] flex-col rounded-2xl bg-white p-6 shadow-2xl dark:bg-gray-900"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="shrink-0 text-lg font-bold text-gray-900 dark:text-gray-100">
              새로운 뉴스 기사 검증하기
            </h2>
            <p className="mt-1 shrink-0 text-xs text-gray-500 dark:text-gray-400">
              기사 URL을 입력하면 1~8단계 파이프라인이 자동으로 검증합니다.
            </p>
            <div className="mt-2 flex shrink-0 flex-wrap gap-1.5">
              <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2.5 py-1 text-[11px] font-medium text-gray-600 dark:bg-gray-800 dark:text-gray-300">
                ⏱️ 건당 몇 분 소요
              </span>
              <span className="inline-flex items-center gap-1 rounded-full bg-gray-100 px-2.5 py-1 text-[11px] font-medium text-gray-600 dark:bg-gray-800 dark:text-gray-300">
                📋 여러 건은 순서대로 자동 처리
              </span>
            </div>

            {!isSubmitting && (
              <div className="mt-4 flex min-h-0 flex-1 flex-col gap-3">
                <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto pr-1">
                  {urlInputs.map((value, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-gray-100 text-xs font-semibold text-gray-500 dark:bg-gray-800 dark:text-gray-400">
                        {i + 1}
                      </span>
                      <input
                        type="url"
                        value={value}
                        onChange={(e) => handleInputChange(i, e.target.value)}
                        placeholder="기사 URL을 입력해주세요"
                        className="h-10 w-full rounded-lg border border-gray-300 px-3 text-sm outline-none focus:border-indigo-500 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100"
                        autoFocus={i === 0}
                      />
                      <button
                        type="button"
                        onClick={() => handleRemoveInput(i)}
                        disabled={urlInputs.length === 1}
                        aria-label="이 입력칸 삭제"
                        className="grid h-7 w-7 shrink-0 place-items-center rounded-full text-gray-400 hover:bg-gray-100 hover:text-gray-600 disabled:invisible dark:hover:bg-gray-800"
                      >
                        ✕
                      </button>
                    </div>
                  ))}
                </div>
                <button
                  type="button"
                  onClick={handleAddInputs}
                  className="shrink-0 rounded-lg border border-dashed border-gray-300 py-2 text-xs font-medium text-gray-500 hover:border-indigo-400 hover:text-indigo-600 dark:border-gray-700 dark:text-gray-400 dark:hover:border-indigo-500 dark:hover:text-indigo-400"
                >
                  + {ADD_BATCH_SIZE}개 더 추가
                </button>
                <div className="flex shrink-0 justify-end gap-2">
                  <button
                    type="button"
                    onClick={handleClose}
                    className="rounded-lg px-4 py-2 text-sm text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
                  >
                    취소
                  </button>
                  <button
                    type="button"
                    onClick={handleSubmit}
                    disabled={urlInputs.every((u) => !u.trim())}
                    className="rounded-lg bg-gradient-to-r from-indigo-600 to-fuchsia-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
                  >
                    검증 시작
                  </button>
                </div>
              </div>
            )}

            {isSubmitting && (
              <div className="mt-4 flex min-h-0 flex-1 flex-col gap-3">
                <p className="shrink-0 text-xs text-gray-500 dark:text-gray-400">
                  {isAllTerminal
                    ? `전체 ${jobs.length}건 처리 완료 (성공 ${doneCount}건, 실패 ${failedCount}건)`
                    : `전체 ${jobs.length}건 중 완료 ${doneCount + failedCount}건 — 서버가 순서대로 처리 중...`}
                </p>
                <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto pr-1">
                  {jobs.map((job, i) => (
                    <div
                      key={i}
                      className="flex items-start gap-2 rounded-lg border border-gray-200 px-3 py-2 text-xs dark:border-gray-700"
                    >
                      <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-gray-100 text-[10px] font-semibold text-gray-500 dark:bg-gray-800 dark:text-gray-400">
                        {i + 1}
                      </span>
                      <span className="mt-0.5 shrink-0">
                        {job.status === "done" && (
                          <span className="grid h-4 w-4 place-items-center rounded-full bg-emerald-100 text-emerald-600 dark:bg-emerald-900/40">
                            ✓
                          </span>
                        )}
                        {job.status === "failed" && (
                          <span className="grid h-4 w-4 place-items-center rounded-full bg-red-100 text-red-600 dark:bg-red-900/40">
                            ✕
                          </span>
                        )}
                        {(job.status === "queued" ||
                          job.status === "fetching" ||
                          job.status === "processing") && (
                          <span className="block h-4 w-4 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
                        )}
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="truncate font-medium text-gray-700 dark:text-gray-200">
                          {job.article_title ?? job.url}
                        </p>
                        <p className="text-gray-400 dark:text-gray-500">
                          {job.status === "done"
                            ? `주장 ${job.claim_count ?? 0}건 처리됨`
                            : job.status === "failed"
                              ? (job.error ?? "검증 중 오류가 발생했습니다.")
                              : statusLabel[job.status]}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
                {isAllTerminal ? (
                  <button
                    type="button"
                    onClick={resetAndClose}
                    className="mt-1 shrink-0 rounded-lg bg-gradient-to-r from-indigo-600 to-fuchsia-600 px-4 py-2 text-sm font-semibold text-white"
                  >
                    확인
                  </button>
                ) : (
                  <div className="shrink-0">
                    <p className="text-xs text-gray-400">
                      모달을 닫아도 백그라운드에서 계속 진행돼요. 끝나는 대로 목록이 자동으로
                      갱신됩니다.
                    </p>
                    <button
                      type="button"
                      onClick={handleClose}
                      className="mt-2 rounded-lg px-4 py-2 text-sm text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
                    >
                      닫기(계속 진행됨)
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
