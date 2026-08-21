import { useEffect, useRef, useState } from "react";

// 2026-08-21 추가: "새로운 뉴스 기사 검증하기" 기능 — URL을 입력하면 agent/api/server.py
// (AWS GPU 서버에서만 실행 가능, VDB/리랭커가 GPU 필요)에 1~8단계 파이프라인 실행을
// 맡기고, 완료되면 그 결과가 반영된 JSON을 다시 불러오도록 부모(App)에 알려준다.
//
// 서버가 몇 분씩 걸릴 수 있어서(HCX 호출 여러 번 + GPU 단계) 요청-응답을 기다리지 않고
// job_id를 받아 폴링하는 비동기 방식으로 간다 — 사용자는 모달을 닫고 다른 걸 봐도 되고,
// 처리가 끝나면 알림만 받는다.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

type JobStatus = "idle" | "queued" | "fetching" | "processing" | "done" | "failed";

interface JobState {
  status: JobStatus;
  // 백엔드(agent/api/server.py)가 snake_case로 응답한다(article_title/claim_count) —
  // 프론트에서 camelCase(articleTitle/claimCount)로 잘못 읽으면 항상 undefined가 되어
  // "주장 0건 처리됨"으로 오표시되는 버그가 있었다(2026-08-20 실제 확인).
  article_title?: string;
  claim_count?: number;
  error?: string;
}

const POLL_INTERVAL_MS = 4000;

interface VerifyNewArticleButtonProps {
  // job이 "done"으로 끝나면 호출 — App이 export된 JSON을 다시 fetch하도록 알림.
  onVerificationDone: () => void;
}

export function VerifyNewArticleButton({ onVerificationDone }: VerifyNewArticleButtonProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [url, setUrl] = useState("");
  const [job, setJob] = useState<JobState>({ status: "idle" });
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);

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

  const pollStatus = (jobId: string) => {
    pollTimerRef.current = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/api/verify/${jobId}`, {
          headers: { "X-API-Key": API_KEY },
        });
        if (!res.ok) throw new Error(`상태 조회 실패 (${res.status})`);
        const data = (await res.json()) as JobState;
        setJob(data);
        if (data.status === "done" || data.status === "failed") {
          stopPolling();
          if (data.status === "done") onVerificationDone();
        }
      } catch (e) {
        stopPolling();
        setJob({ status: "failed", error: e instanceof Error ? e.message : "상태 조회 중 오류" });
      }
    }, POLL_INTERVAL_MS);
  };

  const handleSubmit = async () => {
    if (!url.trim()) return;
    setJob({ status: "queued" });
    try {
      const res = await fetch(`${API_BASE_URL}/api/verify`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "X-API-Key": API_KEY },
        body: JSON.stringify({ url: url.trim() }),
      });
      if (!res.ok) {
        const detail = await res.json().catch(() => ({}));
        throw new Error(detail.detail ?? `요청 실패 (${res.status})`);
      }
      const data = (await res.json()) as { job_id: string };
      pollStatus(data.job_id);
    } catch (e) {
      setJob({
        status: "failed",
        error:
          e instanceof Error
            ? `${e.message} — 검증 서버(AWS GPU)가 켜져있고 SSH 터널이 연결돼 있는지 확인하세요.`
            : "요청 중 오류가 발생했습니다.",
      });
    }
  };

  const handleClose = () => {
    // 진행 중인 job은 서버에서 계속 돌아간다(닫아도 취소 안 됨) — 그냥 모달만 닫고,
    // 폴링은 끝날 때까지 백그라운드에서 계속돼 완료되면 목록이 자동 갱신된다.
    setIsOpen(false);
  };

  const resetAndClose = () => {
    stopPolling();
    setJob({ status: "idle" });
    setUrl("");
    setIsOpen(false);
  };

  const isBusy = job.status === "queued" || job.status === "fetching" || job.status === "processing";

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
        {isBusy && (
          <span className="ml-1 h-2 w-2 animate-pulse rounded-full bg-white" aria-label="처리 중" />
        )}
      </button>

      {isOpen && (
        <div
          className="fixed inset-0 z-30 flex items-center justify-center bg-black/40 px-4"
          onClick={handleClose}
        >
          <div
            className="w-full max-w-md rounded-2xl bg-white p-6 shadow-2xl dark:bg-gray-900"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">
              새로운 뉴스 기사 검증하기
            </h2>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              기사 URL을 입력하면 1~8단계 파이프라인을 돌려서 자동으로 검증합니다.
              (몇 분 정도 걸릴 수 있어요)
            </p>

            {job.status === "idle" && (
              <div className="mt-4 flex flex-col gap-3">
                <input
                  type="url"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="https://www.chosun.com/..."
                  className="w-full rounded-lg border border-gray-300 px-3 py-2 text-sm outline-none focus:border-indigo-500 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-100"
                  autoFocus
                />
                <div className="flex justify-end gap-2">
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
                    disabled={!url.trim()}
                    className="rounded-lg bg-gradient-to-r from-indigo-600 to-fuchsia-600 px-4 py-2 text-sm font-semibold text-white disabled:opacity-40"
                  >
                    검증 시작
                  </button>
                </div>
              </div>
            )}

            {isBusy && (
              <div className="mt-6 flex flex-col items-center gap-3 py-4">
                <div className="h-8 w-8 animate-spin rounded-full border-2 border-indigo-500 border-t-transparent" />
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  {job.status === "queued" && "요청을 등록했습니다..."}
                  {job.status === "fetching" && "기사 원문을 가져오는 중..."}
                  {job.status === "processing" && "1~8단계 파이프라인 처리 중..."}
                </p>
                <p className="text-xs text-gray-400">
                  모달을 닫아도 백그라운드에서 계속 진행돼요. 끝나면 목록이 자동으로 갱신됩니다.
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

            {job.status === "done" && (
              <div className="mt-6 flex flex-col items-center gap-3 py-4">
                <span className="grid h-10 w-10 place-items-center rounded-full bg-emerald-100 text-emerald-600 dark:bg-emerald-900/40">
                  ✓
                </span>
                <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  검증 완료: {job.article_title}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  주장 {job.claim_count ?? 0}건 처리됨 — 목록이 갱신되었습니다.
                </p>
                <button
                  type="button"
                  onClick={resetAndClose}
                  className="mt-2 rounded-lg bg-gradient-to-r from-indigo-600 to-fuchsia-600 px-4 py-2 text-sm font-semibold text-white"
                >
                  확인
                </button>
              </div>
            )}

            {job.status === "failed" && (
              <div className="mt-6 flex flex-col items-center gap-3 py-4">
                <span className="grid h-10 w-10 place-items-center rounded-full bg-red-100 text-red-600 dark:bg-red-900/40">
                  ✕
                </span>
                <p className="text-center text-sm text-gray-700 dark:text-gray-300">
                  {job.error ?? "검증 중 오류가 발생했습니다."}
                </p>
                <button
                  type="button"
                  onClick={resetAndClose}
                  className="mt-2 rounded-lg px-4 py-2 text-sm text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
                >
                  닫기
                </button>
              </div>
            )}
          </div>
        </div>
      )}
    </>
  );
}
