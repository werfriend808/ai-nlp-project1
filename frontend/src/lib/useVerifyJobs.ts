import { useEffect, useRef, useState } from "react";

// 2026-08-26(5): 원래 VerifyNewArticleButton.tsx 안에 있던 job 제출/폴링 로직을 훅으로
// 뽑았다 — 프론트 구조를 "기사 목록 브라우징"에서 "URL 입력 → 그 기사 결과 보기"로 바꾸면서,
// 이 로직을 두 군데(메인 화면의 인라인 입력 폼, 결과 화면의 "다른 기사 검증하기" 모달)에서
// 똑같이 써야 해서 중복을 피하려고 분리했다. 상태(urlInputs/jobs)와 핸들러만 여기 있고,
// 렌더링은 VerifyArticleForm.tsx가 담당한다.
//
// 서버가 몇 분씩 걸릴 수 있어서(HCX 호출 여러 번 + GPU 단계) 요청-응답을 기다리지 않고
// job_id를 받아 폴링하는 비동기 방식으로 간다.
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000";
const API_KEY = import.meta.env.VITE_API_KEY ?? "";

export type JobStatus = "queued" | "fetching" | "processing" | "done" | "failed";

export interface JobEntry {
  url: string;
  jobId?: string;
  status: JobStatus;
  // 백엔드(agent/api/server.py)가 snake_case로 응답한다(article_title/claim_count) —
  // 프론트에서 camelCase로 잘못 읽으면 항상 undefined가 되는 버그가 있었다(2026-08-20 확인).
  article_title?: string;
  claim_count?: number;
  error?: string;
}

const POLL_INTERVAL_MS = 4000;

interface UseVerifyJobsOptions {
  // job 하나가 "done"으로 끝날 때마다 호출 — 이번 제출이 URL 1개짜리(단일)였으면
  // isSingle=true와 함께 완료된 기사 제목을 넘긴다. 호출부는 이걸로 "검증 끝난 기사로
  // 자동 이동"을 결정한다(배치로 여러 개를 넣었을 땐 어느 걸로 이동할지 애매해서 자동
  // 이동하지 않고, 대신 각 job 옆의 "결과 보기" 버튼으로 수동 이동하게 한다).
  onJobDone: (articleTitle: string | undefined, isSingle: boolean) => void;
}

export function useVerifyJobs({ onJobDone }: UseVerifyJobsOptions) {
  const [urlInputs, setUrlInputs] = useState<string[]>([""]);
  const [jobs, setJobs] = useState<JobEntry[]>([]);
  const pollTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // 폴링 콜백이 클로저에 갇힌 옛 jobs를 보고 판단하지 않도록 최신 상태를 ref로도 들고 있는다.
  const jobsRef = useRef<JobEntry[]>([]);
  const notifiedDoneRef = useRef<Set<string>>(new Set());
  const isSingleRef = useRef(false);

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
            // 2026-08-28: claim_count가 0인 job(파이프라인이 검증 가능한 수치 주장을 하나도
            // 못 찾은 기사 — 1단계에서 "통계 인용 기사 아님"으로 걸러졌거나 실제로 수치가
            // 없는 경우)은 DB에 아무 레코드도 안 남아서, "그 기사로 이동"해봐야 보여줄 게
            // 없다(실측: 이동은 되는데 화면이 그대로 입력 폼으로 되돌아가 버려서 마치 버튼이
            // 안 눌리는 것처럼 보임). 그래서 이 경우 article_title을 아예 넘기지 않는다 —
            // 호출부(App.tsx)가 자동 이동을 안 하고, VerifyArticleForm도 "결과 보기" 버튼
            // 대신 "검증 가능한 주장 없음" 안내를 보여준다.
            if (data.status === "done" && !notifiedDoneRef.current.has(job.jobId!)) {
              notifiedDoneRef.current.add(job.jobId!);
              const hasClaims = (data.claim_count ?? 0) > 0;
              onJobDone(hasClaims ? data.article_title : undefined, isSingleRef.current);
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

  const handleAddInputs = (batchSize = 5) => {
    setUrlInputs((prev) => [...prev, ...Array(batchSize).fill("")]);
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

    isSingleRef.current = validUrls.length === 1;
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

  const reset = () => {
    stopPolling();
    setJobs([]);
    jobsRef.current = [];
    setUrlInputs([""]);
  };

  const isSubmitting = jobs.length > 0;
  const isAllTerminal = jobs.length > 0 && jobs.every((j) => j.status === "done" || j.status === "failed");
  const doneCount = jobs.filter((j) => j.status === "done").length;
  const failedCount = jobs.filter((j) => j.status === "failed").length;
  const hasAnyBusy = jobs.some(
    (j) => j.status === "queued" || j.status === "fetching" || j.status === "processing",
  );

  return {
    urlInputs,
    jobs,
    isSubmitting,
    isAllTerminal,
    doneCount,
    failedCount,
    hasAnyBusy,
    handleAddInputs,
    handleRemoveInput,
    handleInputChange,
    handleSubmit,
    reset,
  };
}
