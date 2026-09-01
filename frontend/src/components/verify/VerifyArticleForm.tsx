import type { JobEntry, JobStatus } from "../../lib/useVerifyJobs";

// 2026-08-26(5): 프론트 구조 개편 — "기사 목록에서 골라 보기"에서 "URL을 입력하면 그 기사를
// 보여주기"로 바뀌면서, 원래 VerifyNewArticleButton.tsx 모달 안에 있던 입력 폼을 독립
// 컴포넌트로 뽑았다. 두 군데에서 재사용한다:
//   1) App.tsx의 메인 화면(기사가 선택 안 된 상태) — 모달 없이 그대로 인라인으로 박힘.
//   2) VerifyAnotherArticleButton.tsx — 결과 화면에서 "다른 기사 검증하기" 눌렀을 때 모달 안에 박힘.
//
// useVerifyJobs 훅은 이 컴포넌트가 아니라 "호출부"(App.tsx / VerifyAnotherArticleButton.tsx)가
// 직접 들고 있는다 — 특히 2)번은 모달을 닫아도 폴링이 백그라운드에서 계속돼야 하는데,
// 이 폼 컴포넌트에 훅을 두면 모달 unmount(닫기) 시 상태가 같이 사라져 버린다. 그래서 여기는
// 순수 표시(presentational) 컴포넌트로만 두고, 상태/핸들러는 전부 props로 받는다.
const ADD_BATCH_SIZE = 5;

const STATUS_LABEL: Record<JobStatus, string> = {
  queued: "대기 중",
  fetching: "원문 수집 중",
  processing: "1~8단계 처리 중",
  done: "완료",
  failed: "실패",
};

interface VerifyArticleFormProps {
  urlInputs: string[];
  jobs: JobEntry[];
  isSubmitting: boolean;
  isAllTerminal: boolean;
  doneCount: number;
  failedCount: number;
  onAddInputs: (batchSize?: number) => void;
  onRemoveInput: (index: number) => void;
  onInputChange: (index: number, value: string) => void;
  onSubmit: () => void;
  onReset: () => void;
  // 완료된 job 옆의 "결과 보기"를 눌렀을 때 호출 — 자동 이동(단일 제출)은 호출부(App.tsx)가
  // useVerifyJobs의 onJobDone 콜백에서 직접 처리하므로 이 컴포넌트는 신경 쓰지 않는다.
  onViewArticle: (articleTitle: string) => void;
  autoFocus?: boolean;
}

export function VerifyArticleForm({
  urlInputs,
  jobs,
  isSubmitting,
  isAllTerminal,
  doneCount,
  failedCount,
  onAddInputs,
  onRemoveInput,
  onInputChange,
  onSubmit,
  onReset,
  onViewArticle,
  autoFocus,
}: VerifyArticleFormProps) {
  return (
    <div className="flex flex-col gap-3">
      {!isSubmitting && (
        <>
          <div className="flex flex-col gap-2">
            {urlInputs.map((value, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="grid h-7 w-7 shrink-0 place-items-center rounded-full bg-stone-100 text-xs font-semibold text-stone-500 dark:bg-stone-800 dark:text-stone-400">
                  {i + 1}
                </span>
                <input
                  type="url"
                  value={value}
                  onChange={(e) => onInputChange(i, e.target.value)}
                  placeholder="기사 URL을 입력해주세요"
                  className="h-11 w-full rounded-lg border border-stone-300 px-3 text-sm outline-none focus:border-stone-500 dark:border-stone-700 dark:bg-stone-800 dark:text-stone-100"
                  autoFocus={autoFocus && i === 0}
                />
                <button
                  type="button"
                  onClick={() => onRemoveInput(i)}
                  disabled={urlInputs.length === 1}
                  aria-label="이 입력칸 삭제"
                  className="grid h-7 w-7 shrink-0 place-items-center rounded-full text-stone-400 hover:bg-stone-100 hover:text-stone-600 disabled:invisible dark:hover:bg-stone-800"
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
          <button
            type="button"
            onClick={() => onAddInputs(ADD_BATCH_SIZE)}
            className="rounded-lg border border-dashed border-stone-300 py-2 text-xs font-medium text-stone-500 hover:border-stone-400 hover:text-stone-700 dark:border-stone-700 dark:text-stone-400 dark:hover:border-stone-500 dark:hover:text-stone-300"
          >
            + {ADD_BATCH_SIZE}개 더 추가 (여러 건 한 번에)
          </button>
          <button
            type="button"
            onClick={onSubmit}
            disabled={urlInputs.every((u) => !u.trim())}
            className="rounded-lg bg-stone-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-stone-700 disabled:opacity-40"
          >
            검증 시작
          </button>
        </>
      )}

      {isSubmitting && (
        <div className="flex flex-col gap-3">
          <p className="text-xs text-stone-500 dark:text-stone-400">
            {isAllTerminal
              ? `전체 ${jobs.length}건 처리 완료 (성공 ${doneCount}건, 실패 ${failedCount}건)`
              : `전체 ${jobs.length}건 중 완료 ${doneCount + failedCount}건 — 서버가 순서대로 처리 중...`}
          </p>
          <div className="flex flex-col gap-2">
            {jobs.map((job, i) => (
              <div
                key={i}
                className="flex items-start gap-2 rounded-lg border border-stone-200 px-3 py-2 text-xs dark:border-stone-700"
              >
                <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-stone-100 text-[10px] font-semibold text-stone-500 dark:bg-stone-800 dark:text-stone-400">
                  {i + 1}
                </span>
                <span className="mt-0.5 shrink-0">
                  {job.status === "done" && (
                    <span className="grid h-4 w-4 place-items-center rounded-full bg-match-100 text-match-600 dark:bg-match-900/40">
                      ✓
                    </span>
                  )}
                  {job.status === "failed" && (
                    <span className="grid h-4 w-4 place-items-center rounded-full bg-mismatch-100 text-mismatch-600 dark:bg-mismatch-900/40">
                      ✕
                    </span>
                  )}
                  {(job.status === "queued" || job.status === "fetching" || job.status === "processing") && (
                    <span className="block h-4 w-4 animate-spin rounded-full border-2 border-stone-500 border-t-transparent" />
                  )}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium text-stone-700 dark:text-stone-200">
                    {job.article_title ?? job.url}
                  </p>
                  <p className="text-stone-400 dark:text-stone-500">
                    {job.status === "done"
                      ? job.claim_count
                        ? `주장 ${job.claim_count}건 처리됨`
                        : "검증 가능한 수치 주장을 찾지 못했습니다"
                      : job.status === "failed"
                        ? (job.error ?? "검증 중 오류가 발생했습니다.")
                        : STATUS_LABEL[job.status]}
                  </p>
                </div>
                {/* claim_count가 0이면 DB에 아무 레코드도 안 남아서 볼 결과 자체가 없다 —
                    2026-08-28: 이 경우에도 버튼을 보여주면 눌러도 반응이 없는 것처럼 보인다
                    (실제 버그로 발견됨). */}
                {job.status === "done" && job.article_title && !!job.claim_count && (
                  <button
                    type="button"
                    onClick={() => onViewArticle(job.article_title!)}
                    className="shrink-0 self-center rounded-full bg-stone-100 px-2.5 py-1 text-[11px] font-semibold text-stone-700 hover:bg-stone-200 dark:bg-stone-800 dark:text-stone-200 dark:hover:bg-stone-700"
                  >
                    결과 보기 →
                  </button>
                )}
              </div>
            ))}
          </div>
          {isAllTerminal ? (
            <button
              type="button"
              onClick={onReset}
              className="rounded-lg bg-stone-600 px-4 py-2 text-sm font-semibold text-white hover:bg-stone-700"
            >
              새로 검증하기
            </button>
          ) : (
            <p className="text-xs text-stone-400">
              창을 닫아도 백그라운드에서 계속 진행돼요. 끝나는 대로 반영됩니다.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
