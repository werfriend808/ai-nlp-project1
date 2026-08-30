import { useState } from "react";
import type { JobEntry } from "../../lib/useVerifyJobs";
import { VerifyArticleForm } from "./VerifyArticleForm";

// 2026-08-26(5): 원래 VerifyNewArticleButton.tsx(플로팅 버튼 + 모달)를 결과 화면 전용으로
// 이름을 바꿔 재구성 — 메인 화면은 더 이상 목록이 아니라 URL 입력 자체라서, 이 컴포넌트는
// 이제 "기사 결과를 보고 있는 동안 다른 기사도 검증하고 싶을 때"에만 등장한다(App.tsx가
// selectedGroup이 있을 때만 렌더링).
//
// job 상태(urlInputs/jobs/...)는 이 컴포넌트가 직접 들고 있지 않고 App.tsx의
// useVerifyJobs 인스턴스를 그대로 props로 받는다 — 메인 화면의 인라인 입력 폼과 상태를
// 공유해야, 메인에서 여러 건을 넣고 하나만 보러 이동해도 나머지가 백그라운드에서 계속
// 진행되다가 여기(또는 다시 메인으로 돌아갔을 때)에서도 같은 진행 상황이 보인다. 이
// 컴포넌트가 직접 들고 있는 상태는 모달의 열림/닫힘(isOpen)뿐이다.
interface VerifyAnotherArticleButtonProps {
  urlInputs: string[];
  jobs: JobEntry[];
  isSubmitting: boolean;
  isAllTerminal: boolean;
  doneCount: number;
  failedCount: number;
  hasAnyBusy: boolean;
  onAddInputs: (batchSize?: number) => void;
  onRemoveInput: (index: number) => void;
  onInputChange: (index: number, value: string) => void;
  onSubmit: () => void;
  onReset: () => void;
  // 검증이 끝난 기사로 이동 — 단일 제출이 자동 완료됐을 때(App.tsx가 useVerifyJobs의
  // onJobDone에서 직접 처리), 또는 job 목록에서 "결과 보기"를 직접 눌렀을 때 호출된다.
  onArticleVerified: (articleTitle: string) => void;
}

export function VerifyAnotherArticleButton({
  urlInputs,
  jobs,
  isSubmitting,
  isAllTerminal,
  doneCount,
  failedCount,
  hasAnyBusy,
  onAddInputs,
  onRemoveInput,
  onInputChange,
  onSubmit,
  onReset,
  onArticleVerified,
}: VerifyAnotherArticleButtonProps) {
  const [isOpen, setIsOpen] = useState(false);

  const handleViewArticle = (articleTitle: string) => {
    setIsOpen(false);
    onArticleVerified(articleTitle);
  };

  const handleClose = () => setIsOpen(false);

  return (
    <>
      {/* 고정 버튼 — 스크롤해도 항상 같은 위치(화면 우하단)에 떠 있음. */}
      <button
        type="button"
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 left-1/2 z-20 flex -translate-x-1/2 items-center gap-2 rounded-full bg-stone-600 px-5 py-3 text-sm font-semibold text-white shadow-xl ring-2 ring-white ring-offset-2 ring-offset-stone-200 transition hover:bg-stone-700 active:scale-95 dark:ring-stone-300 dark:ring-offset-stone-950"
      >
        <span aria-hidden className="text-base font-bold leading-none">+</span>
        다른 기사 검증하기
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
            className="flex max-h-[80vh] w-[440px] flex-col gap-1 overflow-y-auto rounded-2xl bg-white p-6 shadow-2xl dark:bg-stone-900"
            onClick={(e) => e.stopPropagation()}
          >
            <h2 className="shrink-0 text-lg font-bold text-stone-900 dark:text-stone-100">
              다른 기사 검증하기
            </h2>
            <p className="mt-1 mb-2 shrink-0 text-xs text-stone-500 dark:text-stone-400">
              기사 URL을 입력하면 자동으로 검증합니다. (건당 몇 분 소요, 여러 건은 순서대로 처리)
            </p>
            <VerifyArticleForm
              urlInputs={urlInputs}
              jobs={jobs}
              isSubmitting={isSubmitting}
              isAllTerminal={isAllTerminal}
              doneCount={doneCount}
              failedCount={failedCount}
              onAddInputs={onAddInputs}
              onRemoveInput={onRemoveInput}
              onInputChange={onInputChange}
              onSubmit={onSubmit}
              onReset={onReset}
              onViewArticle={handleViewArticle}
              autoFocus
            />
            <button
              type="button"
              onClick={handleClose}
              className="mt-2 shrink-0 self-start rounded-lg px-3 py-1.5 text-xs text-stone-500 hover:bg-stone-100 dark:hover:bg-stone-800"
            >
              닫기
            </button>
          </div>
        </div>
      )}
    </>
  );
}
