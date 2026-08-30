// 2026-08-28(2): 입력 화면에 최근 기사 목록을 통째로 펼쳐두면 "첫 화면부터 쭉 나열돼서
// 지저분하다"는 피드백 — 이 압축 카드 하나만 입력 화면에 두고, 누르면 RecentArticlesList가
// 전용 화면(App.tsx의 history 뷰)에 펼쳐지는 구조로 바꿨다.
interface RecentArticlesSummaryCardProps {
  count: number;
  onClick: () => void;
}

export function RecentArticlesSummaryCard({ count, onClick }: RecentArticlesSummaryCardProps) {
  if (count === 0) return null;

  return (
    <button
      type="button"
      onClick={onClick}
      className="flex items-center justify-between gap-3 rounded-2xl border border-stone-200 bg-white px-5 py-4 text-left shadow-sm transition hover:shadow-md dark:border-stone-700 dark:bg-stone-900"
    >
      <div className="flex items-center gap-3">
        <span className="grid h-10 w-10 shrink-0 place-items-center rounded-xl bg-stone-100 text-stone-600 dark:bg-stone-800 dark:text-stone-300">
          <svg viewBox="0 0 24 24" className="h-5 w-5" fill="none">
            <path
              d="M6 3.5h9l4 4V19a1.5 1.5 0 0 1-1.5 1.5h-11A1.5 1.5 0 0 1 5 19V5A1.5 1.5 0 0 1 6.5 3.5Z"
              stroke="currentColor"
              strokeWidth={2}
              strokeLinejoin="round"
            />
            <path d="M15 3.5V8h4" stroke="currentColor" strokeWidth={2} strokeLinejoin="round" />
            <path d="M8.5 12h7M8.5 15.5h4" stroke="currentColor" strokeWidth={2} strokeLinecap="round" />
          </svg>
        </span>
        <div>
          <p className="text-sm font-semibold text-stone-800 dark:text-stone-100">최근 검증한 기사</p>
          <p className="text-xs text-stone-400">{count.toLocaleString()}건</p>
        </div>
      </div>
      <span className="text-stone-300 dark:text-stone-600">→</span>
    </button>
  );
}
