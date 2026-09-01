import { articleDisplayDate, formatDate, type ArticleGroup } from "../../lib/articles";
import { verdictCountLabel } from "../../lib/verdictColors";

// 2026-08-28: "URL 입력 → 그 기사만 보여주기"로 구조를 바꾸면서 예전 기사 목록/통계
// 대시보드(SummaryCard, ArticleListPanel)를 완전히 지웠는데, "내가 검색했던 기사들을 다시
// 볼 수 있으면 좋겠다"는 요청으로 다시 추가한다. 다만 예전 그 수준(필터/정렬/검색/페이지네이션
// +통계)까지는 아니고, 최근 검증한 기사를 제목+판정 요약만 보여주는 간단한 목록으로.
//
// 2026-08-28(2): 처음엔 입력 화면 바로 아래에 이 목록을 펼쳐서 보여줬는데 "첫 화면부터
// 쭉 나열되니 지저분하다"는 피드백 — 입력 화면엔 압축 요약 카드(RecentArticlesSummaryCard,
// App.tsx)만 두고, 그걸 눌러야 이 컴포넌트가 전용 화면에 펼쳐지는 구조로 바꿨다. 그래서
// limit을 옵션으로 뺐다 — 입력 화면(안 쓰임, 요약 카드로 대체)과 전용 화면(limit 없이
// 전체) 두 컨텍스트에서 재사용 가능하게.
interface RecentArticlesListProps {
  groups: ArticleGroup[];
  articleDates: Record<string, string>;
  onSelect: (articleTitle: string) => void;
  limit?: number;
}

interface CountBadgeProps {
  label: string;
  count: number;
  colorClass: string;
}

function CountBadge({ label, count, colorClass }: CountBadgeProps) {
  if (count === 0) return null;
  return (
    <span className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold ${colorClass}`}>
      {label} {count}
    </span>
  );
}

export function RecentArticlesList({ groups, articleDates, onSelect, limit }: RecentArticlesListProps) {
  const sorted = [...groups]
    .sort((a, b) => {
      const da = articleDisplayDate(a, articleDates);
      const db = articleDisplayDate(b, articleDates);
      return db > da ? 1 : db < da ? -1 : 0;
    })
    .slice(0, limit);

  if (sorted.length === 0) {
    return (
      <div className="rounded-2xl border border-stone-200 bg-white p-6 text-center text-sm text-stone-400 shadow-sm dark:border-stone-700 dark:bg-stone-900">
        아직 검증한 기사가 없습니다.
      </div>
    );
  }

  return (
    <div className="overflow-hidden rounded-2xl border border-stone-200 bg-white shadow-sm dark:border-stone-700 dark:bg-stone-900">
      <div className="flex items-center gap-2 border-b border-stone-100 px-5 py-3 dark:border-stone-800">
        <span className="h-2 w-2 rounded-full bg-stone-600" />
        <span className="text-sm font-semibold text-stone-900 dark:text-stone-100">최근 검증한 기사</span>
      </div>
      <ul className="divide-y divide-stone-100 dark:divide-stone-800">
        {sorted.map((group) => {
          const 일치 = group.records.filter((r) => verdictCountLabel(r.verification_result) === "일치").length;
          const 불일치 = group.records.filter((r) => verdictCountLabel(r.verification_result) === "불일치").length;
          const 애매 = group.records.filter((r) => verdictCountLabel(r.verification_result) === "애매").length;

          return (
            <li key={group.articleTitle}>
              <button
                type="button"
                onClick={() => onSelect(group.articleTitle)}
                className="flex w-full items-center justify-between gap-3 px-5 py-3 text-left transition hover:bg-stone-50 dark:hover:bg-stone-800/60"
              >
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-stone-800 dark:text-stone-100">
                    {group.articleTitle}
                  </p>
                  <p className="mt-0.5 text-xs text-stone-400">
                    {formatDate(articleDisplayDate(group, articleDates))}
                  </p>
                </div>
                <div className="flex shrink-0 items-center gap-1.5">
                  <CountBadge label="일치" count={일치} colorClass="bg-match-100 text-match-700 dark:bg-match-900/40 dark:text-match-300" />
                  <CountBadge label="불일치" count={불일치} colorClass="bg-mismatch-100 text-mismatch-700 dark:bg-mismatch-900/40 dark:text-mismatch-300" />
                  <CountBadge label="애매" count={애매} colorClass="bg-caution-100 text-caution-700 dark:bg-caution-900/40 dark:text-caution-300" />
                  <span className="ml-1 text-stone-300 dark:text-stone-600">→</span>
                </div>
              </button>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
