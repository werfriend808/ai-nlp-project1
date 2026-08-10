import type { ArticleGroup } from "../../lib/articles";
import { VerdictSummaryBadges } from "./VerdictSummaryBadges";

interface ArticleListItemProps {
  group: ArticleGroup;
  onSelect: () => void;
}

export function ArticleListItem({ group, onSelect }: ArticleListItemProps) {
  return (
    <button
      type="button"
      onClick={onSelect}
      className="flex w-full flex-col gap-2 rounded-lg border border-gray-200 bg-white p-4 text-left shadow-sm transition hover:border-gray-300 hover:shadow-md dark:border-gray-700 dark:bg-gray-900 dark:hover:border-gray-600"
    >
      <div className="flex items-start justify-between gap-3">
        <h3 className="text-lg font-bold text-gray-950 dark:text-gray-50">
          {group.articleTitle}
        </h3>
        <span className="shrink-0 text-xs text-gray-400">
          주장 {group.records.length}건
        </span>
      </div>
      <VerdictSummaryBadges records={group.records} />
    </button>
  );
}
