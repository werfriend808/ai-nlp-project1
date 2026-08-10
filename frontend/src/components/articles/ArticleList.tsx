import type { ArticleGroup } from "../../lib/articles";
import { ArticleListItem } from "./ArticleListItem";

interface ArticleListProps {
  groups: ArticleGroup[];
  onSelect: (articleTitle: string) => void;
}

export function ArticleList({ groups, onSelect }: ArticleListProps) {
  if (groups.length === 0) {
    return (
      <p className="text-sm text-gray-500 dark:text-gray-400">
        표시할 기사가 없습니다.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {groups.map((group) => (
        <ArticleListItem
          key={group.articleTitle}
          group={group}
          onSelect={() => onSelect(group.articleTitle)}
        />
      ))}
    </div>
  );
}
