import { useMemo, useState } from "react";
import {
  dominantVerdict,
  dominantVerdictCount,
  latestCreatedAt,
  type ArticleGroup,
} from "../../lib/articles";
import { VERDICT_ACCENT_BORDER_CLASS, VERDICT_COUNT_BOX_CLASS, VERDICT_ICON } from "../../lib/verdictColors";

interface ArticleListPanelProps {
  groups: ArticleGroup[];
  articleDates: Record<string, string>;
  onSelect: (articleTitle: string) => void;
}

type FilterTab = "전체" | "불일치" | "검토 필요";
type SortMode = "심각도순" | "최신순";

const FILTER_TABS: FilterTab[] = ["전체", "불일치", "검토 필요"];
const SORT_MODES: SortMode[] = ["심각도순", "최신순"];
const SEVERITY_RANK: Record<"불일치" | "애매" | "일치", number> = { 불일치: 0, 애매: 1, 일치: 2 };

function formatDate(iso: string): string {
  return iso ? iso.slice(0, 10).replaceAll("-", ".") : "—";
}

export function ArticleListPanel({ groups, articleDates, onSelect }: ArticleListPanelProps) {
  const [filter, setFilter] = useState<FilterTab>("전체");
  const [sortMode, setSortMode] = useState<SortMode>("심각도순");

  // 기사 작성일(articleDates, data_set.csv 기반)을 우선 쓰고, 없으면 검증 실행 시각으로
  // 대신한다 — "최근 업데이트"(검증한 시점)가 아니라 "기사 자체의 날짜"를 보여주기 위함.
  const groupDate = (group: ArticleGroup) => articleDates[group.articleTitle] ?? latestCreatedAt(group);

  const filtered = useMemo(() => {
    if (filter === "전체") return groups;
    if (filter === "불일치") return groups.filter((g) => dominantVerdict(g) === "불일치");
    return groups.filter((g) => dominantVerdict(g) !== "일치");
  }, [groups, filter]);

  const sorted = useMemo(() => {
    const copy = [...filtered];
    if (sortMode === "심각도순") {
      copy.sort((a, b) => SEVERITY_RANK[dominantVerdict(a)] - SEVERITY_RANK[dominantVerdict(b)]);
    } else {
      copy.sort((a, b) => (groupDate(b) > groupDate(a) ? 1 : -1));
    }
    return copy;
  }, [filtered, sortMode, articleDates]);

  return (
    <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-900">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-gray-100 px-5 py-4 dark:border-gray-800">
        <h2 className="text-sm font-semibold text-gray-900 dark:text-gray-100">검증 기사 목록</h2>
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex gap-1 rounded-lg bg-gray-100 p-1 dark:bg-gray-800">
            {FILTER_TABS.map((tab) => (
              <button
                key={tab}
                type="button"
                onClick={() => setFilter(tab)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                  filter === tab
                    ? "bg-white text-indigo-600 shadow-sm dark:bg-gray-700 dark:text-indigo-300"
                    : "text-gray-500 hover:text-gray-700 dark:text-gray-400"
                }`}
              >
                {tab}
              </button>
            ))}
          </div>
          <select
            value={sortMode}
            onChange={(e) => setSortMode(e.target.value as SortMode)}
            className="rounded-lg border border-gray-200 bg-white px-2 py-1.5 text-xs font-medium text-gray-600 dark:border-gray-700 dark:bg-gray-800 dark:text-gray-300"
          >
            {SORT_MODES.map((mode) => (
              <option key={mode} value={mode}>
                {mode}
              </option>
            ))}
          </select>
        </div>
      </div>

      {sorted.length === 0 ? (
        <p className="p-6 text-sm text-gray-500 dark:text-gray-400">표시할 기사가 없습니다.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[560px] text-left text-sm">
            <thead>
              <tr className="border-b border-gray-100 text-xs text-gray-400 dark:border-gray-800">
                <th className="px-5 py-2 font-medium">심각도</th>
                <th className="px-3 py-2 font-medium">기사 제목</th>
                <th className="px-3 py-2 font-medium">주장 수</th>
                <th className="px-3 py-2 font-medium">기사 작성일</th>
                <th className="px-5 py-2 font-medium" />
              </tr>
            </thead>
            <tbody>
              {sorted.map((group) => {
                const verdict = dominantVerdict(group);
                return (
                  <tr
                    key={group.articleTitle}
                    onClick={() => onSelect(group.articleTitle)}
                    className={`cursor-pointer border-b border-l-4 border-gray-100 transition last:border-b-0 hover:bg-gray-50 dark:border-gray-800 dark:hover:bg-gray-800/60 ${VERDICT_ACCENT_BORDER_CLASS[verdict]}`}
                  >
                    <td className="px-5 py-3">
                      <span
                        className={`inline-flex items-center gap-1 rounded-full px-2 py-1 text-xs font-semibold ${VERDICT_COUNT_BOX_CLASS[verdict]}`}
                      >
                        <span>{VERDICT_ICON[verdict]}</span>
                        {dominantVerdictCount(group)}
                      </span>
                    </td>
                    <td className="max-w-xs truncate px-3 py-3 font-medium text-gray-900 dark:text-gray-100">
                      {group.articleTitle}
                    </td>
                    <td className="px-3 py-3 whitespace-nowrap text-gray-500 dark:text-gray-400">
                      총 {group.records.length}건
                    </td>
                    <td className="px-3 py-3 whitespace-nowrap text-gray-500 dark:text-gray-400">
                      {formatDate(groupDate(group))}
                    </td>
                    <td className="px-5 py-3 text-right text-gray-300 dark:text-gray-600">→</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
