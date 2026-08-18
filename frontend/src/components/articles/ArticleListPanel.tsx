import { useEffect, useMemo, useState } from "react";
import {
  articleDisplayDate,
  dominantVerdict,
  dominantVerdictCount,
  formatDate,
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
const PAGE_SIZE = 20;
const MAX_PAGE_BUTTONS = 10;

// 페이지 번호 버튼을 최대 MAX_PAGE_BUTTONS개까지만 보여준다. 전체 페이지가 그보다 많으면
// 현재 페이지가 가운데쯤 오도록 윈도우를 옮긴다 (예: 25페이지 중 20페이지에 있으면
// 16~25 같은 식으로 뒤쪽 구간을 보여줌).
function pageWindow(current: number, total: number, max: number): number[] {
  if (total <= max) return Array.from({ length: total }, (_, i) => i + 1);
  const half = Math.floor(max / 2);
  let start = Math.max(1, current - half);
  const end = Math.min(total, start + max - 1);
  start = Math.max(1, end - max + 1);
  return Array.from({ length: end - start + 1 }, (_, i) => start + i);
}

export function ArticleListPanel({ groups, articleDates, onSelect }: ArticleListPanelProps) {
  const [filter, setFilter] = useState<FilterTab>("전체");
  const [sortMode, setSortMode] = useState<SortMode>("심각도순");
  const [page, setPage] = useState(1);

  // 기사 작성일(articleDates, data_set.csv 기반)을 우선 쓰고, 없으면 검증 실행 시각으로
  // 대신한다 — "최근 업데이트"(검증한 시점)가 아니라 "기사 자체의 날짜"를 보여주기 위함.
  const groupDate = (group: ArticleGroup) => articleDisplayDate(group, articleDates);

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

  // 필터/정렬이 바뀌면 목록 자체가 달라지니 1페이지로 되돌린다 — 안 그러면 예를 들어
  // 3페이지를 보다가 필터를 바꿨을 때 존재하지 않는 페이지에 멈춰있을 수 있다.
  useEffect(() => {
    setPage(1);
  }, [filter, sortMode]);

  const totalPages = Math.max(1, Math.ceil(sorted.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const paged = sorted.slice((currentPage - 1) * PAGE_SIZE, currentPage * PAGE_SIZE);
  const pageNumbers = pageWindow(currentPage, totalPages, MAX_PAGE_BUTTONS);

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
              {paged.map((group) => {
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

      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-1 border-t border-gray-100 px-5 py-3 dark:border-gray-800">
          <button
            type="button"
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="flex h-7 w-7 items-center justify-center rounded-md border border-gray-200 text-gray-400 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-30 dark:border-gray-700 dark:hover:bg-gray-800"
          >
            &lsaquo;
          </button>
          {pageNumbers.map((n) => (
            <button
              key={n}
              type="button"
              onClick={() => setPage(n)}
              className={`flex h-7 w-7 items-center justify-center rounded-md text-xs font-medium transition ${
                n === currentPage
                  ? "bg-gray-100 text-gray-900 dark:bg-gray-800 dark:text-gray-100"
                  : "text-gray-400 hover:bg-gray-50 dark:text-gray-500 dark:hover:bg-gray-800"
              }`}
            >
              {n}
            </button>
          ))}
          <button
            type="button"
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="flex h-7 w-7 items-center justify-center rounded-md border border-gray-200 text-gray-400 transition hover:bg-gray-50 disabled:cursor-not-allowed disabled:opacity-30 dark:border-gray-700 dark:hover:bg-gray-800"
          >
            &rsaquo;
          </button>
        </div>
      )}
    </div>
  );
}
