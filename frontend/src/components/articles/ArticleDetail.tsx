import { Fragment, useState } from "react";
import { articleDisplayDate, formatDate, type ArticleGroup } from "../../lib/articles";
import { VERDICT_COUNT_BOX_CLASS, VERDICT_ICON, verdictCountLabel } from "../../lib/verdictColors";
import { VerdictBadge } from "../claims/VerdictBadge";
import { ArticleTextViewer } from "./ArticleTextViewer";
import { InsightsPanel } from "./InsightsPanel";

interface ArticleDetailProps {
  group: ArticleGroup;
  articleText: string | undefined;
  tableOrgIds: Record<string, string>;
  articleDates: Record<string, string>;
  onBack: () => void;
}

const VERDICT_ORDER = ["일치", "불일치", "애매"] as const;

export function ArticleDetail({ group, articleText, tableOrgIds, articleDates, onBack }: ArticleDetailProps) {
  const counts = new Map<(typeof VERDICT_ORDER)[number], number>();
  for (const r of group.records) {
    const label = verdictCountLabel(r.verification_result);
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }

  return (
    <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-[1fr_320px]">
      <div className="flex flex-col gap-4">
        <div>
          <nav className="flex items-center gap-2 text-xs text-gray-400 dark:text-gray-500">
            <button
              type="button"
              onClick={onBack}
              className="inline-flex items-center gap-1 rounded-full border border-gray-200 bg-white px-3 py-1 font-medium text-gray-600 hover:border-indigo-200 hover:bg-indigo-50 hover:text-indigo-600 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-300 dark:hover:border-indigo-800 dark:hover:bg-indigo-900/30 dark:hover:text-indigo-400"
            >
              검증 기사 목록
            </button>
            <span>/</span>
            <span className="max-w-xs truncate">{group.articleTitle}</span>
          </nav>

          <h2 className="mt-2 text-2xl font-bold text-gray-950 dark:text-gray-50">
            {group.articleTitle}
          </h2>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            {group.articleUrl && (
              <a
                href={group.articleUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 rounded-full bg-indigo-50 px-2.5 py-1 text-xs font-medium text-indigo-600 hover:bg-indigo-100 dark:bg-indigo-900/30 dark:text-indigo-300"
              >
                🔗 원문 보기
              </a>
            )}
            {VERDICT_ORDER.filter((label) => counts.has(label)).map((label) => (
              <span
                key={label}
                className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-semibold ${VERDICT_COUNT_BOX_CLASS[label]}`}
              >
                <span>{VERDICT_ICON[label]}</span>
                {label} {counts.get(label)}건
              </span>
            ))}
          </div>
        </div>

        {articleText ? (
          <ArticleTextViewer
            articleText={articleText}
            claims={group.records}
            articleDate={formatDate(articleDisplayDate(group, articleDates))}
          />
        ) : (
          // 원문 텍스트를 못 구한 경우(예: 아직 export 전, 카탈로그 밖 시나리오)의 대체 화면 —
          // 하이라이트 없이 표로라도 결과를 보여준다.
          <ClaimTableFallback group={group} />
        )}
      </div>

      <InsightsPanel group={group} tableOrgIds={tableOrgIds} />
    </div>
  );
}

function ClaimTableFallback({ group }: { group: ArticleGroup }) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <div className="overflow-hidden rounded-2xl border border-gray-200 shadow-sm dark:border-gray-700">
      <table className="min-w-full divide-y divide-gray-200 text-sm dark:divide-gray-700">
        <thead className="bg-gray-50 dark:bg-gray-800">
          <tr>
            <th className="px-3 py-2 text-left font-medium text-gray-500 dark:text-gray-400">
              수치 주장
            </th>
            <th className="px-3 py-2 text-left font-medium text-gray-500 dark:text-gray-400">
              매칭된 통계표
            </th>
            <th className="px-3 py-2 text-left font-medium text-gray-500 dark:text-gray-400">
              판정
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200 bg-white dark:divide-gray-700 dark:bg-gray-900">
          {group.records.map((record) => {
            const isExpanded = expandedId === record.result_id;
            const detailText = record.evidence ?? record.ambiguity_reason;

            return (
              <Fragment key={record.result_id}>
                <tr
                  className={detailText ? "cursor-pointer" : ""}
                  onClick={() =>
                    detailText && setExpandedId(isExpanded ? null : record.result_id)
                  }
                >
                  <td className="max-w-md px-3 py-2 text-gray-900 dark:text-gray-100">
                    {record.claim_sentence}
                  </td>
                  <td className="px-3 py-2 text-gray-500 dark:text-gray-400">
                    {record.kosis_table ?? "—"}
                  </td>
                  <td className="px-3 py-2">
                    <VerdictBadge verdict={record.verification_result} />
                  </td>
                </tr>
                {isExpanded && detailText && (
                  <tr className="bg-gray-50 dark:bg-gray-800">
                    <td colSpan={3} className="px-3 py-3 text-sm text-gray-700 dark:text-gray-300">
                      {detailText}
                    </td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
