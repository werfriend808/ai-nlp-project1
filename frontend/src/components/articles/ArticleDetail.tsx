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

const VERDICT_ORDER = ["불일치", "애매", "일치"] as const;

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
          <nav className="flex items-center gap-2 text-xs text-stone-400 dark:text-stone-500">
            <button
              type="button"
              onClick={onBack}
              className="inline-flex items-center gap-1 rounded-full border border-stone-200 bg-white px-3 py-1 font-medium text-stone-600 hover:border-stone-300 hover:bg-stone-100 hover:text-stone-700 dark:border-stone-700 dark:bg-stone-900 dark:text-stone-300 dark:hover:border-stone-600 dark:hover:bg-stone-800/50 dark:hover:text-stone-300"
            >
              ← 처음으로
            </button>
            <span>/</span>
            <span className="max-w-xs truncate">{group.articleTitle}</span>
          </nav>

          <h2 className="mt-2 text-2xl font-bold text-stone-950 dark:text-stone-50">
            {group.articleTitle}
          </h2>

          <div className="mt-3 flex flex-wrap items-center gap-2">
            {group.articleUrl && (
              <a
                href={group.articleUrl}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 rounded-full bg-stone-100 px-2.5 py-1 text-xs font-medium text-stone-700 hover:bg-stone-200 dark:bg-stone-800/50 dark:text-stone-300"
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
    <div className="overflow-hidden rounded-2xl border border-stone-200 shadow-sm dark:border-stone-700">
      <table className="min-w-full divide-y divide-stone-200 text-sm dark:divide-stone-700">
        <thead className="bg-stone-50 dark:bg-stone-800">
          <tr>
            <th className="px-3 py-2 text-left font-medium text-stone-500 dark:text-stone-400">
              수치 주장
            </th>
            <th className="px-3 py-2 text-left font-medium text-stone-500 dark:text-stone-400">
              매칭된 통계표
            </th>
            <th className="px-3 py-2 text-left font-medium text-stone-500 dark:text-stone-400">
              판정
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-stone-200 bg-white dark:divide-stone-700 dark:bg-stone-900">
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
                  <td className="max-w-md px-3 py-2 text-stone-900 dark:text-stone-100">
                    {record.claim_sentence}
                  </td>
                  <td className="px-3 py-2 text-stone-500 dark:text-stone-400">
                    {record.kosis_table ?? "—"}
                  </td>
                  <td className="px-3 py-2">
                    <VerdictBadge verdict={record.verification_result} />
                  </td>
                </tr>
                {isExpanded && detailText && (
                  <tr className="bg-stone-50 dark:bg-stone-800">
                    <td colSpan={3} className="px-3 py-3 text-sm text-stone-700 dark:text-stone-300">
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
