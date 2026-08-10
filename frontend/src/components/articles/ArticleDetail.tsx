import { Fragment, useState } from "react";
import type { ArticleGroup } from "../../lib/articles";
import { VerdictBadge } from "../claims/VerdictBadge";
import { ArticleTextViewer } from "./ArticleTextViewer";

interface ArticleDetailProps {
  group: ArticleGroup;
  articleText: string | undefined;
  onBack: () => void;
}

export function ArticleDetail({ group, articleText, onBack }: ArticleDetailProps) {
  return (
    <div className="flex flex-col gap-4">
      <div>
        <button
          type="button"
          onClick={onBack}
          className="text-sm text-gray-500 hover:text-gray-900 dark:text-gray-400 dark:hover:text-gray-100"
        >
          ← 목록으로
        </button>
        <h2 className="mt-2 text-2xl font-bold text-gray-950 dark:text-gray-50">
          {group.articleTitle}
        </h2>
        {group.articleUrl && (
          <a
            href={group.articleUrl}
            target="_blank"
            rel="noreferrer"
            className="text-xs text-blue-600 hover:underline dark:text-blue-400"
          >
            원문 보기
          </a>
        )}
      </div>

      {articleText ? (
        <ArticleTextViewer articleText={articleText} claims={group.records} />
      ) : (
        // 원문 텍스트를 못 구한 경우(예: 아직 export 전, 카탈로그 밖 시나리오)의 대체 화면 —
        // 하이라이트 없이 표로라도 결과를 보여준다.
        <ClaimTableFallback group={group} />
      )}
    </div>
  );
}

function ClaimTableFallback({ group }: { group: ArticleGroup }) {
  const [expandedId, setExpandedId] = useState<string | null>(null);

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200 dark:border-gray-700">
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
        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
          {group.records.map((record) => {
            const isExpanded = expandedId === record.result_id;
            const detailText = record.evidence ?? record.ambiguity_reason;

            return (
              <Fragment key={record.result_id}>
                <tr
                  className={`bg-white dark:bg-gray-900 ${detailText ? "cursor-pointer" : ""}`}
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
