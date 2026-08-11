import { Fragment, useMemo, useState } from "react";
import type { VerificationRecord } from "../../types/verification";
import { buildArticleSegments, groupSegmentsByParagraph } from "../../lib/segments";
import { cleanArticleTextForDisplay } from "../../lib/cleanArticleText";
import { VERDICT_HIGHLIGHT_CLASS, VERDICT_MARKER_CLASS, verdictLabel } from "../../lib/verdictColors";
import { VerdictBadge } from "../claims/VerdictBadge";

interface ArticleTextViewerProps {
  articleText: string;
  claims: VerificationRecord[];
}

interface HighlightedClaimProps {
  content: string;
  record: VerificationRecord;
  number: number | undefined;
  onClick: () => void;
}

// 하이라이트 텍스트와 번호 배지를 한 세트로 보이게 렌더링한다. 번호를 인라인 텍스트로
// 넣으면(예전 <sup>) 하이라이트와 분리된 별개 요소처럼 보였어서, 배지를 절대 위치로
// 하이라이트 바로 위에 띄우고 같은 판정 색을 써서 시각적으로 한 덩어리로 묶었다.
function HighlightedClaim({ content, record, number, onClick }: HighlightedClaimProps) {
  const label = verdictLabel(record.verification_result);
  return (
    <span className="relative">
      <span
        aria-hidden="true"
        className={`absolute -top-2.5 -left-1 flex h-3.5 w-3.5 items-center justify-center rounded-full text-[9px] leading-none font-bold ${VERDICT_MARKER_CLASS[label]}`}
      >
        {number}
      </span>
      <span
        onClick={onClick}
        className={`cursor-pointer rounded px-0.5 ${VERDICT_HIGHLIGHT_CLASS[label]}`}
      >
        {content}
      </span>
    </span>
  );
}

// 기사 원문 전체를 그대로 보여주고, 그 안에서 실제로 검증한 수치 주장 부분만 판정 색으로
// 밑줄/하이라이트한다. 클릭하면 그 자리에서 바로 아래에 판정 근거 카드가 펼쳐진다.
// 이렇게 원문 전체를 보여주는 이유: 2단계(claim_extractor)가 기사 속 수치 주장을 빠짐없이
// 다 뽑았는지 사람이 눈으로 바로 확인할 수 있게 하기 위함 — 하이라이트 안 된 수치가 원문에
// 남아있으면 그게 곧 추출 누락(recall) 사례다.
export function ArticleTextViewer({ articleText, claims }: ArticleTextViewerProps) {
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const cleanedText = useMemo(() => cleanArticleTextForDisplay(articleText), [articleText]);
  const { segments, unmatched } = useMemo(
    () => buildArticleSegments(cleanedText, claims),
    [cleanedText, claims],
  );
  const paragraphs = useMemo(() => groupSegmentsByParagraph(segments), [segments]);

  // 몇 번째 주장인지 한눈에 보이게 claims 배열 순서대로 1부터 번호를 매긴다 (원문 하이라이트,
  // 매칭 안 된 목록 둘 다 같은 번호 체계를 공유).
  const claimNumbers = useMemo(() => {
    const map = new Map<string, number>();
    claims.forEach((record, i) => map.set(record.result_id, i + 1));
    return map;
  }, [claims]);

  return (
    <div className="flex flex-col gap-4">
      <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm dark:border-gray-700 dark:bg-gray-900">
        <div className="flex items-center gap-2 border-b border-gray-100 px-6 py-4 dark:border-gray-800">
          <span className="h-2 w-2 rounded-full bg-gradient-to-br from-indigo-500 to-fuchsia-500" />
          <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">기사 원문</span>
        </div>
        <div className="p-6 text-gray-900 dark:text-gray-100">
          {paragraphs.map((paragraphSegments, pIndex) => (
          // <p> 대신 <div>를 쓰는 이유: 아래 판정 카드(<div>, block 요소)가 주장을 클릭하면
          // 문단 중간에 끼어드는데, <p> 안에는 block 요소를 넣을 수 없어 브라우저가 <p>를
          // 강제로 끊어버려 레이아웃이 어긋난다. <div>는 이 제약이 없다.
          <div key={pIndex} className="mb-5 leading-loose last:mb-0">
            {paragraphSegments.map((segment, i) => {
              if (segment.type === "text") {
                return <span key={i}>{segment.content}</span>;
              }

              const { record } = segment;
              const isExpanded = expandedId === record.result_id;

              return (
                <Fragment key={record.result_id}>
                  <HighlightedClaim
                    content={segment.content}
                    record={record}
                    number={claimNumbers.get(record.result_id)}
                    onClick={() => setExpandedId(isExpanded ? null : record.result_id)}
                  />
                  {isExpanded && (
                    <div className="my-3 block rounded-lg border border-gray-200 bg-gray-50 p-4 leading-normal dark:border-gray-700 dark:bg-gray-800">
                      <VerdictBadge verdict={record.verification_result} />
                      <p className="mt-2 text-sm text-gray-700 dark:text-gray-300">
                        {record.evidence ?? record.ambiguity_reason ?? "판정 근거가 기록되지 않았습니다."}
                      </p>
                      {record.kosis_table && (
                        <p className="mt-2 border-t border-gray-200 pt-2 text-xs text-gray-500 dark:border-gray-700 dark:text-gray-400">
                          KOSIS 참조: {record.kosis_table}
                        </p>
                      )}
                    </div>
                  )}
                </Fragment>
              );
            })}
          </div>
          ))}
        </div>
      </div>

      {unmatched.length > 0 && (
        <div className="rounded-lg border border-dashed border-gray-300 p-4 text-sm dark:border-gray-600">
          <p className="mb-3 text-xs text-gray-500 dark:text-gray-400">
            원문에서 정확히 위치를 못 찾은 주장 {unmatched.length}건 (표기 차이로 추정, 원문 안에는
            표시 못 해서 여기 따로 모음):
          </p>
          <ul className="flex flex-col gap-2">
            {unmatched.map((record) => {
              const isExpanded = expandedId === record.result_id;
              const detailText = record.evidence ?? record.ambiguity_reason;
              return (
                <li key={record.result_id}>
                  <HighlightedClaim
                    content={record.claim_sentence}
                    record={record}
                    number={claimNumbers.get(record.result_id)}
                    onClick={() => detailText && setExpandedId(isExpanded ? null : record.result_id)}
                  />
                  {isExpanded && detailText && (
                    <div className="mt-2 rounded-lg border border-gray-200 bg-gray-50 p-4 dark:border-gray-700 dark:bg-gray-800">
                      <VerdictBadge verdict={record.verification_result} />
                      <p className="mt-2 text-sm text-gray-700 dark:text-gray-300">{detailText}</p>
                      {record.kosis_table && (
                        <p className="mt-2 border-t border-gray-200 pt-2 text-xs text-gray-500 dark:border-gray-700 dark:text-gray-400">
                          KOSIS 참조: {record.kosis_table}
                        </p>
                      )}
                    </div>
                  )}
                </li>
              );
            })}
          </ul>
        </div>
      )}
    </div>
  );
}
