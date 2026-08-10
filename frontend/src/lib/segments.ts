import type { VerificationRecord } from "../types/verification";

export interface TextSegment {
  type: "text";
  content: string;
}

export interface ClaimSegment {
  type: "claim";
  content: string;
  record: VerificationRecord;
}

export type ArticleSegment = TextSegment | ClaimSegment;

// claim_sentence는 HCX가 원문에서 뽑아준 값이라 대부분 원문과 정확히 같지만, 둥근따옴표를
// 곧은따옴표로 바꿔 출력하는 경우가 실측 확인됨(예: '하드 케이스' vs '하드 케이스'). 문자
// 개수가 같은 1:1 치환이라 정규화된 문자열에서 찾은 인덱스를 원문에 그대로 써도 어긋나지
// 않는다.
function normalizeQuotes(text: string): string {
  return text.replace(/[‘’]/g, "'").replace(/[“”]/g, '"');
}

// 기사 원문 안에서 각 claim_sentence의 위치를 찾아, 평문/주장 구간이 번갈아 나오는
// 세그먼트 배열로 쪼갠다. 원문에서 못 찾은 주장(공백 표기 차이 등)은 결과에서 빠지고,
// 호출부(ArticleTextViewer)가 별도로 "매칭 안 된 주장" 목록으로 보여준다.
export function buildArticleSegments(
  articleText: string,
  claims: VerificationRecord[],
): { segments: ArticleSegment[]; unmatched: VerificationRecord[] } {
  const normalizedArticle = normalizeQuotes(articleText);

  const located = claims
    .map((record) => ({
      record,
      index: normalizedArticle.indexOf(normalizeQuotes(record.claim_sentence)),
    }))
    .filter((m) => m.index !== -1)
    .sort((a, b) => a.index - b.index);

  const unmatched = claims.filter((c) => !located.some((m) => m.record === c));

  const segments: ArticleSegment[] = [];
  let cursor = 0;
  for (const { record, index } of located) {
    if (index < cursor) continue; // 다른 주장과 겹치는 구간이면 건너뛴다 (이중 하이라이트 방지)
    if (index > cursor) {
      segments.push({ type: "text", content: articleText.slice(cursor, index) });
    }
    const end = index + record.claim_sentence.length;
    segments.push({ type: "claim", content: articleText.slice(index, end), record });
    cursor = end;
  }
  if (cursor < articleText.length) {
    segments.push({ type: "text", content: articleText.slice(cursor) });
  }

  return { segments, unmatched };
}

// 세그먼트 배열을 빈 줄(\n\n 이상) 기준으로 문단 단위 배열로 재구성한다. claim 세그먼트는
// 항상 하나의 문단에만 속하게(중간에 끊기지 않게) 그대로 옮기고, text 세그먼트만 내부의
// 빈 줄 위치에서 새 문단으로 쪼갠다. 렌더링 쪽(ArticleTextViewer)이 문단마다 별도 블록으로
// 그려서 실제 기사처럼 문단 간격을 줄 수 있게 하기 위한 것 — whitespace-pre-wrap만으로는
// 문단 사이 간격이 줄바꿈 한 줄 높이만큼만 생겨 너무 빽빽해 보이는 문제가 있었다.
export function groupSegmentsByParagraph(segments: ArticleSegment[]): ArticleSegment[][] {
  const paragraphs: ArticleSegment[][] = [[]];

  for (const segment of segments) {
    if (segment.type === "claim") {
      paragraphs[paragraphs.length - 1].push(segment);
      continue;
    }
    const parts = segment.content.split(/\n{2,}/);
    parts.forEach((part, i) => {
      if (i > 0) paragraphs.push([]);
      if (part) paragraphs[paragraphs.length - 1].push({ type: "text", content: part });
    });
  }

  return paragraphs.filter((p) => p.length > 0);
}
