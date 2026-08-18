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
// 세그먼트 배열로 쪼갠다. 원문에서 못 찾은 주장(공백 표기 차이 등)이나, 다른 주장과 겹쳐서
// 인라인으로는 못 그리는 주장은 결과에서 빠지고, 호출부(ArticleTextViewer)가 별도로
// "매칭 안 된 주장" 목록으로 보여준다 — 어떤 판정도 화면에서 완전히 사라지면 안 된다.
//
// "겹침"은 실제로 재현된 문제였다: 같은 사실이 살짝 다른 문장으로 두 번 추출되면(예: "합계
// 출산율도 작년 0.75명으로..."가 "여성 한 명이 낳을 것으로 예상되는 아이 수를 뜻하는
// 합계출산율도 작년 0.75명으로..."의 부분 문자열인 경우) 원래는 뒤엣것을 그냥 건너뛰기만
// 해서, 두 claim이 서로 다른 판정(하나는 일치, 하나는 불일치)인데도 뒤엣것의 하이라이트가
// 조용히 통째로 사라졌다(인라인에도 안 보이고 "매칭 안 된 주장" 목록에도 안 들어감 — 위치
// 자체는 찾았으니 "못 찾은 주장"도 아니었기 때문). 이제 겹쳐서 인라인에 못 그린 주장도
// unmatched에 포함시켜서 최소한 어딘가엔 보이게 한다.
//
// claimNumbers: 원문에 실제로 등장하는 순서(위→아래, index 기준)대로 1부터 번호를 매긴
// Map — 예전엔 호출부(ArticleTextViewer)가 claims 배열 순서(=API/DB가 준 순서, 원문
// 등장 순서와 무관)로 번호를 매겨서, 하이라이트는 원문 순서대로 보이는데 번호(①②③...)는
// 그 순서를 안 따라가는 버그가 있었다(실측 확인: 화면엔 ③①④⑤⑥ 순으로 찍힘). 여기서
// 실제로 인라인에 그려지는 순서 그대로 번호를 매기면 하이라이트 순서와 번호가 항상
// 일치한다. 인라인에 못 그리는(못 찾음 + 겹침) 주장은 위치가 없으니, 매칭된 것들 뒤에
// 이어서(원래 배열 순서대로) 번호를 붙인다.
export function buildArticleSegments(
  articleText: string,
  claims: VerificationRecord[],
): { segments: ArticleSegment[]; unmatched: VerificationRecord[]; claimNumbers: Map<string, number> } {
  const normalizedArticle = normalizeQuotes(articleText);

  const located = claims
    .map((record) => ({
      record,
      index: normalizedArticle.indexOf(normalizeQuotes(record.claim_sentence)),
    }))
    .filter((m) => m.index !== -1)
    .sort((a, b) => a.index - b.index);

  const notFound = claims.filter((c) => !located.some((m) => m.record === c));

  const segments: ArticleSegment[] = [];
  const overlapSkipped: VerificationRecord[] = [];
  const rendered: VerificationRecord[] = [];
  let cursor = 0;
  for (const { record, index } of located) {
    if (index < cursor) {
      overlapSkipped.push(record); // 다른 주장과 겹치는 구간 — 인라인 대신 목록으로
      continue;
    }
    if (index > cursor) {
      segments.push({ type: "text", content: articleText.slice(cursor, index) });
    }
    const end = index + record.claim_sentence.length;
    segments.push({ type: "claim", content: articleText.slice(index, end), record });
    cursor = end;
    rendered.push(record);
  }
  if (cursor < articleText.length) {
    segments.push({ type: "text", content: articleText.slice(cursor) });
  }

  const unmatched = [...notFound, ...overlapSkipped];

  const claimNumbers = new Map<string, number>();
  rendered.forEach((record, i) => claimNumbers.set(record.result_id, i + 1));
  unmatched.forEach((record, i) => claimNumbers.set(record.result_id, rendered.length + i + 1));

  return { segments, unmatched, claimNumbers };
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
