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
// 2026-08-21 실측 확인: claim_extractor에 넘어가는 article_text가 일부 기사에서 제목과
// 본문이 구분자 없이 붙어있어서, claim_sentence 앞에 article_title이 그대로 접두어로
// 붙는 경우가 있었다(예: "3월 청년 실업률 7.5%… 4년 만에 최대치 기록 청년 실업률이
// 지난달 7.5%까지 치솟으며..." — 앞부분이 article_title과 완전히 동일). 원문(표시용
// articleText)엔 제목이 안 보이니 그대로는 못 찾는다 — article_title이 있고 claim이
// 그걸로 시작하면 그 접두어를 떼고 다시 찾아본다(백엔드를 안 고쳐도 화면에선 정상
// 노출되게 하는 방어적 처리).
function stripTitlePrefix(sentence: string, articleTitle: string | undefined): string {
  if (!articleTitle) return sentence;
  const normalizedTitle = normalizeQuotes(articleTitle).trim();
  const normalizedSentence = normalizeQuotes(sentence);
  if (normalizedTitle && normalizedSentence.startsWith(normalizedTitle)) {
    return sentence.slice(articleTitle.length).trimStart();
  }
  return sentence;
}

const REGEX_SPECIAL_RE = /[.*+?^${}()|[\]\\]/g;

// claim_sentence와 원문 사이의 공백 표기 차이(괄호 앞뒤 띄어쓰기, "60kg선" vs "60kg 선"
// 등)만으로 매칭에 실패하는 사례가 실측 확인됐다(2026-08-31, 데모 프리셋 row294에서
// 재현 — "8.6kg로 1년 전 대비 4.9%(0.4kg)증가했다"(원문, 공백 없음) vs claim_sentence의
// "...4.9%(0.4kg) 증가했다"(공백 있음)). 예전엔 indexOf로 완전 일치만 찾아서 이런 사소한
// 공백 차이도 "원문에서 못 찾음"으로 빠졌다. sentence를 정규식으로 변환하되 공백 연속
// 구간만 "\s*"(0개 이상 허용)로 느슨하게 바꿔서, 나머지 글자는 원문과 정확히 같은 것만
// 찾는다 — 오탐(엉뚱한 자리 매칭) 위험 없이 공백 차이만 흡수한다.
function buildFlexibleRegex(sentence: string): RegExp {
  const escaped = sentence.replace(REGEX_SPECIAL_RE, "\\$&").replace(/\s+/g, "\\s*");
  return new RegExp(escaped);
}

function findFlexible(article: string, sentence: string): { index: number; length: number } | null {
  const exact = article.indexOf(sentence);
  if (exact !== -1) return { index: exact, length: sentence.length };

  const match = buildFlexibleRegex(sentence).exec(article);
  if (!match) return null;
  return { index: match.index, length: match[0].length };
}

export function buildArticleSegments(
  articleText: string,
  claims: VerificationRecord[],
): {
  segments: ArticleSegment[];
  unmatched: VerificationRecord[];
  notFound: VerificationRecord[];
  overlapSkipped: VerificationRecord[];
  claimNumbers: Map<string, number>;
} {
  const normalizedArticle = normalizeQuotes(articleText);
  const articleTitle = claims[0]?.article_title;

  const located = claims
    .map((record) => {
      const stripped = stripTitlePrefix(record.claim_sentence, articleTitle);
      const found = findFlexible(normalizedArticle, normalizeQuotes(stripped));
      return { record, index: found ? found.index : -1, length: found ? found.length : 0 };
    })
    .filter((m) => m.index !== -1)
    .sort((a, b) => a.index - b.index);

  const notFound = claims.filter((c) => !located.some((m) => m.record === c));

  const segments: ArticleSegment[] = [];
  const overlapSkipped: VerificationRecord[] = [];
  const rendered: VerificationRecord[] = [];
  // 인라인에 그려진 각 구간의 시작 위치 -> 판정. 나중에 겹치는 주장이 나왔을 때, 시작
  // 위치가 완전히 같고 판정도 같으면 "표기 차이"가 아니라 순수 중복 추출(같은 문장이
  // claim_extractor에서 두 번 뽑힘)이라 화면에 또 보여줄 필요가 없다 — 조용히 버린다.
  // 시작 위치는 같은데 판정이 다르면(진짜 의견 차이) 계속 보여준다.
  const renderedByIndex = new Map<number, VerificationRecord["verification_result"]>(); // index -> verification_result
  let cursor = 0;
  for (const { record, index, length } of located) {
    if (index < cursor) {
      const sameSpotVerdict = renderedByIndex.get(index);
      if (sameSpotVerdict !== undefined && sameSpotVerdict === record.verification_result) {
        continue; // 순수 중복 추출 — 조용히 스킵(목록에도 안 보여줌)
      }
      overlapSkipped.push(record); // 다른 주장과 진짜로 겹치는 구간 — 인라인 대신 목록으로
      continue;
    }
    if (index > cursor) {
      segments.push({ type: "text", content: articleText.slice(cursor, index) });
    }
    const end = index + length;
    segments.push({ type: "claim", content: articleText.slice(index, end), record });
    cursor = end;
    rendered.push(record);
    renderedByIndex.set(index, record.verification_result);
  }
  if (cursor < articleText.length) {
    segments.push({ type: "text", content: articleText.slice(cursor) });
  }

  const unmatched = [...notFound, ...overlapSkipped];

  const claimNumbers = new Map<string, number>();
  rendered.forEach((record, i) => claimNumbers.set(record.result_id, i + 1));
  unmatched.forEach((record, i) => claimNumbers.set(record.result_id, rendered.length + i + 1));

  // notFound(원문에서 문장 자체를 못 찾음)와 overlapSkipped(위치는 찾았지만 다른 주장과
  // 겹쳐서 인라인엔 못 그림)는 원인이 달라서 화면에서 다른 문구로 안내해야 한다 —
  // 둘 다 "표기 차이로 추정"이라고 뭉뚱그리면 겹침 케이스엔 안 맞는 설명이 된다.
  return { segments, unmatched, notFound, overlapSkipped, claimNumbers };
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
