// 배치 파이프라인의 _clean_scraped_article_text(agent/pipeline/batch_runner.py)는 "제목
// 위치부터 3000자"만 잘라내서 화면 표시용 잡음(내비게이션 메뉴 반복, 광고/추천 콘텐츠,
// 비디오 플레이어 자막 UI 텍스트 등)은 그대로 남아있다. 이 함수는 그 잡음을 "화면에
// 보여줄 때만" 추가로 걷어낸다 — 실제 파이프라인이 모델에 넘긴 텍스트(article_text 원본)는
// 건드리지 않고, 표시 직전에만 적용한다.
//
// 검증(2026-08-10, 실제 export된 기사 16건 기준): 평균 길이 2726자 → 1714자(-37%)로
// 줄었고, 하이라이트 대상 주장 33건의 매칭 여부는 정리 전후로 동일하게 유지됨(회귀 없음).
const AD_TRAILER_RE = /By Taboola/;
// 타임스탬프 뒤에 댓글수 등 UI 배지 숫자가 하나 더 붙어 스크랩되는 경우가 실측 확인됨
// (예: "…06:07 0 지난달 생산자물가가…", 배지 값은 기사마다 다름 — 0/1/4/23 등). 잘라낼
// 기준 위치를 그 숫자까지 포함해서 잡아야 "0"이 본문 맨 앞에 남는 문제가 안 생긴다.
const BYLINE_TIMESTAMP_RE =
  /입력\s*\d{4}\.\d{2}\.\d{2}\.\s*\d{2}:\d{2}(\s*업데이트\s*\d{4}\.\d{2}\.\d{2}\.\s*\d{2}:\d{2})?(\s*\d+)?/g;
// video.js류 임베드 플레이어가 스크린리더용으로 넣는 자막 설정 UI 텍스트 — 사이트가
// 달라도 이 문구 자체는 라이브러리 공통이라 거의 그대로 반복된다.
const VIDEO_WIDGET_RE = /Video Player is loading\.[\s\S]*?End of dialog window\./g;

// 뉴스 사이트가 "AI 추천"/관련기사 teaser 블록을 통째로 2~3번 반복해서 내보내는 경우가
// 실측 확인됨. 특정 문구를 하드코딩하지 않고, 인접한 두 구간이 글자 단위로 완전히
// 동일하면 하나만 남기는 일반적인 방식으로 잡는다 (긴 블록부터 시도해서 우연한 짧은
// 반복 어구까지 지워지지 않게 함).
function collapseRepeatedBlocks(text: string, minLen = 20, maxLen = 300): string {
  let result = "";
  let i = 0;
  const n = text.length;

  while (i < n) {
    let collapsed = false;
    for (let len = Math.min(maxLen, Math.floor((n - i) / 2)); len >= minLen; len--) {
      const a = text.slice(i, i + len);
      const b = text.slice(i + len, i + len * 2);
      if (a === b) {
        let end = i + len;
        while (text.slice(end, end + len) === a) {
          end += len;
        }
        result += a;
        i = end;
        collapsed = true;
        break;
      }
    }
    if (!collapsed) {
      result += text[i];
      i += 1;
    }
  }

  return result;
}

// 문장 끝(.?!) 뒤 공백 다음이 숫자가 아닐 때만 경계로 본다 — "46.1%로 2023년(46.9%)"처럼
// 소수점 뒤에 곧바로 숫자/기호가 이어지는 경우(공백이 아예 없어 애초에 안 걸림)와, "…9,384,325명"
// 뒤에 공백 없이 이어지는 경우는 그대로 안전하다. 공백 뒤가 숫자로 시작하는 문장(드묾)은
// 못 쪼개고 넘어가지만, 문장 중간을 잘못 끊는 것보다 덜 쪼개는 쪽이 안전하다.
const SENTENCE_BOUNDARY_RE = /(?<=[.?!])\s+(?=[^\d])/;
const PARAGRAPH_GROUP_SIZE = 3;

// data_set.csv 스크랩 본문은 문단 구분(줄바꿈) 없이 완전히 한 줄로 저장돼 있다(팀 제공
// 원본 자체가 그렇다, 2026-08-22 확인). 프론트가 실시간으로 기사 URL에서 다시 긁어와
// 원래 문단(db/fetch_article_text.py, Fusion CMS는 \n\n 유지) 텍스트를 못 가져온 경우
// (URL 만료/네트워크 실패 등)엔 이 CSV 텍스트로 그대로 폴백하는데, 그러면 화면에 문단
// 구분 없는 벽처럼 보인다. 원래 문단 경계를 복원할 방법은 없으니(정보 자체가 유실됨),
// 대신 문장 3개 단위로 묶어 읽기 편하게라도 쪼갠다 — "진짜 문단"은 아니지만 완전히
// 이어붙은 것보다는 훨씬 읽기 쉽다.
function reconstructParagraphs(text: string): string {
  const sentences = text.split(SENTENCE_BOUNDARY_RE);
  if (sentences.length <= 1) return text;

  const paragraphs: string[] = [];
  for (let i = 0; i < sentences.length; i += PARAGRAPH_GROUP_SIZE) {
    paragraphs.push(sentences.slice(i, i + PARAGRAPH_GROUP_SIZE).join(" "));
  }
  return paragraphs.join("\n\n");
}

export function cleanArticleTextForDisplay(text: string): string {
  const adMatch = text.match(AD_TRAILER_RE);
  let trimmed = adMatch?.index !== undefined ? text.slice(0, adMatch.index) : text;

  trimmed = trimmed.replace(VIDEO_WIDGET_RE, " ");

  // 여러 언론사가 공통으로 쓰는 "입력 YYYY.MM.DD. HH:MM" 바이라인 표기 뒤부터가 실제
  // 본문 시작인 경우가 많다. 선두 내비게이션 메뉴가 반복되며 이 패턴이 여러 번 걸릴 수
  // 있어, 가장 마지막(=본문에 가장 가까운) 위치를 기준으로 그 이전을 잘라낸다.
  const matches = [...trimmed.matchAll(BYLINE_TIMESTAMP_RE)];
  if (matches.length > 0) {
    const last = matches[matches.length - 1];
    trimmed = trimmed.slice((last.index ?? 0) + last[0].length);
  }

  trimmed = collapseRepeatedBlocks(trimmed);
  trimmed = trimmed.trim();

  if (!trimmed.includes("\n")) {
    trimmed = reconstructParagraphs(trimmed);
  }

  return trimmed;
}
