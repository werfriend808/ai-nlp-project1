// 배치 파이프라인의 _clean_scraped_article_text(agent/pipeline/batch_runner.py)는 "제목
// 위치부터 3000자"만 잘라내서 화면 표시용 잡음(내비게이션 메뉴 반복, 광고/추천 콘텐츠,
// 비디오 플레이어 자막 UI 텍스트 등)은 그대로 남아있다. 이 함수는 그 잡음을 "화면에
// 보여줄 때만" 추가로 걷어낸다 — 실제 파이프라인이 모델에 넘긴 텍스트(article_text 원본)는
// 건드리지 않고, 표시 직전에만 적용한다.
//
// 검증(2026-08-10, 실제 export된 기사 16건 기준): 평균 길이 2726자 → 1714자(-37%)로
// 줄었고, 하이라이트 대상 주장 33건의 매칭 여부는 정리 전후로 동일하게 유지됨(회귀 없음).
const AD_TRAILER_RE = /By Taboola/;
const BYLINE_TIMESTAMP_RE =
  /입력\s*\d{4}\.\d{2}\.\d{2}\.\s*\d{2}:\d{2}(\s*업데이트\s*\d{4}\.\d{2}\.\d{2}\.\s*\d{2}:\d{2})?/g;
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

  return trimmed.trim();
}
