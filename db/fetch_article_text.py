"""
db/fetch_article_text.py — 기사 URL에서 실제로 렌더링된 본문만 깨끗하게 가져오는 스크래퍼

배치 파이프라인이 쓰는 data_set.csv의 "기사 본문 전체"는 페이지 전체를 그대로 긁은 텍스트라
내비게이션 메뉴/광고/추천 콘텐츠/비디오 위젯 UI 텍스트가 섞여 있다(agent/pipeline/batch_runner.py
의 _clean_scraped_article_text 주석 참고). 프론트 "기사 팩트체크 뷰어"에서 원문을 사람이 읽을
용도로 보여줄 땐 이 잡음이 가독성을 크게 해친다.

trafilatura로 정적 HTML 구조 기반 추출을 시도했으나(2026-08-10), 대상 언론사(조선일보)가
Arc XP(Arc Publishing) CMS로 본문을 클라이언트 사이드 JS로 렌더링하는 SPA라 정적 HTML 구조
분석으로는 본문을 못 찾았다(trafilatura.extract()가 매번 None 반환, 실측 확인됨). 대신 페이지의
<script id="fusion-metadata"> 안에 있는 Fusion.globalContent JSON에 본문이
content_elements(type="text") 배열로 그대로 들어있는 걸 발견해서, 이 JSON을 직접 파싱하는
방식을 우선 시도하고, (다른 CMS를 쓰는 언론사 URL이 섞일 경우를 대비해) 실패하면 trafilatura
추출로, 그마저 실패하면 호출부가 기존 CSV 기반 텍스트로 폴백하도록 None을 반환한다.
"""

from __future__ import annotations

import html
import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Optional

_KST = timezone(timedelta(hours=9))

# content_elements의 텍스트 안에 <b>/<a> 같은 인라인 HTML 태그가 그대로 남아있는 경우가
# 실측 확인됨(CMS가 강조/링크 표시용으로 심어둔 것으로 추정). 프론트가 이 문자열을 HTML로
# 해석하지 않고 그냥 텍스트로 렌더링해서 "<b>" 태그가 화면에 글자 그대로 노출되는 문제가
# 있었다 — 태그만 걷어내고 텍스트는 그대로 둔다.
_HTML_TAG_RE = re.compile(r"<[^>]+>")

try:
    import trafilatura
except ImportError:
    trafilatura = None  # type: ignore[assignment]

FUSION_MARKER = "Fusion.globalContent="


def _extract_json_object(text: str, from_index: int) -> Optional[str]:
    """from_index 이후 첫 '{'부터 중괄호 짝을 맞춰 JSON 객체 하나의 경계를 찾는다.
    문자열 리터럴 안의 중괄호/이스케이프된 따옴표는 카운트에서 제외한다."""
    brace_start = text.find("{", from_index)
    if brace_start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(brace_start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start : i + 1]
    return None


def _extract_from_fusion_json(page_html: str) -> Optional[str]:
    """Arc XP(Fusion) CMS 전용: Fusion.globalContent JSON의 content_elements 중
    type="text"만 순서대로 이어붙인다. 이미지/관련기사 등 다른 타입은 제외돼서 잡음이 섞이지
    않는다."""
    idx = page_html.find(FUSION_MARKER)
    if idx == -1:
        return None

    raw_json = _extract_json_object(page_html, idx + len(FUSION_MARKER))
    if raw_json is None:
        return None

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None

    # content_elements의 텍스트는 &nbsp; 같은 HTML 엔티티가 이스케이프 안 된 채로 남아있거나
    # (실측: "&nbsp;" 리터럴 노출) <b> 같은 인라인 태그가 그대로 섞여있는 경우가 있어
    # (실측: "<b>" 리터럴 노출) 태그 제거 -> 엔티티 unescape 순으로 정리한다.
    paragraphs = [
        html.unescape(_HTML_TAG_RE.sub("", el["content"]))
        for el in data.get("content_elements", [])
        if el.get("type") == "text" and el.get("content")
    ]
    return "\n\n".join(paragraphs) if paragraphs else None


def _extract_fusion_publish_date(page_html: str) -> Optional[date]:
    """Fusion(Arc XP) CMS 전용: 본문 추출에도 쓰는 같은 Fusion.globalContent JSON 안에
    display_date/first_publish_date/created_date(ISO8601, UTC)가 이미 들어있다 — 화면에
    렌더링되는 "기사 등록일"(article-dateline의 dateBox)은 브라우저가 JS로 채우는 값이라
    trafilatura(정적 HTML만 가져옴)로는 못 읽는데, 이 JSON 필드가 사실상 그 값의 원본
    소스다(2026-08-21 조선비즈 기사로 실측: display_date=2024-08-13T23:00:00Z, URL
    경로의 2024/08/14와 KST 변환 후 날짜 일치 확인). UTC라서 KST(+9h)로 변환 후 날짜만
    취한다 — 안 그러면 자정 근처 발행 기사가 하루 밀려서 "이번달"/"작년" 같은 상대 시점
    판정이 틀어질 수 있다."""
    idx = page_html.find(FUSION_MARKER)
    if idx == -1:
        return None

    raw_json = _extract_json_object(page_html, idx + len(FUSION_MARKER))
    if raw_json is None:
        return None

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError:
        return None

    for key in ("display_date", "first_publish_date", "created_date"):
        value = data.get(key)
        if not value:
            continue
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            continue
        return dt.astimezone(_KST).date()
    return None


def fetch_article_for_verification(url: str) -> Optional[dict]:
    """URL 하나에서 실시간 검증 파이프라인(agent/api/server.py)이 필요로 하는 필드를
    전부 채워서 반환한다: {title, text, published_date, url}.

    2026-08-21 추가 — 이 프로젝트는 지금까지 data_set.csv(회사 제공, 제목/본문/작성일이
    이미 컬럼으로 분리돼 있음)로만 파이프라인을 돌렸는데, "URL 입력하면 바로 검증"
    기능은 그 컬럼들이 없어서 URL 하나에서 셋 다 직접 뽑아야 한다. 본문은 기존
    fetch_clean_article_text()의 CMS별 전용 추출(Fusion JSON 등)을 그대로 재사용하고,
    제목/날짜는 trafilatura의 범용 메타데이터 추출(사이트 CMS와 무관하게 OpenGraph/
    JSON-LD 등 표준 메타 태그를 읽음)로 따로 뽑는다 — 본문 추출 방식과 메타데이터
    추출 방식을 분리해야 Fusion XP가 아닌 다른 CMS 기사도 제목/날짜는 뽑을 수 있다.

    실패하면(본문을 못 가져오면) None을 반환한다 — 제목/날짜만 없는 건 기본값으로
    보완하지만(런타임 최신 날짜로 대체), 본문 자체가 없으면 검증할 게 없으므로 실패
    처리한다."""
    if trafilatura is None:
        return None

    try:
        page_html = trafilatura.fetch_url(url)
    except Exception:
        return None
    if not page_html:
        return None

    text = _extract_from_fusion_json(page_html) or trafilatura.extract(page_html)
    if not text:
        return None
    # agent/pipeline/batch_runner.py의 _clean_scraped_article_text와 같은 이유(HCX
    # "40003 Context length exceeded" 실측) — 여기서 뽑은 텍스트는 이미 깨끗하지만
    # (내비게이션 잡음 없음) 그래도 너무 긴 기사 대비 안전하게 길이를 제한한다.
    text = text[:3000]

    title = None
    # Fusion(Arc XP) CMS 사이트는 본문에 쓴 것과 같은 JSON에서 실제 발행 시각을 정확히
    # 뽑을 수 있으니 이걸 우선 시도하고, 아니면 trafilatura의 범용 메타데이터 날짜로
    # 폴백한다(Fusion이 아닌 CMS 사이트 대비).
    published_date = _extract_fusion_publish_date(page_html)
    try:
        meta = trafilatura.extract_metadata(page_html, default_url=url)
        if meta is not None:
            title = getattr(meta, "title", None)
            if published_date is None:
                date_str = getattr(meta, "date", None)
                if date_str:
                    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
                        try:
                            published_date = datetime.strptime(date_str[:10], fmt).date()
                            break
                        except ValueError:
                            continue
    except Exception:
        pass

    if published_date is None:
        # 날짜를 못 뽑으면(메타데이터 없는 사이트 등) 오늘 날짜로 대체한다 — "지난달"/
        # "올해초" 같은 상대 시점 표현의 기준점이 부정확해질 수 있다는 한계가 있지만,
        # 아예 처리를 중단시키는 것보다 최선의 추정치로 계속 진행하는 게 낫다고 판단.
        published_date = date.today()

    return {
        "title": title or url,
        "text": text,
        "published_date": published_date,
        "url": url,
    }


def fetch_clean_article_text(url: str) -> Optional[str]:
    """URL에서 본문만 깨끗하게 가져온다. 실패하면(네트워크 오류, 파싱 실패, trafilatura
    미설치 등) None을 반환한다 — 호출부(export_json.py)가 기존 CSV 기반 텍스트로 폴백할
    수 있게 예외를 던지지 않는다."""
    if trafilatura is None:
        return None

    try:
        page_html = trafilatura.fetch_url(url)
    except Exception:
        return None
    if not page_html:
        return None

    fusion_text = _extract_from_fusion_json(page_html)
    if fusion_text:
        return fusion_text

    return trafilatura.extract(page_html)
