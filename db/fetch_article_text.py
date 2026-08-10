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
from typing import Optional

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
