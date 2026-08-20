"""
tests/test_clean_scraped_article_text.py — agent/pipeline/batch_runner.py의
_clean_scraped_article_text() 단위 테스트 (관련기사/추천 위젯 오염 방지, 2026-08-21)

순수 문자열 처리 함수라 HCX API 호출 없이 즉시 실행 가능.

실행 (프로젝트 루트에서):
    python -m tests.test_clean_scraped_article_text
"""

from __future__ import annotations

from agent.pipeline.batch_runner import _clean_scraped_article_text

TITLE = "상권분석·매출진단 돕는 ‘소상공인 365’ 써보세요"

REAL_BODY = (
    "상권분석·매출진단 돕는 ‘소상공인 365’ 써보세요 김이삭 기자 입력 2025.03.11. 09:00 3 "
    "소상공인시장진흥공단이 운영하는 상권정보시스템 '소상공인 365'가 창업 전 상권분석부터 "
    "매출진단까지 무료로 제공한다고 밝혔다."
)

# 실제 재현된 오염 패턴 — 짧은 기사 뒤에 완전히 무관한 다른 기사 헤드라인 조각들이 이어붙음.
JUNK_TRAILER = (
    " 많이 본 뉴스 포스코 인도 제철소 짓는다... 현지 1위 철강사와 50:50 합작 "
    "매일 철강 수요가 늘고 있는 인도 시장을 선점하기 위한 조치라는 분석이 나온다."
)


def test_junk_trailer_removed():
    contaminated = REAL_BODY + JUNK_TRAILER
    cleaned = _clean_scraped_article_text(TITLE, contaminated, max_len=3000)
    assert "포스코" not in cleaned, f"❌ 관련기사 잡음이 안 잘려나감: {cleaned!r}"
    assert "철강" not in cleaned, f"❌ 관련기사 잡음이 안 잘려나감: {cleaned!r}"
    assert "소상공인 365" in cleaned or "상권정보시스템" in cleaned, (
        f"❌ 진짜 본문까지 같이 잘려나감: {cleaned!r}"
    )
    print("[통과] '많이 본 뉴스' 트레일러가 잘려나가고 진짜 본문은 보존됨")


def test_normal_article_unaffected():
    """junk 마커가 전혀 없는 정상 기사는 한 글자도 안 잘려야 한다(회귀 방지)."""
    normal = REAL_BODY + " 이용은 전액 무료이며, 누구나 홈페이지에서 바로 신청할 수 있다."
    cleaned = _clean_scraped_article_text(TITLE, normal, max_len=3000)
    # byline(_BYLINE_RE)만 제거된 상태와 동일해야 함 — junk 마커가 없으므로 추가로 안 잘림.
    assert "이용은 전액 무료이며" in cleaned, f"❌ junk 마커 없는 정상 문장이 잘못 잘림: {cleaned!r}"
    print("[통과] junk 마커 없는 정상 기사는 안 잘림")


def test_each_marker_variant():
    """4개 마커(By Taboola/많이 본 뉴스/오늘의 멤버십/AI 추천) 전부 개별적으로 작동하는지."""
    markers = ["By Taboola", "많이 본 뉴스", "오늘의 멤버십", "AI 추천"]
    for marker in markers:
        contaminated = REAL_BODY + f" {marker} 완전히 무관한 다른 기사 내용입니다."
        cleaned = _clean_scraped_article_text(TITLE, contaminated, max_len=3000)
        assert "무관한 다른 기사" not in cleaned, f"❌ 마커 {marker!r}가 안 걸림: {cleaned!r}"
    print("[통과] 마커 4종(By Taboola/많이 본 뉴스/오늘의 멤버십/AI 추천) 전부 정상 동작")


if __name__ == "__main__":
    test_junk_trailer_removed()
    test_normal_article_unaffected()
    test_each_marker_variant()
    print("\n[전체 통과] _clean_scraped_article_text 회귀 테스트 3건 모두 통과")
