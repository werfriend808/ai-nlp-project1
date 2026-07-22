"""
agent/mapping/test_mapping.py — 3단계(keyword_search/embedding_search/reranker) 단위 테스트

Day2 체크리스트: "각자 담당 모듈 단위 테스트 마무리 (샘플 10건 이상)" 대응.
table_catalog.json의 7개 카테고리를 모두 커버하는 12개 케이스 + expected_table_id로
top-1 결과가 맞는지 자동 채점한다.

실행:
    python -m agent.mapping.test_mapping   (프로젝트 루트에서)
"""

from __future__ import annotations

from agent.mapping.keyword_search import keyword_search
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import search_and_rerank

try:
    from agent.interfaces import Claim
except ImportError:
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class Claim:  # type: ignore[no-redef]
        sentence: str
        claim_type: str
        period: Optional[str] = None
        unit: Optional[str] = None
        population: Optional[str] = None


# (문장, claim_type, 정답 table_id, 카테고리) — 카테고리 7개 전부 최소 1건 이상 커버
TEST_CASES: list[tuple[str, str, str, str]] = [
    ("지난달 청년 실업률이 6%에 육박했다", "규모", "DT_1DA7001S", "고용/노동"),
    ("고용률이 역대 최고치를 기록했다", "비교", "DT_1DA7001S", "고용/노동"),
    ("취업자 수가 46개월 만에 감소 전환했다", "증감률", "DT_1DA7001S", "고용/노동"),
    ("지난달 소비자물가가 전년 동월 대비 2.2% 올랐다", "증감률", "DT_1J22003", "물가/CPI"),
    ("생활물가가 5개월 연속 올랐다", "증감률", "DT_1J22003", "물가/CPI"),
    ("전국 주민등록인구가 5000만명 아래로 떨어졌다", "규모", "DT_1B04005N", "인구"),
    ("한국 경제성장률이 3개 분기 연속 0%대에 머물렀다", "비교", "DT_200Y102", "경제성장"),
    ("지난해 수출이 6838억달러로 역대 최대를 기록했다", "규모", "DT_1R11006_FRM101", "무역/수출입"),
    ("무역수지가 3년 만에 흑자로 전환했다", "비교", "DT_1R11006_FRM101", "무역/수출입"),
    ("전국 집값이 하락세를 보였다", "비교", "DT_30404_B012", "부동산/주택"),
    ("혼인 건수가 역대 최저를 기록했다", "역대기록", "DT_1B8000G", "출생/사망/혼인"),
    ("합계출산율이 0.7명대로 떨어졌다", "규모", "DT_1B8000G", "출생/사망/혼인"),
]


def run_tests() -> None:
    cache = build_table_embedding_cache()

    total = len(TEST_CASES)
    kw_pass = emb_pass = rerank_pass = 0

    print(f"=== 총 {total}건 테스트 (카테고리 7개 전부 커버) ===\n")

    for sentence, claim_type, expected_id, category in TEST_CASES:
        claim = Claim(sentence=sentence, claim_type=claim_type)

        kw_top = keyword_search(claim, top_k=1)
        emb_top = embedding_search(claim, top_k=1, cache=cache)
        rerank_top = search_and_rerank(
            claim,
            keyword_fn=keyword_search,
            embedding_fn=lambda c: embedding_search(c, cache=cache),
            top_k=1,
        )

        kw_ok = bool(kw_top) and kw_top[0].table_id == expected_id
        emb_ok = bool(emb_top) and emb_top[0].table_id == expected_id
        rerank_ok = bool(rerank_top) and rerank_top[0].table_id == expected_id

        kw_pass += kw_ok
        emb_pass += emb_ok
        rerank_pass += rerank_ok

        mark = lambda ok: "O" if ok else "X"
        print(f"[{category}] {sentence}")
        print(f"  기대값: {expected_id}")
        print(f"  keyword  : {mark(kw_ok)}  ({kw_top[0].table_id if kw_top else '없음'})")
        print(f"  embedding: {mark(emb_ok)}  ({emb_top[0].table_id if emb_top else '없음'})")
        print(f"  rerank   : {mark(rerank_ok)}  ({rerank_top[0].table_id if rerank_top else '없음'})")
        print()

    print("=== 결과 요약 ===")
    print(f"keyword_search top-1 정답률  : {kw_pass}/{total}")
    print(f"embedding_search top-1 정답률: {emb_pass}/{total}  (임베딩 API 미확정 폴백이라 낮게 나오는 게 정상)")
    print(f"reranker(최종) top-1 정답률  : {rerank_pass}/{total}  ← 이게 실질적인 3단계 정확도")


if __name__ == "__main__":
    run_tests()