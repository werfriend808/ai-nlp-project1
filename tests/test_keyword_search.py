"""
tests/test_keyword_search.py — agent/mapping/keyword_search.py 회귀 테스트

2026-08-06, 300개 실제 기사 배치 분석 중 "쉬었음" 관련 claim들이 서로 다른 무관한 표로
잘못 매칭되는 걸 조사하다가 keyword_search.py의 실제 버그를 발견해서 만든 테스트:

"청년이 사상 처음으로 50만명을 넘어섰다" 문장이 공백을 지우고 부분 문자열로 비교하는
_normalize() 방식 때문에 "청년이사상..."이 되어, "이사"(SYNONYMS의 "국내이동"/"인구이동"
동의어이자 DT_1B26001_A01의 catalog keyword)와 우연히 겹쳐 매칭됐다 — 서로 완전히 무관한
두 단어("청년이"/"사상")가 공백 하나만 사이에 두고 이어붙으면서 생긴 오탐.

수정: SYNONYMS 조회(_expand_query_terms)는 공백 제거 없이 원문 그대로 부분 문자열 검사로
바꾸고(raw_term이 전부 내부 공백 없는 단일 단어라 공백 관용이 애초에 필요 없었음),
DT_1B26001_A01의 너무 짧고 애매한 catalog keyword "이사"는 "이사하는"/"이사한"/"이사를"
같은 실제 기사에 흔한 활용형으로 구체화했다. catalog keyword 매칭 자체(_score_table)는
"혼인 건수" vs "혼인건수" 같은 정상적인 내부 띄어쓰기 차이를 여전히 잡아야 해서
_normalize() 기반 방식을 그대로 유지했다.

실행 (프로젝트 루트에서):
    python -m tests.test_keyword_search
"""

from __future__ import annotations

from agent.interfaces import Claim
from agent.mapping.keyword_search import keyword_search


def _search(sentence: str):
    return keyword_search(Claim(sentence=sentence, claim_type="규모"))


def case_01_word_boundary_collision_should_not_match():
    """"청년이 사상"이 공백 제거 시 "이사"와 우연히 겹치던 실제 오탐 재현 — 이제 안 잡혀야 함."""
    results = _search(
        "지난달 별다른 경제활동을 하지 않고 그냥 쉬었다고 응답한 청년이 사상 처음으로 50만명을 넘어섰다."
    )
    table_ids = [r.table_id for r in results]
    assert "DT_1B26001_A01" not in table_ids, (
        f"단어 경계를 넘나드는 '이사' 오탐이 재발함: {table_ids}"
    )


def case_02_genuine_relocation_sentence_still_matches():
    """진짜 "이사" 관련 문장은 수정 후에도 정상적으로 잡혀야 한다 (회귀 방지)."""
    results = _search("최근 서울에서 지방으로 이사하는 사람이 늘고 있다.")
    assert results, "진짜 이사 문장인데 아무 표도 안 잡힘"
    assert results[0].table_id == "DT_1B26001_A01", results[0].table_id


def case_03_internal_whitespace_variant_still_matches():
    """"혼인 건수"(문장)와 "혼인건수"(카탈로그 keyword)처럼 정상적인 내부 띄어쓰기 차이는
    여전히 잡혀야 한다 — SYNONYMS 수정이 catalog keyword 매칭까지 건드리면 안 됨."""
    results = _search("지난달 혼인 건수는 2만153건으로 집계됐다.")
    assert results, "띄어쓰기 차이 때문에 정상 매칭이 깨짐"
    assert results[0].table_id == "DT_1B8000G", results[0].table_id


def case_04_gasoline_price_keyword_now_verified():
    """"휘발유"/"유가"는 실제 기사에 흔한 표현인데 카탈로그엔 KOSIS 분류명 "석유류"만
    있어서 keyword_search가 아예 못 잡던 갭 (2026-08-06, 300개 배치 실측에서 12건 확인 후
    catalog keywords에 "휘발유"/"유가" 등 추가). keyword_search 단독으로 잡히는지 확인 —
    이게 잡혀야 reranker 단계에서 "검증된 후보"로 취급된다."""
    results = _search("이번 주 국내 주유소 휘발유 평균 가격이 국제유가 하락 영향으로 17주 만에 내림세로 돌아섰다.")
    assert results, "휘발유/유가 문장인데 keyword_search가 아무 표도 못 찾음"
    assert results[0].table_id == "DT_1J22112", results[0].table_id
    assert "휘발유" in results[0].source_meta or "유가" in results[0].source_meta, results[0].source_meta


def case_05_seafood_and_diesel_price_keywords_now_verified():
    """"수산물 가격"(축산물은 "물가"/"가격" 둘 다 있는데 수산물은 "물가"만 있었음),
    "경유 판매 가격"(단어 사이에 "판매"가 끼어 기존 "경유 가격" 키워드로는 못 잡히던 경우)
    둘 다 2026-08-09 2,507건 전체 배치 실측에서 확인하고 keywords에 추가."""
    r1 = _search("수산물 가격은 전년 동월 대비 6.4% 올랐다.")
    assert r1 and r1[0].table_id == "DT_1J22112", r1

    r2 = _search("경유 판매 가격도 1L당 1596.6원을 기록하며 직전 주 대비 1.2원 하락했다.")
    assert r2 and r2[0].table_id == "DT_1J22112", r2


def case_06_japan_base_rate_no_longer_matches_domestic_lending_rate():
    """"기준금리"(중앙은행 정책금리)가 SYNONYMS를 통해 "대출금리"(시중은행 대출금리) 표로
    잘못 매칭되던 버그 — 2026-08-09, "일본의 기준금리는 0.5%다"가 국내 예금은행 대출금리
    표로 가던 실제 사례 발견 후 SYNONYMS에서 "기준금리" 항목 제거. 기준금리는 KOSIS로
    검증 불가한 통계라(docs/PENDING_TABLES.md) 억지로 다른 표에 연결하지 않고 매칭
    안 되는 게 정답이다."""
    results = _search("현재 일본의 기준금리는 0.5%다.")
    assert not results, f"기준금리가 여전히 무관한 표로 매칭됨: {results}"


CASES = [
    case_01_word_boundary_collision_should_not_match,
    case_02_genuine_relocation_sentence_still_matches,
    case_03_internal_whitespace_variant_still_matches,
    case_04_gasoline_price_keyword_now_verified,
    case_05_seafood_and_diesel_price_keywords_now_verified,
    case_06_japan_base_rate_no_longer_matches_domestic_lending_rate,
]


def main() -> None:
    results = []
    for case in CASES:
        try:
            case()
            results.append((case.__name__, "PASS", ""))
        except Exception as e:  # noqa: BLE001 - 테스트 러너라 실패 원인만 보고 계속 진행
            results.append((case.__name__, "FAIL", str(e)))

    print(f"총 {len(results)}건 실행")
    print("=" * 70)
    for name, status, detail in results:
        mark = "✅" if status == "PASS" else "❌"
        print(f"{mark} {status}  {name}" + (f" — {detail}" if detail else ""))

    passed = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n{passed}/{len(results)} PASS")


if __name__ == "__main__":
    main()
