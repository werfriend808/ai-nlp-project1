"""
tests/test_table_dimension_hints.py — 4단계 슬롯필링 보강분 단위 테스트

agent/orchestrator/agent_chat.py의 _extract_table_dimension_hints()를 다양한 표
(연령/성별/대출종류/소득분위/산업/지역/주택유형/공종/사망원인 등)로 검증합니다.
LLM/API 호출이 없는 순수 문자열 매칭 함수라 빠르게 반복 실행 가능합니다.

목적: "20대 실업률" 버그 수정(code_map 라벨 직접 매칭)이 이 표 하나에서만 되는 게 아니라
다른 표/축에서도 실제로 일반화되는지 실측한다.

실행 (프로젝트 루트에서):
    python -m tests.test_table_dimension_hints
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.orchestrator.agent_chat import _extract_table_dimension_hints

TABLE_PARAMS_PATH = Path(__file__).parent.parent / "agent" / "kosis" / "table_params.json"
with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
    TABLE_PARAMS = json.load(f)


# ---------------------------------------------------------------------------
# 연령/성별 (기존 버그가 있던 축 — 회귀 확인)
# ---------------------------------------------------------------------------

def case_01_age_20s():
    hints = _extract_table_dimension_hints("20대 실업률 알려줘", "DT_1DA7102S", TABLE_PARAMS)
    assert hints.get("age") == "20대", f"20대 매칭 실패: {hints}"


def case_02_age_60_over():
    hints = _extract_table_dimension_hints("60세이상 실업률 궁금해", "DT_1DA7102S", TABLE_PARAMS)
    assert hints.get("age") == "60세이상", f"60세이상 매칭 실패: {hints}"


def case_03_gender_female():
    hints = _extract_table_dimension_hints("여자 실업률 알려줘", "DT_1DA7102S", TABLE_PARAMS)
    assert hints.get("gender") == "여자", f"여자 매칭 실패: {hints}"


def case_04_no_match_synonym_not_covered_by_this_function():
    """'청년'은 code_map 라벨('청년(15~29세)')과 정확히 안 겹쳐서 이 함수는 못 잡아야 정상
    (이 경우는 _DEMOGRAPHIC_HINTS가 별도로 보완하는 영역)."""
    hints = _extract_table_dimension_hints("청년 실업률 알려줘", "DT_1DA7102S", TABLE_PARAMS)
    assert hints.get("age") is None, f"'청년'이 이 함수에서 잡히면 안 되는데 잡힘: {hints}"


# ---------------------------------------------------------------------------
# 대출종류 (신규 축)
# ---------------------------------------------------------------------------

def case_05_loan_type_household():
    hints = _extract_table_dimension_hints("가계대출 금리 알려줘", "DT_121Y006", TABLE_PARAMS)
    assert hints.get("loan_type") == "가계대출", f"가계대출 매칭 실패: {hints}"


def case_06_loan_type_corporate():
    hints = _extract_table_dimension_hints("기업대출 금리는 얼마야", "DT_121Y006", TABLE_PARAMS)
    assert hints.get("loan_type") == "기업대출", f"기업대출 매칭 실패: {hints}"


# ---------------------------------------------------------------------------
# 소득분위 / 가계수지 항목 (신규 축, 한 문장에 축 2개 동시 매칭)
# ---------------------------------------------------------------------------

def case_07_quintile_and_item_together():
    hints = _extract_table_dimension_hints("소득 1분위 가계수지 알려줘", "DT_1L9U103", TABLE_PARAMS)
    assert hints.get("quintile") == "1분위", f"1분위 매칭 실패: {hints}"
    assert hints.get("item") == "소득", f"소득(item) 매칭 실패: {hints}"


def case_08_item_disposable_income():
    hints = _extract_table_dimension_hints("처분가능소득이 얼마나 늘었어", "DT_1L9U103", TABLE_PARAMS)
    assert hints.get("item") == "처분가능소득", f"처분가능소득 매칭 실패: {hints}"


# ---------------------------------------------------------------------------
# 산업 (신규 축, 서로 다른 표 2개)
# ---------------------------------------------------------------------------

def case_09_industry_service_dt1jh():
    hints = _extract_table_dimension_hints("서비스업생산지수 알려줘", "DT_1JH20202", TABLE_PARAMS)
    assert hints.get("industry") == "서비스업", f"서비스업 매칭 실패: {hints}"


def case_10_industry_construction_dt1jh():
    hints = _extract_table_dimension_hints("건설업 생산은 어때", "DT_1JH20202", TABLE_PARAMS)
    assert hints.get("industry") == "건설업", f"건설업 매칭 실패: {hints}"


def case_11_industry_manufacturing_bsi():
    hints = _extract_table_dimension_hints("제조업 기업경기실사지수 알려줘", "DT_512Y013", TABLE_PARAMS)
    assert hints.get("industry") == "제조업", f"제조업(BSI) 매칭 실패: {hints}"


# ---------------------------------------------------------------------------
# 지역 + 주택유형 (신규 축, 한 문장에 축 2개 동시 매칭)
# ---------------------------------------------------------------------------

def case_12_region_and_housing_type_together():
    hints = _extract_table_dimension_hints("서울 아파트 매매가격 알려줘", "DT_30404_B012", TABLE_PARAMS)
    assert hints.get("region") == "서울", f"서울 매칭 실패: {hints}"
    assert hints.get("housing_type") == "아파트", f"아파트 매칭 실패: {hints}"


# ---------------------------------------------------------------------------
# 공종 (건설기성액, 신규 축)
# ---------------------------------------------------------------------------

def case_13_construction_type():
    hints = _extract_table_dimension_hints("건축 공사 건설기성액 알려줘", "DT_1G18007", TABLE_PARAMS)
    assert hints.get("construction_type") == "건축", f"건축 매칭 실패: {hints}"


# ---------------------------------------------------------------------------
# 사망원인 (신규 축)
# ---------------------------------------------------------------------------

def case_14_death_cause():
    hints = _extract_table_dimension_hints("암 사망자 수 알려줘", "DT_1B34E01", TABLE_PARAMS)
    assert hints.get("cause") == "암", f"암 매칭 실패: {hints}"


# ---------------------------------------------------------------------------
# 부정 케이스: 아무 축 표현도 없는 문장은 빈 dict여야 함
# ---------------------------------------------------------------------------

def case_15_nothing_mentioned_returns_empty():
    hints = _extract_table_dimension_hints("실업률 알려줘", "DT_1DA7102S", TABLE_PARAMS)
    assert hints == {}, f"아무 언급도 없는데 뭔가 채워짐: {hints}"


def case_16_table_not_in_params_returns_empty():
    """table_params.json에 없는 tblId가 들어와도 죽지 않고 빈 dict를 반환해야 함."""
    hints = _extract_table_dimension_hints("아무 문장", "DT_존재안함", TABLE_PARAMS)
    assert hints == {}, f"미등록 tblId에서 예외 없이 빈 dict가 나와야 함: {hints}"


CASES = [
    case_01_age_20s,
    case_02_age_60_over,
    case_03_gender_female,
    case_04_no_match_synonym_not_covered_by_this_function,
    case_05_loan_type_household,
    case_06_loan_type_corporate,
    case_07_quintile_and_item_together,
    case_08_item_disposable_income,
    case_09_industry_service_dt1jh,
    case_10_industry_construction_dt1jh,
    case_11_industry_manufacturing_bsi,
    case_12_region_and_housing_type_together,
    case_13_construction_type,
    case_14_death_cause,
    case_15_nothing_mentioned_returns_empty,
    case_16_table_not_in_params_returns_empty,
]


def main() -> None:
    results = []
    for case in CASES:
        try:
            case()
            results.append((case.__name__, "PASS", ""))
        except Exception as e:
            results.append((case.__name__, "FAIL", f"{type(e).__name__}: {e}"))

    print(f"\n{'=' * 70}")
    print(f"총 {len(results)}건 실행")
    print(f"{'=' * 70}")
    for name, status, detail in results:
        mark = "PASS" if status == "PASS" else "FAIL"
        line = f"[{mark}] {name}"
        if detail:
            line += f"  - {detail}"
        print(line)

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n{n_pass}/{len(results)} PASS")


if __name__ == "__main__":
    main()