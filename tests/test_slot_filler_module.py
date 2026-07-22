"""
tests/test_slot_filler_module.py — D 담당(4단계) 모듈 단위 테스트

agent/orchestrator/slot_filler.py의 fill_slots()를 10건 이상의 케이스로
직접 실행해서 확인합니다. HCX-DASH-002를 실제로 호출하므로 .env에
HCI__API_KEY가 있어야 합니다.

clarify.py는 아직 clarify() 함수가 구현되어 있지 않아 (주석만 있는 상태)
이 테스트에서는 다루지 않습니다 — slot_filler.py만 독립적으로 검증합니다.

실행 (프로젝트 루트에서):
    python -m tests.test_slot_filler_module

각 케이스는 (이름, 실행 함수) 쌍으로 등록되고, 실행 함수가 예외 없이 끝나면 PASS,
assert 실패나 예외가 나면 FAIL로 기록됩니다. 마지막에 요약을 출력합니다.
"""

from __future__ import annotations

from datetime import date

from agent.orchestrator.slot_filler import fill_slots

ARTICLE_DATE = date(2025, 1, 1)


# ---------------------------------------------------------------------------
# 정상 케이스 (단일 슬롯 추출)
# ---------------------------------------------------------------------------

def case_01_region_only():
    slots = fill_slots("서울 통계 알려줘", {}, ARTICLE_DATE)
    assert slots.get("region") == "서울", f"region 추출 실패: {slots}"
    assert slots.get("period") is None, f"period가 잘못 채워짐: {slots}"


def case_02_period_only_absolute_year():
    slots = fill_slots("2024년 통계 알려줘", {}, ARTICLE_DATE)
    assert slots.get("period") == "2024", f"period 추출 실패: {slots}"


def case_03_calc_type_only():
    slots = fill_slots("평균 알려줘", {}, ARTICLE_DATE)
    assert slots.get("calc_type") == "평균", f"calc_type 추출 실패: {slots}"


def case_04_nothing_in_utterance():
    slots = fill_slots("통계 좀 알려줘", {}, ARTICLE_DATE)
    assert not slots.get("period") and not slots.get("region") and not slots.get("calc_type"), (
        f"아무 정보도 없는 발화인데 슬롯이 채워짐: {slots}"
    )


def case_05_all_slots_at_once():
    slots = fill_slots("작년 서울 증감률 알려줘", {}, ARTICLE_DATE)
    assert slots.get("region") == "서울", f"region 추출 실패: {slots}"
    assert slots.get("calc_type") == "증감률", f"calc_type 추출 실패: {slots}"
    assert slots.get("period") == "2024", f"작년→2024 계산 실패: {slots}"


# ---------------------------------------------------------------------------
# 정상 케이스 (상대 시점 계산 — 코드로 처리, LLM 미호출 구간)
# ---------------------------------------------------------------------------

def case_06_relative_last_year():
    slots = fill_slots("작년 통계 줘", {}, ARTICLE_DATE)
    assert slots.get("period") == "2024", f"작년 계산 실패: {slots}"


def case_07_relative_this_year():
    slots = fill_slots("올해 통계 줘", {}, ARTICLE_DATE)
    assert slots.get("period") == "2025", f"올해 계산 실패: {slots}"


def case_08_relative_two_years_ago():
    slots = fill_slots("재작년 통계 줘", {}, ARTICLE_DATE)
    assert slots.get("period") == "2023", f"재작년 계산 실패: {slots}"


def case_09_ambiguous_relative_time_delegated_to_llm():
    """'지난달'처럼 연 단위로 안 떨어지는 표현은 HCX-003에 위임 → 4자리 연도만 나와야 함."""
    slots = fill_slots("지난달 서울 통계 알려줘", {}, ARTICLE_DATE)
    assert slots.get("region") == "서울", f"region 추출 실패: {slots}"
    period = slots.get("period")
    assert period is not None and len(str(period)) == 4 and str(period).isdigit(), (
        f"애매한 시점 표현이 4자리 연도로 정규화되지 않음: {slots}"
    )


# ---------------------------------------------------------------------------
# 정상 케이스 (기존 슬롯과의 병합 — 오염 방지 회귀 테스트)
# ---------------------------------------------------------------------------

def case_10_existing_slots_preserved_on_partial_update():
    existing = {"period": "2024", "calc_type": "합계"}
    slots = fill_slots("부산으로 해줘", existing, ARTICLE_DATE)
    assert slots.get("region") == "부산", f"region이 안 채워짐: {slots}"
    assert slots.get("period") == "2024", f"❌ period가 지역 발화로 오염됨: {slots}"
    assert slots.get("calc_type") == "합계", f"❌ calc_type이 사라짐: {slots}"


def case_11_unrelated_utterance_does_not_pollute_existing():
    existing = {"period": "2023", "region": "전국"}
    slots = fill_slots("오늘 날씨 어때", existing, ARTICLE_DATE)
    assert slots.get("period") == "2023", f"❌ 무관한 발화가 period를 덮어씀: {slots}"
    assert slots.get("region") == "전국", f"❌ 무관한 발화가 region을 덮어씀: {slots}"


# ---------------------------------------------------------------------------
# 엣지 케이스
# ---------------------------------------------------------------------------

def case_12_empty_string_utterance_should_not_crash():
    slots = fill_slots("", {}, ARTICLE_DATE)
    assert isinstance(slots, dict), f"빈 문자열 입력에서 dict가 반환되지 않음: {slots}"


CASES = [
    case_01_region_only,
    case_02_period_only_absolute_year,
    case_03_calc_type_only,
    case_04_nothing_in_utterance,
    case_05_all_slots_at_once,
    case_06_relative_last_year,
    case_07_relative_this_year,
    case_08_relative_two_years_ago,
    case_09_ambiguous_relative_time_delegated_to_llm,
    case_10_existing_slots_preserved_on_partial_update,
    case_11_unrelated_utterance_does_not_pollute_existing,
    case_12_empty_string_utterance_should_not_crash,
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
        mark = "✅" if status == "PASS" else "❌"
        line = f"{mark} {status}  {name}"
        if detail:
            line += f"  — {detail}"
        print(line)

    n_pass = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n{n_pass}/{len(results)} PASS")


if __name__ == "__main__":
    main()
