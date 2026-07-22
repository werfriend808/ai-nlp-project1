"""
tests/test_kosis_module.py — C 담당(5,6단계) 모듈 단위 테스트

agent/kosis/api_client.py + agent/kosis/calculator.py를 10건 이상의 케이스로
직접 실행해서 확인합니다. 실제 KOSIS API를 호출하므로 .env에 KOSIS_API_KEY가
있어야 합니다.

실행 (프로젝트 루트에서):
    python -m tests.test_kosis_module

각 케이스는 (이름, 실행 함수) 쌍으로 등록되고, 실행 함수가 예외 없이 끝나면 PASS,
CalculationError/KosisApiError가 "기대한 실패"로 표시된 케이스에서 나면 그것도 PASS,
그 외 예외나 assert 실패는 FAIL로 기록됩니다. 마지막에 요약 + 발견된 이슈를 출력합니다.
"""

from __future__ import annotations

from agent.kosis.api_client import KosisApiClient, KosisApiError
from agent.kosis.calculator import KosisCalculator, CalculationError

client = KosisApiClient()
calc = KosisCalculator()


# ---------------------------------------------------------------------------
# 정상 케이스 (api_client 단일 조회)
# ---------------------------------------------------------------------------

def case_01_unemployment_2024_youth():
    r = client("DT_1DA7102S", {"period": "2024", "gender": "전체", "age": "청년(15~29세)"})
    assert r.raw_value > 0 and r.unit == "%"


def case_02_unemployment_2023_youth():
    r = client("DT_1DA7102S", {"period": "2023", "gender": "전체", "age": "청년(15~29세)"})
    assert r.raw_value > 0


def case_03_unemployment_male():
    r = client("DT_1DA7102S", {"period": "2024", "gender": "남자", "age": "청년(15~29세)"})
    assert r.raw_value > 0


def case_04_unemployment_female():
    r = client("DT_1DA7102S", {"period": "2024", "gender": "여자", "age": "청년(15~29세)"})
    assert r.raw_value > 0


def case_05_unemployment_all_ages():
    r = client("DT_1DA7102S", {"period": "2020", "gender": "전체", "age": "전체"})
    assert r.raw_value > 0


def case_06_farm_total():
    r = client("DT_1EA1019", {"period": "2024", "age": "전체"})
    assert r.raw_value > 0 and r.unit == "가구"


def case_07_farm_20s():
    r = client("DT_1EA1019", {"period": "2024", "age": "20~24세"})
    assert r.raw_value >= 0


def case_08_farm_elderly_80plus():
    r = client("DT_1EA1019", {"period": "2024", "age": "80세이상"})
    assert r.raw_value > 0


# ---------------------------------------------------------------------------
# 정상 케이스 (calculator)
# ---------------------------------------------------------------------------

def case_09_ratio_elderly_farm():
    numerator = [
        client("DT_1EA1019", {"period": "2024", "age": a})
        for a in ["65~69세", "70~74세", "75~79세", "80세이상"]
    ]
    denominator = client("DT_1EA1019", {"period": "2024", "age": "전체"})
    result = calc.compute_ratio(numerator, denominator)
    assert 0 < result.raw_value < 100


def case_10_sum_all_age_bands_equals_total():
    """무결성 체크: 연령대별(T01~T14) 합계가 전체(T00)와 일치하는지 확인."""
    bands = [
        "20세미만", "20~24세", "25~29세", "30~34세", "35~39세", "40~44세", "45~49세",
        "50~54세", "55~59세", "60~64세", "65~69세", "70~74세", "75~79세", "80세이상",
    ]
    responses = [client("DT_1EA1019", {"period": "2024", "age": a}) for a in bands]
    summed = calc.compute_sum(responses)
    total = client("DT_1EA1019", {"period": "2024", "age": "전체"})
    assert summed.raw_value == total.raw_value, (
        f"연령대 합계({summed.raw_value}) != 전체({total.raw_value})"
    )


def case_11_change_rate_unemployment():
    base = client("DT_1DA7102S", {"period": "2023", "gender": "전체", "age": "청년(15~29세)"})
    target = client("DT_1DA7102S", {"period": "2024", "gender": "전체", "age": "청년(15~29세)"})
    result = calc.compute_change_rate(base, target)
    assert result.calc_type == "증감률"


def case_12_change_unemployment():
    base = client("DT_1DA7102S", {"period": "2019", "gender": "전체", "age": "청년(15~29세)"})
    target = client("DT_1DA7102S", {"period": "2024", "gender": "전체", "age": "청년(15~29세)"})
    result = calc.compute_change(base, target)
    assert result.calc_type == "증감"


# ---------------------------------------------------------------------------
# 엣지 케이스 (의도적으로 깨뜨려서, 에러가 "제대로" 나는지 확인)
# ---------------------------------------------------------------------------

def case_13_ratio_zero_denominator_should_fail():
    from agent.kosis.calculator import KosisApiResponse

    zero = KosisApiResponse(raw_value=0, unit="%", period="2024", org_id="101", itm_id="T80")
    numerator = [KosisApiResponse(raw_value=5, unit="%", period="2024", org_id="101", itm_id="T80")]
    try:
        calc.compute_ratio(numerator, zero)
    except CalculationError:
        return  # 기대한 실패
    raise AssertionError("분모 0인데 예외가 안 났음")


def case_14_sum_mismatched_units_should_fail():
    from agent.kosis.calculator import KosisApiResponse

    a = KosisApiResponse(raw_value=1, unit="가구", period="2024", org_id="101", itm_id="T00")
    b = KosisApiResponse(raw_value=1, unit="%", period="2024", org_id="101", itm_id="T80")
    try:
        calc.compute_sum([a, b])
    except CalculationError:
        return  # 기대한 실패
    raise AssertionError("단위가 다른데 합계 계산이 통과됨")


def case_15_unknown_table_id_should_fail():
    try:
        client("DT_NOT_EXIST", {"period": "2024"})
    except KeyError:
        return  # 기대한 실패
    raise AssertionError("존재하지 않는 table_id인데 예외가 안 났음")


def case_16_region_all_returns_many_rows_should_fail():
    """objL1=ALL로 그대로 보내면(코드 매핑 없이) 57개 지역이 다 나와서 KosisApiError가 나야 정상."""
    try:
        client("DT_1EA1019", {"period": "2024", "age": "전체", "region": "전국 세부지역 없음"})
    except KosisApiError:
        return  # 기대한 실패 (code_map에 없는 값이라 원문 그대로 objL1에 들어가 에러)
    raise AssertionError("잘못된 region 값인데 예외가 안 났음")


CASES = [
    case_01_unemployment_2024_youth,
    case_02_unemployment_2023_youth,
    case_03_unemployment_male,
    case_04_unemployment_female,
    case_05_unemployment_all_ages,
    case_06_farm_total,
    case_07_farm_20s,
    case_08_farm_elderly_80plus,
    case_09_ratio_elderly_farm,
    case_10_sum_all_age_bands_equals_total,
    case_11_change_rate_unemployment,
    case_12_change_unemployment,
    case_13_ratio_zero_denominator_should_fail,
    case_14_sum_mismatched_units_should_fail,
    case_15_unknown_table_id_should_fail,
    case_16_region_all_returns_many_rows_should_fail,
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
