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

from agent.kosis.api_client import KosisApiClient, KosisApiError, _validate_period_format
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
    # 2026-08-21: KOSIS가 시점 확정 후에도 통계를 소급 개정하는 경우가 있어(연령대별
    # 집계와 전체 집계가 서로 다른 시점에 갱신되면 며칠간 1명 단위 오차가 생길 수 있음),
    # 완전 일치 대신 작은 절대 오차(±2명)까지는 허용한다 — 코드 버그가 아니라 정상적인
    # 데이터 소급 개정 노이즈임을 실측으로 확인(당시 973706 vs 973707, 차이 1).
    diff = abs(summed.raw_value - total.raw_value)
    assert diff <= 2, (
        f"연령대 합계({summed.raw_value}) != 전체({total.raw_value}), 차이={diff} (허용 오차 2 초과)"
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


def case_17_future_year_should_fail():
    """아직 발표되지 않은 미래 연도를 요청하면 KOSIS가 자체적으로 [30] 데이터 없음 에러를
    내야 정상 (엣지케이스: 5단계가 7단계에 조용히 엉뚱한 시점 값을 넘기지 않는지 확인)."""
    try:
        client("DT_1DA7102S", {"period": "2099", "gender": "전체", "age": "청년(15~29세)"})
    except KosisApiError:
        return  # 기대한 실패
    raise AssertionError("미래 연도인데 예외가 안 났음")


# ---------------------------------------------------------------------------
# period 형식 검증 (_validate_period_format) — 네트워크 호출 없이 순수 로직만 확인.
# prdSe(Y/M/Q)와 period 자릿수가 안 맞으면 KOSIS에 요청을 보내기도 전에
# KosisApiError로 막혀야 한다 ("202401 vs 2024" 형식 불일치 버그의 재발 방지).
# ---------------------------------------------------------------------------

def case_18_period_format_year_ok():
    _validate_period_format("DT_1DA7102S", "Y", "2024")  # 예외 없이 통과해야 함


def case_19_period_format_month_ok():
    _validate_period_format("DT_402Y014", "M", "202412")


def case_20_period_format_quarter_ok():
    _validate_period_format("DT_1TEC_P112", "Q", "202403")


def case_21_period_format_month_table_with_year_period_should_fail():
    """월간(M) 표에 연도만(4자리) 들어오면 형식 불일치로 에러가 나야 정상 —
    실제로 재현된 '202401 vs 2024' 버그를 사전에 막는 케이스."""
    try:
        _validate_period_format("DT_402Y014", "M", "2024")
    except KosisApiError:
        return  # 기대한 실패
    raise AssertionError("월간 표에 4자리 연도 period인데 예외가 안 났음")


def case_22_period_format_year_table_with_month_period_should_fail():
    try:
        _validate_period_format("DT_1DA7102S", "Y", "202401")
    except KosisApiError:
        return  # 기대한 실패
    raise AssertionError("연간 표에 6자리 period인데 예외가 안 났음")


def case_23_period_format_quarter_invalid_suffix_should_fail():
    """분기 표인데 분기 접미사가 01~04 범위 밖(예: 05)이면 형식 불일치로 에러."""
    try:
        _validate_period_format("DT_1TEC_P112", "Q", "202405")
    except KosisApiError:
        return  # 기대한 실패
    raise AssertionError("분기 표에 잘못된 분기 접미사인데 예외가 안 났음")


def case_24_period_format_unknown_prdse_skipped():
    """F(격년)/D(일단위)처럼 아직 형식 규칙이 정의되지 않은 prdSe는 검증을 건너뛰어야
    한다 — 규칙이 없다고 조용히 틀린 값을 넘기는 게 아니라, 이 함수가 판단할 수 있는
    범위 밖이라는 뜻(자동 변환/보정과는 다름)."""
    _validate_period_format("DT_1SSSA022R", "F", "anything")
    _validate_period_format("DT_731Y001", "D", "20250502")


def case_25_period_format_call_integration_should_fail_before_network():
    """__call__() 레벨에서도 검증이 적용되는지 확인 — 월간 표(DT_402Y014)에 4자리
    연도만 period로 넘기면 실제 HTTP 요청 전에 KosisApiError가 나야 정상.

    2026-08-21 수정: table_params.json의 DT_402Y014는 prdSe=["Y","Q","M"]로 등록돼
    있고(실측 확인: 실제로 prdSe=Y 호출도 성공함 — 표의 "_period_note"("월간만 제공")가
    낡은 주석이었을 뿐), slots에 prd_se를 안 넘기면 _default_prd_se()가 "Y"를 우선
    선택해서 "2024"(4자리)가 오히려 유효한 형식이 돼버려 이 테스트의 전제가 깨졌다.
    이 테스트가 검증하려는 건 "표의 실제 기본 prdSe가 뭐든, 명시적으로 다른 prdSe를
    요청했는데 period 형식이 안 맞으면 막혀야 한다"이므로, prd_se="M"을 명시해서 그
    시나리오를 결정적으로 재현한다."""
    try:
        client("DT_402Y014", {"period": "2024", "prd_se": "M"})
    except KosisApiError as e:
        assert "DT_402Y014" in str(e) and "prdSe" in str(e)
        return  # 기대한 실패
    raise AssertionError("월간 표에 연도만 넘겼는데 __call__에서 예외가 안 났음")


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
    case_17_future_year_should_fail,
    case_18_period_format_year_ok,
    case_19_period_format_month_ok,
    case_20_period_format_quarter_ok,
    case_21_period_format_month_table_with_year_period_should_fail,
    case_22_period_format_year_table_with_month_period_should_fail,
    case_23_period_format_quarter_invalid_suffix_should_fail,
    case_24_period_format_unknown_prdse_skipped,
    case_25_period_format_call_integration_should_fail_before_network,
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
