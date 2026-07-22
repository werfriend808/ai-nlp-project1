"""
agent/kosis/calculator.py — 표 연산 (6단계)

팀 계약(interfaces.py) 기준:
    입력: KosisApiResponse (복수 가능 — 합계/비율/증감 계산 시)
    출력: ComputedResult
    모델 불필요 — 반드시 코드(파이썬 연산)로 계산. LLM은 8단계 "설명"에서만 사용.

※ interfaces.py는 "복수의 KosisApiResponse를 받는다"고만 돼 있고, 그 리스트 안에서
  어떤 게 분자/분모인지, 어떤 게 기준시점/비교시점인지는 명시하지 않습니다. 그래서 아래
  4개 함수는 역할을 인자 이름으로 명확히 구분해서 받습니다(numerator/denominator,
  base/target). 이 컨벤션은 team과 한번 맞춰보는 게 좋습니다 — interfaces.py 자체를
  건드린 건 아니고, ComputedResult/KosisApiResponse 필드는 그대로 씁니다.
"""

from __future__ import annotations

try:
    from interfaces import KosisApiResponse, ComputedResult, CalcType
except ImportError:
    try:
        from agent.interfaces import KosisApiResponse, ComputedResult, CalcType
    except ImportError:  # 단독 실행/테스트용 폴백
        from dataclasses import dataclass
        from typing import Literal, Optional

        CalcType = Literal["합계", "비율", "증감", "증감률"]  # type: ignore[misc]

        @dataclass
        class KosisApiResponse:  # type: ignore[no-redef]
            raw_value: float
            unit: str
            period: str
            org_id: str
            itm_id: str
            obj_l1: Optional[str] = None
            obj_l2: Optional[str] = None
            prd_se: Optional[str] = None

        @dataclass
        class ComputedResult:  # type: ignore[no-redef]
            calc_type: str
            raw_value: float
            unit: str
            period: str


class CalculationError(RuntimeError):
    """단위가 안 맞거나, 분모가 0이거나, 계산에 필요한 값이 없을 때."""


def _check_same_unit(responses: list[KosisApiResponse]) -> str:
    units = {r.unit for r in responses}
    if len(units) > 1:
        raise CalculationError(f"단위가 서로 다른 값들을 더할 수 없습니다: {units}")
    return next(iter(units))


class KosisCalculator:
    """6단계 계산기. calc_type별로 역할이 다른 전용 메서드를 둠."""

    def compute_sum(self, responses: list[KosisApiResponse]) -> ComputedResult:
        """합계: 같은 단위의 값 여러 개를 더함 (예: 연령대별 인구 합계)."""
        if not responses:
            raise CalculationError("합계 계산에는 최소 1개 이상의 KosisApiResponse가 필요합니다.")
        unit = _check_same_unit(responses)
        total = sum(r.raw_value for r in responses)
        return ComputedResult(calc_type="합계", raw_value=total, unit=unit, period=responses[0].period)

    def compute_ratio(
        self, numerator: list[KosisApiResponse], denominator: KosisApiResponse
    ) -> ComputedResult:
        """비율: numerator(복수 가능, 합산됨) / denominator * 100 (%)."""
        if not numerator:
            raise CalculationError("비율 계산에는 numerator가 최소 1개 필요합니다.")
        _check_same_unit(numerator + [denominator])
        num_total = sum(r.raw_value for r in numerator)
        if denominator.raw_value == 0:
            raise CalculationError("분모가 0이라 비율을 계산할 수 없습니다.")
        ratio = num_total / denominator.raw_value * 100
        return ComputedResult(
            calc_type="비율", raw_value=round(ratio, 1), unit="%", period=denominator.period
        )

    def compute_change(self, base: KosisApiResponse, target: KosisApiResponse) -> ComputedResult:
        """증감: target - base (같은 단위 그대로, 절대 증감분)."""
        _check_same_unit([base, target])
        diff = target.raw_value - base.raw_value
        return ComputedResult(
            calc_type="증감", raw_value=diff, unit=base.unit, period=f"{base.period}~{target.period}"
        )

    def compute_change_rate(self, base: KosisApiResponse, target: KosisApiResponse) -> ComputedResult:
        """증감률: (target - base) / base * 100 (%)."""
        if base.raw_value == 0:
            raise CalculationError("기준 시점 값이 0이라 증감률을 계산할 수 없습니다.")
        rate = (target.raw_value - base.raw_value) / base.raw_value * 100
        return ComputedResult(
            calc_type="증감률", raw_value=round(rate, 1), unit="%", period=f"{base.period}~{target.period}"
        )


if __name__ == "__main__":
    # 브리프 워크드 예시(65세 이상 고령 농가 비율 64.2%)를 KosisApiResponse 리스트로 재현.
    # 실제로는 이 4개 KosisApiResponse가 api_client.py를 4번 호출해서 나온 결과라고 가정.
    def r(value, period="2024", unit="가구"):
        return KosisApiResponse(raw_value=value, unit=unit, period=period, org_id="101", itm_id="T00")

    calc = KosisCalculator()

    numerator = [r(33840), r(27218), r(22904), r(22915)]  # 65~69, 70~74, 75~79, 80+
    denominator = r(166558)

    result = calc.compute_ratio(numerator, denominator)
    print(result)
    assert result.raw_value == 64.2, f"브리프 예시(64.2%)와 다릅니다: {result.raw_value}"
    print("✅ 비율 계산 정상 (64.2%)")

    # 증감률 예시 (배추 가격, 두 시점 비교)
    base = r(125.4, period="202401", unit="2020=100")
    target = r(98.7, period="202402", unit="2020=100")
    result2 = calc.compute_change_rate(base, target)
    print(result2)
    assert result2.raw_value == -21.3, f"증감률 예시와 다릅니다: {result2.raw_value}"
    print("✅ 증감률 계산 정상 (-21.3%)")