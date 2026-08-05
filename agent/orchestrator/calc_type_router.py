"""
agent/orchestrator/calc_type_router.py — claim_type(2단계 결과) -> calc_type(6단계 계산종류)
라우팅

배경: agent/interfaces.py의 ClaimType(4종: 규모/증감률/비교/전망)과 agent/kosis/calculator.py의
CalcType(5종: 합계/비율/증감/증감률/최댓값검증) 사이에 매핑 로직이 없었다. 특히 claim_type="규모"
(뉴스기사 100건 실측 조사에서 claim의 61%, 최다수)가 calculator.py 어떤 메서드에도 대응이
안 되는 문제가 있었다 — 자세한 근거는 개인/CALC_TYPE_ROUTING_DESIGN.md 참고.

2단계 완료 즉시 실행 가능한 독립 함수로 분리했다(3단계 표 매핑·4단계 슬롯필링 결과를
기다릴 필요 없음 — claim.claim_type/claim.sentence만 보고 결정):
    - calculator.py(6단계)는 "모델 불필요, 반드시 코드 연산"인 순수 계산 레이어라 판단
      (라우팅) 로직을 섞으면 계약 위반
    - slot_filler(4단계)에 넣으면 되묻기가 늘어나는데, claim 자체가 이미 답을 담고 있는
      경우가 많아 라우터가 먼저 정하고 못 정한 것만 되물어야 함
    - claim_extractor(2단계, LLM)에 넣으면 프롬프트 복잡도·비결정성만 늘어남 — 규칙으로
      되는 판단을 LLM에 맡길 이유 없음

라우팅 테이블:
    규모   + 극값 패턴 있음 -> "최댓값검증" | "최솟값검증" (최고/최대 vs 최저/최소)
    규모   + 극값 패턴 없음 -> "단순조회"
    증감률 -> "증감률"
    비교   -> "증감" (compute_change류는 시간 대신 두 대상을 base/target으로 그대로 재사용 가능)
    전망, 스키마 밖 값(None 포함) -> None (호출부가 5~8단계를 건너뛰고 즉시 판단불가 처리하라는 신호)

agent/interfaces.py의 CalcType 리터럴엔 원래 "단순조회"/"최솟값검증"이 없었는데(파일 상단에
"팀 전체 합의 후 수정" 원칙 명시), 이 라우터 도입을 계기로 팀 확인 후 2026-08-05 두 값을
추가했다("단순조회"는 이미 batch_runner.py가 쓰고 있어서 계약 위반 상태였음).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional

from agent.interfaces import CalcType, Claim
from agent.shared.extreme_value_patterns import ALL_TIME_RE, N_YEARS_SINCE_RE, SINCE_EVENT_RE

_MAX_DIRECTION_RE = re.compile(r"최고|최대|최다")
_MIN_DIRECTION_RE = re.compile(r"최저|최소")


@dataclass(frozen=True)
class ExtremeValueSignal:
    is_extreme: bool
    direction: Optional[Literal["max", "min"]] = None


def detect_extreme_value_claim(sentence: str) -> ExtremeValueSignal:
    """문장이 극값("역대"/"N년 만에"/"코로나(19) 이후") 주장인지 + 최고/최저 방향을 판단한다.

    시작 연도 계산(article_year 필요)은 calc_type 결정 자체엔 불필요해서 여기서는 하지
    않는다 — 실제 시계열 조회는 agent_chat.py의 resolve_max_*_responses류가 담당한다.
    """
    is_extreme = bool(
        ALL_TIME_RE.search(sentence) or N_YEARS_SINCE_RE.search(sentence) or SINCE_EVENT_RE.search(sentence)
    )
    if not is_extreme:
        return ExtremeValueSignal(is_extreme=False)

    has_min = bool(_MIN_DIRECTION_RE.search(sentence))
    has_max = bool(_MAX_DIRECTION_RE.search(sentence))
    if has_min and not has_max:
        return ExtremeValueSignal(is_extreme=True, direction="min")
    # "최고/최대" 단어가 있는 정상 케이스뿐 아니라, 방향 단어가 아예 없는 드문 경우도
    # "max"를 기본값으로 삼는다 — compute_max_check가 기존에 유일하게 구현돼 있던
    # 경로였고, 실측 8건 표본에서도 "최저"류가 명시적으로 있을 때만 min 방향이었다.
    return ExtremeValueSignal(is_extreme=True, direction="max")


def route_calc_type(claim: Claim) -> Optional[CalcType]:
    """claim.claim_type + claim.sentence만으로 calc_type을 결정한다. 3·4단계(표 매핑/슬롯
    필링) 결과 불필요.

    None을 반환하면 "이 claim은 규칙으로 라우팅할 수 없다"는 뜻 — 호출부(예:
    batch_runner.py)가 5~8단계를 건너뛰고 즉시 판단불가로 처리해야 한다. claim_type이
    "전망"이거나 스키마 밖 값(claim_extractor._normalize_claim_type이 이미 None으로
    정규화한 경우 포함)이면 전부 여기로 떨어진다.
    """
    claim_type = claim.claim_type

    if claim_type == "증감률":
        return "증감률"

    if claim_type == "비교":
        return "증감"

    if claim_type == "규모":
        signal = detect_extreme_value_claim(claim.sentence)
        if not signal.is_extreme:
            return "단순조회"
        return "최댓값검증" if signal.direction == "max" else "최솟값검증"

    # claim_type == "전망" 이거나 None/스키마 밖 값 전부 여기로.
    return None


if __name__ == "__main__":
    #   python -m agent.orchestrator.calc_type_router
    samples = [
        Claim(sentence="작년 출생아 수는 23만8천명으로 전년보다 늘었다.", claim_type="규모"),
        Claim(sentence="저축은행 연체율이 9년 만에 최고치를 기록했다.", claim_type="규모"),
        Claim(sentence="제조업 취업자 수가 12년 만에 최저치로 떨어졌다.", claim_type="규모"),
        Claim(sentence="합계 출산율은 역대 최저였다.", claim_type="증감률"),
        Claim(sentence="수도권과 비수도권 인구 격차가 벌어졌다.", claim_type="비교"),
        Claim(sentence="향후 인구는 계속 감소할 것으로 전망된다.", claim_type="전망"),
        Claim(sentence="배경 설명 문장.", claim_type=None),
    ]
    for c in samples:
        print(f"{c.claim_type!r:>8} | {route_calc_type(c)!r:<10} | {c.sentence}")
