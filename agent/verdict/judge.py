"""
agent/verdict/judge.py — 7단계 비교·판정 (일치/불일치/판단불가)

팀 계약(interfaces.py) 기준:
    입력: Claim(기사 수치 주장) + ComputedResult(KOSIS 계산값)
    출력: Verdict
    모델: 1차 필터는 코드 규칙, 애매 경계만 HCX-003(단순)/HCX-007(복합 추론) 위임

※ interfaces.py의 Claim은 sentence(원문 텍스트)만 있고 파싱된 숫자 필드가 없습니다.
  그래서 규칙 기반 1차 필터를 돌리려면 sentence에서 수치를 직접 뽑아내야 하는데
  (_extract_claim_number), 이건 100% 정확할 수 없는 정규식 기반 best-effort입니다.
  추출 실패/애매한 경우는 전부 LLM에 원문 문장 그대로 넘겨서 판단을 맡깁니다 —
  즉 규칙 필터는 "명확한 경우만 걸러내고, 나머지는 보수적으로 LLM에 위임"하는 방향으로
  설계했습니다 (반대로 하면 애매한 걸 코드가 잘못 확정 판정할 위험이 큼).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

try:
    from interfaces import Claim, ComputedResult, Verdict
except ImportError:
    try:
        from agent.interfaces import Claim, ComputedResult, Verdict
    except ImportError:  # 단독 실행/테스트용 폴백
        from dataclasses import dataclass
        from typing import Literal

        VerdictType = Literal["일치", "불일치", "판단불가"]  # type: ignore[misc]
        GapType = Optional[Literal["수치", "기간", "모집단", "과장표현"]]  # type: ignore[misc]

        @dataclass
        class Claim:  # type: ignore[no-redef]
            sentence: str
            claim_type: str
            period: Optional[str] = None
            unit: Optional[str] = None
            population: Optional[str] = None

        @dataclass
        class ComputedResult:  # type: ignore[no-redef]
            calc_type: str
            raw_value: float
            unit: str
            period: str

        @dataclass
        class Verdict:  # type: ignore[no-redef]
            verdict: str
            gap_type: Optional[str] = None
            reason: str = ""

try:
    from agent.preprocessing.hcx_client import call_hcx
except ImportError:
    from preprocessing.hcx_client import call_hcx  # type: ignore[no-redef]


MODEL_SIMPLE = "HCX-003"    # 애매 경계 단일 Claim vs ComputedResult 판정
MODEL_COMPLEX = "HCX-007"   # 여러 Claim/ComputedResult를 종합해야 하는 복합 추론
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "judge_prompt.txt"
SYSTEM_PROMPT = "아래 지시사항을 정확히 따르고, 반드시 지정된 JSON 형식으로만 응답하세요."

# 오차 허용범위: %류 주장은 %p(절대 차이), 그 외 단위는 상대오차(%)로 통일해서 같은 척도로 비교.
NUMERIC_TOLERANCE = 0.1
# tolerance의 5배(=0.5) 넘게 벌어지면 "명확히 큰 차이"로 보고 LLM 호출 없이 바로 불일치 확정.
CLEAR_GAP_MULTIPLIER = 5

_SCALE = {"조": 1e12, "억": 1e8, "만": 1e4, "천": 1e3}
_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(조|억|만|천)?")
_YEAR_RE = re.compile(r"(?:19|20)\d{2}")
_DIGITS_RE = re.compile(r"\d+")

# "1.1% 감소했다"처럼 claim_type="증감률" 문장은 숫자 자체엔 부호가 없고 방향은 단어로만
# 표현됨. ComputedResult 쪽(calculator.py compute_change/compute_change_rate)은 감소를
# 음수로 표현하므로, 같은 척도로 비교하려면 이 단어들을 보고 부호를 붙여줘야 함.
_DECREASE_WORDS = ("감소", "하락", "줄어", "축소", "내림", "내렸", "떨어", "낮아", "줄었")
_INCREASE_WORDS = ("증가", "상승", "올랐", "늘어", "확대", "늘었", "높아", "올림")


class JudgeError(RuntimeError):
    """LLM 판정 응답을 JSON으로 파싱하지 못한 경우."""


def _load_prompt_template(path: Path = PROMPT_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. D가 판정 프롬프트를 먼저 작성해야 합니다.")
    return path.read_text(encoding="utf-8")


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise JudgeError(f"응답에서 JSON 객체를 찾지 못했습니다: {text!r}")
    return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# 1차 필터 — 코드 규칙 (LLM 호출 없음)
# ---------------------------------------------------------------------------

def _to_value(num_str: str, scale: Optional[str]) -> float:
    value = float(num_str.replace(",", ""))
    return value * _SCALE[scale] if scale else value


def _extract_claim_number(claim: Claim) -> Optional[float]:
    """claim.sentence에서 핵심 수치를 정규식으로 뽑아냄 (best-effort).

    claim.unit이 있으면 그 단위 바로 앞 숫자를 우선 사용하고, 없으면 문장에서 찾은
    마지막 숫자를 사용합니다(주장 문장은 보통 "~55.8kg을 기록했다"처럼 핵심 수치가
    문장 뒤쪽에 옴). "2024년"처럼 연도로 보이는 4자리 숫자는 후보에서 제외합니다.
    """
    candidates: list[float] = []
    for m in _NUMBER_RE.finditer(claim.sentence):
        num_str, scale = m.groups()
        cleaned = num_str.replace(",", "")
        tail = claim.sentence[m.end() : m.end() + 1]
        if not scale and tail == "년" and _YEAR_RE.fullmatch(cleaned):
            continue  # "2024년" 같은 연도 표기는 비교 대상 수치가 아니므로 제외
        candidates.append(_to_value(num_str, scale))

    if not candidates:
        return None

    if claim.unit:
        unit_match = re.search(
            rf"(\d[\d,]*(?:\.\d+)?)\s*(조|억|만|천)?\s*{re.escape(claim.unit)}", claim.sentence
        )
        if unit_match:
            value = _to_value(unit_match.group(1), unit_match.group(2))
            return _apply_direction(value, claim)

    return _apply_direction(candidates[-1], claim)


def _apply_direction(value: float, claim: Claim) -> float:
    """claim_type="증감률"인 경우에만 감소/증가 단어를 보고 부호를 붙임."""
    if claim.claim_type != "증감률" or value < 0:
        return value
    if any(w in claim.sentence for w in _DECREASE_WORDS):
        return -value
    return value


def _is_percent(claim: Claim, computed: ComputedResult) -> bool:
    return computed.unit == "%" or claim.unit == "%" or claim.claim_type in ("증감률", "비율")


def _numeric_gap(claim_value: float, claim: Claim, computed: ComputedResult) -> float:
    """%류는 %p(절대 차이), 그 외는 상대오차(%)로 통일해서 반환."""
    if _is_percent(claim, computed):
        return abs(claim_value - computed.raw_value)
    if computed.raw_value == 0:
        return abs(claim_value - computed.raw_value)
    return abs(claim_value - computed.raw_value) / abs(computed.raw_value) * 100


def _period_granularity(period: Optional[str]) -> Optional[str]:
    """period 문자열이 "월" 단위인지 "년" 단위인지 best-effort로 추정.

    (참고: Day2 파이프라인 연결 테스트에서 실제로 발견된 문제 — 기사는 "전년 동월 대비"
    처럼 월 단위로 주장하는데, KOSIS 조회가 연간 평균 기준(prdSe=Y)으로만 이뤄져서
    비교 기준 자체가 다른 채로 판정될 뻔한 케이스가 있었음. 이 함수는 그 갭을 규칙
    단계에서 감지해서 LLM에 위임하기 위한 것.)
    """
    if not period:
        return None
    if "월" in period and "개월" not in period:
        return "월"
    for run in _DIGITS_RE.findall(period):
        if len(run) == 6:
            return "월"
    for run in _DIGITS_RE.findall(period):
        if len(run) == 4:
            return "년"
    return None


def _rule_based_verdict(claim: Claim, computed: ComputedResult) -> Optional[Verdict]:
    """규칙만으로 명확히 판단되면 Verdict를 반환하고, 애매하면 None(LLM 위임)을 반환."""
    claim_value = _extract_claim_number(claim)
    if claim_value is None:
        return None  # 수치 추출 실패 → 규칙으로 못 정함, LLM에 원문 문장 그대로 위임

    gap = _numeric_gap(claim_value, claim, computed)
    clear_threshold = NUMERIC_TOLERANCE * CLEAR_GAP_MULTIPLIER
    gap_unit = "%p" if _is_percent(claim, computed) else "%(상대오차)"

    if gap > clear_threshold:
        return Verdict(
            verdict="불일치",
            gap_type="수치",
            reason=(
                f"기사 수치({claim_value})와 통계 계산값({computed.raw_value}{computed.unit}) "
                f"차이가 {gap:.2f}{gap_unit}로 허용 오차를 크게 초과함 (규칙 기반 판정, LLM 미호출)."
            ),
        )

    if gap <= NUMERIC_TOLERANCE:
        # 엣지케이스 방어: computed.period가 아예 없으면(5단계 필드 누락 등) "시점 불일치 없음"
        # 으로 잘못 확정하지 말고 규칙 필터를 포기하고 LLM에 위임한다. (실제로 이 가드가 없으면
        # computed.period=""일 때도 "일치"로 확정 판정해버리는 버그가 있었음 — 검증 완료.)
        if not computed.period:
            return None
        claim_gran = _period_granularity(claim.period)
        computed_gran = _period_granularity(computed.period)
        period_mismatch = claim_gran is not None and computed_gran is not None and claim_gran != computed_gran
        unit_mismatch = (
            bool(claim.unit) and bool(computed.unit) and claim.unit != computed.unit
            and not _is_percent(claim, computed)
        )
        if not period_mismatch and not unit_mismatch:
            return Verdict(
                verdict="일치",
                gap_type=None,
                reason=(
                    f"기사 수치({claim_value})와 통계 계산값({computed.raw_value}{computed.unit}) "
                    f"차이가 허용 오차({NUMERIC_TOLERANCE}{gap_unit}) 이내이고 시점·단위 불일치도 "
                    "없음 (규칙 기반 판정, LLM 미호출)."
                ),
            )

    return None  # 애매한 경계(수치는 비슷한데 시점/단위가 다르거나 경계 근처) → LLM 위임


# ---------------------------------------------------------------------------
# 애매 경계 — HCX-003 판정
# ---------------------------------------------------------------------------

def _judge_with_llm(claim: Claim, computed: ComputedResult, *, model: str = MODEL_SIMPLE) -> Verdict:
    template = _load_prompt_template()
    prompt = (
        template.replace("{claim_sentence}", claim.sentence)
        .replace("{claim_type}", claim.claim_type)
        .replace("{claim_period}", claim.period or "명시 안 됨")
        .replace("{claim_unit}", claim.unit or "명시 안 됨")
        .replace("{claim_population}", claim.population or "명시 안 됨")
        .replace("{computed_calc_type}", computed.calc_type)
        .replace("{computed_value}", str(computed.raw_value))
        .replace("{computed_unit}", computed.unit)
        .replace("{computed_period}", computed.period)
    )

    reply = call_hcx(model=model, system_prompt=SYSTEM_PROMPT, user_content=prompt)

    try:
        parsed = _extract_json_object(reply)
        verdict = str(parsed["verdict"])
        if verdict not in ("일치", "불일치", "판단불가"):
            raise ValueError(f"알 수 없는 verdict 값: {verdict!r}")
        gap_type = parsed.get("gap_type")
        if gap_type in (None, "None", "none", ""):
            gap_type = None
        elif gap_type not in ("수치", "기간", "모집단", "과장표현"):
            raise ValueError(f"알 수 없는 gap_type 값: {gap_type!r}")
        return Verdict(verdict=verdict, gap_type=gap_type, reason=str(parsed.get("reason", "")))
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise JudgeError(f"응답 파싱 실패: {reply!r}") from e


def judge(claim: Claim, computed: ComputedResult, *, model: str = MODEL_SIMPLE) -> Verdict:
    """7단계 메인 진입점 (단일 Claim vs 단일 ComputedResult).

    1차로 코드 규칙(_rule_based_verdict)을 먼저 시도하고, 명확히 정해지지 않는
    애매한 경우에만 HCX-003을 호출합니다.
    """
    rule_verdict = _rule_based_verdict(claim, computed)
    if rule_verdict is not None:
        return rule_verdict
    return _judge_with_llm(claim, computed, model=model)


# ---------------------------------------------------------------------------
# 복잡 케이스 — HCX-007 승격
# ---------------------------------------------------------------------------

def needs_hybrid_reasoning(claims: list[Claim], computed_results: list[ComputedResult]) -> bool:
    """단일 Claim-ComputedResult 쌍으로 못 다루는 복합 케이스인지 판단.

    승격 조건 (둘 중 하나라도 해당하면 HCX-007로 올림):
      1. Claim이 여러 개 얽혀서 함께 봐야 함 (예: claim_type="비교"가 다른 Claim을 가리켜
         "A가 B의 절반" 같은 상호 비교 주장을 함 — 개별로는 판정 불가)
      2. 하나의 주장을 검증하는 데 ComputedResult가 여러 개 필요함 (여러 표를 종합해야
         계산이 완성되는 경우, 예: 두 통계표 비율 재계산·복수 시점 추세 종합)
    """
    if len(claims) > 1 and any(c.claim_type == "비교" for c in claims):
        return True
    if len(computed_results) > 1:
        return True
    return False


def judge_complex(
    claims: list[Claim], computed_results: list[ComputedResult], *, model: str = MODEL_COMPLEX
) -> Verdict:
    """복합 케이스 판정: 여러 Claim/ComputedResult를 한 번에 HCX-007에 넘겨 종합 추론시킴.

    judge_prompt.txt의 단일 Claim용 템플릿 대신, 목록을 그대로 나열한 프롬프트를 씁니다
    (단일 쌍 few-shot으로는 "여러 표 종합"까지 예시를 늘리기엔 프롬프트가 지나치게 커져서
    별도 구성 — 필요시 team 상의 후 전용 few-shot 프롬프트 파일로 분리 가능).
    """
    claims_block = "\n".join(
        f"- \"{c.sentence}\" (claim_type={c.claim_type}, period={c.period}, "
        f"unit={c.unit}, population={c.population})"
        for c in claims
    )
    computed_block = "\n".join(
        f"- {r.calc_type} {r.raw_value}{r.unit} ({r.period} 기준)" for r in computed_results
    )
    prompt = (
        "아래는 한 기사에서 나온 여러 개의 수치 주장과, 그걸 검증하기 위해 조회한 여러 개의 "
        "국가 공식 통계(KOSIS) 계산값입니다. 여러 주장/통계를 함께 종합적으로 추론해서 "
        "전체적으로 '일치/불일치/판단불가'를 하나로 판정하세요.\n\n"
        f"기사 주장 목록:\n{claims_block}\n\n"
        f"통계 계산값 목록:\n{computed_block}\n\n"
        '출력 형식 (JSON만 출력, 다른 텍스트 금지):\n'
        '{"verdict": "일치|불일치|판단불가", "gap_type": "수치|기간|모집단|과장표현" 또는 null, '
        '"reason": "여러 주장/통계를 종합한 판단 근거"}'
    )

    reply = call_hcx(model=model, system_prompt=SYSTEM_PROMPT, user_content=prompt, max_tokens=1536)

    try:
        parsed = _extract_json_object(reply)
        verdict = str(parsed["verdict"])
        if verdict not in ("일치", "불일치", "판단불가"):
            raise ValueError(f"알 수 없는 verdict 값: {verdict!r}")
        gap_type = parsed.get("gap_type")
        if gap_type in (None, "None", "none", ""):
            gap_type = None
        return Verdict(verdict=verdict, gap_type=gap_type, reason=str(parsed.get("reason", "")))
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise JudgeError(f"응답 파싱 실패: {reply!r}") from e


def judge_all(claims: list[Claim], computed_results: dict[int, ComputedResult]) -> list[Verdict]:
    """claims 인덱스 -> ComputedResult 매핑을 받아, 복합 케이스는 HCX-007로 한 번에,
    단순 케이스는 claim별로 judge()를 호출해서 Verdict 리스트를 돌려줌.

    (오케스트레이터가 여러 Claim을 모아서 7단계를 한 번에 돌릴 때 쓰라고 만든 편의 함수.
    복합 승격 시 관련된 모든 claim에 동일한 Verdict를 매깁니다.)
    """
    matched = [(i, c, computed_results[i]) for i, c in enumerate(claims) if i in computed_results]
    if not matched:
        return []

    if needs_hybrid_reasoning([c for _, c, _ in matched], [r for _, _, r in matched]):
        verdict = judge_complex([c for _, c, _ in matched], [r for _, _, r in matched])
        return [verdict for _ in matched]

    return [judge(c, r) for _, c, r in matched]


if __name__ == "__main__":
    #   python -m agent.verdict.judge

    # 케이스 1 — 규칙만으로 "일치" (LLM 호출 없음)
    claim1 = Claim(
        sentence="작년 국민 1인당 쌀 소비량은 1년 전보다 1.1% 감소했다.",
        claim_type="증감률",
        period="2024년",
        unit="%",
        population="국민 1인당",
    )
    computed1 = ComputedResult(calc_type="증감률", raw_value=-1.15, unit="%", period="2023~2024")
    result1 = judge(claim1, computed1)
    print(f"[케이스1 - 일치 예상] {result1}")
    assert result1.verdict == "일치"

    # 케이스 2 — 규칙만으로 "불일치" (기사 10% vs 통계 7.2%, 예시 그대로, LLM 호출 없음)
    claim2 = Claim(
        sentence="청년 실업률이 지난달 10%까지 치솟았다.",
        claim_type="규모",
        period="2025년 1월",
        unit="%",
        population="청년",
    )
    computed2 = ComputedResult(calc_type="합계", raw_value=7.2, unit="%", period="2025")
    result2 = judge(claim2, computed2)
    print(f"[케이스2 - 불일치 예상] {result2}")
    assert result2.verdict == "불일치" and result2.gap_type == "수치"

    # 케이스 3 — 수치는 비슷한데 기간 단위(월 vs 연)가 달라 애매 경계 → HCX-003 호출
    claim3 = Claim(
        sentence="지난달 소비자물가가 전년 동월 대비 2.2% 오른 것으로 나타났다.",
        claim_type="증감률",
        period="2024년 12월",
        unit="%",
        population="소비자물가",
    )
    computed3 = ComputedResult(calc_type="증감률", raw_value=2.3, unit="%", period="2023~2024")
    print("[케이스3 - 애매 경계, HCX-003 호출 시도]")
    try:
        result3 = judge(claim3, computed3)
        print(f"  결과: {result3}")
    except Exception as e:
        print(f"  HCX-003 호출 실패 ({type(e).__name__}: {e}) — API 키/네트워크 확인 필요")

    # 케이스 4 — 복합 케이스(Claim 2개 + ComputedResult 2개) → HCX-007 승격 조건 확인
    claim4a = Claim(
        sentence="올해 최저임금은 시간당 10,030원이다.", claim_type="규모", period="2025년", unit="원", population="최저임금"
    )
    claim4b = Claim(
        sentence="이는 작년보다 1.7% 오른 수치다.", claim_type="비교", period="2025년", unit="%", population="최저임금"
    )
    computed4a = ComputedResult(calc_type="합계", raw_value=10030, unit="원", period="2025")
    computed4b = ComputedResult(calc_type="증감률", raw_value=1.7, unit="%", period="2024~2025")
    escalate = needs_hybrid_reasoning([claim4a, claim4b], [computed4a, computed4b])
    print(f"[케이스4 - 승격 조건 확인] needs_hybrid_reasoning={escalate}")
    assert escalate is True
    print("  → 조건 충족 확인됨 (claim_type='비교' 포함 + Claim 2개). 실제 HCX-007 호출은 judge_complex()로 별도 실행.")

    # 케이스 5 — 엣지케이스 방어: computed.period 누락(5단계 필드 누락 등) 시 규칙으로
    # 잘못 확정하지 않고 LLM에 위임하는지 확인 (수정 전엔 "일치"로 잘못 확정되던 버그였음)
    claim5 = Claim(
        sentence="작년 국민 1인당 쌀 소비량은 1년 전보다 1.1% 감소했다.",
        claim_type="증감률", period="2024년", unit="%", population="국민 1인당",
    )
    computed5 = ComputedResult(calc_type="증감률", raw_value=-1.15, unit="%", period="")
    rb5 = _rule_based_verdict(claim5, computed5)
    print(f"[케이스5 - period 누락 방어] 규칙 기반 1차 필터 결과: {rb5}")
    assert rb5 is None, "period 누락인데도 규칙으로 확정해버림 (엣지케이스 방어 회귀)"
    print("  → 통과: 시점 정보 없음을 '불일치 없음'으로 오인하지 않고 LLM에 위임함.")
