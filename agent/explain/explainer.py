"""
agent/explain/explainer.py — 8단계 검증 결과 설명 생성 (LLM 기반 최종 설명)

팀 계약(interfaces.py) 기준:
    입력: Claim + TableCandidate + ComputedResult + Verdict
    출력: Explanation
    모델: HCX-007 또는 RAG Reasoning 모델 (최종 1회 호출)

※ 판정(7단계 judge.py)은 이미 끝난 상태로 들어옵니다. 이 단계는 "왜 그렇게 판정했는지"를
  사람이 읽을 수 있게 풀어 설명하는 것이 유일한 역할이며, 판정 자체를 다시 내리거나 뒤집지
  않습니다. 설명에는 반드시 (1)근거통계 (2)계산방식 (3)판정이유 (4)한계 4가지가 포함돼야
  하고, 특히 verdict="판단불가"일 때 한계(limitation)가 비어있으면 얼버무린 것으로 보고
  ExplainerError를 던집니다 (검증자 리뷰 단계에서 이게 핵심 정보이기 때문).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

try:
    from interfaces import Claim, TableCandidate, ComputedResult, Verdict, Explanation
except ImportError:
    try:
        from agent.interfaces import Claim, TableCandidate, ComputedResult, Verdict, Explanation
    except ImportError:  # 단독 실행/테스트용 폴백
        from dataclasses import dataclass, field
        from typing import Literal

        VerdictType = Literal["일치", "불일치", "판단불가"]  # type: ignore[misc]

        @dataclass
        class Claim:  # type: ignore[no-redef]
            sentence: str
            claim_type: str
            period: Optional[str] = None
            unit: Optional[str] = None
            population: Optional[str] = None

        @dataclass
        class TableCandidate:  # type: ignore[no-redef]
            table_id: str
            table_name: str
            score: float
            required_slots: list = field(default_factory=list)
            source_meta: Optional[str] = None

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

        @dataclass
        class Explanation:  # type: ignore[no-redef]
            claim_sentence: str
            table_name: str
            calc_summary: str
            verdict: str
            explanation_text: str
            limitation: Optional[str] = None

try:
    from agent.preprocessing.hcx_client import call_hcx
except ImportError:
    from preprocessing.hcx_client import call_hcx  # type: ignore[no-redef]


MODEL = "HCX-007"  # RAG Reasoning 모델로 교체할 경우 explain(..., model="...")로 덮어쓰면 됨
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "explainer_prompt.txt"
SYSTEM_PROMPT = "아래 지시사항을 정확히 따르고, 반드시 지정된 JSON 형식으로만 응답하세요."


class ExplainerError(RuntimeError):
    """LLM 설명 생성 응답을 JSON으로 파싱하지 못했거나, 필수 요소(특히 판단불가의 한계)가 빠진 경우."""


def _load_prompt_template(path: Path = PROMPT_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. 설명 생성 프롬프트를 먼저 작성해야 합니다.")
    return path.read_text(encoding="utf-8")


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ExplainerError(f"응답에서 JSON 객체를 찾지 못했습니다: {text!r}")
    return json.loads(match.group(0))


def _build_calc_summary(computed: ComputedResult) -> str:
    """계산 과정 요약 (예: "2023~2024 기준 증감률 -3.2%"). Explanation.calc_summary에 그대로 씀."""
    return f"{computed.period} 기준 {computed.calc_type} {computed.raw_value}{computed.unit}"


def explain(
    claim: Claim,
    table: TableCandidate,
    computed: ComputedResult,
    verdict: Verdict,
    *,
    model: str = MODEL,
) -> Explanation:
    """8단계 메인 진입점. 이미 나온 판정(Verdict)을 사람이 읽을 수 있는 설명으로 풀어냄."""
    template = _load_prompt_template()
    prompt = (
        template.replace("{claim_sentence}", claim.sentence)
        .replace("{table_name}", table.table_name)
        .replace("{source_meta}", table.source_meta or "출처 정보 없음")
        .replace("{computed_calc_type}", computed.calc_type)
        .replace("{computed_value}", str(computed.raw_value))
        .replace("{computed_unit}", computed.unit)
        .replace("{computed_period}", computed.period)
        .replace("{verdict}", verdict.verdict)
        .replace("{gap_type}", verdict.gap_type or "없음")
        .replace("{verdict_reason}", verdict.reason)
    )

    reply = call_hcx(model=model, system_prompt=SYSTEM_PROMPT, user_content=prompt, max_tokens=1024)

    try:
        parsed = _extract_json_object(reply)
        explanation_text = str(parsed["explanation_text"]).strip()
        limitation = parsed.get("limitation")
        if limitation in (None, "None", "none", ""):
            limitation = None
        else:
            limitation = str(limitation).strip()
    except (KeyError, json.JSONDecodeError) as e:
        raise ExplainerError(f"응답 파싱 실패: {reply!r}") from e

    if not explanation_text:
        raise ExplainerError("explanation_text가 비어있습니다 — LLM이 설명을 생성하지 못함")

    if verdict.verdict == "판단불가" and not limitation:
        # 15:00~15:30 체크리스트 대응: "판단불가" 케이스는 왜 판단이 어려운지가 얼버무려지지
        # 않고 명확히 나와야 함. limitation이 비면 사람 리뷰 없이 조용히 넘기지 않고 에러.
        raise ExplainerError(
            f"판단불가 케이스인데 limitation이 비어있습니다 (얼버무림 의심): {parsed!r}"
        )

    return Explanation(
        claim_sentence=claim.sentence,
        table_name=table.table_name,
        calc_summary=_build_calc_summary(computed),
        verdict=verdict.verdict,
        explanation_text=explanation_text,
        limitation=limitation,
    )


if __name__ == "__main__":
    #   python -m agent.explain.explainer

    # 케이스 1 — 일치
    claim1 = Claim(
        sentence="지난달 소비자물가가 전년 동월 대비 2.2% 오른 것으로 나타났다.",
        claim_type="증감률", period="2024년 12월", unit="%", population="소비자물가",
    )
    table1 = TableCandidate(
        table_id="DT_1J22003", table_name="소비자물가지수(2020=100)", score=1.0,
        required_slots=["시점"], source_meta="통계청 소비자물가조사",
    )
    computed1 = ComputedResult(calc_type="증감률", raw_value=2.3, unit="%", period="2023~2024")
    verdict1 = Verdict(verdict="일치", gap_type=None, reason="0.1%p 차이는 오차 범위 내이고 방향·규모가 기사 주장과 실질적으로 같다.")
    exp1 = explain(claim1, table1, computed1, verdict1)
    print(f"[케이스1 - 일치]\n{exp1}\n")
    assert exp1.verdict == "일치" and exp1.explanation_text

    # 케이스 2 — 불일치(과장표현)
    claim2 = Claim(
        sentence="최저임금이 작년보다 무려 두 배 가까이 올랐다.",
        claim_type="증감률", period="2025년", unit="%", population="최저임금",
    )
    table2 = TableCandidate(
        table_id="DT_MINWAGE", table_name="최저임금 현황", score=1.0,
        required_slots=["시점"], source_meta="고용노동부",
    )
    computed2 = ComputedResult(calc_type="증감률", raw_value=12.0, unit="%", period="2024~2025")
    verdict2 = Verdict(verdict="불일치", gap_type="과장표현", reason="실제 증가율은 12.0%로 '두 배 가까이'라는 표현과 크게 어긋나는 과장이다.")
    exp2 = explain(claim2, table2, computed2, verdict2)
    print(f"[케이스2 - 불일치]\n{exp2}\n")
    assert exp2.verdict == "불일치" and exp2.explanation_text

    # 케이스 3 — 판단불가: limitation이 반드시 채워지는지 확인 (핵심 검증 포인트)
    claim3 = Claim(
        sentence="역대 최대 규모의 예산이 편성됐다.",
        claim_type="규모", period="2025년", unit=None, population="예산",
    )
    table3 = TableCandidate(
        table_id="DT_BUDGET", table_name="국가 예산안 총지출 규모", score=1.0,
        required_slots=["시점"], source_meta="기획재정부",
    )
    computed3 = ComputedResult(calc_type="규모", raw_value=656.6, unit="조원", period="2025")
    verdict3 = Verdict(verdict="판단불가", gap_type="수치", reason="기사 문장에 비교 가능한 구체적 수치가 없어 통계값과 직접 대조할 수 없다.")
    exp3 = explain(claim3, table3, computed3, verdict3)
    print(f"[케이스3 - 판단불가]\n{exp3}\n")
    assert exp3.verdict == "판단불가"
    assert exp3.limitation, "판단불가인데 limitation이 비어있음 (방어 로직 실패)"
    print("  → 통과: '판단불가' 케이스에서 한계(limitation)가 얼버무려지지 않고 채워짐.")
