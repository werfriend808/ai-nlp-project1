"""
tests/test_table_dimension_hints_llm.py — 4단계 슬롯필링 LLM 2차 폴백 단위 테스트

agent/orchestrator/agent_chat.py의 _extract_table_dimension_hints_llm()을 검증합니다.
1차(코드 기반, test_table_dimension_hints.py)와 달리 실제 HCX-DASH-002 API를 호출하므로
느리고(케이스당 수 초) 완전히 결정적이지 않을 수 있으나, 아래 3개 케이스는 2026-08-05
개발 중 실제로 문제가 발견/확인된 것들이라 회귀 확인용으로 반드시 필요합니다.

- GNI 케이스(긍정): 1차가 원리상 못 잡는 진짜 동의어를 2차가 정확히 채워야 함.
- "20대→전체" 케이스(부정): 정확한 값이 후보에 없을 때 억지로 포괄값을 채우면 안 됨
  (_GENERIC_FALLBACK_LABELS 방어 확인 — 처음 발견 당시 LLM이 "전체"를 잘못 채웠었음).
- "비농가" 환각 케이스(부정): 후보에 실존하는 값이어도 문장에 근거가 없으면 채우면 안 됨
  (quote 원문 대조 방어 확인 — 위와 다른 방어 메커니즘이 걸러야 하는 케이스).

실행 (프로젝트 루트에서, HCX_API_KEY 필요):
    python -m tests.test_table_dimension_hints_llm
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.orchestrator.agent_chat import _extract_table_dimension_hints_llm

TABLE_PARAMS_PATH = Path(__file__).parent.parent / "agent" / "kosis" / "table_params.json"
with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
    TABLE_PARAMS = json.load(f)


def case_01_gni_synonym_should_fill():
    """'국민총소득' vs code_map의 'GNI달러' — 표현이 달라 1차로는 못 잡는 진짜 동의어."""
    sentence = "우리나라 작년 1인당 국민총소득(GNI)이 2년 연속 일본을 앞선 것으로 집계됐다(3만6745달러)."
    hints = _extract_table_dimension_hints_llm(sentence, "DT_200Y001", TABLE_PARAMS, ["gni_item"])
    assert hints.get("gni_item") == "1인당GNI달러", f"GNI 동의어 매칭 실패: {hints}"


def case_02_age_20s_should_not_fallback_to_generic():
    """DT_1B04005N의 age 후보는 '전체'/'계'뿐(20대 코드 자체가 없음) — 절대 채우면 안 됨.

    2026-08-05 최초 발견 당시 LLM이 이 케이스에서 '전체'(완전히 다른 값)를 답으로 냈던
    실제 버그 재현 케이스. _GENERIC_FALLBACK_LABELS로 방어 중."""
    sentence = "지난해 20대 인구는 630만2000명으로, 전년보다 19만3000명 줄었다."
    hints = _extract_table_dimension_hints_llm(sentence, "DT_1B04005N", TABLE_PARAMS, ["age"])
    assert "age" not in hints, f"20대 케이스에서 age가 채워지면 안 됨(포괄값 오답 위험): {hints}"


def case_03_bi_nongga_hallucination_should_not_fill():
    """DT_1DA7001S의 gender 후보 중 '비농가'는 실존값이지만 이 문장과 전혀 무관 — 절대
    채우면 안 됨. 2026-08-05 최초 발견 당시 LLM이 근거 없이 '비농가'를 답으로 냈던 실제
    환각 재현 케이스. quote(근거 문구) 원문 대조로 방어 중 — 2번과 다른 방어 메커니즘."""
    sentence = "인구 대비 취업자 수 비율을 뜻하는 고용률은 지난달 61.4%로 1년 전(61.7%)에 비해 0.3%포인트 떨어졌다."
    hints = _extract_table_dimension_hints_llm(sentence, "DT_1DA7001S", TABLE_PARAMS, ["gender"])
    assert "gender" not in hints, f"근거 없는 '비농가' 환각이 채워지면 안 됨: {hints}"


CASES = [
    case_01_gni_synonym_should_fill,
    case_02_age_20s_should_not_fallback_to_generic,
    case_03_bi_nongga_hallucination_should_not_fill,
]


def main() -> None:
    print(f"총 {len(CASES)}건 실행 (실제 HCX-DASH-002 API 호출, 느릴 수 있음)")
    print("=" * 70)
    results = []
    for case in CASES:
        try:
            case()
            results.append((case.__name__, "PASS", ""))
        except Exception as e:  # noqa: BLE001 - 테스트 러너라 실패 원인만 보고 계속 진행
            results.append((case.__name__, "FAIL", str(e)))

    for name, status, detail in results:
        mark = "✅" if status == "PASS" else "❌"
        print(f"{mark} {status}  {name}" + (f" — {detail}" if detail else ""))

    passed = sum(1 for _, s, _ in results if s == "PASS")
    print(f"\n{passed}/{len(results)} PASS")


if __name__ == "__main__":
    main()