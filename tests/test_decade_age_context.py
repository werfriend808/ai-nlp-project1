"""
tests/test_decade_age_context.py — 나이대("숫자0대") 오탐 방지 문맥 확인 단위 테스트

agent/orchestrator/agent_chat.py의 _resolve_decade_age_codes()가 "숫자0대" 패턴을 진짜
나이대로 오인하는 문제를 방지하기 위해 추가한 방어 3단계를 검증한다:
  1) 정규식 자체(_DECADE_EXPR_RE)에 (?<!\\d) 추가 — "1082만1480대"처럼 큰 숫자 끝자리가
     매칭되는 걸 애초에 막음.
  2) 1차 규칙(_check_decade_age_context) — 뒤에 조직/순위 명사(기업/은행 등)가 오면 확실히
     아님, 뒤에 나이 관련 단어(후반/인구 등)나 조사로 끝나면 확실히 맞음.
  3) 2차 LLM(_confirm_decade_age_with_llm) — 위 두 규칙 다 신호가 없는 애매한 경우만 확인.

전부 2026-08-04 실측(실제 기사 1005건/claim 3181건 분석, full_coverage_result.jsonl)에서
발견된 진짜 오탐/정탑 문장을 그대로 재현 케이스로 사용한다.

실행 (프로젝트 루트에서, 케이스 04는 실제 HCX API 호출):
    python -m tests.test_decade_age_context
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.orchestrator.agent_chat import _resolve_decade_age_codes

TABLE_PARAMS_PATH = Path(__file__).parent.parent / "agent" / "kosis" / "table_params.json"
with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
    TABLE_PARAMS = json.load(f)

AGE_CODE_MAP = TABLE_PARAMS["DT_1B04005N"]["dimensions"]["age"]["code_map"]


def case_01_large_number_tail_should_not_match():
    """"1082만1480대"(자동차 판매 대수) — 정규식 lookbehind로 애초에 매칭 자체가 안 돼야 함."""
    sentence = (
        "도요타자동차는 30일 자회사 다이하쓰와 히노를 포함한 총판매량이 역대 최대였던 "
        "2023년(1123만대) 대비 3.7% 줄어든 1082만1480대를 나타냈다고 밝혔다."
    )
    result = _resolve_decade_age_codes(sentence, AGE_CODE_MAP)
    assert result is None, f"큰 숫자 끝자리가 나이대로 오탐됨: {result}"


def case_02_ranking_expression_should_be_rejected_by_rule():
    """"상위 10대 기업" — 1차 규칙(뒤에 '기업')이 확실히 아님으로 걸러야 함(LLM 호출 불필요)."""
    sentence = "올해 3분기 수출액 상위 10대 기업의 무역집중도가 40%로 역대 최고치를 기록했다."
    result = _resolve_decade_age_codes(sentence, AGE_CODE_MAP)
    assert result is None, f"'10대 기업'(순위)이 나이대로 오탐됨: {result}"


def case_03_genuine_age_with_particle_should_resolve_by_rule():
    """"40대의" — 조사로 끝나서 1차 규칙이 확실히 맞음으로 처리, 실제 코드가 채워져야 함."""
    sentence = "작년 4분기 말 기준 40대의 1인당 평균 은행 대출 잔액은 1억1073만원으로 역대 가장 높았다."
    result = _resolve_decade_age_codes(sentence, AGE_CODE_MAP)
    assert result is not None, "진짜 나이대 표현('40대의')이 걸러짐(과잉 방어)"
    label, codes = result
    assert "40" in label
    assert len(codes) == 2  # 40~44세, 45~49세


def case_04_ambiguous_real_case_should_be_confirmed_by_llm():
    """"20대 일자리" — 1차 규칙 신호가 없는 애매한 경우라 2차 LLM이 확인해서 맞다고 판단해야 함."""
    sentence = (
        "작년 3분기 30세 미만인 10·20대 일자리가 14만6000개 줄어 2017년 관련 통계 집계 "
        "이후 최대 폭으로 줄었다고 밝혔다."
    )
    result = _resolve_decade_age_codes(sentence, AGE_CODE_MAP)
    assert result is not None, "실제 나이대 표현('20대 일자리')이 LLM 확인 단계에서 잘못 거부됨"


CASES = [
    case_01_large_number_tail_should_not_match,
    case_02_ranking_expression_should_be_rejected_by_rule,
    case_03_genuine_age_with_particle_should_resolve_by_rule,
    case_04_ambiguous_real_case_should_be_confirmed_by_llm,
]


def main() -> None:
    print(f"총 {len(CASES)}건 실행 (케이스 04는 실제 HCX API 호출)")
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