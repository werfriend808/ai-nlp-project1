"""
agent/pipeline/chatbot.py — 실전 2: 자연어를 KOSIS API 요청으로 완성하는 대화형 챗봇

기사 파이프라인(1~8단계)과는 별개로, 사용자가 자연어로 통계를 물어보면:
  1) 문장으로 통계표 후보를 찾고 (3단계 매핑 재사용 — keyword_search + embedding_search + rerank)
  2) 슬롯(period/region/calc_type)을 채우고, 부족하면 되묻고 (4단계 slot_filler/clarify 재사용)
  3) 슬롯이 다 차면 KOSIS를 조회하고 필요하면 계산해서 답변 (5·6단계 api_client/calculator 재사용)

batch_runner.py의 run_stage_4/run_stage_5_6 로직(이미 1~6단계 연결 테스트로 검증됨)을
input() 기반 대화 루프로 감싼 것이라, 새로 만든 부분은 "대화 루프" 자체뿐입니다.

# KNOWN BUG: clarify_rules.REQUIRED_SLOTS = ["period", "region", "calc_type"]가 표 종류와
# 무관하게 항상 고정이라, region 축이 없는 표(예: 실업률표)에도 "어느 지역 기준인가요?"를
# 무조건 물어봅니다. Day2 파이프라인 연결 테스트에서 이미 확인된 이슈(tests/pipeline_integration_log.md
# 이슈 2 참고)라 오늘은 일부러 안 고치고 뼈대 완성을 우선함 — 다음 작업으로 예정.

사용법 (프로젝트 루트에서, .env에 HCX_API_KEY·KOSIS_API_KEY 필요):
    python -m agent.pipeline.chatbot
"""

from __future__ import annotations

import json
from datetime import date
from typing import Optional

from agent.interfaces import Claim, TableCandidate
from agent.kosis.api_client import KosisApiClient, KosisApiError, TABLE_PARAMS_PATH
from agent.kosis.calculator import CalculationError, KosisCalculator
from agent.mapping.embedding_search import build_table_embedding_cache, embedding_search
from agent.mapping.keyword_search import keyword_search
from agent.mapping.reranker import search_and_rerank
from agent.orchestrator.clarify_rules import get_next_clarify_step
from agent.orchestrator.slot_filler import fill_slots, is_valid_period, normalize_time_expressions
from agent.pipeline.batch_runner import build_kosis_slots

MAX_CLARIFY_ROUNDS = 2  # 슬롯당(질문 1건당) 최대 되묻기 횟수 — 넘으면 포기하고 안내 문구 출력
EXIT_WORDS = {"종료", "quit", "exit", "그만"}


def _wrap_as_claim(user_text: str) -> Claim:
    """3단계 매핑(search_and_rerank)은 Claim.sentence만 보고 매칭하므로(claim_type은 안 씀),
    기사에서 뽑은 Claim이 아니라 사용자의 자연어 질문을 그대로 넣어도 동작한다."""
    return Claim(sentence=user_text, claim_type="규모")


def _find_table(user_text: str, embedding_cache: dict) -> Optional[TableCandidate]:
    candidates = search_and_rerank(
        _wrap_as_claim(user_text),
        keyword_fn=keyword_search,
        embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
    )
    return candidates[0] if candidates else None


def _collect_slots(user_text: str) -> Optional[dict]:
    """4단계: fill_slots + clarify. 부족하면 최대 MAX_CLARIFY_ROUNDS번 되묻고,
    그래도 안 채워지면 포기(None)한다.

    되묻기 답변은 fill_slots()로 다시 LLM 추출을 시키지 않고, get_next_clarify_step()이
    "지금 정확히 어느 슬롯을 물었는지"를 알려주므로 그 슬롯에 답변을 직접 채운다.
    (fill_slots의 LLM 추출이 "증감률"처럼 짧은 단답 한 단어에서 자주 null을 반환하는
    문제가 있음 — few-shot 예시를 추가해도 재현됨. 되묻기 상황에서는 어차피 슬롯이
    뭔지 이미 알고 있으니, 굳이 다시 추출을 맡기지 않는 게 더 안정적이다.)
    """
    slots = fill_slots(user_text, {}, date.today())
    step = get_next_clarify_step(slots)

    rounds = 0
    while step.next_slot_to_ask and rounds < MAX_CLARIFY_ROUNDS:
        print(f"봇: {step.clarify_question}")
        reply = input("나: ").strip()

        if step.next_slot_to_ask == "period":
            normalized = normalize_time_expressions({"period": reply}, date.today())
            value = normalized.get("period")
            if value and is_valid_period(value):
                slots["period"] = value
        else:
            slots[step.next_slot_to_ask] = reply

        step = get_next_clarify_step(slots)
        rounds += 1

    if step.next_slot_to_ask:
        print("봇: 해당 통계는 지원하지 않거나 질문을 더 구체화해주세요.")
        return None
    return slots


def _run_lookup(
    table_id: str,
    table_name: str,
    generic_slots: dict,
    table_params: dict,
    client: KosisApiClient,
    calculator: KosisCalculator,
) -> str:
    """5·6단계: 슬롯을 표별 파라미터로 변환해 KOSIS 조회 + 필요하면 계산까지 수행."""
    kosis_slots = build_kosis_slots(table_id, generic_slots, table_params)
    if kosis_slots is None:
        return f"'{table_name}' 통계는 아직 조회 설정이 준비되지 않았습니다. (table_params.json에 없음)"

    calc_type = generic_slots.get("calc_type")
    try:
        if calc_type in ("증감", "증감률") and kosis_slots.get("period"):
            base_slots = dict(kosis_slots, period=str(int(kosis_slots["period"]) - 1))
            base_resp = client(table_id, base_slots)
            target_resp = client(table_id, kosis_slots)
            calc_fn = calculator.compute_change_rate if calc_type == "증감률" else calculator.compute_change
            result = calc_fn(base_resp, target_resp)
            return (
                f"{table_name} 기준, {base_resp.period}년 {base_resp.raw_value}{base_resp.unit} → "
                f"{target_resp.period}년 {target_resp.raw_value}{target_resp.unit}로 "
                f"{result.calc_type} {result.raw_value}{result.unit}입니다."
            )
        resp = client(table_id, kosis_slots)
        return f"{table_name} 기준, {resp.period}년 값은 {resp.raw_value}{resp.unit}입니다."
    except (KosisApiError, CalculationError) as e:
        return f"조회/계산 중 문제가 생겼습니다: {e}"


def chat() -> None:
    print("KOSIS 통계 챗봇입니다. 궁금한 통계를 자연어로 물어보세요. ('종료'로 끝내기)")

    try:
        client = KosisApiClient()
    except RuntimeError as e:
        print(f"[중단] {e}")
        return

    calculator = KosisCalculator()
    with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)
    embedding_cache = build_table_embedding_cache()

    while True:
        user_text = input("\n나: ").strip()
        if not user_text or user_text.lower() in EXIT_WORDS:
            print("봇: 이용해주셔서 감사합니다.")
            break

        table = _find_table(user_text, embedding_cache)
        if table is None:
            print("봇: 관련된 통계표를 찾지 못했습니다. 다르게 질문해 주시겠어요?")
            continue

        slots = _collect_slots(user_text)
        if slots is None:
            continue

        answer = _run_lookup(table.table_id, table.table_name, slots, table_params, client, calculator)
        print(f"봇: {answer}")


if __name__ == "__main__":
    chat()
