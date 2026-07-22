from agent.interfaces import Slots
from agent.orchestrator.clarify_rules import get_next_clarify_step


def clarify(current_slots: Slots):
    """
    슬롯이 부족하면 되물을 질문을 반환, 다 채워졌으면 None 반환.
    LLM 호출 없음 — clarify_rules.py의 템플릿 기반 로직만 사용.
    """
    step = get_next_clarify_step(current_slots)
    if step.clarify_question:
        return step.clarify_question
    return None

if __name__ == "__main__":
    # 케이스 1: 슬롯 다 채워진 경우
    full_slots = {"period": "2024", "region": "전국", "calc_type": "증감률"}
    print("케이스1 (다 채워짐):", clarify(full_slots))

    # 케이스 2: 슬롯 일부 비어있는 경우
    partial_slots = {"period": "2024"}
    print("케이스2 (일부 비어있음):", clarify(partial_slots))

    # 케이스 3: 슬롯 전부 비어있는 경우
    empty_slots = {}
    print("케이스3 (전부 비어있음):", clarify(empty_slots))