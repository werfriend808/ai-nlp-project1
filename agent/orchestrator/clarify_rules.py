from dataclasses import dataclass
from typing import Optional
from agent.interfaces import Slots
 
 
# ---------------------------------------------------------------------------
# 1. 필수 슬롯 정의
# ---------------------------------------------------------------------------
 
REQUIRED_SLOTS = ["period", "region", "calc_type"]
 

 # 필요하면 이후에 추가: "population", "unit" 등
 
 
 # ---------------------------------------------------------------------------
# 2. 슬롯별 되묻기 문구 (템플릿 우선 — LLM 호출 없이 고정 문구로 처리)<<요구하는게맞는지ㅁㄹ
# ---------------------------------------------------------------------------
 
CLARIFY_QUESTIONS = {
    "period": "어느 연도 기준으로 알려드릴까요?",
    "region": "어느 지역 기준인가요?",
    "calc_type": "증감률을 원하시나요, 합계를 원하시나요?",
}
  
# ---------------------------------------------------------------------------
# 3. 우선순위 규칙
# ---------------------------------------------------------------------------
# 여러 슬롯이 동시에 비어있으면, 아래 리스트 순서대로 먼저 나온 것부터 되묻는다.
#
# 근거:
#   - period가 없으면 다른 슬롯이 채워져 있어도 계산 자체가 불가능하므로 1순위
#   - region은 그다음으로 통계 조회 범위를 결정하는 핵심 조건이라 2순위
#   - calc_type은 값이 없어도 기본값(예: 단순 조회)으로 대체 가능한 경우가 있어 3순위
 
PRIORITY_ORDER = ["period", "region", "calc_type"]
 
  
# ---------------------------------------------------------------------------
# 4. 실제 로직 — 다음에 물어볼 질문 하나를 결정하는 함수
# ---------------------------------------------------------------------------
 
@dataclass
class NextClarifyStep:
    missing_slots: list[str]           # 현재 비어있는 슬롯 전체 목록
    next_slot_to_ask: Optional[str]     # 이번에 물어볼 슬롯 (없으면 None = 다 채워짐)
    clarify_question: Optional[str]     # 사용자에게 보여줄 실제 질문 문구
 
 
def get_next_clarify_step(current_slots: Slots) -> NextClarifyStep:
    """
    현재까지 채워진 slots를 보고, 다음에 뭘 물어봐야 하는지 결정.
 
    사용 예:
        slots = {"period": "2024"}  # region, calc_type 아직 없음
        step = get_next_clarify_step(slots)
        print(step.next_slot_to_ask)     # "region"
        print(step.clarify_question)     # "어느 지역 기준인가요?"
    """
    missing = [slot for slot in REQUIRED_SLOTS if slot not in current_slots or not current_slots[slot]]
 
    if not missing:
        return NextClarifyStep(missing_slots=[], next_slot_to_ask=None, clarify_question=None)
 
    # PRIORITY_ORDER 순서대로 훑으면서, missing에 있는 것 중 가장 우선순위 높은 것 하나 고름
    for slot in PRIORITY_ORDER:
        if slot in missing:
            return NextClarifyStep(
                missing_slots=missing,
                next_slot_to_ask=slot,
                clarify_question=CLARIFY_QUESTIONS[slot],
            )
 
    # PRIORITY_ORDER에 없는 슬롯이 missing에 있는 경우 (예외 케이스)
    # -> 여기 걸리면 하드코딩 규칙으로 못 정한 것이므로 LLM에게 판단을 넘기는 지점
    fallback_slot = missing[0]
    return NextClarifyStep(
        missing_slots=missing,
        next_slot_to_ask=fallback_slot,
        clarify_question=CLARIFY_QUESTIONS.get(
            fallback_slot,
            f"{fallback_slot} 값을 알려주시겠어요?"
        ),
    )
 
 
# ---------------------------------------------------------------------------
# 5. 간단 테스트 (직접 실행해서 확인 가능)
# ---------------------------------------------------------------------------
 
if __name__ == "__main__":
    # 케이스 1: 아무것도 안 채워진 경우 -> period부터 물어봐야 함
    print(get_next_clarify_step({}))
 
    # 케이스 2: period만 채워진 경우 -> region 물어봐야 함
    print(get_next_clarify_step({"period": "2024"}))
 
    # 케이스 3: period, region 채워진 경우 -> calc_type 물어봐야 함
    print(get_next_clarify_step({"period": "2024", "region": "서울"}))
 
    # 케이스 4: 다 채워진 경우 -> 더 물어볼 것 없음
    print(get_next_clarify_step({"period": "2024", "region": "서울", "calc_type": "증감률"}))