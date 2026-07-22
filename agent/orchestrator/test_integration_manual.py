from datetime import date
from agent.orchestrator.slot_filler import fill_slots
from agent.orchestrator.clarify import clarify

ARTICLE_DATE = date(2025, 1, 1)

# (설명, 발화, 시작 슬롯 상태)
SCENARIOS = [
    ("한 번에 다 채워지는 경우", "작년 서울 대비 전국 증감률 알려줘", {}),
    ("지역만 있음", "서울 통계 알려줘", {}),
    ("시점만 있음", "올해 기준으로 알려줘", {}),
    ("계산종류만 있음", "평균 알려줘", {}),
    ("아무것도 없음", "통계 좀 알려줘", {}),
    ("이미 일부 채워진 상태에서 보완", "부산으로 해줘", {"period": "2024", "calc_type": "합계"}),
    ("재작년 표현", "재작년 전국 순위 알려줘", {}),
    ("지난달처럼 애매한 표현(HCX-003 위임)", "지난달 서울 통계 알려줘", {}),
]

for desc, utterance, existing in SCENARIOS:
    print(f"\n[{desc}] 발화: \"{utterance}\" / 기존 슬롯: {existing}")
    slots = fill_slots(utterance, existing, ARTICLE_DATE)
    question = clarify(slots)
    print(f"  → 채워진 슬롯: {slots}")
    print(f"  → 되묻기 질문: {question if question else '(없음, 충분히 채워짐)'}")
