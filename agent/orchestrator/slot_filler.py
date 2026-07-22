import os
import json
import re
import requests
from dotenv import load_dotenv
from agent.interfaces import Slots
from agent.orchestrator.clarify_rules import REQUIRED_SLOTS
import pandas as pd
from datetime import datetime, date


load_dotenv()
API_KEY = os.getenv("HCX_API_KEY")
MODEL = "HCX-DASH-002"
URL = f"https://clovastudio.stream.ntruss.com/v3/chat-completions/{MODEL}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# 슬롯 이름만으로는 LLM이 뜻을 오해할 수 있어 여기서 설명 보강 (프롬프트 텍스트에만 쓰임)
SLOT_DESCRIPTIONS = {
    "period": "질문에서 언급된 시점/기간 (예: 2024, 작년, 지난달 등 원문 그대로)",
    "region": "질문에서 언급된 지역명 (예: 서울, 전국, 부산)",
    "calc_type": "요청된 계산/지표 종류 (예: 증감률, 합계, 평균, 순위)",
}


def call_hcx(prompt: str) -> str:
    """HCX-DASH-002 호출 후 응답 텍스트(content)만 뽑아서 리턴"""
    payload = {"messages": [{"role": "user", "content": prompt}]}
    response = requests.post(URL, headers=HEADERS, json=payload)
    response.raise_for_status()  # 상태코드 200 아니면 여기서 에러 터짐
    data = response.json()
    return data["result"]["message"]["content"]


def build_extraction_prompt(user_input: str) -> str:
    slot_lines = "\n".join(
        f"- {slot}: {SLOT_DESCRIPTIONS.get(slot, '설명 없음')}"
        for slot in REQUIRED_SLOTS
    )
    return f"""다음 문장에서 아래 슬롯 값을 추출해서 JSON으로만 응답하세요.
값이 문장에 없으면 반드시 null로 표시하세요. 설명이나 다른 텍스트는 절대 포함하지 마세요.

중요 규칙:
- 지역명(예: 서울, 부산, 전국)은 반드시 region에만 넣으세요. period나 calc_type에 넣지 마세요.
- 시점 표현(예: 작년, 2024, 지난달)은 반드시 period에만 넣으세요.
- 계산 방식(예: 증감률, 합계, 평균, 순위)은 반드시 calc_type에만 넣으세요.
- 문장에 해당 슬롯 정보가 없으면 절대 추측하지 말고 null로 응답하세요.

슬롯 목록:
{slot_lines}

예시:
문장: "서울 통계 알려줘"
응답: {{"period": null, "region": "서울", "calc_type": null}}

문장: "작년 증감률 알려줘"
응답: {{"period": "작년", "region": null, "calc_type": "증감률"}}

문장: "{user_input}"

응답 형식 (JSON만, 다른 텍스트 없이):
{{"period": ..., "region": ..., "calc_type": ...}}
"""


def extract_json_fallback(raw: str) -> dict:
    """LLM이 JSON 앞뒤에 설명을 덧붙이는 경우, {...} 구간만 잘라서 재시도"""
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}


RELATIVE_TIME_KEYWORDS = ["작년", "올해", "지난달", "이번달", "재작년", "지난주", "이번주"]

# 연도로 바로 환산 가능한 표현만 여기 정의 (계산 = 코드로)
RELATIVE_YEAR_OFFSET = {
    "재작년": -2,
    "작년": -1,
    "올해": 0,
}


def normalize_time_expressions(extracted: dict, article_date: date) -> dict:
    period_val = extracted.get("period")
    if not period_val:
        return extracted

    period_str = str(period_val)

    # 1) 연 단위 표현은 파이썬으로 직접 계산 (LLM 호출 안 함)
    for kw, offset in RELATIVE_YEAR_OFFSET.items():
        if kw in period_str:
            extracted["period"] = str(article_date.year + offset)
            return extracted

    # 2) 월/주 단위처럼 연 단위로 딱 떨어지지 않는 애매한 표현만 LLM(HCX-003)에 위임
    if any(kw in period_str for kw in RELATIVE_TIME_KEYWORDS):
        prompt = f"""이 기사는 {article_date.year}년 {article_date.month}월에 작성되었습니다.
"{period_val}"라는 표현을 이 기사 작성 시점 기준으로 절대 연도로 바꾸세요.

규칙:
- 반드시 숫자 4자리(예: 2024)만 응답하세요.
- 설명, 문장, 다른 텍스트를 절대 포함하지 마세요.
- 오직 연도 숫자만 출력하세요.
"""
        absolute = call_hcx(prompt)
        match = re.search(r"\d{4}", absolute)
        extracted["period"] = match.group() if match else absolute.strip()

    return extracted

def is_valid_period(value) -> bool:
    """period 값이 그럴듯한지 검증 (연도 4자리이거나 상대시점 키워드 포함)"""
    if value is None:
        return True
    s = str(value)
    if re.fullmatch(r"\d{4}", s):
        return True
    if any(kw in s for kw in RELATIVE_TIME_KEYWORDS):
        return True
    return False


def fill_slots(user_input: str, existing_slots: dict, article_date: date) -> dict:
    prompt = build_extraction_prompt(user_input)
    raw = call_hcx(prompt)

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        extracted = extract_json_fallback(raw)

    extracted = normalize_time_expressions(extracted, article_date)

    merged = dict(existing_slots)
    for slot in REQUIRED_SLOTS:
        value = extracted.get(slot)
        if value is None:
            continue
        if slot == "period" and not is_valid_period(value):
            # 이상한 값이면 무시하고 기존 값 유지 (덮어쓰지 않음)
            continue
        merged[slot] = value

    return merged

if __name__ == "__main__":
    from datetime import date

    print("=== 회귀 테스트 1: 기존 period가 지역 발화로 오염되는지 ===")
    result1 = fill_slots("부산으로 해줘", {"period": "2024", "calc_type": "합계"}, date(2025, 1, 1))
    print(result1)
    assert result1.get("period") == "2024", "❌ period가 오염됨!"
    assert result1.get("region") == "부산", "❌ region이 안 채워짐!"
    print("✅ period 보존 확인\n")

    print("=== 회귀 테스트 2: 지역만 있는 문장에서 period가 비어있는지 ===")
    result2 = fill_slots("서울 통계 알려줘", {}, date(2025, 1, 1))
    print(result2)
    assert result2.get("period") is None, "❌ period에 지역명이 채워짐!"
    assert result2.get("region") == "서울", "❌ region이 안 채워짐!"
    print("✅ period 오염 없음 확인\n")

    print("=== 기존 테스트: 작년 → 2024년 계산 확인 ===")
    result3 = fill_slots("작년에 이런 사건이 몇 건 있었어?", {}, date(2025, 1, 1))
    print(result3)
    assert result3.get("period") == "2024", "❌ 작년 계산이 틀림!"
    print("✅ 작년 계산 정확\n")

    print("모든 테스트 통과 🎉")