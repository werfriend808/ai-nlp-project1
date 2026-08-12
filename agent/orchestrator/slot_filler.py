import os
import json
import re
import requests
from dotenv import load_dotenv
from agent.interfaces import Slots
from agent.orchestrator.clarify_rules import REQUIRED_SLOTS
import pandas as pd
from datetime import datetime, date
from typing import Optional


load_dotenv()
API_KEY = os.getenv("HCX_API_KEY")
MODEL = "HCX-DASH-002"
URL = f"https://clovastudio.stream.ntruss.com/v3/chat-completions/{MODEL}"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# 슬롯 이름만으로는 LLM이 뜻을 오해할 수 있어 여기서 설명 보강 (프롬프트 텍스트에만 쓰임)
# calc_type 값 체계: 2026-08-05 정리 — 6단계 calculator.py가 실제로 계산할 수 있는 값
# (증감/증감률/비율/최댓값검증)으로 맞췄다. "합계"는 여기 포함 안 함 — 사용자가 요청하는
# 계산 종류가 아니라, 나이대(20대 등) 다중코드처럼 표 구조상 어쩔 수 없이 필요한 내부 조립
# 방식이라 별도 패턴 감지 로직(agent_chat.py의 나이대/극값 감지)이 처리하고, 이 슬롯에서는
# 다루지 않는다. 예전엔 "평균"/"순위"도 예시로 들었는데 실제로 어디서도 처리 안 되는(죽은)
# 값이라 빼고, 실제로 쓰이는 "비율"/"최댓값검증"으로 교체.
SLOT_DESCRIPTIONS = {
    "period": "질문에서 언급된 시점/기간 (예: 2024, 작년, 지난달 등 원문 그대로)",
    "region": "질문에서 언급된 지역명 (예: 서울, 전국, 부산)",
    "calc_type": "요청된 계산/지표 종류. 반드시 다음 4가지 중 하나여야 함: 증감, 증감률, 비율, 최댓값검증",
}


def call_hcx(prompt: str) -> str:
    """HCX-DASH-002 호출 후 응답 텍스트(content)만 뽑아서 리턴

    2026-08-11 재현성 문제 대응: 이 함수는 agent/preprocessing/hcx_client.py의 공용
    호출부를 안 쓰고 여기서 별도로 requests를 직접 호출하고 있었다 — temperature도
    seed도 하나도 안 넘겨서 CLOVA Studio 기본값(temperature=0.5, seed=무작위)을 그대로
    쓰던 상태였다. 1·2·8단계와 동일하게 temperature=0.0 + seed 고정으로 맞춘다.
    """
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "seed": 42,
    }
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
- 계산 방식(예: 증감, 증감률, 비율, 최댓값검증)은 반드시 calc_type에만 넣으세요.
- 문장에 해당 슬롯 정보가 없으면 절대 추측하지 말고 null로 응답하세요.

슬롯 목록:
{slot_lines}

예시:
문장: "서울 통계 알려줘"
응답: {{"period": null, "region": "서울", "calc_type": null}}

문장: "작년 증감률 알려줘"
응답: {{"period": "작년", "region": null, "calc_type": "증감률"}}

문장: "고령 인구 비율 알려줘"
응답: {{"period": null, "region": null, "calc_type": "비율"}}

문장: "역대 최고치가 맞는지 알려줘"
응답: {{"period": null, "region": null, "calc_type": "최댓값검증"}}

문장: "1~10일 수출은 160억 달러로 전년동기 대비 3.8% 증가했다"
응답: {{"period": "전년동기", "region": null, "calc_type": "증감률"}}

문장: "서울 알려줘"
응답: {{"period": null, "region": "서울", "calc_type": null}}

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


# "전년"/"전월"/"전분기"/"전기"/"당월"/"당해"/"1년새": 2026-08-05 실제 기사 1007건 슬롯필링
# 커버리지 실측에서, 한국 경제기사에 아주 흔한 이 표현들이 이 목록에 없어서 "지난해"와
# 똑같은 이유(is_valid_period()가 못 알아보는 값으로 버림)로 대량(약 130건, 전체의 13%)
# 유실되고 있는 게 확인되어 추가함. "지난분기"/"직전분기"/"당분기"/"다음달"은 뜻이 명확한
# 동의어라 같이 추가(2026-08-05). "최근"/"당해년도"는 문맥마다 가리키는 기간이 달라 애매해서
# 의도적으로 제외 — 넣으면 LLM이 틀린 연도를 확정해버릴 위험이 있음.
RELATIVE_TIME_KEYWORDS = [
    "작년", "지난해", "올해", "지난달", "이번달", "재작년", "지난주", "이번주",
    "전년", "전월", "전달", "전분기", "지난분기", "직전분기", "당분기", "전기",
    "당월", "당해", "1년새", "다음달",
]

# 연도로 바로 환산 가능한 표현만 여기 정의 (계산 = 코드로)
# "지난해"는 "작년"과 같은 뜻인데 빠져있어서, LLM 추출 자체는 "지난해 12월"처럼 정확히
# 뽑아내는데도 is_valid_period()가 못 알아보는 값으로 취급해 버려지는 문제가 있었음
# (pipeline_integration_log.md 이슈 1-5, 2026-08-04 확인 후 수정).
# "전년"/"전년도"/"전전년"/"전전년도"/"직전연도"/"금년"도 각각 "작년"/"재작년"/"올해"와
# 같은 뜻인데 마찬가지로 빠져있었음(2026-08-05, 실측으로 발견).
RELATIVE_YEAR_OFFSET = {
    "재작년": -2,
    "전전년도": -2,
    "전전년": -2,
    "작년": -1,
    "지난해": -1,
    "전년도": -1,
    "전년": -1,
    "직전연도": -1,
    "올해": 0,
    "금년": 0,
}

# "1년 전"/"7년 전"처럼 숫자가 바뀌는 표현은 고정 키워드 사전으로 못 잡아서 정규식으로 따로
# 처리한다(agent_chat.py의 "N년 만에" 처리와 같은 원리 — 계산 가능한 건 코드로, LLM 호출 없음).
# "여"(-쯤/-남짓)가 "4년여 전"처럼 낄 수 있어서 agent_chat.py의 "N년여 만에"와 같은 이유로 허용.
_N_YEARS_AGO_RE = re.compile(r"(\d+)년\s*여?\s*전")

# 절대 연도인데 LLM이 "2024" 대신 "2024년"처럼 접미사를 붙여서 응답하는 포맷 흔들림이
# 있어서(2026-08-06, test_slot_filler_module.py case_02에서 실측 확인), is_valid_period()가
# 4자리 숫자만 허용하다 보니 멀쩡한 절대 연도가 그냥 버려지던 문제를 여기서 먼저 벗겨낸다.
_ABSOLUTE_YEAR_WITH_SUFFIX_RE = re.compile(r"(\d{4})년")

# ---------------------------------------------------------------------------
# 표 주기(prdSe) 인식 — 2026-08-12 실측 확인: 매칭된 표가 월간(M)/분기(Q)인데도
# slot_filler는 표 주기를 아예 모른 채 무조건 연도 4자리만 만들어서, KOSIS API가
# 요구하는 형식(YYYYMM/YYYY+분기2자리)과 안 맞아 카탈로그 64개 중 32개(50%)가 API 호출
# 단계에서 형식 오류로 100% 실패하던 문제를 고친다. prdSe는 table_params.json에 표별로
# 미리 검증돼 저장돼 있는 값이라(3단계가 이미 매칭한 표의 table_id로 조회), 호출부가
# 이 값을 여기까지 넘겨줘야 한다.
# ---------------------------------------------------------------------------

_MONTH_OFFSET_KEYWORDS = {
    "지난달": -1, "전달": -1, "전월": -1, "당월": 0, "이번달": 0, "다음달": 1,
}
_QUARTER_OFFSET_KEYWORDS = {
    "지난분기": -1, "전분기": -1, "직전분기": -1, "당분기": 0,
}

# "지난해 말"/"연말"처럼 연 단위 표현에 "말/초" 수식어가 붙으면, "지난해"라는 substring만
# 보고 "기사 작성월과 같은 달"로 계산하면 안 된다 — "말"은 그 해 12월(4분기), "초"는 그 해
# 1월(1분기)을 구체적으로 가리키는 표현이다(2026-08-12 실측: "지난해 말 200개→225개"
# 비교가 "전년 동월"(작성월 기준) 로직 때문에 실제로는 6월 데이터와 비교되어 잘못된 판정이
# 난 사례로 확인). "중순"은 가리키는 달이 사람마다 다르게 해석될 여지가 있어 이번엔 다루지
# 않는다.
_YEAR_END_MARKERS = ("말", "연말")
_YEAR_START_MARKERS = ("초", "연초")


def _add_months(year: int, month: int, delta: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + delta
    return total // 12, total % 12 + 1


def _quarter_of(month: int) -> int:
    return (month - 1) // 3 + 1


def _add_quarters(year: int, quarter: int, delta: int) -> tuple[int, int]:
    total = year * 4 + (quarter - 1) + delta
    return total // 4, total % 4 + 1


def _format_month(year: int, month: int) -> str:
    return f"{year:04d}{month:02d}"


def _format_quarter(year: int, quarter: int) -> str:
    return f"{year:04d}{quarter:02d}"


def normalize_time_expressions(
    extracted: dict, article_date: date, *, prd_se: Optional[str] = None
) -> dict:
    """상대 시점 표현을 절대 시점 문자열로 바꾼다.

    prd_se(표 주기, "Y"/"M"/"Q")를 안 넘기면(기본값 None) 이전과 완전히 동일하게
    무조건 연도 4자리만 만든다 — 기존 호출부(연간 표만 다루던 코드)는 안 바꿔도 그대로
    동작한다. prd_se="M"/"Q"를 넘긴 호출부만 아래 새 분기를 탄다.
    """
    period_val = extracted.get("period")
    if not period_val:
        return extracted

    period_str = str(period_val)

    # -1) "2024년"처럼 절대 연도에 "년" 접미사가 붙은 경우 먼저 벗겨낸다 (상대 표현 아님).
    # 표 주기와 무관하게 그대로 둔다 — "2024년"은 특정 달을 가리키는 표현이 아니라 연간
    # 집계 개념이라, 월/분기로 억지로 쪼개면 오히려 틀린 값이 된다(별도 판단 필요 영역).
    absolute_year_match = _ABSOLUTE_YEAR_WITH_SUFFIX_RE.fullmatch(period_str)
    if absolute_year_match:
        extracted["period"] = absolute_year_match.group(1)
        return extracted

    # 0) "N년 전"(숫자 가변) — 위와 같은 이유로 연 단위 그대로 유지.
    n_years_ago_match = _N_YEARS_AGO_RE.search(period_str)
    if n_years_ago_match:
        extracted["period"] = str(article_date.year - int(n_years_ago_match.group(1)))
        return extracted

    # 1-M) 표가 월간이면, 월 단위 표현("지난달" 등)과 "작년"류(연 단위 표현)를 파이썬으로
    # 직접 월 단위로 계산한다(LLM 호출 없음). "작년"처럼 원래 연 단위인 표현도 월간 표에서는
    # 한국 경제기사 관용("전년동월 대비")에 따라 "같은 달, N년 전"으로 해석한다 — 완벽한
    # 해법은 아니고("작년 한 해 전체"를 뜻하는 극소수 케이스는 못 잡음), 실측상 압도적
    # 다수가 이 해석과 일치해서 기본값으로 채택.
    if prd_se == "M":
        for kw, delta in _MONTH_OFFSET_KEYWORDS.items():
            if kw in period_str:
                y, m = _add_months(article_date.year, article_date.month, delta)
                extracted["period"] = _format_month(y, m)
                return extracted
        year_offset = next((o for kw, o in RELATIVE_YEAR_OFFSET.items() if kw in period_str), None)
        has_end_marker = any(marker in period_str for marker in _YEAR_END_MARKERS)
        has_start_marker = any(marker in period_str for marker in _YEAR_START_MARKERS)
        if year_offset is not None or has_end_marker or has_start_marker:
            # "연말"/"연초"처럼 "말/초" 수식어만 있고 다른 연도 표현이 없으면 올해(offset=0)를
            # 가리키는 것으로 기본 처리한다.
            year = article_date.year + (year_offset or 0)
            if has_end_marker:
                month = 12
            elif has_start_marker:
                month = 1
            else:
                month = article_date.month
            extracted["period"] = _format_month(year, month)
            return extracted

    # 1-Q) 표가 분기면 마찬가지로 분기 단위로 직접 계산.
    if prd_se == "Q":
        current_quarter = _quarter_of(article_date.month)
        for kw, delta in _QUARTER_OFFSET_KEYWORDS.items():
            if kw in period_str:
                y, q = _add_quarters(article_date.year, current_quarter, delta)
                extracted["period"] = _format_quarter(y, q)
                return extracted
        year_offset = next((o for kw, o in RELATIVE_YEAR_OFFSET.items() if kw in period_str), None)
        has_end_marker = any(marker in period_str for marker in _YEAR_END_MARKERS)
        has_start_marker = any(marker in period_str for marker in _YEAR_START_MARKERS)
        if year_offset is not None or has_end_marker or has_start_marker:
            year = article_date.year + (year_offset or 0)
            if has_end_marker:
                quarter = 4
            elif has_start_marker:
                quarter = 1
            else:
                quarter = current_quarter
            extracted["period"] = _format_quarter(year, quarter)
            return extracted

    # 1) 연 단위 표현은 파이썬으로 직접 계산 (LLM 호출 안 함) — prd_se가 Y/None이거나,
    # 위 M/Q 분기에서 못 잡은 나머지(예: 월간 표인데 "지난주"처럼 월/분기 어느 쪽도 아닌
    # 표현)의 폴백.
    for kw, offset in RELATIVE_YEAR_OFFSET.items():
        if kw in period_str:
            extracted["period"] = str(article_date.year + offset)
            return extracted

    # 2) 그 외 애매한 표현("지난주" 등)만 LLM(HCX-003)에 위임 — prd_se에 따라 요청 형식을
    # 동적으로 바꾼다. 예전엔 무조건 "숫자 4자리만"으로 강제해서, 월간/분기 표에 필요한
    # 6자리 형식(YYYYMM/YYYY+분기)을 요청할 방법이 아예 없었다.
    if any(kw in period_str for kw in RELATIVE_TIME_KEYWORDS):
        if prd_se == "M":
            format_instruction = "6자리 연월(YYYYMM, 예: 202407)"
            digit_count = 6
        elif prd_se == "Q":
            format_instruction = "6자리 연도+분기코드(YYYY+01~04, 예: 202403=2024년 3분기)"
            digit_count = 6
        else:
            format_instruction = "숫자 4자리(예: 2024)"
            digit_count = 4

        prompt = f"""이 기사는 {article_date.year}년 {article_date.month}월에 작성되었습니다.
"{period_val}"라는 표현을 이 기사 작성 시점 기준으로 절대 시점으로 바꾸세요.

규칙:
- 반드시 {format_instruction}만 응답하세요.
- 설명, 문장, 다른 텍스트를 절대 포함하지 마세요.
- 오직 숫자만 출력하세요.
"""
        absolute = call_hcx(prompt)
        match = re.search(rf"\d{{{digit_count}}}", absolute)
        extracted["period"] = match.group() if match else absolute.strip()

    return extracted

def is_valid_period(value) -> bool:
    """period 값이 그럴듯한지 검증 (연도 4자리, 월/분기 6자리, 또는 상대시점 키워드 포함)"""
    if value is None:
        return True
    s = str(value)
    if re.fullmatch(r"\d{4}", s):
        return True
    # 6자리는 YYYYMM(월간) 또는 YYYY+분기코드(분기) 둘 다 이 자리에서는 형식만 보고
    # 통과시킨다 — 어느 쪽 의미인지는 prdSe 문맥에 달려있고(3~5단계가 판단), 여기선 그냥
    # normalize_time_expressions가 만들어낸 6자리 결과를 거부하지만 않으면 된다.
    if re.fullmatch(r"\d{6}", s):
        return True
    if any(kw in s for kw in RELATIVE_TIME_KEYWORDS):
        return True
    return False


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", "", text)


def is_region_grounded(value, user_input: str) -> bool:
    """region 값이 실제로 원문 문장에 등장하는지 확인한다 (공백 무시 비교).

    표별 세부 슬롯(agent_chat.py의 dimension_hints LLM 2차 폴백)은 근거 quote가 원문에
    있는지 대조해서 환각을 막는데, 이 공용 region 슬롯엔 그런 방어가 전혀 없었다 —
    2026-08-06 300개 배치 실측에서 "7월 1~10일 수출이... 증가했다"처럼 지역 언급 자체가
    없는 문장에도 계속 region="서울"이 지어내지는 걸 확인(5단계_진행가능 결과 중 지역명이
    있는 13건 중 4건이 이런 식으로 문장에 없는 값이었음). period처럼 값 자체의 형식이
    아니라 "원문에 실제로 있는가"만 확인하면 되므로 정규식이 아니라 단순 포함 검사로
    충분하다."""
    if value is None:
        return True
    return _normalize_whitespace(str(value)) in _normalize_whitespace(user_input)


def fill_slots(
    user_input: str, existing_slots: dict, article_date: date, *, prd_se: Optional[str] = None
) -> dict:
    """prd_se: 매칭된 표의 주기("Y"/"M"/"Q", table_params.json 조회 결과). 안 넘기면(기본값
    None) 기존과 동일하게 연 단위로만 시점을 계산한다 — 호출부가 표 정보를 아직 안 넘기는
    경우(예: 표 매칭 전 단계)에도 그대로 동작해야 하므로 하위 호환을 위해 옵션으로 둔다."""
    prompt = build_extraction_prompt(user_input)
    raw = call_hcx(prompt)

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        extracted = extract_json_fallback(raw)

    extracted = normalize_time_expressions(extracted, article_date, prd_se=prd_se)

    merged = dict(existing_slots)
    for slot in REQUIRED_SLOTS:
        value = extracted.get(slot)
        if value is None:
            continue
        if slot == "period" and not is_valid_period(value):
            # 이상한 값이면 무시하고 기존 값 유지 (덮어쓰지 않음)
            continue
        if slot == "region" and not is_region_grounded(value, user_input):
            # 원문에 없는 지역명(환각)이면 무시하고 기존 값 유지 (덮어쓰지 않음)
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

    print("=== 기존 테스트: 작년 → 2024년 계산 확인 (prd_se 안 넘기면 그대로 연 단위) ===")
    result3 = fill_slots("작년에 이런 사건이 몇 건 있었어?", {}, date(2025, 1, 1))
    print(result3)
    assert result3.get("period") == "2024", "❌ 작년 계산이 틀림!"
    print("✅ 작년 계산 정확\n")

    print("=== 2026-08-12 신규: 표 주기(prd_se) 인식 회귀 테스트 ===")
    # 케이스 M-1: 월간 표(M)에서 "지난달" → 기사 작성월 기준 전월, YYYYMM 6자리
    r_m1 = normalize_time_expressions({"period": "지난달"}, date(2025, 3, 15), prd_se="M")
    assert r_m1["period"] == "202502", f"❌ 지난달(월간) 계산 틀림: {r_m1}"
    print(f"[M-1] 지난달(2025년3월 작성 기준) -> {r_m1['period']} ✅")

    # 케이스 M-2: 월간 표에서 "작년"(연 단위 표현) → 한국 경제기사 관용대로 "전년 동월"
    r_m2 = normalize_time_expressions({"period": "작년"}, date(2025, 3, 15), prd_se="M")
    assert r_m2["period"] == "202403", f"❌ 작년(월간, 전년동월) 계산 틀림: {r_m2}"
    print(f"[M-2] 작년(월간표, 2025년3월 작성 기준) -> {r_m2['period']} (전년동월) ✅")

    # 케이스 M-3: 연도 경계(1월 작성 시 "지난달"은 전년 12월)
    r_m3 = normalize_time_expressions({"period": "지난달"}, date(2025, 1, 15), prd_se="M")
    assert r_m3["period"] == "202412", f"❌ 연도 경계 지난달 계산 틀림: {r_m3}"
    print(f"[M-3] 지난달(2025년1월 작성, 연도 경계) -> {r_m3['period']} ✅")

    # 케이스 Q-1: 분기 표(Q)에서 "지난분기" → 기사 작성월이 속한 분기 기준 전분기
    r_q1 = normalize_time_expressions({"period": "지난분기"}, date(2025, 4, 10), prd_se="Q")
    assert r_q1["period"] == "202501", f"❌ 지난분기 계산 틀림: {r_q1}"
    print(f"[Q-1] 지난분기(2025년4월=2분기 작성 기준) -> {r_q1['period']} ✅")

    # 케이스 Q-2: 연도 경계(1분기 작성 시 "지난분기"는 전년 4분기)
    r_q2 = normalize_time_expressions({"period": "지난분기"}, date(2025, 2, 10), prd_se="Q")
    assert r_q2["period"] == "202404", f"❌ 분기 연도 경계 계산 틀림: {r_q2}"
    print(f"[Q-2] 지난분기(2025년2월=1분기 작성, 연도 경계) -> {r_q2['period']} ✅")

    # 케이스 5-6-1/5-6-2 (어제 실배치 실패 재현): DT_121Y006/DT_343_2010_S0027 둘 다
    # 월간(M) 표인데 슬롯필링이 연도만 채워서 KosisApiError가 났던 사례 — prd_se="M"을
    # 넘기면 더 이상 4자리 연도가 안 나와야 한다.
    r_regr1 = normalize_time_expressions({"period": "작년"}, date(2025, 6, 1), prd_se="M")
    assert re.fullmatch(r"\d{6}", r_regr1["period"]), f"❌ 월간표 회귀: {r_regr1}"
    print(f"[회귀] DT_121Y006류(월간) 재현 — 이제 6자리로 나옴: {r_regr1['period']} ✅")

    print("\n=== is_valid_period가 6자리(YYYYMM/YYYY+분기)도 통과시키는지 확인 ===")
    assert is_valid_period("202502") is True, "❌ 6자리 period가 무효 처리됨!"
    assert is_valid_period("2025") is True
    print("✅ 6자리 period 형식 통과 확인\n")

    print("=== 2026-08-12 신규: '지난해 말/초' 수식어 인식 버그 수정 회귀 테스트 ===")
    # 실제 배치에서 재현된 사례: "지난해 말"(작년 12월)이 "지난해"라는 substring만 보고
    # "전년 동월"(기사 작성월과 같은 달)로 잘못 계산되던 버그 — "코스피 3000..." 기사에서
    # "지난해 말 200개→225개" 비교가 실제로는 6월 데이터와 비교되어 잘못된 판정이 난
    # 사례로 확인됨(2026-08-12).
    r_end = normalize_time_expressions({"period": "지난해 말"}, date(2025, 6, 21), prd_se="M")
    assert r_end["period"] == "202412", f"❌ '지난해 말' 계산 틀림: {r_end}"
    print(f"[말(끝) 수식어] '지난해 말'(2025년6월 작성 기준) -> {r_end['period']} ✅ (12월)")

    r_start = normalize_time_expressions({"period": "작년 초"}, date(2025, 6, 21), prd_se="M")
    assert r_start["period"] == "202401", f"❌ '작년 초' 계산 틀림: {r_start}"
    print(f"[초(시작) 수식어] '작년 초'(2025년6월 작성 기준) -> {r_start['period']} ✅ (1월)")

    # "연말"처럼 다른 연도 표현 없이 "말/초" 수식어만 있으면 올해(offset=0)를 가리키는
    #것으로 기본 처리한다.
    r_end_q = normalize_time_expressions({"period": "연말"}, date(2025, 6, 21), prd_se="Q")
    assert r_end_q["period"] == "202504", f"❌ 분기 표 '연말'(수식어만 있음) 계산 틀림: {r_end_q}"
    print(f"[분기 표, 수식어만 있는 경우] '연말'(2025년6월 작성 기준) -> {r_end_q['period']} ✅ (올해 4분기)")

    # 회귀 방지: 수식어 없는 기존 "작년" 동작(전년 동월)은 그대로 유지되는지
    r_plain = normalize_time_expressions({"period": "작년"}, date(2025, 6, 21), prd_se="M")
    assert r_plain["period"] == "202406", f"❌ 수식어 없는 '작년' 회귀: {r_plain}"
    print(f"[회귀 확인] 수식어 없는 '작년' -> {r_plain['period']} ✅ (전년 동월, 그대로 유지)")
    print("  → 통과: '말/초' 수식어가 있으면 12월/1월로, 없으면 기존처럼 전년 동월로 계산됨.")

    print("\n=== '전년동월'/'전년동기'는 이미 '전년' 부분 문자열 매칭으로 정상 동작 확인 ===")
    r_ydy = normalize_time_expressions({"period": "전년동월"}, date(2025, 6, 21), prd_se="M")
    assert r_ydy["period"] == "202406", r_ydy
    r_ydq = normalize_time_expressions({"period": "전년동기"}, date(2025, 6, 21), prd_se="Q")
    assert r_ydq["period"] == "202402", r_ydq
    print(f"[전년동월/동기] '전년동월'+M -> {r_ydy['period']}, '전년동기'+Q -> {r_ydq['period']} ✅")
    print("  → 통과: 별도 사전 추가 없이 기존 '전년' 키가 이미 커버하고 있었음을 확인.")

    print("\n모든 테스트 통과 🎉")