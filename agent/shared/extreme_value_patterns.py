"""
agent/shared/extreme_value_patterns.py — "역대 최고"/"9년 만에 최대"/"코로나 이후 최저" 같은
극값(extremum) 주장에서 공통으로 쓰는 정규식 패턴 + 순수 판별 함수 모음.

원래 agent/orchestrator/agent_chat.py(실전2 챗봇)에만 있던 걸 여기로 옮겼다(2026-08-05,
calc_type 라우팅 작업). 이제 agent_chat.py와 agent/orchestrator/calc_type_router.py(2단계
직후 calc_type 라우팅, 표/API 정보 없이 문장만 봄) 둘 다 여기서 import한다.

시계열을 실제로 fetch하는 로직(KosisApiClient/table_id 필요)은 agent_chat.py에 그대로
남아있다 — 여기엔 "문장에서 시작 연도를 뽑는" 순수 함수만 둔다.
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# "코로나 이후 최고" — 사건을 기준점으로 삼는 표현. "코로나 이후"는 매번 달라지는 상대
# 시점이 아니라 고정된 역사적 사실이라, LLM에 맡기지 않고 정확한 사전 하나로 처리한다
# (2026-08-06 설계, agent_chat.py에서 원래 결정된 내용).
# ---------------------------------------------------------------------------
TIME_ANCHOR_YEARS: dict[str, int] = {
    "코로나": 2020,
    "코로나19": 2020,
    "팬데믹": 2020,
}
SINCE_EVENT_RE = re.compile(r"(코로나(?:19)?|팬데믹)\s*이후")

# "8년 만에 최대"처럼 숫자로 명시된 기간 표현. "코로나 이후"(고정 이벤트)와 달리 매번
# 다른 숫자가 오는 진짜 상대 표현이라 기사 작성 연도 기준으로 계산해야 한다.
# "여"(-쯤/-남짓): "4년여 만에"처럼 "년"과 "만에" 사이에 낄 수 있는데 원래 정규식이 못
# 잡던 실제 사례를 2026-08-05 실측(코스피 변동성 기사)에서 발견해서 허용하도록 수정.
# "만에/만의/만이다/만이었다": "16년 만의 최고치"(관형형)처럼 "만에"가 아닌 다른 활용형도
# 실제 기사에 흔하게 나오는 걸 2026-08-06 300개 배치 실측에서 발견해서 확장. "10년 만기
# 채권"처럼 무관한 "만기"는 이 4개 활용형에 없어서 오탐 안 됨.
N_YEARS_SINCE_RE = re.compile(r"(\d+)년\s*여?\s*만(?:에|의|이다|이었다)")

# "역대 최대"처럼 기준 시점 자체가 없는 극값 주장.
ALL_TIME_RE = re.compile(r"역대")


def _normalize_whitespace(text: str) -> str:
    """공백 전부 제거 — agent_chat.py의 동명 함수와 동일한 정규화(코로나/N년 표현엔 원래
    공백이 거의 없어서 두 정규식(SINCE_EVENT_RE/N_YEARS_SINCE_RE) 모두 \\s*라 사실상
    결과는 같지만, 행동 차이를 만들지 않기 위해 원본과 동일하게 맞춘다). agent_chat.py의
    _normalize_whitespace는 극값 판별 이외의 다른 곳에서도 널리 쓰여서 그쪽은 그대로 두고,
    여기선 이 모듈 전용의 최소 사본만 둔다."""
    return re.sub(r"\s+", "", text)


def resolve_since_event_start_year(question: str) -> Optional[int]:
    """"코로나 이후"처럼 사건을 기준점으로 삼는 표현에서 시작 연도를 찾는다. 매칭 안 되면 None."""
    normalized_question = _normalize_whitespace(question)
    m = SINCE_EVENT_RE.search(normalized_question)
    if not m:
        return None
    anchor = m.group(1)
    key = "코로나" if anchor.startswith("코로나") else anchor
    return TIME_ANCHOR_YEARS.get(key)


def resolve_n_years_since_start_year(question: str, article_year: int) -> Optional[int]:
    """"8년 만에 최대"처럼 숫자로 명시된 기간 표현에서 시작 연도를 계산한다
    (기사 작성 연도 - N). 매칭 안 되면 None."""
    normalized_question = _normalize_whitespace(question)
    m = N_YEARS_SINCE_RE.search(normalized_question)
    if not m:
        return None
    return article_year - int(m.group(1))


def is_extreme_value_claim(text: str) -> bool:
    """문장이 "역대"/"N년 만에"/"코로나(19) 이후" 중 하나라도 포함하는 극값 주장인지만
    판단한다 (시작 연도 계산 없이 존재 여부만 빠르게 확인하고 싶을 때 사용)."""
    return bool(
        ALL_TIME_RE.search(text) or N_YEARS_SINCE_RE.search(text) or SINCE_EVENT_RE.search(text)
    )
