"""
agent/orchestrator/agent_chat.py — 실전2: 자연어를 KOSIS API 요청으로 완성하는 대화형 챗봇 에이전트

실전2 요구사항 요약:
    수십만 건 통계표를 전부 검색하는 건 1주 분량으로 무리 -> "API 요청에 필요한 값을 대화로
    채워가는" 문제로 좁힌다. 사용자는 자연어로 묻고, 에이전트는 부족한 값을 되물어 채운 뒤
    표를 찾아 계산한다.

툴 4종 (기존 3·4·5·6단계 모듈을 그대로 재사용):
    1) 재질의        — clarify_rules.py의 질문 문구는 재사용하되, "무엇을 되물을지"는
                       표별 dimensions + 원 질문의 비교 의도를 보고 에이전트가 직접 판단
                       (아래 _wants_comparison/_needs_region 참고)
    2) 통계표 매핑    — mapping/reranker.py의 search_and_rerank()
    3) API 호출      — kosis/api_client.py의 KosisApiClient
    4) 표 연산       — kosis/calculator.py의 KosisCalculator

"통계 없음/불명확" 판단:
    3단계 매핑 결과가 비었거나, 검증 안 된 임베딩 전용(신뢰도 낮음) 매칭이면 억지로 답하지
    않고 솔직하게 "통계 없음/불명확"으로 응답한다 (batch_runner.py의 표매칭_불충분 가드레일과
    동일한 원칙).

⚠️ 실전2 1주 범위에서 의도적으로 좁힌 부분 (알려진 한계):
    - calc_type이 "증감"/"증감률"이면 두 시점을 비교해서 계산하고, 그 외(합계/비율/단순조회
      등)는 전부 "단순조회"(값 그대로)로 처리한다. compute_sum/compute_ratio는 "어느 카테고리를
      묶어서 합/비율을 낼지"까지 대화로 좁혀야 하는데, 이건 표별 dimensions를 대화형으로
      순회하는 로직이 더 필요해서 이번 1주 범위에서는 뺐다.
    - 표별 세부 축(성별/연령/주택유형 등)은 D의 generic slots(period/region/calc_type)에
      없으면 table_params.json의 default_value로만 채워진다 (build_kosis_slots, 기존
      batch_runner.py와 동일 로직 재사용). "어느 연령대 기준으로 알려드릴까요?" 같은 표별
      맞춤 재질의는 다음 단계 과제로 남겨둠.
    - 표에 따라 period 형식이 연/분기/월로 다른데(예: DT_200Y102는 분기, DT_30404_B012/B013은
      월), 지금 슬롯은 연도만 채워서 형식이 안 맞으면 KOSIS가 에러를 반환한다 — 이 경우도
      "통계 없음/불명확"으로 정직하게 답한다 (table_params.json에 이미 문서화된 알려진 갭).

실행:
    python -m agent.orchestrator.agent_chat
"""

from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path
from typing import Optional

from agent.interfaces import Claim, TableCandidate, ComputedResult, KosisApiResponse
from agent.mapping.keyword_search import keyword_search
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import search_and_rerank, load_document_texts
from agent.orchestrator.slot_filler import fill_slots, call_hcx, extract_json_fallback
from agent.orchestrator.clarify_rules import CLARIFY_QUESTIONS
from agent.kosis.api_client import KosisApiClient, KosisApiError
from agent.kosis.calculator import KosisCalculator, CalculationError
from agent.shared.extreme_value_patterns import (
    TIME_ANCHOR_YEARS as _TIME_ANCHOR_YEARS,
    SINCE_EVENT_RE as _SINCE_EVENT_RE,
    N_YEARS_SINCE_RE as _N_YEARS_SINCE_RE,
    ALL_TIME_RE as _ALL_TIME_RE,
    resolve_since_event_start_year as _resolve_since_event_start_year,
    resolve_n_years_since_start_year as _resolve_n_years_since_start_year,
)

TABLE_PARAMS_PATH = Path(__file__).parent.parent / "kosis" / "table_params.json"

# "실업률 알려줘"처럼 단순 조회 의도인 질문엔 calc_type을 아예 안 물어보고 바로 단순조회로
# 추론한다. 이 키워드들이 원 질문에 있을 때만 "증감/비교" 의도로 보고 calc_type을 되묻는다.
_COMPARISON_KEYWORDS = (
    "증감", "증가", "감소", "변화", "대비", "비교", "보다",
    "늘었", "줄었", "올랐", "내렸", "상승", "하락", "추이",
)

# 표에 이 이름의 축이 있을 때만 "지역" 슬롯을 되묻는다 (없는 표에 지역을 묻는 걸 방지).
_REGION_LIKE_DIMENSIONS = ("region", "country")

# D의 generic slots(period/region/calc_type)엔 "연령대"/"성별" 개념이 없어서, 표를 맞게
# 찾아도 원 질문의 "청년"/"여성" 같은 인구통계 한정어가 API 호출에 반영이 안 되고 항상
# default_value(전체)로 조회되는 문제가 있었다 (실제로 "청년 실업률" 질문이 전체 인구
# 실업률 2.8%를 돌려준 사례로 확인됨 — 청년(15~29세)은 5.9%로 실제론 크게 다름).
# 원 질문에서 이 한정어를 감지해서 table_params.json의 age/gender dimension 값으로
# 직접 채워 넣는다 (표별 dimensions를 대화로 순회하는 정식 구현 전까지의 임시 조치).
_DEMOGRAPHIC_HINTS: dict[str, dict[str, str]] = {
    "청년": {"age": "청년(15~29세)"},
    "청년층": {"age": "청년(15~29세)"},  # golden set 실제 기사(A040-02 "청년층 고용률")에서 미매칭 확인, 2026-08-04 추가
    "고령": {"age": "60세이상"},
    "노인": {"age": "60세이상"},
    "여성": {"gender": "여자"},
    "여자": {"gender": "여자"},
    "남성": {"gender": "남자"},
    "남자": {"gender": "남자"},
}


def _normalize_whitespace(text: str) -> str:
    """공백 제거 비교용 정규화. keyword_search.py의 _normalize()와 동일한 이유 —
    기사 문장은 "원화 기준"/"국가 채무"처럼 code_map 라벨("원화기준"/"국가채무")과
    띄어쓰기가 다른 경우가 흔해서, 공백을 무시하고 비교한다."""
    return re.sub(r"\s+", "", text)


def _extract_table_dimension_hints(question: str, table_id: str, table_params: dict) -> dict:
    """매칭된 표의 dimensions(code_map)를 보고, 질문 문장에 그 코드 라벨이 이미 그대로
    나와있으면 채운다 (예: DT_1DA7102S의 age code_map에 있는 "20대"가 질문에 있으면 그대로 사용).

    _DEMOGRAPHIC_HINTS(청년/여성 등 7개 하드코딩)와 달리 표마다 다른 축(연령/성별/대출종류/
    소득분위/업종 등)에 전부 자동으로 적용된다 — 표가 늘어나도 이 함수는 안 고쳐도 됨.
    다만 표현이 code_map 라벨과 정확히 안 겹치는 경우(예: "청년" vs "청년(15~29세)")는 여기서
    못 잡으므로, _DEMOGRAPHIC_HINTS 같은 동의어 사전이 계속 보완해야 한다 — 이 함수가
    _extract_demographic_hints를 대체하는 게 아니라 보완하는 관계.

    실제 golden set(59건) 검증 중 "원화 기준"(문장) vs "원화기준"(code_map), "국가 채무" vs
    "국가채무" 같은 순수 띄어쓰기 차이로 못 잡는 사례가 다수 발견되어(2026-08-03),
    keyword_search.py와 동일하게 공백 무시 비교를 적용한다.
    """
    hints: dict = {}
    normalized_question = _normalize_whitespace(question)
    dimensions = table_params.get(table_id, {}).get("dimensions", {})
    for dim_name, dim in dimensions.items():
        code_map = dim.get("code_map", {})
        # 라벨이 여러 개 겹쳐 매칭될 수 있으니(예: "20대"와 "20~24세" 둘 다 후보), 더 구체적인
        # (긴) 라벨을 우선한다. 정규화 후 길이 기준으로 정렬해 공백 차이로 순서가 뒤바뀌지 않게 함.
        for label in sorted(code_map.keys(), key=lambda l: len(_normalize_whitespace(l)), reverse=True):
            if label and _normalize_whitespace(label) in normalized_question:
                hints[dim_name] = label
                break
    return hints


# "전체"류 라벨을 LLM 후보에 주면, 문장에 특정 값(예: "20대")이 있는데 후보엔 그 정확한 값이
# 없을 때 LLM이 null 대신 이 포괄값을 "그럴듯한 답"으로 골라버리는 경우가 실제로 확인됨
# (2026-08-05, DT_1B04005N: "20대 인구" 문장 → age 후보가 ['계','전체']뿐이라 LLM이 '전체'를
# 답함 — 이건 "전체 인구"라는 완전히 다른 값이라 처음 발견한 "20대 실업률" 버그와 같은 성격의
# 오답임). 이런 라벨이 진짜 정답인 케이스는 애초에 아무것도 안 채워져도 default_value가 같은
# 값을 채우므로, 후보에서 빼도 손해가 없다 — "억지로 아무거나 고르는" 위험만 제거된다.
_GENERIC_FALLBACK_LABELS = {"전체", "계", "합계"}


def _build_dimension_llm_prompt(question: str, dimensions: dict, unresolved_dims: list[str]) -> str:
    lines = []
    for dim_name in unresolved_dims:
        all_candidates = dimensions.get(dim_name, {}).get("code_map", {}).keys()
        candidates = sorted(c for c in all_candidates if c not in _GENERIC_FALLBACK_LABELS)
        if candidates:
            lines.append(f'- {dim_name}: 후보 = {candidates}')
    candidates_block = "\n".join(lines)
    example_dim = unresolved_dims[0]
    example_json = json.dumps(
        {example_dim: {"value": "후보 중 하나 또는 null", "quote": "그 근거가 된 문장 속 표현 그대로, value가 null이면 null"}},
        ensure_ascii=False,
    )
    return f"""다음 문장에서 아래 각 축(dimension)에 해당하는 값을 반드시 "후보" 목록 중에서만 골라
JSON으로 응답하세요. 후보 목록에 없는 새 값을 만들어내지 마세요.

중요 규칙:
- 각 축마다 value(후보 중 하나 또는 null)와 quote(그렇게 판단한 근거가 되는, 문장에 실제로
  나온 표현을 그대로 복사)를 같이 답하세요. quote는 지어내지 말고 문장에서 그대로 가져오세요.
- 문장에 해당 내용이 없거나, 후보 중 확실히 맞는 게 없거나, 근거가 되는 표현을 문장에서
  못 찾겠으면 value와 quote 둘 다 반드시 null로 응답하세요. 애매하면 추측하지 말고 null.
- 설명이나 다른 텍스트는 절대 포함하지 마세요.

축 목록과 후보:
{candidates_block}

문장: "{question}"

응답 형식 (JSON만, 다른 텍스트 없이, 예시): {example_json}
"""


def _extract_table_dimension_hints_llm(
    question: str, table_id: str, table_params: dict, unresolved_dims: list[str]
) -> dict:
    """_extract_table_dimension_hints(코드 기반, 1차)가 못 채운 축만 LLM(HCX-DASH-002)에
    넘겨서 보완한다 (2차 폴백). code_map에 있는 라벨과 문장 표현이 정확히 안 겹치는 진짜
    동의어/표현차이 케이스(예: "1인당 국민총소득(GNI)" vs code_map의 "1인당GNI달러")를 위한 것 —
    2026-08-03 golden set 검증에서 이런 케이스가 실제로 확인됐지만 건수가 적어(1건) 보류했었는데,
    2026-08-05 사용자 요청으로 실제 구현.

    환각 방지 3단계:
    1) 그 표의 code_map에 실제로 존재하는 라벨(전체/계/합계 제외)이 아니면 버린다.
    2) LLM이 같이 답한 quote(근거 문구)가 실제로 원문 문장에 있는지 확인하고, 없으면 버린다 —
       "value는 후보 안에 있지만 문장에 아무 근거가 없는데도 그럴듯하게 골라버리는" 진짜 환각을
       잡기 위한 것 (2026-08-05 확인: "고용률..." 문장에 성별 언급이 전혀 없는데도 gender 후보
       중 하나인 "비농가"를 근거 없이 답한 사례 — code_map 소속 여부 검사만으로는 못 걸러짐).
    3) 문장이 "20대"/"70대 이상" 같은 다중코드(10세 단위) 패턴이면, 그 축은 LLM한테 아예
       안 물어본다 — 2026-08-06 확인: DT_1B04005N.age에 5세 단위 21개 구간을 채운 뒤로
       "20대 인구" 문장에 LLM이 "20~24세"(20대의 절반뿐)를 답으로 내놓는 새 오답이 생김.
       "20대"는 resolve_decade_age_responses()가 별도로 처리해야 하는 값이라, 단일값 후보
       중 하나를 고르는 이 함수가 손대면 안 됨."""
    if not unresolved_dims:
        return {}
    dimensions = table_params.get(table_id, {}).get("dimensions", {})

    def _has_real_candidates(dim_name: str) -> bool:
        code_map = dimensions.get(dim_name, {}).get("code_map", {})
        return any(c not in _GENERIC_FALLBACK_LABELS for c in code_map)

    def _is_decade_multi_code_case(dim_name: str) -> bool:
        code_map = dimensions.get(dim_name, {}).get("code_map", {})
        return _resolve_decade_age_codes(question, code_map) is not None

    unresolved_dims = [
        d for d in unresolved_dims if _has_real_candidates(d) and not _is_decade_multi_code_case(d)
    ]
    if not unresolved_dims:
        return {}

    prompt = _build_dimension_llm_prompt(question, dimensions, unresolved_dims)
    try:
        raw = call_hcx(prompt)
    except Exception:
        return {}

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        extracted = extract_json_fallback(raw)

    normalized_question = _normalize_whitespace(question)
    hints: dict = {}
    for dim_name in unresolved_dims:
        entry = extracted.get(dim_name)
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        quote = entry.get("quote")
        if value is None or not quote:
            continue
        candidates = dimensions[dim_name].get("code_map", {})
        if value not in candidates or value in _GENERIC_FALLBACK_LABELS:
            continue
        if _normalize_whitespace(str(quote)) not in normalized_question:
            continue  # quote가 원문에 실제로 없으면(지어낸 근거) 버림
        hints[dim_name] = value
    return hints


def _extract_demographic_hints(question: str) -> dict:
    hints: dict = {}
    for keyword, slot_values in _DEMOGRAPHIC_HINTS.items():
        if keyword in question:
            hints.update(slot_values)
    return hints


# ---------------------------------------------------------------------------
# "20대"/"70대 이상" 같은 10세 단위 다중코드 합산 (DT_1B04005N age 등)
#
# KOSIS는 이 표를 5세 단위로만 분류해서 "20대"에 해당하는 단일 코드가 없다. "20대"마다
# 하드코딩하는 대신, code_map에 이미 등록된 5세 단위 라벨("20~24세" 등)을 파싱해서 그
# 10세 구간에 포함되는 코드들을 자동으로 찾아낸다 — 10대~100세이상까지 전부 이 함수
# 하나로 커버되고, 다른 표에 같은 5세 단위 구조가 있어도 그대로 재사용 가능하다
# (2026-08-06 설계).
# ---------------------------------------------------------------------------
_AGE_BAND_RANGE_RE = re.compile(r"^(\d+)~(\d+)세$")
_AGE_100_PLUS_RE = re.compile(r"^100세이상$")
# (?<!\d): 매칭되는 한 자리 숫자 앞에 다른 숫자가 있으면(예: "1480대"의 "80대") 매칭하지
# 않는다 — 실제 기사 실측(2026-08-04, full_coverage_result.jsonl 3181건 분석)에서
# "1082만1480대"(자동차 판매 대수)처럼 큰 숫자의 끝자리가 나이대로 오탐되는 사례를 발견해서
# 추가한 정규식 레벨 방어. "20대"/"10·20대"처럼 진짜 나이대는 앞에 다른 숫자가 안 붙는다.
_DECADE_EXPR_RE = re.compile(r"(?<!\d)(\d)0대(\s*이상)?")


def _parse_age_band_label(label: str) -> Optional[tuple[int, Optional[int]]]:
    """code_map의 5세 단위 라벨을 (시작나이, 끝나이) 튜플로 변환한다. "100세이상"처럼
    끝이 없으면 끝나이는 None(무한대 취급). 매칭 안 되면 None."""
    m = _AGE_BAND_RANGE_RE.match(label)
    if m:
        return int(m.group(1)), int(m.group(2))
    if _AGE_100_PLUS_RE.match(label):
        return 100, None
    return None


# ---------------------------------------------------------------------------
# "숫자0대"가 진짜 나이대인지 문맥 확인 (1차 규칙 + 2차 LLM 더블체크)
#
# 실측(2026-08-04, 실제 기사 1005건/claim 3181건 분석)에서 같은 정규식이 나이대가 아닌
# 경우에도 오탐되는 걸 확인함: "상위 10대 기업"(순위), "K2 전차 총 180대"(대수/단위).
# 정규식 하나로는 이 둘을 못 걸러서, 표별 슬롯 힌트(LLM 2차)와 같은 원칙으로 처리한다 —
# 확실한 신호가 있는 대다수는 규칙(공짜)으로 끝내고, 신호가 없는 애매한 소수만 LLM에 묻는다.
# ---------------------------------------------------------------------------
_DECADE_AGE_FOLLOW_WHITELIST = (
    "후반", "초반", "중반", "이상", "미만", "인구", "실업률", "취업률", "취업자",
    "근로자", "직장인", "남성", "여성", "남자", "여자", "가구주", "세대주",
)
_DECADE_AGE_FOLLOW_BLACKLIST = ("기업", "은행", "국가", "기관", "대학", "도시", "종목", "그룹", "브랜드")
_DECADE_AGE_PRECEDE_BLACKLIST = (
    "전차", "차량", "트럭", "헬기", "전투기", "버스", "컴퓨터", "노트북", "항공기", "탱크", "장비",
)
# 조사로 자연스럽게 끝나는 경우("20대는", "20대가")도 나이대로 확정 — 순위/대수 표현은
# 보통 뒤에 바로 다른 명사(기업/전차 등)가 오지 조사만 딱 붙어 끝나지 않는다.
_DECADE_AGE_PARTICLE_CHARS = "는가이을를의도만에과와랑"


def _check_decade_age_context(normalized_question: str, match: re.Match) -> Optional[bool]:
    """"숫자0대" 매칭 앞뒤 문맥만 보고 확실히 판단되면 True/False, 확실하지 않으면(애매) None.

    None을 반환하면 호출하는 쪽이 LLM(2차)에게 물어봐야 한다 — 여기서 애매한 걸 억지로
    True/False로 단정하지 않는다."""
    follow = normalized_question[match.end() : match.end() + 6]
    precede = normalized_question[max(0, match.start() - 6) : match.start()]

    if any(w in follow for w in _DECADE_AGE_FOLLOW_BLACKLIST):
        return False
    if any(w in precede for w in _DECADE_AGE_PRECEDE_BLACKLIST):
        return False
    if any(w in follow for w in _DECADE_AGE_FOLLOW_WHITELIST):
        return True
    if not follow or follow[0] in _DECADE_AGE_PARTICLE_CHARS:
        return True
    return None


def _confirm_decade_age_with_llm(question: str, matched_expr: str) -> bool:
    """규칙(1차)으로 확실히 판단 안 되는 애매한 경우만 호출하는 2차 확인. 실패/근거없음/
    근거가 원문에 없으면(환각) 전부 안전하게 False(나이대 아님)로 처리한다 — 표별
    LLM 힌트(_extract_table_dimension_hints_llm)와 같은 "확신 없으면 억지로 추측하지
    않는다" 원칙."""
    prompt = f"""다음 문장에서 "{matched_expr}"라는 표현이 나이대(연령대)를 뜻하는지 판단하세요.

규칙:
- 나이대를 뜻하면(예: "20대 취업자 수", "40대 남성") value를 true로 답하세요.
- 개수를 세는 단위(예: "전차 180대")나 순위(예: "상위 10대 기업")처럼 나이가 아니면
  value를 false로 답하세요.
- 판단 근거가 되는, 문장에 실제로 나온 표현을 quote에 그대로 복사하세요. 지어내지 마세요.
- 설명이나 다른 텍스트는 절대 포함하지 마세요.

문장: "{question}"

응답 형식 (JSON만, 다른 텍스트 없이): {{"value": true 또는 false, "quote": "근거 문구"}}
"""
    try:
        raw = call_hcx(prompt)
    except Exception:
        return False

    try:
        extracted = json.loads(raw)
    except json.JSONDecodeError:
        extracted = extract_json_fallback(raw)

    value = extracted.get("value")
    quote = extracted.get("quote")
    if value is not True or not quote:
        return False
    if _normalize_whitespace(str(quote)) not in _normalize_whitespace(question):
        return False
    return True


def _resolve_decade_age_codes(question: str, age_code_map: dict) -> Optional[tuple[str, list[str]]]:
    """문장에서 "20대"/"70대 이상" 같은 표현을 찾아, age_code_map(5세 단위 라벨들) 중 그
    구간에 해당하는 코드 목록을 반환한다. 매칭 없으면 None.

    반환값: (매칭된 표현 원문, 해당 구간 코드 리스트) — 코드가 2개 이상이면 호출하는 쪽이
    api_client를 코드별로 여러 번 호출한 뒤 calculator.compute_sum()으로 합산해야 한다."""
    normalized_question = _normalize_whitespace(question)
    m = _DECADE_EXPR_RE.search(normalized_question)
    if not m:
        return None

    context_check = _check_decade_age_context(normalized_question, m)
    if context_check is False:
        return None
    if context_check is None and not _confirm_decade_age_with_llm(question, m.group(0)):
        return None

    decade_start = int(m.group(1)) * 10
    is_or_above = bool(m.group(2))

    matched_codes = []
    for label, code in age_code_map.items():
        band = _parse_age_band_label(label)
        if band is None:
            continue
        band_start, band_end = band
        if is_or_above:
            if band_start >= decade_start:
                matched_codes.append(code)
        elif band_start >= decade_start and (band_end is not None and band_end < decade_start + 10):
            matched_codes.append(code)

    if not matched_codes:
        return None
    return m.group(0), matched_codes


def resolve_decade_age_responses(
    table_id: str, question: str, base_slots: dict, table_params: dict, client
) -> Optional[tuple[str, list]]:
    """"20대"/"70대 이상" 등을 감지해서, 필요한 코드들을 client.fetch_by_codes()로 한 번에
    가져온다 (합산은 호출하는 쪽이 calculator.compute_sum()으로). 감지 안 되면 None.

    2026-08-06: 원래 코드별로 client()를 여러 번 호출했었는데, resolve_max_since_event_
    responses()에 fetch_series()를 적용한 것과 일관되게 맞추려고 fetch_by_codes() 한 번
    호출로 리팩터링(KOSIS가 objL2="25+30"처럼 "+"로 묶은 요청도 한 번에 응답해줌을 확인)."""
    age_code_map = table_params.get(table_id, {}).get("dimensions", {}).get("age", {}).get("code_map", {})
    resolved = _resolve_decade_age_codes(question, age_code_map)
    if resolved is None:
        return None
    label, codes = resolved
    responses = client.fetch_by_codes(table_id, base_slots, "age", codes)
    return label, responses


# ---------------------------------------------------------------------------
# "코로나 이후 최고"/"역대 최대" 같은 극값(extremum) 주장 검증
#
# 이런 주장은 단일 KosisApiResponse 하나로는 못 다룬다 — "기준 시점부터 지금까지의 모든
# 과거 값"을 실제로 조회해서, 현재 값이 정말 그 중 최댓값인지 비교해야 한다(6단계
# calculator.compute_max_check() 참고). "코로나 이후"처럼 사건을 기준점으로 삼는 표현은
# 그 사건이 언제 시작됐는지를 먼저 알아야 하는데, 이건 "작년"/"지난달"처럼 매번 달라지는
# 상대 시점이 아니라 고정된 역사적 사실이라 정확한 사전 하나로 충분하다 — LLM에 맡기면
# 매번 값이 오락가락(비결정적)할 수 있는 걸 굳이 LLM에 맡길 이유가 없다(2026-08-06 설계).
#
# 정규식 상수(_TIME_ANCHOR_YEARS/_SINCE_EVENT_RE/_N_YEARS_SINCE_RE/_ALL_TIME_RE)와 순수
# 판별 함수(_resolve_since_event_start_year/_resolve_n_years_since_start_year)는
# agent/shared/extreme_value_patterns.py로 옮겼다(2026-08-05, calc_type_router.py도 같이
# 씀 — 위 import 구문 참고). 실제 시계열 fetch(client.fetch_series 호출)는 표/API 정보가
# 필요해서 여기 그대로 남는다.
# ---------------------------------------------------------------------------


def resolve_max_since_event_responses(
    table_id: str, question: str, base_slots: dict, client, *, current_year: int
) -> Optional[tuple[int, list]]:
    """"코로나 이후 최고" 등을 감지해서, 시작 연도부터 current_year까지의 시계열을
    client.fetch_series()로 한 번에 가져온다 (최댓값 검증은 호출하는 쪽이
    calculator.compute_max_check()로). 감지 안 되면 None.

    2026-08-06: 원래 연도별로 client()를 여러 번 호출했었는데, KOSIS가 넓은 기간
    요청(startPrdDe~endPrdDe)을 한 번의 호출로 전부 돌려준다는 걸 확인해서
    fetch_series() 한 번 호출로 리팩터링(실측: 26개 연도가 1번 호출로 옴)."""
    start_year = _resolve_since_event_start_year(question)
    if start_year is None:
        return None
    responses = client.fetch_series(table_id, base_slots, str(start_year), str(current_year))
    return start_year, responses


def resolve_max_n_years_responses(
    table_id: str, question: str, base_slots: dict, client, *, article_year: int, current_year: int
) -> Optional[tuple[int, list]]:
    """"8년 만에 최대" 등을 감지해서, (기사 작성 연도-N)부터 current_year까지의 시계열을
    한 번에 가져온다. 감지 안 되면 None."""
    start_year = _resolve_n_years_since_start_year(question, article_year)
    if start_year is None:
        return None
    responses = client.fetch_series(table_id, base_slots, str(start_year), str(current_year))
    return start_year, responses


def resolve_max_all_time_responses(
    table_id: str, question: str, base_slots: dict, client, *, current_year: int, earliest_year: int = 1960
) -> Optional[list]:
    """"역대 최대"처럼 기준 시점 자체가 없는 극값 주장을 감지해서, 그 표가 실제로 갖고
    있는 가장 오래된 시점부터 전부 가져온다. earliest_year는 안전하게 이른 기본값(1960)일
    뿐 — KOSIS는 실제 데이터가 없는 앞부분은 그냥 알아서 제외하고 있는 것만 돌려주므로
    표마다 정확한 시작 연도를 몰라도 안전하다(실측: DT_1DA7102S에 1999년부터 요청해도
    실제 데이터가 있는 2000년부터로 정상 응답). 감지 안 되면 None."""
    if not _ALL_TIME_RE.search(question):
        return None
    return client.fetch_series(table_id, base_slots, str(earliest_year), str(current_year))


def _wants_comparison(question: str) -> bool:
    return any(kw in question for kw in _COMPARISON_KEYWORDS)


def _format_number(value: float) -> str:
    """9304400.0 -> "9,304,400", 5.9 -> "5.9", 291919.23905 -> "291,919.24" 처럼
    정수는 소수점 없이, 소수는 천단위 구분+소수점 2자리로 사람이 읽기 편하게 표시."""
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def _normalize_short_reply(text: str) -> str:
    """D의 slot_filler 프롬프트는 few-shot 예시가 전부 "~알려줘" 꼴이라, 동사 없는 단답
    (예: "증감률", "합계")은 정보 없음(null)으로 판단해버리는 경향이 있다 (실제 확인됨).
    짧은 단답에 동사를 붙여서 인식률을 보정한다 — slot_filler.py 자체는 건드리지 않음."""
    stripped = text.strip()
    if len(stripped) <= 8 and not any(stripped.endswith(end) for end in ("줘", "요", "다", "까", "함")):
        return f"{stripped} 알려줘"
    return text


# "23년도"/"2023년도"/"24년"처럼 접미사 붙은 연도 표현을, D의 슬롯 추출 LLM이 간혹(비결정적
# 으로) "2024"로 안 바꾸고 "23년도" 그대로 반환해버려서 is_valid_period() 검증에 걸려
# 통째로 버려지는 문제가 실제로 재현됨 ("24년도"는 되는데 "23년도"는 안 되는 식으로 같은
# 패턴에서 오락가락함). LLM에 넘기기 전에 미리 깔끔한 4자리 연도로 바꿔서 이 비결정성 자체를
# 피한다 — slot_filler.py는 건드리지 않음.
_YEAR_SUFFIX_RE = re.compile(r"(\d{2}|\d{4})\s*년도?(?=\s|$)")
# "년/년도" 접미사도 없이 그냥 "26"처럼 숫자만 딱 온 답변 — "어느 연도 기준으로 알려드릴까요?"
# 라는 질문에 대한 답으로 이것 말고는 해석할 여지가 없는 경우라, 접미사 없이도 연도로 인정한다.
_BARE_YEAR_RE = re.compile(r"^(\d{2}|\d{4})$")


def _normalize_year_expression(text: str) -> str:
    stripped = text.strip()
    bare_match = _BARE_YEAR_RE.fullmatch(stripped)
    if bare_match:
        digits = bare_match.group(1)
        return f"20{digits}" if len(digits) == 2 else digits

    def _replace(match: re.Match) -> str:
        digits = match.group(1)
        year = f"20{digits}" if len(digits) == 2 else digits
        return year

    return _YEAR_SUFFIX_RE.sub(_replace, text)


# "26년6월"/"2026년 6월"처럼 월 단위까지 명시한 질문은 "YYYYMM" 6자리로 뽑아낸다.
# D의 slot_filler는 연도(4자리) 단위 추출만 하도록 설계돼 있어서(is_valid_period가 4자리만
# 통과시킴), 6자리 값을 fill_slots에 맡기면 검증에 걸려 버려진다 — 그래서 이건 fill_slots를
# 거치지 않고 원 발화에서 직접 뽑아 slots["period"]에 나중에 덮어쓴다.
_MONTH_EXPR_RE = re.compile(r"(\d{2}|\d{4})년\s*(\d{1,2})월")


def _extract_month_period(text: str) -> Optional[str]:
    m = _MONTH_EXPR_RE.search(text)
    if not m:
        return None
    year_digits, month_digits = m.groups()
    year = f"20{year_digits}" if len(year_digits) == 2 else year_digits
    month = month_digits.zfill(2)
    return f"{year}{month}"


# ---------------------------------------------------------------------------
# 툴 1: 통계표 매핑 — 실전1(3단계)의 search_and_rerank를 그대로 재사용
# ---------------------------------------------------------------------------
def tool_search_table(user_question: str, embedding_cache: dict, document_texts: dict) -> list[TableCandidate]:
    claim = Claim(sentence=user_question, claim_type="규모")
    return search_and_rerank(
        claim,
        keyword_fn=keyword_search,
        embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
        top_k=3,
        document_texts=document_texts,
    )


# ---------------------------------------------------------------------------
# 툴 2: 재질의 — D의 clarify_rules.CLARIFY_QUESTIONS(질문 문구)는 재사용하되, "어떤 값을
# 되묻고 어떤 값을 추론할지"는 표별 dimensions + 원 질문의 비교 의도를 보고 에이전트가 직접
# 판단한다 (clarify.py의 "모든 표에 무조건 3개 다 되묻기" 로직은 안 씀 — 실제 대화에서
# "실업률 알려줘"처럼 단순조회 의도인데도 지역/계산방식까지 억지로 되묻는 문제가 있었음).
# ---------------------------------------------------------------------------
def tool_reask(slots: dict, *, needs_region: bool, needs_calc_type: bool) -> Optional[str]:
    if not slots.get("period"):
        return CLARIFY_QUESTIONS["period"]
    if needs_region and not slots.get("region"):
        return CLARIFY_QUESTIONS["region"]
    if needs_calc_type and not slots.get("calc_type"):
        return CLARIFY_QUESTIONS["calc_type"]
    return None


# ---------------------------------------------------------------------------
# 툴 3: API 호출 — D의 generic slots(period/region/calc_type)를 표별 dimensions로
# 변환하는 로직은 batch_runner.py의 build_kosis_slots와 동일 (표별 세부 축은 default_value)
# ---------------------------------------------------------------------------
def build_kosis_slots(table_id: str, generic_slots: dict, table_params: dict) -> Optional[dict]:
    if table_id not in table_params:
        return None
    base = table_params[table_id]
    kosis_slots: dict = {"period": generic_slots.get("period")}
    for dim_name, dim in base.get("dimensions", {}).items():
        value = generic_slots.get(dim_name)
        kosis_slots[dim_name] = value if value is not None else dim.get("default_value")
    return kosis_slots


def tool_call_kosis(
    table_id: str, slots: dict, table_params: dict, client: KosisApiClient
) -> KosisApiResponse:
    kosis_slots = build_kosis_slots(table_id, slots, table_params)
    if kosis_slots is None:
        raise KeyError(f"'{table_id}'가 table_params.json에 없습니다.")
    period = kosis_slots.get("period") or ""

    if len(period) == 6:
        # "202606"처럼 월 단위 6자리 period면, 이 표의 기본 prdSe(보통 Y)와 무관하게
        # 월(M) 단위로 조회하도록 api_client.py에 명시적으로 알려준다.
        kosis_slots["prd_se"] = "M"
        return client(table_id, kosis_slots)

    if len(period) == 4 and period.isdigit() and int(period) == date.today().year:
        # "올해"(아직 안 끝난 해)는 연간 확정치가 원천적으로 없는 경우가 많다. 이 표가
        # 월 단위 데이터도 갖고 있을 수 있으니, 최근 몇 개월을 거꾸로 시도해서 조회되는
        # 첫 값을 대신 쓴다 (예: "2026 알려줘" -> 2026년 데이터가 없으면 2026-06, 2026-05...
        # 순으로 시도). 이 표가 애초에 월 단위를 지원 안 하면 전부 실패하고 아래
        # 원래 연간 조회로 넘어가서 기존과 동일하게 에러를 낸다.
        today = date.today()
        for months_back in range(4):
            month = today.month - months_back
            year = today.year
            if month <= 0:
                month += 12
                year -= 1
            fallback_slots = dict(kosis_slots, period=f"{year}{month:02d}", prd_se="M")
            try:
                return client(table_id, fallback_slots)
            except (KosisApiError, KeyError):
                continue

    return client(table_id, kosis_slots)


# ---------------------------------------------------------------------------
# 툴 4: 표 연산 — calculator.py 그대로 재사용
# ---------------------------------------------------------------------------
def tool_calculate(
    calc_type: Optional[str],
    calculator: KosisCalculator,
    *,
    base: Optional[KosisApiResponse] = None,
    target: Optional[KosisApiResponse] = None,
    single: Optional[KosisApiResponse] = None,
) -> ComputedResult:
    if calc_type == "증감" and base and target:
        return calculator.compute_change(base, target)
    if calc_type == "증감률" and base and target:
        return calculator.compute_change_rate(base, target)
    assert single is not None
    return ComputedResult(
        calc_type="단순조회", raw_value=single.raw_value, unit=single.unit, period=single.period
    )


# ---------------------------------------------------------------------------
# 에이전트 오케스트레이터 — 대화 상태를 들고 있으면서 4개 툴 호출 순서를 결정
# ---------------------------------------------------------------------------
class ChatSession:
    def __init__(self, *, verbose: bool = True):
        self.verbose = verbose
        self.client = KosisApiClient()
        self.calculator = KosisCalculator()
        with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
            self.table_params = json.load(f)
        self.embedding_cache = build_table_embedding_cache()
        self.document_texts = load_document_texts()

        self.slots: dict = {}
        self.table: Optional[TableCandidate] = None
        self.wants_comparison: bool = False

    def reset(self) -> None:
        self.slots = {}
        self.table = None
        self.wants_comparison = False

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _needs_region(self) -> bool:
        """지역 축이 있어도 code_map에 "전국" 하나뿐이면(예: 인구이동/사교육비/사회조사
        표는 세부 지역 코드를 아직 안 채워둠) 물어봐야 실질적인 의미가 없으므로 묻지 않는다."""
        assert self.table is not None
        dims = self.table_params.get(self.table.table_id, {}).get("dimensions", {})
        for dim_name in _REGION_LIKE_DIMENSIONS:
            dim = dims.get(dim_name)
            if dim and len(set(dim.get("code_map", {}).values())) > 1:
                return True
        return False

    def handle(self, user_input: str) -> str:
        """사용자 발화 1건을 받아서 에이전트 응답(str)을 돌려준다."""

        # 아직 표를 못 찾았으면: 이번 발화로 통계표 매핑부터 시도
        if self.table is None:
            self._log(f'  [tool: 통계표_매핑] query="{user_input}"')
            candidates = tool_search_table(user_input, self.embedding_cache, self.document_texts)
            if not candidates:
                return "죄송합니다, 관련된 통계표를 찾지 못했습니다. (통계 없음/불명확)"

            top = candidates[0]
            if top.source_meta and "unverified" in top.source_meta:
                return (
                    f"'{top.table_name}' 표가 후보로 나왔지만 신뢰도가 낮아 확답드리기 어렵습니다. "
                    "(통계 없음/불명확 — 질문을 더 구체적으로 해주시면 다시 찾아볼게요)"
                )

            self.table = top
            self.wants_comparison = _wants_comparison(user_input)
            # dimension_hints: 매칭된 표의 code_map을 직접 보고 채우는 범용 로직(신규) — 표가
            # 뭐든(연령/성별/대출종류/소득분위 등) 자동으로 동작한다.
            # demographic_hints: 기존 7개 하드코딩(청년/여성 등) — code_map 라벨과 표현이 정확히
            # 안 겹치는 경우(예: "청년" vs "청년(15~29세)")를 보완한다.
            # dimension_hints가 code_map과 직접 매칭된 값이라 더 정확하므로 우선 적용한다.
            dimension_hints = _extract_table_dimension_hints(user_input, top.table_id, self.table_params)
            demographic_hints = _extract_demographic_hints(user_input)
            combined_hints = {**demographic_hints, **dimension_hints}

            # 2차: 1차(코드 기반)가 못 채운 축만 LLM에 넘겨서 보완 시도 (진짜 동의어/표현차이용).
            all_dims = self.table_params.get(top.table_id, {}).get("dimensions", {})
            unresolved_dims = [d for d in all_dims if d not in combined_hints]
            llm_hints = _extract_table_dimension_hints_llm(
                user_input, top.table_id, self.table_params, unresolved_dims
            )
            combined_hints = {**llm_hints, **combined_hints}  # 코드 기반 결과가 항상 우선

            self.slots.update(combined_hints)
            self._log(f"  [tool: 통계표_매핑] 후보 확정 -> {top.table_name} ({top.table_id})")
            self._log(f"  [tool: 통계표_매핑] 비교 의도 감지: {self.wants_comparison}")
            if combined_hints:
                self._log(f"  [tool: 통계표_매핑] 표별 슬롯 힌트 감지 -> {combined_hints}")
            if llm_hints:
                self._log(f"  [tool: 통계표_매핑] LLM 보완 힌트 -> {llm_hints}")

        # 이번 발화에서 슬롯 추출 (되묻기 답변이든 첫 질문 자체든 동일하게 시도)
        normalized = _normalize_short_reply(_normalize_year_expression(user_input))
        self._log(f'  [tool: 슬롯추출] "{normalized}"')
        self.slots = fill_slots(normalized, self.slots, date.today())

        # "26년6월"처럼 월까지 명시했으면, fill_slots(연도 4자리만 인식)가 채운 값을
        # 덮어써서 정확한 월 단위(YYYYMM)로 조회되게 한다.
        month_period = _extract_month_period(user_input)
        if month_period:
            self.slots["period"] = month_period
            self._log(f"  [tool: 슬롯추출] 월 단위 시점 감지 -> {month_period}")

        question = tool_reask(
            self.slots, needs_region=self._needs_region(), needs_calc_type=self.wants_comparison
        )
        if question:
            self._log(f'  [tool: 재질의] -> "{question}"')
            return question

        # 슬롯이 다 채워짐 -> API 호출 + 표 연산
        try:
            calc_type = self.slots.get("calc_type")
            self._log(f"  [tool: API_호출] table={self.table.table_id} slots={self.slots}")

            if calc_type in ("증감", "증감률") and self.slots.get("period"):
                base_slots = dict(self.slots, period=str(int(self.slots["period"]) - 1))
                base_resp = tool_call_kosis(self.table.table_id, base_slots, self.table_params, self.client)
                target_resp = tool_call_kosis(self.table.table_id, self.slots, self.table_params, self.client)
                self._log(f"  [tool: 표연산] calc_type={calc_type}")
                computed = tool_calculate(calc_type, self.calculator, base=base_resp, target=target_resp)
            else:
                resp = tool_call_kosis(self.table.table_id, self.slots, self.table_params, self.client)
                self._log("  [tool: 표연산] calc_type 없음/미지원 -> 단순조회")
                computed = tool_calculate(calc_type, self.calculator, single=resp)

        except (KeyError, KosisApiError, CalculationError) as e:
            answer = f"조회/계산 중 문제가 발생했습니다: {e} (통계 없음/불명확)"
            self.reset()
            return answer

        answer = (
            f"'{self.table.table_name}' 기준, {computed.period} {computed.calc_type} "
            f"{_format_number(computed.raw_value)}{computed.unit} 입니다."
        )
        self.reset()
        return answer


def run_cli(*, verbose: bool = True) -> None:
    print("=== KOSIS 챗봇 (실전2 데모) ===")
    print("궁금한 것을 물어보세요. 종료하려면 'q' 입력.\n")

    session = ChatSession(verbose=verbose)
    while True:
        try:
            user_input = input("사용자> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.lower() in ("q", "quit", "exit"):
            break
        if not user_input:
            continue
        answer = session.handle(user_input)
        print(f"에이전트> {answer}\n")


if __name__ == "__main__":
    import sys

    # 발표/시연용으로 툴 호출 로그 없이 깔끔하게 보고 싶으면: python -m agent.orchestrator.agent_chat --clean
    run_cli(verbose="--clean" not in sys.argv)
