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
from agent.mapping.reranker import search_and_rerank
from agent.orchestrator.slot_filler import fill_slots
from agent.orchestrator.clarify_rules import CLARIFY_QUESTIONS
from agent.kosis.api_client import KosisApiClient, KosisApiError
from agent.kosis.calculator import KosisCalculator, CalculationError

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
    "고령": {"age": "60세이상"},
    "노인": {"age": "60세이상"},
    "여성": {"gender": "여자"},
    "여자": {"gender": "여자"},
    "남성": {"gender": "남자"},
    "남자": {"gender": "남자"},
}


def _extract_demographic_hints(question: str) -> dict:
    hints: dict = {}
    for keyword, slot_values in _DEMOGRAPHIC_HINTS.items():
        if keyword in question:
            hints.update(slot_values)
    return hints


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
def tool_search_table(user_question: str, embedding_cache: dict) -> list[TableCandidate]:
    claim = Claim(sentence=user_question, claim_type="규모")
    return search_and_rerank(
        claim,
        keyword_fn=keyword_search,
        embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
        top_k=3,
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
            candidates = tool_search_table(user_input, self.embedding_cache)
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
            demographic_hints = _extract_demographic_hints(user_input)
            self.slots.update(demographic_hints)
            self._log(f"  [tool: 통계표_매핑] 후보 확정 -> {top.table_name} ({top.table_id})")
            self._log(f"  [tool: 통계표_매핑] 비교 의도 감지: {self.wants_comparison}")
            if demographic_hints:
                self._log(f"  [tool: 통계표_매핑] 인구통계 한정어 감지 -> {demographic_hints}")

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
