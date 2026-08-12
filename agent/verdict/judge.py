"""
agent/verdict/judge.py — 7단계 비교·판정 (일치/불일치/판단불가)

팀 계약(interfaces.py) 기준:
    입력: Claim(기사 수치 주장) + ComputedResult(KOSIS 계산값)
    출력: Verdict
    모델: 1차 필터는 코드 규칙, 애매 경계만 HCX-003(단순)/HCX-007(복합 추론) 위임

※ interfaces.py의 Claim은 sentence(원문 텍스트)만 있고 파싱된 숫자 필드가 없습니다.
  그래서 규칙 기반 1차 필터를 돌리려면 sentence에서 수치를 직접 뽑아내야 하는데
  (_extract_claim_number), 이건 100% 정확할 수 없는 정규식 기반 best-effort입니다.
  추출 실패/애매한 경우는 전부 LLM에 원문 문장 그대로 넘겨서 판단을 맡깁니다 —
  즉 규칙 필터는 "명확한 경우만 걸러내고, 나머지는 보수적으로 LLM에 위임"하는 방향으로
  설계했습니다 (반대로 하면 애매한 걸 코드가 잘못 확정 판정할 위험이 큼).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

try:
    from interfaces import Claim, ComputedResult, Verdict
except ImportError:
    try:
        from agent.interfaces import Claim, ComputedResult, Verdict
    except ImportError:  # 단독 실행/테스트용 폴백
        from dataclasses import dataclass
        from typing import Literal

        VerdictType = Literal["일치", "불일치", "판단불가"]  # type: ignore[misc]
        GapType = Optional[Literal["수치", "기간", "모집단", "과장표현"]]  # type: ignore[misc]

        @dataclass
        class Claim:  # type: ignore[no-redef]
            sentence: str
            claim_type: str
            period: Optional[str] = None
            unit: Optional[str] = None
            population: Optional[str] = None

        @dataclass
        class ComputedResult:  # type: ignore[no-redef]
            calc_type: str
            raw_value: float
            unit: str
            period: str

        @dataclass
        class Verdict:  # type: ignore[no-redef]
            verdict: str
            gap_type: Optional[str] = None
            reason: str = ""

try:
    from agent.preprocessing.hcx_client import call_hcx
except ImportError:
    from preprocessing.hcx_client import call_hcx  # type: ignore[no-redef]


MODEL_SIMPLE = "HCX-003"    # 애매 경계 단일 Claim vs ComputedResult 판정
MODEL_COMPLEX = "HCX-007"   # 여러 Claim/ComputedResult를 종합해야 하는 복합 추론
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "judge_prompt.txt"
SYSTEM_PROMPT = "아래 지시사항을 정확히 따르고, 반드시 지정된 JSON 형식으로만 응답하세요."

# 오차 허용범위: %류 주장은 %p(절대 차이), 그 외 단위는 상대오차(%)로 통일해서 같은 척도로 비교.
NUMERIC_TOLERANCE = 0.1
# tolerance의 5배(=0.5) 넘게 벌어지면 "명확히 큰 차이"로 보고 LLM 호출 없이 바로 불일치 확정.
CLEAR_GAP_MULTIPLIER = 5

# "조/억/만"은 4자리씩 끊는 큰 자릿수 단위(그룹 단위) — 그룹끼리는 덧셈으로 이어진다.
# "천/백"은 그 그룹 안에서 1~9999 범위의 계수를 만드는 작은 단위 — 계수는 그룹 단위와
# 곱해진다(예: "3천만" = 3천(계수) × 만(그룹단위) = 30,000,000). _find_compound_numbers
# 참고.
_BIG_UNITS = {"조": 1e12, "억": 1e8, "만": 1e4}
_BIG_ORDER = ["조", "억", "만"]  # 인덱스가 클수록 작은 단위
_SMALL_UNITS = {"천": 1e3, "백": 1e2}
_SMALL_ORDER = ["천", "백"]  # 인덱스가 클수록 작은 단위
# 계수 조각(예: "3천", "4백") 바로 뒤에 공백만 사이에 두고 "조/억/만" 글자가 곧바로 오면
# (그 사이에 숫자가 끼어 있지 않으면) 그 계수를 그룹 단위 배수로 곱한다는 신호다. 이
# 글자는 앞에 숫자가 안 붙어 있어 _NUMBER_RE 자체로는 매칭되지 않으므로 별도로 찾는다.
_TRAILING_BIG_UNIT_RE = re.compile(r"\s*(조|억|만)")
_NUMBER_RE = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(조|억|만|천|백)?")
_DIGITS_RE = re.compile(r"\d+")
# 숫자 바로 뒤에 이 마커가 오면 "23만8천명"의 "명"처럼 claim.unit과 일치하는 값이 아닌 한
# 기간/시점 표현("46개월 만에", "3년 만에", "2024년")이지 비교 대상 수치가 아니다.
_DURATION_MARKER_RE = re.compile(r"^(개월|년|일|주|시간|분기|세)")
# "국내 어린이(0~14세) 수가 역대 최저치"처럼 나이대 범위("0~14세")는 그 자체로 하나의
# 묶음이라, 위 _DURATION_MARKER_RE만으로는 범위의 시작 숫자("0")까지는 못 거른다(끝 숫자
# "14"만 "세" 마커로 걸러짐). 그러면 claim.value가 없어 정규식 폴백으로 넘어갔을 때 "0"이
# 엉뚱하게 claim 수치로 뽑혀서, 실제로는 수치 없는 주장인데도 규칙 필터가 확신에 찬
# "불일치"를 내버리는 버그가 있었음(2026-08-05, 판정 골든셋 채점 중 재현). 범위 표현
# 전체를 후보에서 제외해서 막는다.
_AGE_RANGE_RE = re.compile(r"\d+\s*[~\-]\s*\d+\s*세")

# "1.1% 감소했다"처럼 claim_type="증감률" 문장은 숫자 자체엔 부호가 없고 방향은 단어로만
# 표현됨. ComputedResult 쪽(calculator.py compute_change/compute_change_rate)은 감소를
# 음수로 표현하므로, 같은 척도로 비교하려면 이 단어들을 보고 부호를 붙여줘야 함.
#
# "적자전환"/"흑자로 돌아섰다" 같은 문맥 기반 증감 표현은 계속 새로 나올 수 있어 키워드
# 나열로는 근본적으로 다 못 잡는다 — 메인 해법은 claim_extractor_prompt.txt의 few-shot
# 예시로 2단계 LLM이 comparison_operator를 직접 채우게 하는 것이고(claim.comparison_operator가
# _apply_direction에서 최우선으로 쓰임), 여기 목록은 2단계가 못 채웠을 때의 최후 방어선일
# 뿐이라 "적자"/"흑자" 최소한만 추가한다(2026-08-12).
_DECREASE_WORDS = ("감소", "하락", "줄어", "축소", "내림", "내렸", "떨어", "낮아", "줄었", "적자")
_INCREASE_WORDS = ("증가", "상승", "올랐", "늘어", "확대", "늘었", "높아", "올림", "흑자")


class JudgeError(RuntimeError):
    """LLM 판정 응답을 JSON으로 파싱하지 못한 경우."""


def _load_prompt_template(path: Path = PROMPT_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. D가 판정 프롬프트를 먼저 작성해야 합니다.")
    return path.read_text(encoding="utf-8")


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise JudgeError(f"응답에서 JSON 객체를 찾지 못했습니다: {text!r}")
    return json.loads(match.group(0))


# HCX가 gap_type을 한글 대신 영어로 응답하는 경우가 간헐적으로 있음(실측: "population" —
# 판정 골든셋 채점 중 재현, 2026-08-05). 프롬프트 재작성으로 100% 막긴 어려워서, 파싱
# 단계에서 흔한 영단어를 한글로 정규화해 불필요한 JudgeError를 줄인다.
_GAP_TYPE_ALIASES = {
    "population": "모집단",
    "numeric": "수치",
    "number": "수치",
    "value": "수치",
    "period": "기간",
    "time": "기간",
    "exaggeration": "과장표현",
}


def _normalize_gap_type(gap_type: object) -> Optional[str]:
    if gap_type in (None, "None", "none", ""):
        return None
    key = str(gap_type).strip().lower()
    return _GAP_TYPE_ALIASES.get(key, gap_type)


# ---------------------------------------------------------------------------
# 1차 필터 — 코드 규칙 (LLM 호출 없음)
# ---------------------------------------------------------------------------

def _parse_term(
    matches: list[re.Match], i: int, text: str
) -> tuple[float, int, int, Optional[int], int]:
    """matches[i]부터 시작하는 "한 단위 묶음"을 계산한다.

    "조/억/만"은 4자리씩 끊는 큰 자릿수(그룹) 단위, "천/백"은 그 그룹 안에서
    1~9999 범위의 계수를 만드는 작은 단위다. 매칭된 숫자에 "조/억/만"이 곧바로 붙어
    있으면(예: "1억", "23만") 그 자체로 하나의 그룹이라 바로 반환한다. "천/백"이
    붙어 있으면(예: "3천", "4백") 뒤이어 공백 없이 더 작은 "천/백" 단위가 이어지는
    한 계수로 먼저 합산하고(예: "2천4백" -> 2400, 오늘 오전 로직), 그 계수 바로 뒤에
    (중간에 숫자 없이) "조/억/만" 글자가 곧바로 오면 그 단위 값을 계수에 곱한다(예:
    "3천만" -> 3000 × 10000 = 30,000,000). 곱할 단위가 없으면 계수를 그대로 반환한다.

    반환: (값, 시작 인덱스, 끝 인덱스, 그룹 랭크(조=0/억=1/만=2, 없으면 None), 다음에
    읽을 matches 인덱스). 그룹 랭크가 None이면 "조/억/만" 그룹에 속하지 않는 값이라는
    뜻으로, 상위 호출부에서 그룹끼리의 덧셈 체인(예: "1억2천만"의 억+만)이나 단위
    없는 잔여 숫자 병합(예: "338만 4523")을 판단하는 데 쓰인다.
    """
    num_str, scale = matches[i].groups()
    start, end = matches[i].start(), matches[i].end()
    coeff = float(num_str.replace(",", ""))
    i += 1

    if scale in _BIG_UNITS:
        return coeff * _BIG_UNITS[scale], start, end, _BIG_ORDER.index(scale), i

    run_value = coeff * _SMALL_UNITS[scale] if scale in _SMALL_UNITS else coeff
    small_rank = _SMALL_ORDER.index(scale) if scale in _SMALL_UNITS else -1
    while small_rank >= 0 and i < len(matches):
        gap = text[end : matches[i].start()]
        if gap.strip():  # 공백 아닌 문자(쉼표 등)가 끼어 있으면 별개 숫자
            break
        nxt_num, nxt_scale = matches[i].groups()
        nxt_rank = _SMALL_ORDER.index(nxt_scale) if nxt_scale in _SMALL_UNITS else -1
        if nxt_rank <= small_rank:  # 단위가 순서대로 작아지지 않으면 별개 숫자
            break
        run_value += float(nxt_num.replace(",", "")) * _SMALL_UNITS[nxt_scale]
        end = matches[i].end()
        small_rank = nxt_rank
        i += 1

    big_match = _TRAILING_BIG_UNIT_RE.match(text, end)
    if big_match:
        run_value *= _BIG_UNITS[big_match.group(1)]
        end = big_match.end()
        return run_value, start, end, _BIG_ORDER.index(big_match.group(1)), i

    return run_value, start, end, None, i


def _find_compound_numbers(text: str) -> list[tuple[float, int, int]]:
    """text에서 숫자를 찾아 한글 복합 숫자를 하나의 값으로 합산한다.

    두 가지 서로 다른 규칙을 함께 처리한다:
    1) 곱셈형 그룹: "3천만"처럼 계수(3천=3000)에 그룹 단위(만=10000)가 곱해지는
       표현("3천만" -> 30,000,000), 그리고 "1억2천만"처럼 그런 그룹들이 조/억/만
       순서로 이어지며 덧셈으로 합쳐지는 표현("1억"+"2천만" -> 120,000,000). 계수
       계산과 그룹 단위 곱은 _parse_term이 담당한다.
    2) 덧셈형 잔여 숫자: "338만 4523"처럼 그룹 뒤에 그 그룹 단위보다 작은 잔여
       숫자가 단위 없이 그냥 붙는(공백 있든 없든) 표현. 병합은 한 번만 허용한다 —
       병합한 뒤에는 더 이상 이어붙이지 않는다. 그래야 "360, 100"(쉼표+공백으로
       나열된 별개 숫자 두 개)처럼 원래부터 독립적인 숫자까지 합쳐버리는 걸 막을 수
       있다. 두 조각 사이에 쉼표 등 공백 아닌 문자가 끼어 있어도 병합하지 않는다.

    (예전엔 _NUMBER_RE로 한 조각씩만 보고 숫자 하나에 단위 글자 하나만 붙는다고
    가정해서, "3천만"을 만나면 "3천"까지만 매칭하고 그 뒤의 "만"은 앞에 숫자가 없어
    매칭되지 않아 조용히 버려지는 버그가 있었음 — "3천만원"이 3,000으로 읽혀
    ×10,000 배수가 통째로 사라지는 사례로 재현됨, 2026-08-12.)

    반환: [(합산값, 시작 인덱스, 끝 인덱스), ...] — 인덱스는 claim.unit 매칭이나
    연도 제외 판단에 이어서 쓰기 위해 그대로 넘김.
    """
    matches = [m for m in _NUMBER_RE.finditer(text) if m.group(1)]
    results: list[tuple[float, int, int]] = []
    i = 0
    n = len(matches)
    while i < n:
        total, start, end, rank, i = _parse_term(matches, i, text)
        while rank is not None and i < n:
            gap = text[end : matches[i].start()]
            if gap.strip():  # 공백 아닌 문자(쉼표 등)가 끼어 있으면 별개 숫자
                break
            nxt_total, _nxt_start, nxt_end, nxt_rank, nxt_i = _parse_term(matches, i, text)
            if nxt_rank is not None:
                if nxt_rank <= rank:  # 단위가 순서대로 작아지지 않으면 별개 그룹
                    break
                total += nxt_total
                end = nxt_end
                rank = nxt_rank
                i = nxt_i
                continue
            # 단위 없는 잔여 숫자 — 현재 그룹 단위보다 작은 값일 때만 나머지로
            # 병합하고, 병합 여부와 무관하게 더 이상 이어붙이지 않는다.
            if nxt_total < _BIG_UNITS[_BIG_ORDER[rank]]:
                total += nxt_total
                end = nxt_end
                i = nxt_i
            break
        results.append((total, start, end))
    return results


def _extract_claim_number(claim: Claim) -> Optional[float]:
    """claim의 핵심 수치를 반환.

    claim.value(2단계 claim_extractor가 이미 뽑아준 값)가 있으면 그걸 그대로 신뢰해서 씀 —
    2단계가 문장을 직접 이해하고 뽑은 값이 정규식 추측보다 신뢰도가 높음. claim.value가
    없으면(옛날 방식으로 만든 Claim이거나 2단계 추출 실패) 정규식 기반 best-effort로 폴백
    (아래 로직 — claim.unit 앞 숫자 우선, 없으면 문장 마지막 숫자, "2024년" 같은 연도 제외).
    """
    if claim.value is not None:
        return _apply_direction(claim.value, claim)

    chunks = _find_compound_numbers(claim.sentence)
    age_range_spans = [m.span() for m in _AGE_RANGE_RE.finditer(claim.sentence)]
    chunks = [
        (value, start, end)
        for value, start, end in chunks
        if not any(span_start <= start and end <= span_end for span_start, span_end in age_range_spans)
    ]

    if claim.unit:
        unit_pattern = re.compile(rf"^\s*{re.escape(claim.unit)}")
        for value, _start, end in chunks:
            if unit_pattern.match(claim.sentence[end:]):
                return _apply_direction(value, claim)

    candidates: list[float] = []
    for value, start, end in chunks:
        marker_match = _DURATION_MARKER_RE.match(claim.sentence[end : end + 3])
        if marker_match and marker_match.group(1) != claim.unit:
            # "2024년"(연도 표기), "46개월"/"3년 만에"(기간 경과 표현) 둘 다 여기 걸림 —
            # claim이 실제로 그 단위를 주장하는 게 아니라면(예: claim.unit="개월") 비교 대상
            # 수치가 아니므로 제외한다. (예전엔 "년"+4자리 연도 표기만 걸렀는데, "46개월"처럼
            # 개월 단위 기간 표현은 못 걸러서 "46"을 실제 수치로 오인하는 버그가 있었음 —
            # 실제 배치 실행에서 "청년층 취업자 수는 46개월 만에 감소로 전환했다"가
            # 상대오차 1143%라는 무의미한 "불일치" 판정으로 이어지는 걸로 재현됨.)
            continue
        candidates.append(value)

    if not candidates:
        return None

    return _apply_direction(candidates[-1], claim)


def _apply_direction(value: float, claim: Claim) -> float:
    """부호를 붙임 — claim.comparison_operator(2단계가 뽑아준 값)가 있으면 그걸 최우선으로
    쓰고(가장 신뢰할 수 있는 신호), 없으면 claim_type="증감률" 문장에 한해 감소/증가 단어를
    보고 부호를 붙이는 기존 방식으로 폴백."""
    if value < 0:
        return value
    if claim.comparison_operator == "감소":
        return -value
    if claim.comparison_operator is not None:
        return value
    if claim.claim_type != "증감률":
        return value
    if any(w in claim.sentence for w in _DECREASE_WORDS):
        return -value
    return value


def _is_percent(claim: Claim, computed: ComputedResult) -> bool:
    """% 단위 여부는 claim.unit이 명시돼 있으면 그걸 최우선으로 본다. claim.unit이 아예
    없을 때만 claim_type("증감률"/"비율")으로 percent 여부를 추정한다.

    (예전엔 claim.unit="명"처럼 명확히 %가 아닌 단위가 있어도 claim_type=="증감률"이면
    무조건 percent로 취급해서, "23만8천명"(명 단위, 실제 차이 0.13%)을 "317%p 차이"로
    잘못 계산해 오탐하는 버그가 있었음 — 실제 배치 실행에서 재현됨.)
    """
    if claim.unit:
        return claim.unit == "%"
    return computed.unit == "%" or claim.claim_type in ("증감률", "비율")


def _numeric_gap(claim_value: float, claim: Claim, computed: ComputedResult) -> float:
    """항상 절대 차이를 반환 (%류는 %p, 그 외는 원래 단위 그대로)."""
    return abs(claim_value - computed.raw_value)


def _decimal_places(value: float) -> int:
    """value를 문자열로 표현했을 때 소수점 아래 유효자릿수. repr()은 float을 원래
    입력과 왕복 변환 가능한 최단 표현으로 돌려주므로("49.1" -> repr 49.1, 49.10 아님)
    기사에 실제로 적힌 소수점 정밀도를 그대로 복원할 수 있다."""
    text = repr(float(value))
    if "e" in text or "E" in text:
        return 0
    if "." not in text:
        return 0
    frac = text.split(".", 1)[1].rstrip("0")
    return len(frac)


# 실측된 정상 반올림 사례(9,384,000 / 28,041,000 / 3,557,000)는 전부 끝자리 0이 정확히
# 3개(천 단위 반올림)였다. 끝자리 0 개수에 상한을 안 두면 "100,000명"(0이 5개)처럼 흔한
# 어림수에서 허용오차가 ±50,000(상대오차 50%)까지 폭주해서 사실상 판정 기능이 무력화되는
# 버그가 있었음(2026-08-11, 엣지케이스 예측 후 실측으로 확인) — 원인은 끝자리 0이 많을수록
# "반올림 폭이 크다"고 그대로 믿어버린 것. 3자리(천 단위)까지만 반올림 근거로 인정하고
# 그 이상은 "우연히 딱 떨어지는 값"일 수도 있다고 보수적으로 취급해서 상한을 둔다.
_MAX_TRAILING_ZEROS_FOR_TOLERANCE = 3


def _trailing_zero_count(value: float) -> int:
    """정수부 끝에 붙은 0 개수(최대 _MAX_TRAILING_ZEROS_FOR_TOLERANCE까지만). "9384000"처럼
    큰 수 끝이 0으로 딱 떨어지면 반올림된 값(예: 천 단위 반올림)일 가능성이 높다고 본다."""
    int_text = str(abs(int(round(value))))
    stripped = int_text.rstrip("0")
    raw_count = len(int_text) - len(stripped) if int_text != "0" else 0
    return min(raw_count, _MAX_TRAILING_ZEROS_FOR_TOLERANCE)


def _implied_tolerance(claim_value: float, is_percent: bool) -> float:
    """claim_value가 기사에 실제로 적힌 유효자릿수를 보고 "이 정도 반올림 오차는
    허용"이라는 임계값을 동적으로 추정한다.

    (2026-08-05, 판정 골든셋 채점 중 실측 근거로 도입 — 고정된 NUMERIC_TOLERANCE 하나로는
    "91,155건 vs 91,151건"(4건 차이, 사람은 불일치로 봄)과 "9,384,000명 vs 9,384,325명"
    (325명 차이, 사람은 반올림으로 보고 일치로 봄)을 구분할 수 없었다. 실제로는 절대
    오차 크기가 아니라 "기사 숫자가 끝자리 0으로 반올림된 값처럼 보이는가"가 기준이었음
    — 91,155는 끝자리가 0이 아니라 정확한 값으로, 9,384,000은 끝이 000이라 반올림된
    값으로 사람이 판단한 것으로 재현됨.)

    - %류(퍼센트/퍼센트포인트): 소수점 자릿수만 본다. "10%"처럼 정수로만 적혀도 "±5"
      같은 느슨한 값이 아니라 NUMERIC_TOLERANCE(0.1%p)를 그대로 쓴다 — 퍼센트는 정수로
      적어도 통상 정밀한 값이지 "10의 자리로 반올림"한 게 아니기 때문.
    - 그 외(명/건 등 절대량): 정수부 끝자리 0 개수로 반올림 단위를 추정. 끝자리가 0이
      아니면(예: 91155) 정확한 값으로 보고 허용오차를 최소 단위(0.5)로 좁힌다.
    """
    if is_percent:
        decimals = _decimal_places(claim_value)
        return 0.5 * (10 ** -decimals) if decimals > 0 else NUMERIC_TOLERANCE
    trailing_zeros = _trailing_zero_count(claim_value)
    return 0.5 * (10 ** trailing_zeros) if trailing_zeros > 0 else 0.5


_RATE_CALC_TYPES = ("증감", "증감률")
_LEVEL_CALC_TYPES = ("단순조회", "합계", "규모")


def _claim_computed_type_mismatch(claim: Claim, computed: ComputedResult) -> bool:
    """claim이 주장하는 종류(claim_type)와 computed가 실제로 계산한 종류(calc_type)가
    다른 값(수준값 vs 변화율)을 가리키면 True.

    (실제 배치 실행에서 재현된 버그: claim_type="규모"인 "실업률이 6%에 육박"을
    computed.calc_type="증감률"인 계산값(실업률 자체의 전년비 변화율 3.7%)과 그대로 숫자
    비교해서 "6 vs 3.7%p 차이, 불일치"로 확정한 사례가 있었음 — 최종 판정은 우연히 맞았지만
    비교 근거 자체가 잘못된 값이었음. 종류가 다르면 숫자가 우연히 비슷하든 다르든 규칙으로
    확정하지 않고 LLM에 위임해서 문장을 직접 보고 판단하게 한다.)
    """
    if claim.claim_type == "규모" and computed.calc_type in _RATE_CALC_TYPES:
        return True
    if claim.claim_type == "증감률" and computed.calc_type in _LEVEL_CALC_TYPES:
        return True
    return False


def _period_granularity(period: Optional[str]) -> Optional[str]:
    """period 문자열이 "월" 단위인지 "년" 단위인지 best-effort로 추정.

    (참고: Day2 파이프라인 연결 테스트에서 실제로 발견된 문제 — 기사는 "전년 동월 대비"
    처럼 월 단위로 주장하는데, KOSIS 조회가 연간 평균 기준(prdSe=Y)으로만 이뤄져서
    비교 기준 자체가 다른 채로 판정될 뻔한 케이스가 있었음. 이 함수는 그 갭을 규칙
    단계에서 감지해서 LLM에 위임하기 위한 것.)
    """
    if not period:
        return None
    if "월" in period and "개월" not in period:
        return "월"
    for run in _DIGITS_RE.findall(period):
        if len(run) == 6:
            return "월"
    for run in _DIGITS_RE.findall(period):
        if len(run) == 4:
            return "년"
    return None


_YEAR_RE = re.compile(r"(?:19|20)\d{2}")


def _extract_years(period: Optional[str]) -> list[int]:
    if not period:
        return []
    return [int(m.group()) for m in _YEAR_RE.finditer(period)]


def _period_year_mismatch(claim_period: Optional[str], computed_period: Optional[str], *, max_gap: int = 3) -> bool:
    """claim이 가리키는 시점과 실제로 KOSIS에 조회된 시점(computed)의 연도가 서로 너무
    멀리 떨어져 있으면 True — 표가 다루는 연도 범위 밖의 과거/미래 시점을 claim이
    가리키는데, 4단계 슬롯필링이 그 시점을 못 알아듣고 조용히 다른(보통 최근) 연도로
    대체해서 조회해버린 경우를 잡기 위함이다(2026-08-12 실측: "1955~1960년 인구 증가율"
    claim이 is_valid_period()가 범위 표현("1955~1960년")을 못 알아듣고 걸러버려서
    "2024년" 데이터와 그대로 비교되어 잘못된 "불일치"가 나온 사례로 확인).

    claim.period/computed.period 양쪽에서 연도(4자리, 1900~2099년대)를 전부 뽑아
    가장 가까운 한 쌍의 차이가 max_gap을 넘으면 불일치로 본다. 둘 중 하나라도 연도를
    못 뽑으면(예: "작년"처럼 이미 절대 연도로 정규화 안 된 표현) 판단 근거가 없으므로
    False(불일치 아님 취급 — 상위 로직이 계속 진행)를 반환한다."""
    claim_years = _extract_years(claim_period)
    computed_years = _extract_years(computed_period)
    if not claim_years or not computed_years:
        return False
    return min(abs(cy - py) for cy in claim_years for py in computed_years) > max_gap


def _format_period_label(period: Optional[str], prd_se: Optional[str] = None) -> str:
    """period 문자열을 LLM 프롬프트에 넣기 전에 사람이 헷갈리지 않게 포맷한다.

    prd_se(표 주기)를 안 넘기면(기본값 None) 원래 문자열 그대로 반환한다 — 6자리 숫자가
    월(YYYYMM)인지 분기(YYYY+분기코드)인지 원래 구분할 방법이 없어서, 예전엔 LLM이 그냥
    "2024년 1월부터 2월까지"처럼 임의로(대부분 틀리게) 월로 해석해버리는 문제가 있었다
    (2026-08-12, 분기 표 period="202402"가 실제로는 "2024년 2분기"인데 판정 설명에
    "1~2월"로 잘못 서술된 사례로 실측 확인). prd_se를 넘긴 호출부만 이 분기를 탄다."""
    if not period:
        return period or "명시 안 됨"
    if "~" in period:
        return "~".join(_format_period_label(p, prd_se) for p in period.split("~"))
    if prd_se == "Q" and re.fullmatch(r"\d{6}", period):
        return f"{period[:4]}년 {int(period[4:])}분기"
    if prd_se == "M" and re.fullmatch(r"\d{6}", period):
        return f"{period[:4]}년 {int(period[4:])}월"
    return period


def _rule_based_verdict(claim: Claim, computed: ComputedResult) -> Optional[Verdict]:
    """규칙만으로 명확히 판단되면 Verdict를 반환하고, 애매하면 None(LLM 위임)을 반환."""
    claim_value = _extract_claim_number(claim)
    if claim_value is None:
        return None  # 수치 추출 실패 → 규칙으로 못 정함, LLM에 원문 문장 그대로 위임

    if _claim_computed_type_mismatch(claim, computed):
        return None  # 수준값 vs 변화율처럼 종류가 다른 값 비교 → 규칙으로 확정하지 않고 LLM 위임

    is_pct = _is_percent(claim, computed)
    gap = _numeric_gap(claim_value, claim, computed)
    tolerance = _implied_tolerance(claim_value, is_pct)
    clear_threshold = tolerance * CLEAR_GAP_MULTIPLIER
    gap_unit = "%p" if is_pct else (claim.unit or computed.unit or "")

    # 단위가 서로 다르면(예: 톤 vs 천달러) gap 자체가 서로 다른 척도를 뺀 무의미한 숫자라,
    # 격차가 아무리 커도 "명확한 불일치"로 규칙 확정하면 안 된다. 예전엔 "일치" 분기(아래)
    # 에만 이 체크가 있고 "불일치" 분기엔 없어서, 단위가 안 맞는 비교도 그냥 숫자 차이가
    # 크다는 이유만으로 확정 불일치를 내버리는 비대칭 버그가 있었다(2026-08-12, 실제
    # 배치에서 "미국산 원유 수입량 2151만톤" claim이 "683,609,488천달러" 통계값과 그대로
    # 비교되어 잘못된 "불일치"로 확정된 사례로 실측 확인).
    unit_mismatch = (
        bool(claim.unit) and bool(computed.unit) and claim.unit != computed.unit
        and not is_pct
    )
    year_mismatch = _period_year_mismatch(claim.period, computed.period)

    if gap > clear_threshold and not unit_mismatch and not year_mismatch:
        return Verdict(
            verdict="불일치",
            gap_type="수치",
            reason=(
                f"기사 수치({claim_value})와 통계 계산값({computed.raw_value}{computed.unit}) "
                f"차이가 {gap:.2f}{gap_unit}로 허용 오차(반올림 단위 추정 ±{tolerance:g}{gap_unit})를 "
                "크게 초과함 (규칙 기반 판정, LLM 미호출)."
            ),
        )

    if gap <= tolerance and not unit_mismatch and not year_mismatch:
        # 엣지케이스 방어: computed.period가 아예 없으면(5단계 필드 누락 등) "시점 불일치 없음"
        # 으로 잘못 확정하지 말고 규칙 필터를 포기하고 LLM에 위임한다. (실제로 이 가드가 없으면
        # computed.period=""일 때도 "일치"로 확정 판정해버리는 버그가 있었음 — 검증 완료.)
        if not computed.period:
            return None
        claim_gran = _period_granularity(claim.period)
        computed_gran = _period_granularity(computed.period)
        period_mismatch = claim_gran is not None and computed_gran is not None and claim_gran != computed_gran
        if not period_mismatch:
            return Verdict(
                verdict="일치",
                gap_type=None,
                reason=(
                    f"기사 수치({claim_value})와 통계 계산값({computed.raw_value}{computed.unit}) "
                    f"차이가 허용 오차(반올림 단위 추정 ±{tolerance:g}{gap_unit}) 이내이고 시점·단위 "
                    "불일치도 없음 (규칙 기반 판정, LLM 미호출)."
                ),
            )

    return None  # 애매한 경계(수치는 비슷한데 시점/단위가 다르거나 경계 근처) → LLM 위임


# ---------------------------------------------------------------------------
# 애매 경계 — HCX-003 판정
# ---------------------------------------------------------------------------

def _judge_with_llm(
    claim: Claim, computed: ComputedResult, *, model: str = MODEL_SIMPLE, prd_se: Optional[str] = None
) -> Verdict:
    template = _load_prompt_template()
    prompt = (
        template.replace("{claim_sentence}", claim.sentence)
        .replace("{claim_type}", claim.claim_type)
        .replace("{claim_period}", claim.period or "명시 안 됨")
        .replace("{claim_unit}", claim.unit or "명시 안 됨")
        .replace("{claim_population}", claim.population or "명시 안 됨")
        .replace("{computed_calc_type}", computed.calc_type)
        .replace("{computed_value}", str(computed.raw_value))
        .replace("{computed_unit}", computed.unit)
        .replace("{computed_period}", _format_period_label(computed.period, prd_se))
    )

    # 2026-08-11 재현성 문제 대응: temperature 기본값(hcx_client의 0.2)을 두면 같은
    # claim+계산값을 넣어도 판정이 흔들릴 수 있어 0.0으로 고정 (seed는 hcx_client
    # 기본값 42가 자동 적용됨).
    reply = call_hcx(model=model, system_prompt=SYSTEM_PROMPT, user_content=prompt, temperature=0.0)

    try:
        parsed = _extract_json_object(reply)
        verdict = str(parsed["verdict"])
        if verdict not in ("일치", "불일치", "판단불가"):
            raise ValueError(f"알 수 없는 verdict 값: {verdict!r}")
        gap_type = _normalize_gap_type(parsed.get("gap_type"))
        if gap_type is not None and gap_type not in ("수치", "기간", "모집단", "과장표현"):
            raise ValueError(f"알 수 없는 gap_type 값: {gap_type!r}")
        return Verdict(verdict=verdict, gap_type=gap_type, reason=str(parsed.get("reason", "")))
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise JudgeError(f"응답 파싱 실패: {reply!r}") from e


def judge(
    claim: Claim, computed: ComputedResult, *, model: str = MODEL_SIMPLE, prd_se: Optional[str] = None
) -> Verdict:
    """7단계 메인 진입점 (단일 Claim vs 단일 ComputedResult).

    1차로 코드 규칙(_rule_based_verdict)을 먼저 시도하고, 명확히 정해지지 않는
    애매한 경우에만 HCX-003을 호출합니다.

    prd_se(표 주기, "Y"/"M"/"Q"): LLM에게 넘기는 판정 설명 프롬프트에서 computed.period를
    사람이 헷갈리지 않게(월/분기 구분) 포맷하는 데만 쓰인다 — 안 넘기면(기본값 None) 기존과
    동일하게 원본 문자열을 그대로 노출한다(하위 호환).
    """
    rule_verdict = _rule_based_verdict(claim, computed)
    if rule_verdict is not None:
        return rule_verdict
    return _judge_with_llm(claim, computed, model=model, prd_se=prd_se)


# ---------------------------------------------------------------------------
# 복잡 케이스 — HCX-007 승격
# ---------------------------------------------------------------------------

def needs_hybrid_reasoning(claims: list[Claim], computed_results: list[ComputedResult]) -> bool:
    """단일 Claim-ComputedResult 쌍으로 못 다루는 복합 케이스인지 판단.

    승격 조건 (둘 중 하나라도 해당하면 HCX-007로 올림):
      1. Claim이 여러 개 얽혀서 함께 봐야 함 (예: claim_type="비교"가 다른 Claim을 가리켜
         "A가 B의 절반" 같은 상호 비교 주장을 함 — 개별로는 판정 불가)
      2. 하나의 주장을 검증하는 데 ComputedResult가 여러 개 필요함 (여러 표를 종합해야
         계산이 완성되는 경우, 예: 두 통계표 비율 재계산·복수 시점 추세 종합)
    """
    if len(claims) > 1 and any(c.claim_type == "비교" for c in claims):
        return True
    if len(computed_results) > 1:
        return True
    return False


def judge_complex(
    claims: list[Claim], computed_results: list[ComputedResult], *, model: str = MODEL_COMPLEX
) -> Verdict:
    """복합 케이스 판정: 여러 Claim/ComputedResult를 한 번에 HCX-007에 넘겨 종합 추론시킴.

    judge_prompt.txt의 단일 Claim용 템플릿 대신, 목록을 그대로 나열한 프롬프트를 씁니다
    (단일 쌍 few-shot으로는 "여러 표 종합"까지 예시를 늘리기엔 프롬프트가 지나치게 커져서
    별도 구성 — 필요시 team 상의 후 전용 few-shot 프롬프트 파일로 분리 가능).
    """
    claims_block = "\n".join(
        f"- \"{c.sentence}\" (claim_type={c.claim_type}, period={c.period}, "
        f"unit={c.unit}, population={c.population})"
        for c in claims
    )
    computed_block = "\n".join(
        f"- {r.calc_type} {r.raw_value}{r.unit} ({r.period} 기준)" for r in computed_results
    )
    prompt = (
        "아래는 한 기사에서 나온 여러 개의 수치 주장과, 그걸 검증하기 위해 조회한 여러 개의 "
        "국가 공식 통계(KOSIS) 계산값입니다. 여러 주장/통계를 함께 종합적으로 추론해서 "
        "전체적으로 '일치/불일치/판단불가'를 하나로 판정하세요.\n\n"
        f"기사 주장 목록:\n{claims_block}\n\n"
        f"통계 계산값 목록:\n{computed_block}\n\n"
        '출력 형식 (JSON만 출력, 다른 텍스트 금지):\n'
        '{"verdict": "일치|불일치|판단불가", "gap_type": "수치|기간|모집단|과장표현" 또는 null, '
        '"reason": "여러 주장/통계를 종합한 판단 근거"}'
    )

    reply = call_hcx(
        model=model, system_prompt=SYSTEM_PROMPT, user_content=prompt, max_tokens=1536, temperature=0.0
    )

    try:
        parsed = _extract_json_object(reply)
        verdict = str(parsed["verdict"])
        if verdict not in ("일치", "불일치", "판단불가"):
            raise ValueError(f"알 수 없는 verdict 값: {verdict!r}")
        gap_type = _normalize_gap_type(parsed.get("gap_type"))
        return Verdict(verdict=verdict, gap_type=gap_type, reason=str(parsed.get("reason", "")))
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise JudgeError(f"응답 파싱 실패: {reply!r}") from e


def judge_all(claims: list[Claim], computed_results: dict[int, ComputedResult]) -> list[Verdict]:
    """claims 인덱스 -> ComputedResult 매핑을 받아, 복합 케이스는 HCX-007로 한 번에,
    단순 케이스는 claim별로 judge()를 호출해서 Verdict 리스트를 돌려줌.

    (오케스트레이터가 여러 Claim을 모아서 7단계를 한 번에 돌릴 때 쓰라고 만든 편의 함수.
    복합 승격 시 관련된 모든 claim에 동일한 Verdict를 매깁니다.)
    """
    matched = [(i, c, computed_results[i]) for i, c in enumerate(claims) if i in computed_results]
    if not matched:
        return []

    if needs_hybrid_reasoning([c for _, c, _ in matched], [r for _, _, r in matched]):
        verdict = judge_complex([c for _, c, _ in matched], [r for _, _, r in matched])
        return [verdict for _ in matched]

    return [judge(c, r) for _, c, r in matched]


if __name__ == "__main__":
    #   python -m agent.verdict.judge

    # 케이스 1 — 규칙만으로 "일치" (LLM 호출 없음)
    claim1 = Claim(
        sentence="작년 국민 1인당 쌀 소비량은 1년 전보다 1.1% 감소했다.",
        claim_type="증감률",
        period="2024년",
        unit="%",
        population="국민 1인당",
    )
    computed1 = ComputedResult(calc_type="증감률", raw_value=-1.15, unit="%", period="2023~2024")
    result1 = judge(claim1, computed1)
    print(f"[케이스1 - 일치 예상] {result1}")
    assert result1.verdict == "일치"

    # 케이스 2 — 규칙만으로 "불일치" (기사 10% vs 통계 7.2%, 예시 그대로, LLM 호출 없음)
    claim2 = Claim(
        sentence="청년 실업률이 지난달 10%까지 치솟았다.",
        claim_type="규모",
        period="2025년 1월",
        unit="%",
        population="청년",
    )
    computed2 = ComputedResult(calc_type="합계", raw_value=7.2, unit="%", period="2025")
    result2 = judge(claim2, computed2)
    print(f"[케이스2 - 불일치 예상] {result2}")
    assert result2.verdict == "불일치" and result2.gap_type == "수치"

    # 케이스 3 — 수치는 비슷한데 기간 단위(월 vs 연)가 달라 애매 경계 → HCX-003 호출
    claim3 = Claim(
        sentence="지난달 소비자물가가 전년 동월 대비 2.2% 오른 것으로 나타났다.",
        claim_type="증감률",
        period="2024년 12월",
        unit="%",
        population="소비자물가",
    )
    computed3 = ComputedResult(calc_type="증감률", raw_value=2.3, unit="%", period="2023~2024")
    print("[케이스3 - 애매 경계, HCX-003 호출 시도]")
    try:
        result3 = judge(claim3, computed3)
        print(f"  결과: {result3}")
    except Exception as e:
        print(f"  HCX-003 호출 실패 ({type(e).__name__}: {e}) — API 키/네트워크 확인 필요")

    # 케이스 4 — 복합 케이스(Claim 2개 + ComputedResult 2개) → HCX-007 승격 조건 확인
    claim4a = Claim(
        sentence="올해 최저임금은 시간당 10,030원이다.", claim_type="규모", period="2025년", unit="원", population="최저임금"
    )
    claim4b = Claim(
        sentence="이는 작년보다 1.7% 오른 수치다.", claim_type="비교", period="2025년", unit="%", population="최저임금"
    )
    computed4a = ComputedResult(calc_type="합계", raw_value=10030, unit="원", period="2025")
    computed4b = ComputedResult(calc_type="증감률", raw_value=1.7, unit="%", period="2024~2025")
    escalate = needs_hybrid_reasoning([claim4a, claim4b], [computed4a, computed4b])
    print(f"[케이스4 - 승격 조건 확인] needs_hybrid_reasoning={escalate}")
    assert escalate is True
    print("  → 조건 충족 확인됨 (claim_type='비교' 포함 + Claim 2개). 실제 HCX-007 호출은 judge_complex()로 별도 실행.")

    # 케이스 5 — 엣지케이스 방어: computed.period 누락(5단계 필드 누락 등) 시 규칙으로
    # 잘못 확정하지 않고 LLM에 위임하는지 확인 (수정 전엔 "일치"로 잘못 확정되던 버그였음)
    claim5 = Claim(
        sentence="작년 국민 1인당 쌀 소비량은 1년 전보다 1.1% 감소했다.",
        claim_type="증감률", period="2024년", unit="%", population="국민 1인당",
    )
    computed5 = ComputedResult(calc_type="증감률", raw_value=-1.15, unit="%", period="")
    rb5 = _rule_based_verdict(claim5, computed5)
    print(f"[케이스5 - period 누락 방어] 규칙 기반 1차 필터 결과: {rb5}")
    assert rb5 is None, "period 누락인데도 규칙으로 확정해버림 (엣지케이스 방어 회귀)"
    print("  → 통과: 시점 정보 없음을 '불일치 없음'으로 오인하지 않고 LLM에 위임함.")

    # 케이스 6 — 회귀 방지: "23만8천"처럼 조/억/만/천이 이어지는 한글 복합 숫자를
    # 제대로 합산하는지 + claim.unit="명"인데 claim_type="증감률"이라고 percent로
    # 오판하지 않는지 (배치 실행 중 실제로 재현된 두 버그의 조합 케이스)
    claim6 = Claim(
        sentence="작년 출생아 수는 23만8천명으로 전년보다 늘어난 것으로 나타났다.",
        claim_type="증감률", period="2024년", unit="명", population="출생아 수",
    )
    computed6 = ComputedResult(calc_type="단순조회", raw_value=238317.0, unit="명 건", period="2024")
    claim6_value = _extract_claim_number(claim6)
    print(f"[케이스6 - 복합 숫자 파싱] '23만8천명' 추출값: {claim6_value}")
    assert claim6_value == 238000.0, f"복합 숫자 파싱 회귀 (기대 238000.0, 실제 {claim6_value})"
    rb6 = _rule_based_verdict(claim6, computed6)
    print(f"[케이스6 - is_percent 오판 방어] 규칙 기반 1차 필터 결과: {rb6}")
    assert rb6 is None, "명 단위인데도 %p로 오판해서 규칙으로 확정해버림 (회귀)"
    print("  → 통과: 238000(추출값)이 정확히 합산됐고, '명' 단위를 %로 오판하지 않아 LLM에 위임함.")

    # 케이스 7 — 회귀 방지: claim.value가 없고 문장에 실제 claim 수치가 없는데
    # "(0~14세)" 같은 나이대 범위 표현만 있을 때, 범위 숫자를 엉뚱하게 claim 수치로
    # 오인해서 규칙 필터가 확신에 찬 오답을 내지 않는지 (판정 골든셋 채점 중 실제로
    # 재현된 버그 — "0"이 claim 수치로 뽑혀서 큰 값과 비교돼 거짓 "불일치"가 나왔었음)
    claim7 = Claim(
        sentence="지난달 말 기준 국내 어린이(0~14세) 수가 역대 최저치를 기록했다는 소식이 전해졌다.",
        claim_type="규모", period="2025년", unit=None, population="어린이",
    )
    claim7_value = _extract_claim_number(claim7)
    print(f"[케이스7 - 나이대 범위 오인 방어] 추출값: {claim7_value}")
    assert claim7_value is None, f"나이대 범위(0~14세)를 claim 수치로 잘못 뽑음 (회귀, 실제 {claim7_value})"
    print("  → 통과: 수치 없는 주장을 '0'/'14'로 오인하지 않고 None(LLM 위임) 반환.")

    # 케이스 8 — 회귀 방지: HCX가 gap_type을 영어("population")로 응답해도 한글로
    # 정규화되는지 (판정 골든셋 채점 중 실제로 재현된 JudgeError 원인)
    normalized = _normalize_gap_type("population")
    print(f"[케이스8 - gap_type 영어 응답 정규화] 'population' -> {normalized!r}")
    assert normalized == "모집단", f"gap_type 영어 정규화 회귀 (기대 '모집단', 실제 {normalized!r})"
    print("  → 통과: 영어 응답도 허용된 한글 gap_type으로 정규화됨.")

    # 케이스 9 — 회귀 방지: "100,000명"처럼 끝자리 0이 많은 어림수에서 허용오차가
    # 폭주하지 않는지 (엣지케이스 예측 후 실측으로 확인된 버그 — 상한 없이 끝자리 0
    # 개수를 그대로 반올림 단위로 쓰면 "10만명" 주장의 허용오차가 ±50,000(상대오차 50%)
    # 까지 벌어져 사실상 판정이 무력화됐었음. 2026-08-11)
    tol_100k = _implied_tolerance(100000.0, is_percent=False)
    print(f"[케이스9 - 어림수 허용오차 폭주 방어] 100,000 -> 허용오차 ±{tol_100k}")
    assert tol_100k <= 500, f"어림수 허용오차 폭주 회귀 (기대 500 이하, 실제 {tol_100k})"
    print("  → 통과: 끝자리 0이 많아도(5개) 허용오차가 천 단위 상한(500) 안으로 제한됨.")

    # 케이스 10 — 회귀 방지: "338만 4523명"/"338만4523명"처럼 만 단위 뒤에 단위 없는
    # 1만 미만 잔여 숫자가 공백이 있든 없든 붙는 한글 복합 숫자를 하나의 값으로 합산하는지
    # (수정 전엔 둘 다 "338만"(=3380000)만 인식하고 뒤의 "4523"을 별개 숫자로 떼어내서
    # 3384523이 아니라 [3380000, 4523] 두 개로 쪼개지는 버그였음. 2026-08-12 수정)
    value_with_space = _find_compound_numbers("338만 4523명")[0][0]
    print(f"[케이스10 - 잔여 숫자 병합(공백 있음)] '338만 4523명' -> {value_with_space}")
    assert value_with_space == 3384523.0, (
        f"공백 있는 잔여 숫자 병합 회귀 (기대 3384523.0, 실제 {value_with_space})"
    )
    value_no_space = _find_compound_numbers("338만4523명")[0][0]
    print(f"[케이스10 - 잔여 숫자 병합(공백 없음)] '338만4523명' -> {value_no_space}")
    assert value_no_space == 3384523.0, (
        f"공백 없는 잔여 숫자 병합 회귀 (기대 3384523.0, 실제 {value_no_space})"
    )
    print("  → 통과: 공백 유무와 무관하게 만 단위 뒤 잔여 숫자가 하나의 값으로 합산됨.")

    # 케이스 11 — 회귀 방지: 쉼표로 구분된 표현은 케이스10의 잔여 숫자 병합과 헷갈리지
    # 않고 그대로 유지되는지 ("360, 100"은 쉼표+공백으로 나열된 별개 숫자 두 개라
    # 합치면 존재하지 않는 숫자를 만들어내므로 절대 합치면 안 됨)
    thousand_sep = _find_compound_numbers("360,000명")
    assert len(thousand_sep) == 1 and thousand_sep[0][0] == 360000.0, (
        f"쉼표 천단위 구분 회귀 (실제 {thousand_sep})"
    )
    multi_sep = _find_compound_numbers("365,234,215명")
    assert len(multi_sep) == 1 and multi_sep[0][0] == 365234215.0, (
        f"쉼표 다단 구분 회귀 (실제 {multi_sep})"
    )
    listed_numbers = _find_compound_numbers("360, 100명")
    listed_values = [v for v, _, _ in listed_numbers]
    assert listed_values == [360.0, 100.0], (
        f"쉼표+공백 분리 유지 회귀 (기대 [360.0, 100.0], 실제 {listed_values})"
    )
    print(
        "[케이스11 - 쉼표 표현 안 깨짐] "
        f"'360,000명'->{thousand_sep[0][0]}, '365,234,215명'->{multi_sep[0][0]}, "
        f"'360, 100명'->{listed_values} (모두 유지)"
    )
    print("  → 통과: 쉼표 구분 숫자와 쉼표+공백 나열 숫자 모두 잔여 숫자 병합과 헷갈리지 않음.")

    # 케이스 12 — 회귀 방지: 기존에 이미 정상 동작하던 "23만8천명"(단위 있는 조각끼리만
    # 이어지는 복합 숫자)이 이번 수정으로 깨지지 않는지
    existing_compound = _find_compound_numbers("23만8천명")[0][0]
    print(f"[케이스12 - 기존 복합 숫자 파싱 안 깨짐] '23만8천명' -> {existing_compound}")
    assert existing_compound == 238000.0, (
        f"기존 복합 숫자 파싱 회귀 (기대 238000.0, 실제 {existing_compound})"
    )
    print("  → 통과: 기존에 정상 동작하던 케이스는 그대로 유지됨.")

    # 케이스 13 — (여유 작업) "백" 단위 지원: "2천4백명"처럼 천/백 단위가 이어지는
    # 복합 숫자를 정상 합산하는지. ("2천4백만"처럼 만 단위가 앞의 "2천4백" 전체에
    # 곱해지는 trailing multiplier 표현은 그 뒤 케이스14에서 곱셈형 결합 단위 파싱
    # 재설계로 함께 지원됨, 2026-08-12.)
    hundred_case = _find_compound_numbers("2천4백명")[0][0]
    print(f"[케이스13 - '백' 단위 지원] '2천4백명' -> {hundred_case}")
    assert hundred_case == 2400.0, f"'백' 단위 지원 회귀 (기대 2400.0, 실제 {hundred_case})"
    print("  → 통과: 천/백 단위가 이어지는 복합 숫자가 정상 합산됨.")

    # 케이스 14 — 곱셈형 결합 단위 버그 수정: "3천만"처럼 "천/백"으로 만든 계수
    # 뒤에 "만/억/조"가 곧바로 붙으면 그 단위 배수를 곱해야 하는데, 예전엔 "만" 앞에
    # 숫자가 안 붙어 있어 _NUMBER_RE에 아예 매칭되지 않고 조용히 버려져서 ×10,000
    # 배수가 통째로 사라지는 버그가 있었음(예: "3천만원"이 3,000으로 읽힘).
    # 오늘 오전에 고친 덧셈형 잔여 숫자 병합("338만 4523")과는 다른 구조의 문제라
    # 별도 파싱(_parse_term)으로 재설계함, 2026-08-12.
    mult_cases = {
        "3천만원": 30000000.0,
        "5천만 명": 50000000.0,
        "1억2천만원": 120000000.0,
        "2백만원": 2000000.0,
        "4천억원": 400000000000.0,
    }
    for text, expected in mult_cases.items():
        actual = _find_compound_numbers(text)[0][0]
        print(f"[케이스14 - 곱셈형 결합 단위] '{text}' -> {actual}")
        assert actual == expected, (
            f"곱셈형 결합 단위 파싱 실패 ('{text}' 기대 {expected}, 실제 {actual})"
        )
    print("  → 통과: 계수(천/백) 뒤에 곧바로 붙는 만/억/조 배수가 정확히 곱해짐,")
    print("     그리고 억+만처럼 그룹끼리 이어지는 표현도 그룹별로 곱한 뒤 덧셈으로 합산됨.")

    # 케이스 15 — 곱셈형 그룹과 덧셈형 잔여 숫자가 함께 나오는 경우: "1억2천만 3500명"
    # 처럼 곱셈형 그룹(1억2천만=120,000,000) 뒤에 단위 없는 잔여 숫자(3500)가 더
    # 붙는 실제 기사 표현도 두 규칙이 서로 안 깨지고 합산되는지 확인.
    combo_value = _find_compound_numbers("1억2천만 3500명")[0][0]
    print(f"[케이스15 - 곱셈형 그룹 + 덧셈형 잔여 숫자] '1억2천만 3500명' -> {combo_value}")
    assert combo_value == 120003500.0, (
        f"곱셈형+덧셈형 조합 파싱 실패 (기대 120003500.0, 실제 {combo_value})"
    )
    print("  → 통과: 곱셈형 그룹 합산 뒤에 잔여 숫자가 한 번 더 정확히 병합됨.")

    # 케이스 16 — 회귀 방지: 케이스10/11(덧셈형 잔여 숫자 병합, 쉼표 분리 유지)이
    # 이번 곱셈형 재설계로 깨지지 않는지 재확인
    regress_with_space = _find_compound_numbers("338만 4523명")[0][0]
    regress_no_space = _find_compound_numbers("338만4523명")[0][0]
    regress_listed = [v for v, _, _ in _find_compound_numbers("360, 100명")]
    regress_simple = _find_compound_numbers("1200만원")[0][0]
    print(
        "[케이스16 - 곱셈형 재설계 이후 회귀 확인] "
        f"'338만 4523명'->{regress_with_space}, '338만4523명'->{regress_no_space}, "
        f"'360, 100명'->{regress_listed}, '1200만원'->{regress_simple}"
    )
    assert regress_with_space == 3384523.0, "338만 4523 병합 회귀"
    assert regress_no_space == 3384523.0, "338만4523 병합 회귀"
    assert regress_listed == [360.0, 100.0], "쉼표+공백 분리 유지 회귀"
    assert regress_simple == 12000000.0, "단순 만 단위 파싱 회귀"
    print("  → 통과: 덧셈형 잔여 숫자 병합/쉼표 분리 유지/단순 만 단위 파싱 모두 그대로 유지됨.")

    # 케이스 17 — "적자전환"처럼 문맥 기반 증감 표현은 키워드 나열로 다 못 잡으므로,
    # 메인 해법은 claim_extractor_prompt.txt의 few-shot 예시로 2단계 LLM이
    # comparison_operator를 직접 채우게 하는 것(2026-08-12). 여기서는 2단계가
    # comparison_operator="감소"를 정확히 채웠다고 가정했을 때(value는 설계대로
    # 부호 없이 320억원 크기 그대로), _apply_direction이 그 신호를 최우선으로 써서
    # 음수로 뒤집는지 확인한다 — LLM 호출 없이 코드 경로만 검증 가능한 부분.
    claim17 = Claim(
        sentence="영업이익이 -320억원 적자전환했다.",
        claim_type="규모", period="올해 3분기", unit="원", population="영업이익",
        value=32000000000.0, comparison_operator="감소",
    )
    claim17_value = _extract_claim_number(claim17)
    print(f"[케이스17 - '적자전환' comparison_operator 부호 반영] 추출값: {claim17_value}")
    assert claim17_value == -32000000000.0, (
        f"comparison_operator='감소' 부호 반영 회귀 (기대 -32000000000.0, 실제 {claim17_value})"
    )
    print("  → 통과: comparison_operator='감소'가 채워져 있으면 '적자전환' 같은 문맥 표현도")
    print("     키워드 목록 없이 음수로 정확히 뒤집힘 (2단계 few-shot이 메인 해법, 아래 참고).")

    # 케이스 18 — 회귀 방지: 단위(unit) 검증 비대칭 버그 수정 확인. 실제 배치에서 재현된
    # "미국산 원유 수입량 2151만톤" claim(unit="톤")이 "683,609,488천달러"(unit="천달러")
    # 통계값과 비교되어, 단위가 완전히 다른데도 그냥 숫자 차이가 크다는 이유만으로 확정
    # "불일치"가 나던 버그 — 톤과 달러는 애초에 비교 불가능하므로 규칙으로 확정하지 않고
    # LLM에 위임(None)해야 한다.
    claim18 = Claim(
        sentence="지난해 한국이 수입한 미국산 원유는 약 2151만t을 기록했다.",
        claim_type="규모", period="2024년", unit="톤", population="미국산 원유 수입량",
        value=21510000.0,
    )
    computed18 = ComputedResult(calc_type="단순조회", raw_value=683609488.0, unit="천달러", period="2024")
    result18 = _rule_based_verdict(claim18, computed18)
    print(f"[케이스18 - 단위 비대칭 버그 수정] 톤 vs 천달러 비교 결과: {result18}")
    assert result18 is None, (
        f"단위 불일치 회귀 — 규칙 기반으로 확정 판정하면 안 되는데 {result18}이 나옴"
    )
    print("  → 통과: 단위가 다르면 격차가 커도 '불일치'로 확정하지 않고 LLM에 위임함")
    print("     ('일치' 쪽에만 있던 단위 검증을 '불일치' 쪽에도 대칭으로 추가, 2026-08-12).")

    # 케이스 19 — 회귀 방지: 표 데이터 범위 밖 시점 비교 버그 수정 확인. 실제 배치에서
    # 재현된 "1955~1960년 인구 증가율" claim(claim.period="1955~1960년")이 슬롯필링
    # 단계에서 그 범위 표현을 못 알아듣고 "2024년"(computed.period="2023~2024") 데이터로
    # 대체 조회되어, 완전히 다른 시대의 값끼리 비교되고도 확정 "불일치"가 나던 버그.
    claim19 = Claim(
        sentence="이전까진 5년 새 6%가량 늘던 인구수는 1955~1960년 16% 넘게 늘었다.",
        claim_type="증감률", period="1955~1960년", unit="%", population="전국 인구", value=16.0,
    )
    computed19 = ComputedResult(calc_type="증감률", raw_value=-0.2, unit="%", period="2023~2024")
    result19 = _rule_based_verdict(claim19, computed19)
    print(f"[케이스19 - 연도 범위 밖 시점 비교 버그 수정] 1955~1960 vs 2023~2024 비교 결과: {result19}")
    assert result19 is None, (
        f"연도 범위 밖 비교 회귀 — 규칙 기반으로 확정 판정하면 안 되는데 {result19}이 나옴"
    )
    print("  → 통과: 시점이 서로 너무 멀리 떨어져 있으면(63년 차이) '불일치'로 확정하지 않고")
    print("     LLM에 위임함 — 이 케이스는 사실 원인이 슬롯필링이 범위 표현을 놓친 것이므로")
    print("     LLM도 결국 판단불가로 처리하는 게 맞지만, 최소한 '틀린 확정 판정'은 막음.")

    # 회귀 방지: 정상적인 장기 비교(claim과 computed가 같은 연도를 하나라도 공유)는 그대로
    # 확정 판정되는지 — "30년 전인 1994년의 절반 수준" 같은 정상 케이스가 이번 수정으로
    # 덩달아 막히면 안 된다.
    claim19b = Claim(
        sentence="작년 소비량은 30년 전인 1994년(108.3kg)의 절반 수준이다.",
        claim_type="비교", period="2024년", unit="kg", population="국민 1인당", value=55.8,
    )
    computed19b = ComputedResult(calc_type="증감", raw_value=-52.5, unit="kg", period="1994~2024")
    result19b = _rule_based_verdict(claim19b, computed19b)
    print(f"[케이스19b - 정상 장기비교 회귀 확인] 2024 vs 1994~2024(연도 겹침) -> {result19b is not None}")
    assert result19b is not None, "정상 장기 비교(연도가 computed.period 안에 포함)까지 막혀버림 — 회귀"
    print("  → 통과: computed.period 안에 claim이 가리키는 연도가 포함돼 있으면(2024) 정상 판정됨.")

    print("\n=== 2026-08-12 신규: 분기/월 period 라벨링 버그 수정 회귀 테스트 ===")
    assert _format_period_label("202402", prd_se="Q") == "2024년 2분기", _format_period_label("202402", prd_se="Q")
    assert _format_period_label("202402", prd_se="M") == "2024년 2월", _format_period_label("202402", prd_se="M")
    assert _format_period_label("202402") == "202402", "prd_se 안 넘기면 원본 그대로여야 함(하위 호환)"
    print("[period 라벨 포맷] '202402'+Q -> '2024년 2분기', +M -> '2024년 2월', prd_se 없으면 원본 유지 ✅")
    print("  → 통과: 분기 표의 202402가 판정 설명 프롬프트에서 더 이상 '월'로 오인식되지 않음")
    print("     (2026-08-12, 中 1분기 GDP 기사에서 실측 확인된 버그).")
