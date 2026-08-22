"""
agent/preprocessing/claim_extractor.py — 2단계 수치 주장 문장 추출

팀 계약(interfaces.py) 기준:
    입력: 기사 본문(str)  (※ 1단계 결과가 아니라 원본 기사 본문을 다시 받음)
    출력: Claim의 리스트 (문장 하나하나 따로 호출 X, 기사 전체 넣고 한 번에)

모델: HCX-DASH-002 (2026-07-24 HCX-003에서 교체 — agent/preprocessing/eval_claim_extractor_model.py로
  실측한 결과, HCX-003은 긴 기사(prompt 길이 약 17k자 이상)에서 컨텍스트 길이 초과로 절반이
  아예 실패했고, HCX-DASH-002는 같은 샘플에서 100% 성공 + 2.6배 빠름 + 품질 지표도 비슷하거나
  더 나음. 자세한 수치는 tests/claim_extractor_model_eval_log.md 참고.)
프롬프트: prompts/claim_extractor_prompt.txt (few-shot 3개 포함, {article_text} 자리에 본문을 채워 넣음)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from .hcx_client import call_hcx
from .claim_candidate_scanner import find_missed_candidates

try:
    from interfaces import Claim
except ImportError:
    try:
        from agent.interfaces import Claim
    except ImportError:  # 단독 실행/테스트용 폴백
        from dataclasses import dataclass
        from typing import Optional

        @dataclass
        class Claim:  # type: ignore[no-redef]
            sentence: str
            claim_type: str
            period: Optional[str] = None
            unit: Optional[str] = None
            population: Optional[str] = None
            statistic_expression: Optional[str] = None
            value: Optional[float] = None
            value_type: Optional[str] = None
            comparison_operator: Optional[str] = None
            comparison_target: Optional[str] = None
            comparison_value: Optional[float] = None
            region: Optional[str] = None
            source_org: Optional[str] = None
            source_report: Optional[str] = None
            age: Optional[str] = None
            gender: Optional[str] = None
            search_query: Optional[str] = None


MODEL = "HCX-DASH-002"
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "claim_extractor_prompt.txt"
SYSTEM_PROMPT = "아래 지시사항을 정확히 따르고, 반드시 지정된 JSON 배열 형식으로만 응답하세요."

# recover_missed_claims() 전용 — claim_extractor_prompt.txt(전체 기사용, few-shot 7개 포함)를
# 그대로 재사용하지 않는 이유: 여기선 이미 claim_candidate_scanner가 걸러낸 몇 개 문장만
# 재검토하면 되므로, 전체 프롬프트를 다시 보내면 토큰 낭비가 크다. 짧은 프롬프트로 "이
# 문장이 진짜 claim인지"만 판단시킨다 — LLM의 판단력은 그대로 쓰되(하이브리드 설계),
# 스캐너가 넓게 잡은 후보(제품 스펙 등 노이즈 포함)를 여기서 거른다.
_RECOVERY_PROMPT_TEMPLATE = """아래 문장들은 규칙 기반 스캐너가 숫자를 포함하고 있어서
"수치 기반 주장"일 수 있다고 표시했지만, 1차 추출에서는 뽑히지 않은 것들입니다. 각 문장을
다시 검토해서, 실제로 검증 가치 있는 수치 주장이면 구조화하고, 아니면(제품 스펙, 법령상
기준값, 개별 기업/제품의 가격·실적처럼 국가 공식 통계가 아닌 경우) 결과에서 제외하세요.

## 판단 기준
- 구체적인 수치(값·비율·순위 등)를 포함하고, 실제로 조사·집계된 통계 주장이어야 합니다.
- 제품 스펙(좌석 수, 트렁크 용량 등)이나 법령상 기준값은 제외합니다.
- sentence는 절대 요약하거나 다시 쓰지 말고, 아래 문장 원문을 그대로 사용합니다.
- ⚠️ 한 문장 안에 서로 다른 그룹(연령대·지역 등)의 수치가 나란히 여러 개 나와도(예: "90세
  이상 사망자가 6만1200명으로 늘었고, 50대 사망자도 2만5800명으로 늘었다") 앞부분만 뽑고
  뒷부분을 버리면 안 됩니다 — 그 문장이 담고 있는 통계 주장 중 하나를 대표해서(예: 첫
  번째 수치 기준으로) claim_type/value를 채우되, sentence는 문장 전체를 그대로 사용해서
  절대 빠뜨리지 마세요. "복합 문장이라 애매하다"는 이유로 통째로 제외하지 마세요.
- ⚠️ OECD 평균 대비, 전년 대비처럼 다른 대상과 비교하는 문장이어도 공식 기관(통계청 등)이
  발표한 수치라면 검증 가치 있는 claim입니다 — 비교 표현이 있다는 이유만으로 제외하지
  마세요.

## 출력 형식 (JSON 배열만 출력, 다른 텍스트 금지 — claim_extractor와 동일 스키마)
[
  {
    "sentence": "...", "claim_type": "규모|증감률|비교|전망",
    "period": "..." 또는 null, "unit": "..." 또는 null, "population": "..." 또는 null,
    "statistic_expression": "..." 또는 null, "value": 숫자 또는 null,
    "value_type": "수준값|증감폭" 또는 null (claim_type이 "규모"일 때만 채움 — value가 특정 시점의
      총량이면 "수준값", value 자체가 증가/감소한 변화폭이면 "증감폭"),
    "comparison_operator": "증가|감소|동일|초과|미만" 또는 null,
    "comparison_target": "..." 또는 null, "comparison_value": 숫자 또는 null,
    "region": "..." 또는 null, "source_org": "..." 또는 null, "source_report": "..." 또는 null
  }
]
검증 가치 있는 문장이 하나도 없으면 빈 배열 []을 출력합니다.

## 예시 (복합 문장 처리)
입력 문장: "90세 이상 고령층 사망자가 6만1200명으로 전년 대비 3800명 늘었고, 50대 사망자도
2만5800명으로 600명 늘었다."
출력: [{"sentence": "90세 이상 고령층 사망자가 6만1200명으로 전년 대비 3800명 늘었고, 50대
사망자도 2만5800명으로 600명 늘었다.", "claim_type": "규모", "period": null, "unit": "명",
"population": "90세 이상 고령층", "statistic_expression": "전년 대비 3800명 증가",
"value": 61200, "value_type": "수준값", "comparison_operator": "증가",
"comparison_target": "전년", "comparison_value": 3800, "region": null, "source_org": null,
"source_report": null}]
(뒤에 나오는 "50대 사망자" 수치는 population/statistic_expression에 다 담지 못하더라도,
sentence 자체는 원문 전체를 유지해서 문장이 통째로 유실되지 않게 합니다.)
(claim_type은 "규모"입니다 — value가 특정 시점의 총량(61200)이고 그 변화분(3800)은
comparison_value에 따로 담기 때문입니다. "전년 대비 늘었다"는 비교 표현이 있다고
claim_type을 "증감률"로 바꾸지 마세요 — 퍼센트(%) 변화율이 실제로 주장된 문장에만
"증감률"을 쓰고, 이 예시처럼 절대량(명)의 변화는 "규모"+comparison_value로 표현합니다.)

## 검토할 문장들
{candidate_sentences}
"""

# 실측 확인(2026-08-04, verify_claim_extractor_on_golden.py): 기사가 이 길이를 넘어가면
# temperature=0으로 결정적으로 만들어도 gold 골든셋 대비 recall이 28.8%까지 떨어짐 —
# few-shot 예시들은 전부 이 정도 이하 짧은 발췌문이었고 거기선 잘 뽑히는 걸로 봐서,
# "모델이 이 크기 이상을 한 번에 훑을 때 일부만 뽑고 마는" 커버리지 문제로 판단.
# 그래서 이보다 길면 아래에서 문단 단위로 잘라 각각 호출한 뒤 합친다.
CHUNK_SIZE = 3000


class ClaimExtractorError(RuntimeError):
    """추출 응답을 JSON 배열로 파싱하지 못한 경우."""


def _load_prompt_template(path: Path = PROMPT_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. A가 few-shot 프롬프트를 먼저 작성해야 합니다.")
    return path.read_text(encoding="utf-8")


def _sanitize_smart_quotes(text: str) -> str:
    """HCX가 문자열 구분자로 스마트 쿼트(“ ”)를 섞어 쓰는 경우를 보정한다.

    실측 사례: `: “...` 처럼 여는 쿼트 자체가 스마트 쿼트인 경우, `...다"”,`처럼
    제대로 닫힌 뒤 스마트 쿼트가 하나 더 붙어 나오는 경우 둘 다 발생함.
    """
    text = re.sub(r'(?<=[:\[]\s)[“”]', '"', text)
    text = re.sub(r'["“”]+(?=[,\]}])', '"', text)
    return text


def _extract_json_array(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ClaimExtractorError(f"응답에서 JSON 배열을 찾지 못했습니다: {text!r}")
    raw = match.group(0)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(_sanitize_smart_quotes(raw))


def _to_optional_float(value: object) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


# interfaces.py의 ClaimType(2단계 계약, 4종). claim_extractor.py 안에서만 방어적으로
# 참조하는 로컬 사본이라 팀 합의 없이 여기서 그대로 씀 — interfaces.py 자체는 안 건드림.
_KNOWN_CLAIM_TYPES = {"규모", "증감률", "비교", "전망"}


def _normalize_claim_type(raw: object) -> Optional[str]:
    """HCX가 claim_type을 null로 주면 예전엔 claim_type=str(item["claim_type"])가 그대로
    문자열 "None"을 만들어버렸다(다운스트림이 "None"을 진짜 값처럼 취급하는 버그,
    calc_type_router 도입 배경이 된 100건 실측 조사에서 9/84건 재현됨). 여기서 null과
    "배경"/"규정"처럼 4종 스키마 밖 값을 전부 None으로 통일해서, 다운스트림(calc_type
    라우팅 등)이 "판단불가"로 처리할 수 있는 명확한 신호로 만든다."""
    if raw is None:
        return None
    text = str(raw)
    return text if text in _KNOWN_CLAIM_TYPES else None


# interfaces.py의 ComparisonOperator(5종: 증가/감소/동일/초과/미만) 밖에서 실측 조사(100건
# 표본) 중 반복 관측된 동의어 하나만 정규화한다. "하락"(3건)은 "감소"의 동의어가 명확해서
# 매핑하지만, "혼합"/"완화"/"약화"/"회복"/"2년 연속"(1건씩)은 표본이 너무 작고 방향이
# 모호해서 지금은 새 스키마를 만들지 않고 원문 그대로 둔다(개인/CALC_TYPE_ROUTING_DESIGN.md
# Q3-6/7 참고).
_COMPARISON_OPERATOR_SYNONYMS = {"하락": "감소"}

# 위 100건 표본 검토 후에도 "1위"/"악화"처럼 방향 동의어도 아니고 팀이 검토한 적도 없는
# 값이 그대로 저장되는 사례가 실측 확인됨(2026-08-20, verifications_export.json) — "악화"는
# 이미 허용된 "약화"와 다른 단어라 혼동 주의. claim_type과 달리 이 필드는 여태 스키마
# 검증이 아예 없어서 LLM이 뭘 주든 그대로 통과시켰다. 이제 "5종 스키마 값" 또는 "팀이
# 이미 검토해서 원문 유지하기로 정한 애매한 동의어" 둘 중 하나가 아니면 None으로 떨어뜨린다
# — _AMBIGUOUS_BUT_ACCEPTED는 위 주석의 기존 팀 결정을 그대로 옮긴 것이라, 이 결정 자체는
# 안 건드리고 "1위" 같은 완전히 다른 종류의 오류만 추가로 걸러낸다.
_AMBIGUOUS_BUT_ACCEPTED = {"혼합", "완화", "약화", "회복", "2년 연속"}
_KNOWN_COMPARISON_OPERATORS = {"증가", "감소", "동일", "초과", "미만"}


def _normalize_comparison_operator(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = _COMPARISON_OPERATOR_SYNONYMS.get(str(raw), str(raw))
    if text in _KNOWN_COMPARISON_OPERATORS or text in _AMBIGUOUS_BUT_ACCEPTED:
        return text
    return None


# interfaces.py의 ValueType(2종: 수준값/증감폭, 2026-08-16 추가). claim_type과 달리 이 필드가
# 스키마 밖 값이어도 claim 자체를 버릴 이유는 없다 — None이면 calc_type_router가 기존처럼
# "단순조회"로 안전하게 폴백한다(하위 호환).
_KNOWN_VALUE_TYPES = {"수준값", "증감폭"}


def _normalize_value_type(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw)
    return text if text in _KNOWN_VALUE_TYPES else None


def _item_to_claim(item: dict) -> Claim:
    return Claim(
        sentence=str(item["sentence"]),
        claim_type=_normalize_claim_type(item.get("claim_type")),
        period=item.get("period"),
        unit=item.get("unit"),
        population=item.get("population"),
        statistic_expression=item.get("statistic_expression"),
        value=_to_optional_float(item.get("value")),
        value_type=_normalize_value_type(item.get("value_type")),
        comparison_operator=_normalize_comparison_operator(item.get("comparison_operator")),
        comparison_target=item.get("comparison_target"),
        comparison_value=_to_optional_float(item.get("comparison_value")),
        region=item.get("region"),
        source_org=item.get("source_org"),
        source_report=item.get("source_report"),
        age=item.get("age"),
        gender=item.get("gender"),
        search_query=item.get("search_query"),
    )


def _parse_claims(reply: str) -> list[Claim]:
    parsed = _extract_json_array(reply)
    return [c for c in (_item_to_claim(item) for item in parsed) if c.claim_type is not None]


def _iter_top_level_objects(array_text: str):
    """array_text 안의 `{...}` 객체들을, 문자열 내부의 구두점은 무시하면서 하나씩 잘라낸다.

    배열 전체가 깨져도(마지막 객체가 maxTokens에 걸려 중간에 끊긴 경우 등) 앞의
    완결된 객체들은 그대로 살릴 수 있게 하기 위한 용도.
    """
    depth = 0
    in_string = False
    escape = False
    start = None
    for i, ch in enumerate(array_text):
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                yield array_text[start:i + 1]
                start = None


def _salvage_claims(reply: str) -> list[Claim]:
    """배열 전체 파싱이 실패해도, 개별적으로 파싱 가능한 객체만 건져서 Claim으로 만든다.

    (1) 응답이 maxTokens에 걸려 배열 마지막 객체가 중간에 끊긴 경우,
    (2) 스마트 쿼트 보정 후에도 특정 객체 하나만 여전히 깨진 경우
    둘 다, 그 객체 하나만 버리고 나머지는 살린다.
    """
    start = reply.find("[")
    if start == -1:
        return []

    sanitized = _sanitize_smart_quotes(reply[start:])
    claims: list[Claim] = []
    for obj_text in _iter_top_level_objects(sanitized):
        try:
            claim = _item_to_claim(json.loads(obj_text))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if claim.claim_type is not None:
            claims.append(claim)
    return claims


def _split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """text를 chunk_size 근처에서 문장 끝("다.") 경계를 찾아 자른다 (문장 중간이 잘리는 것 방지).

    겹침(overlap) 없이 순서대로 이어붙이는 방식 — 겹치면 같은 claim이 두 청크에서 중복
    추출될 수 있는데, 그러면 병합 시 중복 제거 로직이 따로 필요해진다. 대신 겹침이 없으면
    "이는 ~"처럼 청크 경계 바로 다음에서 앞 청크 문장을 참조하는 극히 드문 경우만 맥락을
    일부 잃는데, 그래도 그 claim의 수치 자체는 청크 안에 있어 대부분 그대로 뽑힌다.
    """
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            boundary = text.rfind("다.", start + chunk_size // 2, end)
            if boundary != -1:
                end = boundary + 2
        chunks.append(text[start:end])
        start = end
    return chunks


def _extract_claims_single(
    chunk_text: str, *, model: str, max_tokens: int, temperature: float
) -> list[Claim]:
    """청크 하나(또는 짧아서 안 쪼갠 기사 전체)를 HCX에 보내 Claim 리스트로 파싱한다.

    실패 처리 3단계:
    1) 한 번 더 같은 요청을 재시도 (드문 비결정적 생성 오류 대응).
    2) 그래도 안 되면, 마지막 응답에서 개별적으로 파싱 가능한 객체만 건져낸다
       (maxTokens에 걸려 배열이 중간에 끊긴 경우, 특정 객체만 스마트 쿼트로
       깨진 경우 등 — 재시도로는 안 고쳐지는 경우가 많아서 별도 처리).
    3) 그것도 하나도 못 건지면 ClaimExtractorError.
    """
    template = _load_prompt_template()
    prompt = template.replace("{article_text}", chunk_text)

    last_reply = ""
    for _ in range(2):
        last_reply = call_hcx(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_content=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        try:
            return _parse_claims(last_reply)
        except (ClaimExtractorError, KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue

    salvaged = _salvage_claims(last_reply)
    if salvaged:
        return salvaged

    raise ClaimExtractorError(f"응답 파싱 실패(재시도+구제 포함): {last_reply!r}")


def extract_claims(
    article_text: str, *, model: str = MODEL, max_tokens: int = 4096, temperature: float = 0.0
) -> list[Claim]:
    """기사 본문 하나를 받아 수치 기반 주장 문장들을 Claim 리스트로 돌려줍니다.

    temperature 기본값을 hcx_client 기본(0.2)보다 낮춰 0.0으로 둔다 — 동일 입력인데도
    실행마다 어떤 claim을 뽑을지가 달라지는 비결정성이 실측 확인됨
    (verify_claim_extractor_on_golden.py, 2026-08-04).

    CHUNK_SIZE(3000자)보다 긴 기사는 문단 단위로 잘라 청크별로 따로 호출한 뒤 결과를
    합친다. temperature=0으로도 recall이 28.8%에 그쳐서(few-shot 예시 보강도 무효과),
    원인이 비결정성이 아니라 "모델이 긴 입력을 한 번에 훑을 때 일부만 뽑고 마는" 커버리지
    문제로 판단했고, 각 청크를 few-shot 예시와 비슷한 짧은 크기로 줄여서 정면 대응한다.
    청크 사이에 겹침은 없어서(위 _split_into_chunks 참고) 중복 제거 로직은 불필요.

    실패 처리(청크별로 독립 적용): 한 청크가 ClaimExtractorError로 실패해도 나머지 청크는
    계속 처리한다 — 기사 전체를 버리는 것보다, 실패한 청크분만 놓치는 게 낫다.
    모든 청크가 실패하면 ClaimExtractorError를 그대로 올린다.
    """
    chunks = _split_into_chunks(article_text)

    all_claims: list[Claim] = []
    errors: list[Exception] = []
    for chunk in chunks:
        try:
            all_claims.extend(
                _extract_claims_single(chunk, model=model, max_tokens=max_tokens, temperature=temperature)
            )
        except ClaimExtractorError as e:
            errors.append(e)

    if not all_claims and errors:
        raise errors[0]

    return all_claims


# 2026-08-17 실측: claim_extractor는 temperature=0에서도 같은 기사를 다시 돌리면 1차 추출
# 결과가 달라진다(이미 알려진 비결정성) — 그리고 recover_missed_claims 자체도 같은 이유로
# 비결정적이라, 스캐너가 정확히 찾아낸 후보를 줘도 복구 호출 한 번으로 전부 못 건지는 경우가
# 실제로 재현됐다(4개 놓친 기사를 재실행하니 1차 추출과 복구 조합이 매번 달라짐). 그래서
# "스캔→복구"를 한 번이 아니라 몇 번 반복한다 — 놓친 게 없는 대부분의 기사는 1회차에서
# 바로 끝나 추가 비용이 전혀 없고(scan은 정규식이라 공짜), 놓친 게 있는 소수 기사만 최대
# _MAX_RECOVERY_ROUNDS번까지 짧은 재확인 호출을 반복한다. 어느 회차든 새로 건진 게 0개면
# 더 반복해도 소용없다고 보고 바로 멈춘다(비용 상한).
_MAX_RECOVERY_ROUNDS = 2


def recover_missed_claims(
    article_text: str,
    extracted_claims: list[Claim],
    *,
    model: str = MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.0,
) -> list[Claim]:
    """extract_claims() 이후, 규칙 기반 스캐너(claim_candidate_scanner)가 찾은 후보 중
    LLM이 1차 추출에서 놓친 문장이 있으면 그 문장들만 다시 HCX에 물어봐서 복구한다
    (하이브리드 추출 설계의 3단계 — recall 안전망). 최대 _MAX_RECOVERY_ROUNDS회까지
    "다시 스캔 → 다시 복구"를 반복해서, 복구 자체가 한 번에 다 못 건지는 경우까지 보강한다.

    스캐너는 정밀도가 낮게(recall 우선) 설계돼 있어서, 놓친 후보를 그대로 claim으로
    확정하지 않고 여기서 다시 LLM 판단을 한 번 더 거친다 — 제품 스펙·개별 기업 수치 같은
    노이즈는 이 단계에서 걸러진다.

    실패 시(HCX 호출 오류, JSON 파싱 실패 등) 그때까지 모은 claims를 그대로 반환한다 —
    복구는 best-effort이고, 실패해도 이전 라운드까지의 결과는 지켜야 배치가 안 죽는다.
    """
    claims = extracted_claims
    for _ in range(_MAX_RECOVERY_ROUNDS):
        before = len(claims)
        claims = _recover_missed_claims_once(
            article_text, claims, model=model, max_tokens=max_tokens, temperature=temperature
        )
        if len(claims) == before:
            break  # 이번 회차에서 놓친 게 없었거나(스캔 결과 0건) 하나도 못 건졌으면 종료
    return claims


def _recover_missed_claims_once(
    article_text: str,
    extracted_claims: list[Claim],
    *,
    model: str,
    max_tokens: int,
    temperature: float,
) -> list[Claim]:
    """recover_missed_claims()의 한 회차 분량 — 스캔 1번 + (필요하면) 복구 호출 1번."""
    already_sentences = [c.sentence for c in extracted_claims]
    missed = find_missed_candidates(article_text, already_sentences)
    if not missed:
        return extracted_claims

    prompt = _RECOVERY_PROMPT_TEMPLATE.replace(
        "{candidate_sentences}", "\n".join(f"- {s}" for s in missed)
    )
    reply = ""
    try:
        reply = call_hcx(
            model=model,
            system_prompt=SYSTEM_PROMPT,
            user_content=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        recovered = _parse_claims(reply)
    except (ClaimExtractorError, KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        # maxTokens에 걸려 배열 마지막 객체가 중간에 끊기면 _parse_claims가 통째로
        # 실패한다 — 후보 문장이 많을 때(사망자 나이대별 여러 그룹처럼 한 회차에 건질 게
        # 많은 경우) 실측 재현됨. 앞쪽에서 이미 완결된 객체들은 _salvage_claims로 건져서
        # 그 회차를 통째로 날리지 않는다.
        salvaged = _salvage_claims(reply) if reply else []
        if not salvaged:
            print(f"[claim_extractor] 놓친 claim 복구 실패 ({type(e).__name__}: {e}) → 이번 회차는 원래 결과만 사용")
            return extracted_claims
        recovered = salvaged

    return extracted_claims + recovered


# 2026-08-21 실측(DB 실데이터, id=17): claim.sentence 맨 앞에 기사 제목이 그대로 붙어
# 나오는 경우가 있었다("3월 청년 실업률 7.5%… 4년 만에 최대치 기록 청년 실업률이
# 지난달 7.5%까지 치솟으며..." — 앞부분이 article_title과 완전히 동일). 원인은
# batch_runner._clean_scraped_article_text()가 스크랩 원본에서 제목과 본문 사이에
# 줄바꿈을 넣어 구분해주더라도, HCX가 그 경계를 항상 지키진 않고 "제목+첫 문장"을
# 하나의 claim으로 묶어버리는 경우가 있기 때문 — 입력단 정리만으로는 100% 못 막는다.
# 그래서 출력단(claim_extractor가 반환하는 sentence)에서 한 번 더 방어적으로 제목
# 접두어를 잘라낸다. article_title을 안 넘기면(호출부가 모르는 경우) 아무것도 안 한다.
def strip_title_prefix(sentence: str, article_title: Optional[str]) -> str:
    """claim.sentence가 article_title로 시작하면 그 접두어를 떼고 반환한다."""
    if not article_title:
        return sentence
    title = article_title.strip()
    if title and sentence.startswith(title):
        return sentence[len(title):].lstrip()
    return sentence


def strip_title_prefix_from_claims(claims: list[Claim], article_title: Optional[str]) -> list[Claim]:
    """extract_claims()/recover_missed_claims()가 반환한 claim 리스트 전체에
    strip_title_prefix()를 적용한다 — run_article()에서 두 호출 직후 한 번만 부르면 됨."""
    for claim in claims:
        claim.sentence = strip_title_prefix(claim.sentence, article_title)
    return claims


if __name__ == "__main__":
    #   python -m agent.preprocessing.claim_extractor
    sample = (
        "통계청이 23일 발표한 '2024년 양곡소비량조사 결과'에 따르면, "
        "작년 국민 1인당 쌀 소비량은 1년 전보다 1.1%(0.6kg) 감소한 55.8kg을 기록했다. "
        "작년 소비량은 30년 전인 1994년(108.3kg)의 절반 수준이다."
    )
    for claim in extract_claims(sample):
        print(claim)
