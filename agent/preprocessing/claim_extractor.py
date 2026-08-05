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
            comparison_operator: Optional[str] = None
            comparison_target: Optional[str] = None
            comparison_value: Optional[float] = None
            region: Optional[str] = None
            source_org: Optional[str] = None
            source_report: Optional[str] = None


MODEL = "HCX-DASH-002"
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "claim_extractor_prompt.txt"
SYSTEM_PROMPT = "아래 지시사항을 정확히 따르고, 반드시 지정된 JSON 배열 형식으로만 응답하세요."

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


def _normalize_comparison_operator(raw: object) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw)
    return _COMPARISON_OPERATOR_SYNONYMS.get(text, text)


def _item_to_claim(item: dict) -> Claim:
    return Claim(
        sentence=str(item["sentence"]),
        claim_type=_normalize_claim_type(item.get("claim_type")),
        period=item.get("period"),
        unit=item.get("unit"),
        population=item.get("population"),
        statistic_expression=item.get("statistic_expression"),
        value=_to_optional_float(item.get("value")),
        comparison_operator=_normalize_comparison_operator(item.get("comparison_operator")),
        comparison_target=item.get("comparison_target"),
        comparison_value=_to_optional_float(item.get("comparison_value")),
        region=item.get("region"),
        source_org=item.get("source_org"),
        source_report=item.get("source_report"),
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
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
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


if __name__ == "__main__":
    #   python -m agent.preprocessing.claim_extractor
    sample = (
        "통계청이 23일 발표한 '2024년 양곡소비량조사 결과'에 따르면, "
        "작년 국민 1인당 쌀 소비량은 1년 전보다 1.1%(0.6kg) 감소한 55.8kg을 기록했다. "
        "작년 소비량은 30년 전인 1994년(108.3kg)의 절반 수준이다."
    )
    for claim in extract_claims(sample):
        print(claim)
