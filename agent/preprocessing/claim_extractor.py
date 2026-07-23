"""
agent/preprocessing/claim_extractor.py — 2단계 수치 주장 문장 추출

팀 계약(interfaces.py) 기준:
    입력: 기사 본문(str)  (※ 1단계 결과가 아니라 원본 기사 본문을 다시 받음)
    출력: Claim의 리스트 (문장 하나하나 따로 호출 X, 기사 전체 넣고 한 번에)

모델: HCX-003
프롬프트: prompts/claim_extractor_prompt.txt (few-shot 3개 포함, {article_text} 자리에 본문을 채워 넣음)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

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


MODEL = "HCX-003"
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "claim_extractor_prompt.txt"
SYSTEM_PROMPT = "아래 지시사항을 정확히 따르고, 반드시 지정된 JSON 배열 형식으로만 응답하세요."


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


def _item_to_claim(item: dict) -> Claim:
    return Claim(
        sentence=str(item["sentence"]),
        claim_type=str(item["claim_type"]),
        period=item.get("period"),
        unit=item.get("unit"),
        population=item.get("population"),
    )


def _parse_claims(reply: str) -> list[Claim]:
    parsed = _extract_json_array(reply)
    return [_item_to_claim(item) for item in parsed]


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
            claims.append(_item_to_claim(json.loads(obj_text)))
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return claims


def extract_claims(article_text: str, *, model: str = MODEL, max_tokens: int = 2048) -> list[Claim]:
    """기사 본문 하나를 받아 수치 기반 주장 문장들을 Claim 리스트로 돌려줍니다.

    실패 처리 3단계:
    1) 한 번 더 같은 요청을 재시도 (드문 비결정적 생성 오류 대응).
    2) 그래도 안 되면, 마지막 응답에서 개별적으로 파싱 가능한 객체만 건져낸다
       (maxTokens에 걸려 배열이 중간에 끊긴 경우, 특정 객체만 스마트 쿼트로
       깨진 경우 등 — 재시도로는 안 고쳐지는 경우가 많아서 별도 처리).
    3) 그것도 하나도 못 건지면 ClaimExtractorError.
    """
    template = _load_prompt_template()
    prompt = template.replace("{article_text}", article_text)

    last_reply = ""
    for _ in range(2):
        last_reply = call_hcx(
            model=model, system_prompt=SYSTEM_PROMPT, user_content=prompt, max_tokens=max_tokens
        )
        try:
            return _parse_claims(last_reply)
        except (KeyError, ValueError, TypeError, json.JSONDecodeError):
            continue

    salvaged = _salvage_claims(last_reply)
    if salvaged:
        return salvaged

    raise ClaimExtractorError(f"응답 파싱 실패(재시도+구제 포함): {last_reply!r}")


if __name__ == "__main__":
    #   python -m agent.preprocessing.claim_extractor
    sample = (
        "통계청이 23일 발표한 '2024년 양곡소비량조사 결과'에 따르면, "
        "작년 국민 1인당 쌀 소비량은 1년 전보다 1.1%(0.6kg) 감소한 55.8kg을 기록했다. "
        "작년 소비량은 30년 전인 1994년(108.3kg)의 절반 수준이다."
    )
    for claim in extract_claims(sample):
        print(claim)
