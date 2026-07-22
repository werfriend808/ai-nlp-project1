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


def _extract_json_array(text: str) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        raise ClaimExtractorError(f"응답에서 JSON 배열을 찾지 못했습니다: {text!r}")
    return json.loads(match.group(0))


def extract_claims(article_text: str, *, model: str = MODEL) -> list[Claim]:
    """기사 본문 하나를 받아 수치 기반 주장 문장들을 Claim 리스트로 돌려줍니다."""
    template = _load_prompt_template()
    prompt = template.replace("{article_text}", article_text)

    reply = call_hcx(model=model, system_prompt=SYSTEM_PROMPT, user_content=prompt)

    try:
        parsed = _extract_json_array(reply)
        return [
            Claim(
                sentence=str(item["sentence"]),
                claim_type=str(item["claim_type"]),
                period=item.get("period"),
                unit=item.get("unit"),
                population=item.get("population"),
            )
            for item in parsed
        ]
    except (KeyError, ValueError, TypeError, json.JSONDecodeError) as e:
        raise ClaimExtractorError(f"응답 파싱 실패: {reply!r}") from e


if __name__ == "__main__":
    #   python -m agent.preprocessing.claim_extractor
    sample = (
        "통계청이 23일 발표한 '2024년 양곡소비량조사 결과'에 따르면, "
        "작년 국민 1인당 쌀 소비량은 1년 전보다 1.1%(0.6kg) 감소한 55.8kg을 기록했다. "
        "작년 소비량은 30년 전인 1994년(108.3kg)의 절반 수준이다."
    )
    for claim in extract_claims(sample):
        print(claim)
