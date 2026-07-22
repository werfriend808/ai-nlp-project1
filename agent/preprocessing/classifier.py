"""
agent/preprocessing/classifier.py — 1단계 기사 관련도 분류

팀 계약(interfaces.py) 기준:
    입력: 기사 본문(str)
    출력: ClassificationResult (label, score, reason)

모델: HCX-DASH-002
프롬프트: prompts/classifier_prompt.txt (few-shot 8개 포함, {article_text} 자리에 본문을 채워 넣음)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .hcx_client import call_hcx

try:
    from interfaces import ClassificationResult
except ImportError:
    try:
        from agent.interfaces import ClassificationResult
    except ImportError:  # 단독 실행/테스트용 폴백
        from dataclasses import dataclass

        @dataclass
        class ClassificationResult:  # type: ignore[no-redef]
            label: bool
            score: float
            reason: str


MODEL = "HCX-DASH-002"
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "classifier_prompt.txt"
SYSTEM_PROMPT = "아래 지시사항을 정확히 따르고, 반드시 지정된 JSON 형식으로만 응답하세요."


class ClassifierError(RuntimeError):
    """분류 응답을 JSON으로 파싱하지 못한 경우."""


def _load_prompt_template(path: Path = PROMPT_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. A가 few-shot 프롬프트를 먼저 작성해야 합니다.")
    return path.read_text(encoding="utf-8")


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ClassifierError(f"응답에서 JSON 객체를 찾지 못했습니다: {text!r}")
    return json.loads(match.group(0))


def classify(article_text: str, *, model: str = MODEL) -> ClassificationResult:
    """기사 본문 하나를 받아 1단계 분류 결과(label/score/reason)를 돌려줍니다."""
    template = _load_prompt_template()
    prompt = template.replace("{article_text}", article_text)

    reply = call_hcx(model=model, system_prompt=SYSTEM_PROMPT, user_content=prompt)

    try:
        parsed = _extract_json_object(reply)
        return ClassificationResult(
            label=bool(parsed["label"]),
            score=float(parsed["score"]),
            reason=str(parsed["reason"]),
        )
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise ClassifierError(f"응답 파싱 실패: {reply!r}") from e


if __name__ == "__main__":
    #   python -m agent.preprocessing.classifier
    sample = (
        "통계청이 23일 발표한 '2024년 양곡소비량조사 결과'에 따르면, "
        "작년 국민 1인당 쌀 소비량은 1년 전보다 1.1%(0.6kg) 감소한 55.8kg을 기록했다."
    )
    print(classify(sample))
