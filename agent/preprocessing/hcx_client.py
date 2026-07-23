"""
agent/preprocessing/hcx_client.py — CLOVA Studio Chat Completions 호출 공통 헬퍼

1단계(classifier.py)와 2단계(claim_extractor.py)가 공통으로 쓰는 HCX 호출 로직만 담당합니다.
프롬프트 조립·응답 파싱은 각 모듈이 하고, 이 파일은 "HTTP 호출 + 에러 처리"만 합니다.

※ CLOVA Studio는 모델마다 API 버전이 다릅니다 (실제로 HCX-003을 v3 엔드포인트로 호출하면
  "40084 Unsupported API for model" 에러가 남). HCX-DASH-002/HCX-005/HCX-007처럼 최신 모델은
  v3(/v3/chat-completions/{model}), HCX-003/HCX-DASH-001처럼 이전 세대 모델은 v1
  (/v1/chat-completions/{model})을 써야 해서, 아래에서 모델명으로 버전을 자동 선택합니다.

사전 준비물:
    HCX_API_KEY — .env(HCX_API_KEY=...)에 넣거나 환경변수로 설정
    (pip install python-dotenv 해두면 .env 자동 로드)

참고 문서:
    v3 (HCX-DASH-002 등): https://api.ncloud-docs.com/docs/en/clovastudio-chatcompletionsv3
    v1 (HCX-003 등):       https://api.ncloud-docs.com/docs/en/clovastudio-chatcompletions
"""

from __future__ import annotations

import os
import uuid
from typing import Optional

import requests

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

CLOVASTUDIO_HOST = "https://clovastudio.stream.ntruss.com"

# v3 엔드포인트를 지원하지 않는 이전 세대 모델 (v1 엔드포인트로 호출해야 함)
LEGACY_V1_MODELS = {"HCX-003", "HCX-DASH-001"}

# HCX-007(하이브리드 추론 모델)은 요청에 maxTokens를 넣으면 "40001 Invalid parameter:
# maxTokens"로 거부됨 (실제로 확인됨 — 100/1024/32768 등 어떤 값을 넣어도 동일하게 실패,
# 필드 자체를 빼야 200 OK). 추론 단계 때문에 길이를 모델이 자체 결정하는 것으로 보임.
NO_MAX_TOKENS_MODELS = {"HCX-007"}


class HcxApiError(RuntimeError):
    """CLOVA Studio 호출 실패 (HTTP 에러 또는 예상과 다른 응답 형식)."""


def call_hcx(
    model: str,
    system_prompt: str,
    user_content: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    api_key: Optional[str] = None,
    timeout: int = 30,
) -> str:
    """system/user 메시지 한 쌍을 CLOVA Studio에 보내고 assistant 응답 텍스트만 돌려줍니다.

    model 예: "HCX-DASH-002", "HCX-003"
    """
    api_key = api_key or os.environ.get("HCX_API_KEY")
    if not api_key:
        raise RuntimeError(
            "HCX_API_KEY가 없습니다. .env(HCX_API_KEY=...)에 넣거나 "
            "call_hcx(..., api_key=...)로 직접 넘겨주세요."
        )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-NCP-CLOVASTUDIO-REQUEST-ID": str(uuid.uuid4()),
        "Content-Type": "application/json",
    }
    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "topP": 0.8,
        "temperature": temperature,
    }
    if model not in NO_MAX_TOKENS_MODELS:
        body["maxTokens"] = max_tokens

    api_version = "v1" if model in LEGACY_V1_MODELS else "v3"
    url = f"{CLOVASTUDIO_HOST}/{api_version}/chat-completions/{model}"

    response = requests.post(url, headers=headers, json=body, timeout=timeout)
    response.raise_for_status()
    data = response.json()

    try:
        return data["result"]["message"]["content"]
    except (KeyError, TypeError) as e:
        raise HcxApiError(f"예상과 다른 응답 형식입니다: {data}") from e
