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

import concurrent.futures
import os
import time
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

# LEGACY_V1_MODELS(구버전 v1 엔드포인트)는 temperature=0.0을 그대로 보내면 "400 Bad
# Request"로 거부된다(실제로 확인됨, 2026-08-11 — judge.py가 재현성을 위해 temperature=0.0을
# 넘기면서 7단계 LLM 위임 경로가 전부 깨지는 걸로 재현됨). v3 엔드포인트(HCX-007 등)는
# temperature=0.0을 그대로 받아들여서 v1 모델에만 해당하는 문제다. 0에 아주 가까운 값으로
# 살짝 올려서, 재현성은 사실상 유지하면서 API 거부는 피한다.
_MIN_TEMPERATURE_FOR_V1 = 0.01

# 2026-08-19: export_for_rerank.py로 기사 30건을 연달아 돌리면 CLOVA Studio가 429(Too Many
# Requests)를 반복해서 대부분의 classify/claim_extractor 호출이 그냥 스킵되는 문제가 실측됨
# (30개 기사 중 claim 6건만 생존). 429는 일시적 과다호출 신호일 뿐이라 재시도하면 대개
# 통과하므로, Retry-After 헤더(있으면 그 값을, 없으면 지수 백오프)를 보고 잠깐 기다렸다가
# 재시도한다.
_MAX_429_RETRIES = 5
_BACKOFF_BASE_SECONDS = 2.0

# 2026-08-20: requests의 timeout=은 "청크 사이 대기시간"만 잰다 — CLOVA Studio가 스트리밍
# 응답을 아주 천천히(청크마다 60초는 안 넘기지만 전체는 몇 분씩) 흘려보내면 실제로는 절대
# 안 끝나는 호출이 생긴다(30건 배치 실행 때 실측: 호출 하나가 몇 분씩 멈춰있는 현상 반복).
# 별도 스레드에서 요청을 돌리고 그 스레드 자체에 진짜 wall-clock 제한(전체 호출 1회당,
# 429 재시도 포함)을 걸어서 이 문제를 막는다.
DEFAULT_HARD_TIMEOUT_SECONDS = 120.0
_HCX_CALL_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=8, thread_name_prefix="hcx-call"
)


class HcxApiError(RuntimeError):
    """CLOVA Studio 호출 실패 (HTTP 에러 또는 예상과 다른 응답 형식)."""


def _post_with_hard_timeout(url, headers, body, read_timeout, deadline):
    """requests.post를 별도 스레드에서 실행하고 deadline(time.monotonic() 기준 절대시각)까지만
    기다린다. 스레드 자체는 소켓이 열려있는 한 백그라운드에서 계속 돌 수 있지만(파이썬은
    스레드를 강제 종료할 수 없음), 호출한 쪽은 deadline이 지나면 즉시 제어권을 돌려받는다."""
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("HCX 호출 전체 제한시간을 초과했습니다.")
    future = _HCX_CALL_EXECUTOR.submit(
        requests.post, url, headers=headers, json=body, timeout=read_timeout
    )
    try:
        return future.result(timeout=remaining)
    except concurrent.futures.TimeoutError as e:
        raise TimeoutError("HCX 호출 전체 제한시간을 초과했습니다.") from e


def call_hcx(
    model: str,
    system_prompt: str,
    user_content: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
    seed: Optional[int] = 42,
    api_key: Optional[str] = None,
    timeout: int = 60,
    hard_timeout: float = DEFAULT_HARD_TIMEOUT_SECONDS,
) -> str:
    """system/user 메시지 한 쌍을 CLOVA Studio에 보내고 assistant 응답 텍스트만 돌려줍니다.

    model 예: "HCX-DASH-002", "HCX-003"

    seed: 2026-08-11 재현성 문제 대응. CLOVA Studio API는 seed를 안 보내면(=0과 동일)
    출력이 매번 무작위화된다 — temperature=0.0으로 낮춰도 이것과는 별개 문제라, 같은
    기사를 여러 번 넣었을 때 claim 개수/내용이 흔들리는 현상(실측: 4건→2건→4건)의
    유력한 원인이었다. 1~4294967295 사이 고정값을 주면 같은 입력에 항상 같은 결과가
    나온다(공식 문서: "출력 일관성 수준을 조정"). None으로 넘기면 기존처럼 seed를
    아예 안 보낸다(무작위 필요한 특수 상황 대비).

    timeout: requests의 청크 사이 대기시간(초) — 개별 read 사이 간격만 잰다.
    hard_timeout: 이 call_hcx() 호출 전체(429 재시도 포함)에 대한 실제 wall-clock
    제한(초). 스트리밍 응답이 청크는 꾸준히 오지만 전체적으로 아주 느릴 때 timeout=만으로는
    안 막히는 무한 대기를 막기 위함 — 이 시간을 넘기면 HcxApiError를 던진다.
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
    api_version = "v1" if model in LEGACY_V1_MODELS else "v3"
    effective_temperature = (
        max(temperature, _MIN_TEMPERATURE_FOR_V1) if api_version == "v1" else temperature
    )

    body = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "topP": 0.8,
        "temperature": effective_temperature,
    }
    if model not in NO_MAX_TOKENS_MODELS:
        body["maxTokens"] = max_tokens

    url = f"{CLOVASTUDIO_HOST}/{api_version}/chat-completions/{model}"
    deadline = time.monotonic() + hard_timeout

    try:
        response = _post_with_hard_timeout(url, headers, body, timeout, deadline)
        for attempt in range(_MAX_429_RETRIES):
            if response.status_code != 429:
                break
            retry_after = response.headers.get("Retry-After")
            wait_seconds = (
                float(retry_after) if retry_after else _BACKOFF_BASE_SECONDS * (2**attempt)
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("HCX 호출 전체 제한시간을 초과했습니다.")
            time.sleep(min(wait_seconds, remaining))
            response = _post_with_hard_timeout(url, headers, body, timeout, deadline)
    except TimeoutError as e:
        raise HcxApiError(
            f"HCX 호출이 전체 제한시간({hard_timeout:.0f}초)을 넘겨서 중단했습니다. "
            "CLOVA Studio 응답이 비정상적으로 느린 상태일 수 있습니다."
        ) from e

    response.raise_for_status()
    data = response.json()

    try:
        return data["result"]["message"]["content"]
    except (KeyError, TypeError) as e:
        raise HcxApiError(f"예상과 다른 응답 형식입니다: {data}") from e
