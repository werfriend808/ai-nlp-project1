"""agent/observability.py — 관측성: 파이프라인 단계별 구조화 로그.

3단계(표 매칭) 검색/리랭킹 결과, 5단계(KOSIS 조회) 캐시 히트 여부, KOSIS API 호출
지연시간을 JSON Lines(logs/pipeline_events.jsonl, 줄마다 이벤트 하나)로 남긴다.
지금까지 각 단계는 print()로만 진행상황을 남겨서 사람이 로그를 눈으로 훑는 것 말고는
집계/분석(예: "이번 배치에서 detail_cache 히트율이 몇 %였나", "reranker 지연시간이
느려진 시점이 언제부터인가")을 할 방법이 없었다 — 이 모듈은 print()를 대체하지 않고
같은 정보를 구조화된 형태로 추가로 남긴다.

로깅 실패(디스크 꽉 참, 권한 문제 등)가 파이프라인 본 동작을 절대 막으면 안 되므로
log_event()는 어떤 예외도 삼키고 조용히 무시한다 — 관측성 도구가 관측 대상을 죽이면
안 된다는 원칙."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# 2026-08-22: 기본 위치는 프로젝트 루트/logs/pipeline_events.jsonl. 배치 실행 환경(로컬/
# 원격 GPU 서버)마다 다른 경로에 쓰고 싶으면 KOSIS_PIPELINE_LOG_PATH로 덮어쓴다.
LOG_PATH = Path(
    os.environ.get(
        "KOSIS_PIPELINE_LOG_PATH",
        str(Path(__file__).parent.parent / "logs" / "pipeline_events.jsonl"),
    )
)
# 기본은 켜짐 — 끄고 싶으면(예: 디스크 I/O조차 아끼고 싶은 매우 빠듯한 환경) 0/false로.
_ENABLED = os.environ.get("KOSIS_PIPELINE_LOGGING", "1").strip().lower() not in ("0", "false", "no")


def log_event(event_type: str, **fields) -> None:
    """event_type + 임의의 필드를 JSON Lines 한 줄로 append한다. 실패해도 예외를 올리지
    않는다 — 호출부는 이 함수가 절대 안 죽는다고 가정하고 try/except 없이 불러도 된다."""
    if not _ENABLED:
        return
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        record = {"ts": datetime.now(timezone.utc).isoformat(), "event": event_type, **fields}
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception:
        pass


class Timer:
    """with Timer() as t: ... 그 다음 t.elapsed_ms로 구간 소요시간(ms)을 읽는다."""

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        self.elapsed_ms: float = 0.0
        return self

    def __exit__(self, *exc_info) -> bool:
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000
        return False
