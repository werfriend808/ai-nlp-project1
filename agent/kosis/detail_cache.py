"""agent/kosis/detail_cache.py — PHASE 6: KOSIS 표 상세정보(분류축 이름+값 코드맵)
lazy 캐시.

배경: build_kosis_slots()(agent/pipeline/batch_runner.py)는 표가 table_params.json
(64개 수동 카탈로그)에 없으면 무조건 None을 반환해서, VDB(28.7만 개)에서만 찾은 표는
검색은 됐어도 실제 값 조회가 불가능했다. 이 모듈은 검증 파이프라인이 최종 후보로 좁힌
표에 대해서만(Top 3~5) KOSIS API로 상세정보를 조회하고 Supabase에 캐시해서, 같은 표가
나중에 다른 claim의 후보로 다시 나오면 API 호출 없이 재사용한다.

캐시 미스가 아니라 "영구 실패"(표 자체가 없음 등)도 status로 캐시한다 — 안 그러면
죽은 표를 후보로 만날 때마다 매번 느린 API(최대 15초, 재시도 포함)를 또 두드리게 된다
(2026-08-21 실측: DT_118N_SAUP31 같은 사례).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import psycopg2
import psycopg2.extras

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

from agent.kosis.enrich_objl import ObjlFetchError, fetch_table_detail
from agent.observability import Timer, log_event

TABLE_NAME = "kosis_table_detail_cache"
# 표 구조(축/코드)는 KOSIS가 자주 바꾸는 게 아니라서 길게 잡는다 — 실패 캐시도 같은 TTL로
# 두면, 개편으로 되살아난 표를 90일 넘게 놓치지 않으면서도 재시도 낭비를 막는다.
CACHE_TTL_DAYS = 90

_conn = None


class DetailCacheUnavailableError(RuntimeError):
    """Supabase(kosis_table_detail_cache)에 연결할 수 없음."""


def _get_connection():
    global _conn
    if _conn is None or _conn.closed:
        db_url = os.environ.get("SUPABASE_DB_URL")
        if not db_url:
            raise DetailCacheUnavailableError(
                "SUPABASE_DB_URL이 없습니다. .env에 Supabase 프로젝트의 Postgres 연결 문자열을 넣으세요."
            )
        _conn = psycopg2.connect(db_url)
        _conn.autocommit = True
    return _conn


def _is_stale(fetched_at: Optional[datetime]) -> bool:
    if fetched_at is None:
        return True
    now = datetime.now(timezone.utc)
    return (now - fetched_at) > timedelta(days=CACHE_TTL_DAYS)


def _save(
    tbl_id: str,
    org_id: str,
    *,
    axis_names: Optional[list[str]],
    code_maps: Optional[dict],
    axis_num_to_name: Optional[dict],
    prd_se_list: Optional[list[str]],
    status: str,
    error_detail: Optional[str] = None,
    retry_count: int = 0,
) -> None:
    conn = _get_connection()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            insert into {TABLE_NAME}
                (tbl_id, org_id, axis_names, prd_se_list, status, error_detail, retry_count, fetched_at, updated_at)
            values (%s, %s, %s, %s, %s, %s, %s, now(), now())
            on conflict (tbl_id) do update set
                org_id = excluded.org_id,
                axis_names = excluded.axis_names,
                prd_se_list = excluded.prd_se_list,
                status = excluded.status,
                error_detail = excluded.error_detail,
                retry_count = excluded.retry_count,
                fetched_at = now(),
                updated_at = now();
            """,
            (
                tbl_id,
                org_id,
                json.dumps(
                    {"axis_names": axis_names, "code_maps": code_maps, "axis_num_to_name": axis_num_to_name},
                    ensure_ascii=False,
                )
                if axis_names is not None
                else None,
                prd_se_list,
                status,
                error_detail,
                retry_count,
            ),
        )


def get_table_detail(tbl_id: str, org_id: str, *, api_key: Optional[str] = None) -> dict:
    """표 하나의 상세정보를 캐시 우선으로 가져온다.

    반환: {"status": "ok" | "error_no_data" | "error_other", "axis_names": [...] | None,
           "code_maps": {...} | None, "axis_num_to_name": {...} | None,
           "prd_se": str | None, "from_cache": bool}
    실패해도 예외를 던지지 않는다(호출부가 status를 보고 이 후보를 verification에서
    제외할지 판단) — 단 Supabase 자체에 연결이 안 되면 DetailCacheUnavailableError.

    axis_num_to_name/prd_se(2026-08-21 추가): PHASE 7(dynamic_slot_mapping)이 찾은
    axis_name->code를 실제 KOSIS API 파라미터(objL{n}, prdSe)로 잇는 데 필요하다 — 처음
    구현 때는 axis_names/code_maps만 반환해서, 캐시가 있어도 호출부가 objL 번호나 표
    주기를 몰라 실제 조회로 못 이어졌다(2026-08-21에 발견해 같이 고침)."""
    conn = _get_connection()
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"select * from {TABLE_NAME} where tbl_id = %s", (tbl_id,))
        row = cur.fetchone()

    if row is not None and not _is_stale(row["fetched_at"]):
        if row["status"] == "ok" and row["axis_names"]:
            payload = row["axis_names"]  # jsonb -> dict (psycopg2가 자동 파싱)
            prd_se_list = row.get("prd_se_list") or []
            log_event("detail_cache", tbl_id=tbl_id, status="ok", from_cache=True)
            return {
                "status": "ok",
                "axis_names": payload.get("axis_names"),
                "code_maps": payload.get("code_maps"),
                "axis_num_to_name": payload.get("axis_num_to_name"),
                "prd_se": prd_se_list[0] if prd_se_list else None,
                "from_cache": True,
            }
        if row["status"] != "ok":
            log_event("detail_cache", tbl_id=tbl_id, status=row["status"], from_cache=True)
            return {
                "status": row["status"], "axis_names": None, "code_maps": None,
                "axis_num_to_name": None, "prd_se": None, "from_cache": True,
            }

    t_fetch = Timer()
    try:
        with t_fetch:
            detail = fetch_table_detail(org_id, tbl_id, api_key=api_key)
        _save(
            tbl_id, org_id,
            axis_names=detail["axis_names"], code_maps=detail["code_maps"],
            axis_num_to_name=detail["axis_num_to_name"],
            prd_se_list=[detail["prd_se"]], status="ok",
        )
        log_event(
            "detail_cache", tbl_id=tbl_id, status="ok", from_cache=False,
            fetch_latency_ms=round(t_fetch.elapsed_ms, 1),
        )
        return {
            "status": "ok", "axis_names": detail["axis_names"],
            "code_maps": detail["code_maps"], "axis_num_to_name": detail["axis_num_to_name"],
            "prd_se": detail["prd_se"], "from_cache": False,
        }
    except ObjlFetchError as e:
        msg = str(e)
        status = "error_no_data" if "존재하지 않습니다" in msg else "error_other"
        retry_count = (row["retry_count"] + 1) if row else 1
        _save(
            tbl_id, org_id, axis_names=None, code_maps=None, axis_num_to_name=None,
            prd_se_list=None, status=status, error_detail=msg, retry_count=retry_count,
        )
        log_event(
            "detail_cache", tbl_id=tbl_id, status=status, from_cache=False,
            fetch_latency_ms=round(t_fetch.elapsed_ms, 1), error=msg,
        )
        return {
            "status": status, "axis_names": None, "code_maps": None,
            "axis_num_to_name": None, "prd_se": None, "from_cache": False,
        }
