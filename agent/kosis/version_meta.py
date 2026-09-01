"""agent/kosis/version_meta.py — 표 버전(STAT_ID+표이름 클러스터) 메타데이터 조회.

같은 통계가 KOSIS에서 tblId를 갈아치우며 재발행되는 경우(원인 C, 근접중복·버전분화 —
예: DT_1DA7E06S 구판 vs DT_1DA7E06S_NEW 신판)를 가려내기 위한 것. 크롤 원본
(agent/kosis/crawl_output/tables.jsonl)에 STAT_ID/TBL_NM/SEND_DE가 이미 있었는데
kosis_vdb_tables 적재 시 버려졌던 걸 backfill_version_meta.py로 별도 테이블
(kosis_table_version_meta)에 복원해서 쓴다 — kosis_vdb_tables 자체는 건드리지 않는다
(287,498건에 HNSW 벡터 인덱스가 걸려있어서, 무관한 컬럼만 갱신해도 각 행이 다시 쓰이며
인덱스까지 갱신돼 실측상 감당 안 되게 느렸다, backfill_version_meta.py 참고).

실측 확인(2026-08-25): STAT_ID 단독은 같은 설문조사(예: 경제활동인구조사) 전체를
묶어서 너무 굵다(1,432개 STAT_ID가 287,498개 표에 분산 — 그룹당 평균 200개, 서로
무관한 표들까지 섞임). (STAT_ID, TBL_NM) 조합으로 묶으면 32,400개의 진짜 버전
클러스터가 깔끔하게 나온다(예: DT_1DA7E06S/DT_1DA7E06S_NEW 정확히 둘만 묶임). SEND_DE
(마지막 자료 제공일)로 그 안에서 "지금도 갱신되는 표"를 가려낸다.
"""
from __future__ import annotations

import os
import threading
from typing import Optional

import psycopg2
import psycopg2.extras

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

TABLE_NAME = "kosis_table_version_meta"

_conn = None
_LOCK = threading.RLock()
# 프로세스 수명 동안 유지되는 단순 캐시 — stat_id/tbl_nm/send_de는 크롤 시점에 고정된
# 정적 값이라(detail_cache.py의 axis_names처럼 API 응답이 바뀔 일이 없음) TTL 불필요.
_CACHE: dict[str, Optional[dict]] = {}


def _get_connection():
    global _conn
    if _conn is None or _conn.closed:
        db_url = os.environ.get("SUPABASE_DB_URL")
        if not db_url:
            return None
        _conn = psycopg2.connect(db_url)
        _conn.autocommit = True
    return _conn


def get_version_meta_batch(table_ids: list[str]) -> dict[str, dict]:
    """table_ids 각각의 {"stat_id", "tbl_nm", "send_de"}를 반환한다(캐시 우선, 미스만
    한 번에 조회). DB 연결 불가/조회 실패/컬럼이 비어있는 표는 결과 dict에서 그냥
    빠진다 — 호출부가 fail-open으로 건너뛰도록."""
    missing = [t for t in table_ids if t not in _CACHE]
    if missing:
        conn = _get_connection()
        if conn is None:
            for t in missing:
                _CACHE[t] = None
        else:
            with _LOCK, conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    f"select tbl_id, stat_id, tbl_nm, send_de from {TABLE_NAME} where tbl_id = ANY(%s)",
                    (missing,),
                )
                rows = {r["tbl_id"]: dict(r) for r in cur.fetchall()}
            for t in missing:
                row = rows.get(t)
                _CACHE[t] = row if row and row.get("stat_id") and row.get("tbl_nm") else None

    return {t: _CACHE[t] for t in table_ids if _CACHE.get(t)}
