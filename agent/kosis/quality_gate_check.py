"""agent/kosis/quality_gate_check.py -- 재구축 중 주기적(10,000건 단위) 품질 게이트.

기존 enrichment/period fix 로직은 전혀 건드리지 않는다 -- reembed_worker.py/
reembed_v2_worker.py가 이미 DB에 써넣은 결과를 읽기 전용으로 감사(audit)만 한다.

체크 항목:
  - TABLE/ITEM/AXIS/AXIS VALUE 각 컬럼 NULL 분류(EXPECTED_NULL vs 그 외)
  - JOIN 오류(구조적으로는 FK 제약이 이미 막지만, 이중 확인 차원에서 재검증)
  - 중복(PK/UNIQUE 제약이 이미 막지만, 이중 확인)
  - 이전 체크포인트 대비 NULL/오류율 급증 여부 -> anomaly

결과를 kosis_rebuild_quality_log 테이블에 남기고, 이상치 발견 시 exit code 1을 반환한다
(오케스트레이터가 이 exit code로 다음 배치를 계속할지 멈출지 판단한다).

사용법:
    python -m agent.kosis.quality_gate_check SERVER_A
    python -m agent.kosis.quality_gate_check SERVER_A --json  # 기계 판독용 출력 추가
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import psycopg2
import psycopg2.extras

QUALITY_LOG_SCHEMA = """
create table if not exists kosis_rebuild_quality_log (
    id bigserial primary key,
    server_role text not null,
    processed int not null,
    success int not null,
    genuine_success int,
    failed int not null,
    skipped int not null,
    item_rows int not null,
    axis_rows int not null,
    axis_value_rows int not null,
    table_null_json jsonb,
    item_null_json jsonb,
    axis_null_json jsonb,
    axis_value_null_json jsonb,
    join_errors int not null default 0,
    duplicate_rows int not null default 0,
    anomaly boolean not null default false,
    anomaly_reasons jsonb,
    created_at timestamptz not null default now()
);
create index if not exists idx_kosis_rebuild_quality_log_role_time
    on kosis_rebuild_quality_log (server_role, created_at);
alter table kosis_rebuild_quality_log add column if not exists genuine_success int;
"""

# EXPECTED_NULL: 코드가 애초에 채우지 않는 컬럼(20건 스모크 테스트에서 이미 확인됨) --
# 이 컬럼들의 NULL은 이상치 판정에서 제외한다.
TABLE_EXPECTED_NULL_COLS = {"topic", "classification", "survey_name", "description"}
# items.axis_id는 설계상 NULL(itmId 축은 objL 체계와 별개) -- 코드 주석에 명시된 사실.
ITEM_EXPECTED_NULL_COLS = {"axis_id"}

# 이상 판정 임계값(상대/절대 둘 다 확인 -- 표본이 작을 때 상대비율만 보면 과민반응하므로).
NULL_RATE_ABS_THRESHOLD = 0.05      # success 표 기준 5% 초과면 절대 임계 위반
NULL_RATE_JUMP_MULTIPLIER = 3.0     # 직전 체크포인트 대비 3배 이상 급증


def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def get_connection():
    return psycopg2.connect(os.environ["SUPABASE_DB_URL"])


def ensure_schema(conn):
    with conn.cursor() as cur:
        cur.execute(QUALITY_LOG_SCHEMA)
    conn.commit()


def run_check(conn, role: str) -> dict:
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        "select status, count(*) as n from kosis_reembed_checkpoint_qwen "
        "where server_role = %s group by status",
        (role,),
    )
    status_counts = {r["status"]: r["n"] for r in cur.fetchall()}
    processed = sum(status_counts.values())
    success = status_counts.get("success", 0)
    failed = status_counts.get("failed", 0)
    skipped = status_counts.get("processing", 0) + status_counts.get("pending", 0)

    # 이 role이 처리한 table_id만 대상으로 집계(다른 role/과거 테스트 데이터와 섞이지 않도록).
    # 주의: checkpoint.status='success'는 "처리 시도가 끝났다"는 뜻이라 excluded_too_large/
    # error_*도 여기 포함된다(reembed_worker_fast.flush_batch가 결과와 무관하게 항상
    # checkpoint를 success로 찍음). unit/period 같은 TABLE 컬럼의 NULL 비율은 반드시
    # kosis_vdb_tables_qwen.metadata_status='success'(진짜 enrichment 성공)로만 걸러서
    # 계산해야 한다 -- 안 그러면 excluded_too_large(에는 unit/period가 항상 NULL)가
    # 섞여 들어가 NULL 비율이 실제보다 과대평가된다(2026-08-27 실측: 8% excluded 섞이니
    # unit NULL 3.76% -> 11.2%로 뻥튀기됨, 오탐으로 파이프라인이 반복 정지했었음).
    cur.execute(
        "select table_id from kosis_reembed_checkpoint_qwen where server_role=%s and status='success'",
        (role,),
    )
    success_ids = [r["table_id"] for r in cur.fetchall()]

    cur.execute(
        "select t.table_id from kosis_vdb_tables_qwen t "
        "join kosis_reembed_checkpoint_qwen c on c.table_id = t.table_id "
        "where c.server_role=%s and c.status='success' and t.metadata_status='success'",
        (role,),
    )
    genuine_success_ids = [r["table_id"] for r in cur.fetchall()]

    if not success_ids:
        return {
            "server_role": role, "processed": processed, "success": 0, "genuine_success": 0,
            "failed": failed, "skipped": skipped, "item_rows": 0, "axis_rows": 0, "axis_value_rows": 0,
            "table_null": {}, "item_null": {}, "axis_null": {}, "axis_value_null": {},
            "join_errors": 0, "duplicate_rows": 0, "anomaly": False, "anomaly_reasons": [],
        }

    cur.execute(
        "select count(*) from kosis_vdb_items_qwen where table_id = any(%s)", (success_ids,)
    )
    item_rows = cur.fetchone()["count"]
    cur.execute(
        "select count(*) from kosis_vdb_axes_qwen where table_id = any(%s)", (success_ids,)
    )
    axis_rows = cur.fetchone()["count"]
    cur.execute(
        "select count(*) from kosis_vdb_axis_values_qwen where table_id = any(%s)", (success_ids,)
    )
    axis_value_rows = cur.fetchone()["count"]

    genuine_success = len(genuine_success_ids)
    table_cols = ["institution_name", "table_name", "topic", "classification", "survey_name",
                  "description", "unit", "period_start", "period_end", "embedding_text", "embedding"]
    table_null = {}
    with conn.cursor() as c2:
        for col in table_cols:
            c2.execute(
                f"select count(*) from kosis_vdb_tables_qwen where {col} is null and table_id = any(%s)",
                (genuine_success_ids,),
            )
            table_null[col] = c2.fetchone()[0]

    item_cols = ["item_id", "item_name", "axis_id"]
    item_null = {}
    with conn.cursor() as c2:
        for col in item_cols:
            c2.execute(
                f"select count(*) from kosis_vdb_items_qwen where {col} is null and table_id = any(%s)",
                (success_ids,),
            )
            item_null[col] = c2.fetchone()[0]

    axis_cols = ["axis_id", "axis_name", "axis_order"]
    axis_null = {}
    with conn.cursor() as c2:
        for col in axis_cols:
            c2.execute(
                f"select count(*) from kosis_vdb_axes_qwen where {col} is null and table_id = any(%s)",
                (success_ids,),
            )
            axis_null[col] = c2.fetchone()[0]

    axis_value_cols = ["value_id", "value_name", "code"]
    axis_value_null = {}
    with conn.cursor() as c2:
        for col in axis_value_cols:
            c2.execute(
                f"select count(*) from kosis_vdb_axis_values_qwen where {col} is null and table_id = any(%s)",
                (success_ids,),
            )
            axis_value_null[col] = c2.fetchone()[0]

    # JOIN 오류: FK 제약이 이미 구조적으로 막지만(orphan INSERT 자체가 불가능), 이중 확인.
    with conn.cursor() as c2:
        c2.execute(
            "select count(*) from kosis_vdb_items_qwen i "
            "left join kosis_vdb_tables_qwen t on i.table_id = t.table_id where t.table_id is null"
        )
        join_err_items = c2.fetchone()[0]
        c2.execute(
            "select count(*) from kosis_vdb_axes_qwen a "
            "left join kosis_vdb_tables_qwen t on a.table_id = t.table_id where t.table_id is null"
        )
        join_err_axes = c2.fetchone()[0]
        c2.execute(
            "select count(*) from kosis_vdb_axis_values_qwen v "
            "left join kosis_vdb_axes_qwen a on v.table_id = a.table_id and v.axis_id = a.axis_id "
            "where a.table_id is null"
        )
        join_err_values = c2.fetchone()[0]
    join_errors = join_err_items + join_err_axes + join_err_values

    # 중복: PK/UNIQUE 제약이 이미 막지만, 이중 확인(그룹핑으로 재검증).
    with conn.cursor() as c2:
        c2.execute("select table_id, count(*) c from kosis_vdb_tables_qwen group by table_id having count(*) > 1")
        dup_tables = c2.fetchall()
        c2.execute("select table_id, item_id, count(*) c from kosis_vdb_items_qwen group by table_id, item_id having count(*) > 1")
        dup_items = c2.fetchall()
    duplicate_rows = len(dup_tables) + len(dup_items)

    anomaly_reasons = []
    if join_errors > 0:
        anomaly_reasons.append(f"JOIN 오류 {join_errors}건 (FK 제약을 우회한 비정상 상태 -- 즉시 조사 필요)")
    if duplicate_rows > 0:
        anomaly_reasons.append(f"중복 {duplicate_rows}건 (PK/UNIQUE 제약을 우회한 비정상 상태 -- 즉시 조사 필요)")

    for col, n in table_null.items():
        if col in TABLE_EXPECTED_NULL_COLS:
            continue
        rate = n / genuine_success if genuine_success else 0
        if rate > NULL_RATE_ABS_THRESHOLD:
            anomaly_reasons.append(f"TABLE.{col} NULL 비율 {rate:.1%} (임계 {NULL_RATE_ABS_THRESHOLD:.0%} 초과, 진짜 success {genuine_success}건 기준)")

    # 직전 체크포인트 대비 급증 여부
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as c2:
        c2.execute(
            "select * from kosis_rebuild_quality_log where server_role=%s "
            "order by created_at desc limit 1",
            (role,),
        )
        prev = c2.fetchone()

    if prev:
        prev_genuine_success = prev.get("genuine_success") or prev["success"] or 1
        for col, n in table_null.items():
            if col in TABLE_EXPECTED_NULL_COLS:
                continue
            prev_n = (prev["table_null_json"] or {}).get(col, 0)
            prev_rate = prev_n / prev_genuine_success
            cur_rate = n / genuine_success if genuine_success else 0
            if prev_rate > 0 and cur_rate > prev_rate * NULL_RATE_JUMP_MULTIPLIER:
                anomaly_reasons.append(
                    f"TABLE.{col} NULL 비율 급증: {prev_rate:.2%} -> {cur_rate:.2%} "
                    f"({NULL_RATE_JUMP_MULTIPLIER}배 이상)"
                )
        prev_failed_rate = (prev["failed"] or 0) / (prev["processed"] or 1)
        cur_failed_rate = failed / processed if processed else 0
        if prev_failed_rate > 0 and cur_failed_rate > prev_failed_rate * NULL_RATE_JUMP_MULTIPLIER:
            anomaly_reasons.append(
                f"failed 비율 급증: {prev_failed_rate:.2%} -> {cur_failed_rate:.2%}"
            )

    return {
        "server_role": role, "processed": processed, "success": success,
        "genuine_success": genuine_success, "failed": failed,
        "skipped": skipped, "item_rows": item_rows, "axis_rows": axis_rows,
        "axis_value_rows": axis_value_rows, "table_null": table_null, "item_null": item_null,
        "axis_null": axis_null, "axis_value_null": axis_value_null, "join_errors": join_errors,
        "duplicate_rows": duplicate_rows, "anomaly": len(anomaly_reasons) > 0,
        "anomaly_reasons": anomaly_reasons,
    }


def save_log(conn, r: dict):
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into kosis_rebuild_quality_log
                (server_role, processed, success, genuine_success, failed, skipped, item_rows, axis_rows,
                 axis_value_rows, table_null_json, item_null_json, axis_null_json,
                 axis_value_null_json, join_errors, duplicate_rows, anomaly, anomaly_reasons)
            values (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                r["server_role"], r["processed"], r["success"], r.get("genuine_success"),
                r["failed"], r["skipped"],
                r["item_rows"], r["axis_rows"], r["axis_value_rows"],
                json.dumps(r["table_null"]), json.dumps(r["item_null"]),
                json.dumps(r["axis_null"]), json.dumps(r["axis_value_null"]),
                r["join_errors"], r["duplicate_rows"], r["anomaly"],
                json.dumps(r["anomaly_reasons"], ensure_ascii=False),
            ),
        )
    conn.commit()


def print_report(r: dict):
    print(f"=== 품질 게이트: {r['server_role']} (processed={r['processed']}) ===")
    print(f"success={r['success']} (진짜 enrichment success={r.get('genuine_success')}) "
          f"failed={r['failed']} skipped/pending={r['skipped']}")
    print(f"item_rows={r['item_rows']} axis_rows={r['axis_rows']} axis_value_rows={r['axis_value_rows']}")
    print("TABLE NULL:", r["table_null"])
    print("ITEM NULL:", r["item_null"])
    print("AXIS NULL:", r["axis_null"])
    print("AXIS VALUE NULL:", r["axis_value_null"])
    print(f"JOIN 오류: {r['join_errors']}  중복: {r['duplicate_rows']}")
    if r["anomaly"]:
        print("*** ANOMALY DETECTED ***")
        for reason in r["anomaly_reasons"]:
            print("  -", reason)
    else:
        print("이상 없음 (정상)")


def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["SERVER_A", "SERVER_B"])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    conn = get_connection()
    ensure_schema(conn)
    r = run_check(conn, args.role)
    save_log(conn, r)
    print_report(r)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, default=str))
    conn.close()
    sys.exit(1 if r["anomaly"] else 0)


if __name__ == "__main__":
    main()
