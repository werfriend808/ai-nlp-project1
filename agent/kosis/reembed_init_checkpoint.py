"""agent/kosis/reembed_init_checkpoint.py — kosis_reembed_checkpoint_qwen을 이 서버가
담당할 partition(tables.jsonl의 절반)으로 초기화한다. 이미 채워져 있으면 건너뛴다(멱등).

사용법:
    python -m agent.kosis.reembed_init_checkpoint SERVER_A
    python -m agent.kosis.reembed_init_checkpoint SERVER_B
"""
from __future__ import annotations

import json
import os
import sys

import psycopg2
from psycopg2.extras import execute_values

TABLES_PATH = "agent/kosis/crawl_output/tables.jsonl"


def _load_env():
    env_path = os.path.join(os.path.dirname(__file__), "..", "..", ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def main():
    _load_env()
    role = sys.argv[1] if len(sys.argv) > 1 else None
    if role not in ("SERVER_A", "SERVER_B"):
        raise SystemExit("usage: reembed_init_checkpoint.py SERVER_A|SERVER_B")

    with open(TABLES_PATH) as f:
        lines = f.readlines()
    total = len(lines)
    half = total // 2
    if role == "SERVER_A":
        start, end = 0, half - 1
    else:
        start, end = half, total - 1

    db_url = os.environ["SUPABASE_DB_URL"]
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "select count(*) from kosis_reembed_checkpoint_qwen where server_role = %s",
                (role,),
            )
            existing = cur.fetchone()[0]
        if existing > 0:
            print(f"{role}: checkpoint already has {existing} rows — skipping init (idempotent).")
            return

        rows = []
        for i in range(start, end + 1):
            rec = json.loads(lines[i])
            tbl_id = rec.get("TBL_ID")
            rows.append((tbl_id, role, i, "pending"))

        with conn.cursor() as cur:
            for b in range(0, len(rows), 5000):
                execute_values(
                    cur,
                    """
                    insert into kosis_reembed_checkpoint_qwen (table_id, server_role, line_no, status)
                    values %s
                    on conflict (table_id) do nothing;
                    """,
                    rows[b:b + 5000],
                )
                conn.commit()
                print(f"  checkpoint init: {min(b+5000, len(rows))}/{len(rows)}")
        print(f"{role}: initialized {len(rows)} checkpoint rows (line {start}..{end}).")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
