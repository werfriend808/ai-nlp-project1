"""agent/kosis/preload_tables.py -- tables.jsonl의 TABLE 스켈레톤(원본 필드만)을
kosis_vdb_tables_qwen에 먼저 적재한다. enrichment(ITEM/AXIS/AXIS VALUE, embedding_text,
embedding)는 이후 reembed_worker.py가 채운다(on conflict do update로 같은 행을 덮어씀).

목적: enrichment(수일 소요)를 시작하기 전에 "JSONL row count == DB TABLE row count"를
먼저 검증해서, 원본 데이터 자체의 문제(인코딩, 파싱 등)를 조기에 잡는다. 기존
enrichment/period fix 로직은 전혀 건드리지 않는다 -- 이 스크립트는 순수 적재만 한다.

사용법:
    python -m agent.kosis.preload_tables
"""
from __future__ import annotations

import json
import os

import psycopg2
import psycopg2.extras

TABLES_PATH = "agent/kosis/crawl_output/tables.jsonl"
BATCH = 5000


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
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    conn.autocommit = False

    with open(TABLES_PATH, encoding="utf-8") as f:
        lines = f.readlines()
    total = len(lines)
    print(f"tables.jsonl: {total}줄")

    rows = []
    for line in lines:
        d = json.loads(line)
        rows.append((
            d.get("TBL_ID"), d.get("STAT_ID"), d.get("ORG_ID"), d.get("TBL_NM"),
            d.get("SEND_DE"), "pending",
        ))

    with conn.cursor() as cur:
        for b in range(0, len(rows), BATCH):
            psycopg2.extras.execute_values(
                cur,
                """
                insert into kosis_vdb_tables_qwen
                    (table_id, stat_id, org_id, table_name, send_date, metadata_status)
                values %s
                on conflict (table_id) do nothing;
                """,
                rows[b:b + BATCH],
            )
            conn.commit()
            print(f"  preload: {min(b + BATCH, len(rows))}/{len(rows)}")

    with conn.cursor() as cur:
        cur.execute("select count(*) from kosis_vdb_tables_qwen")
        db_count = cur.fetchone()[0]
        cur.execute("select count(distinct table_id) from kosis_vdb_tables_qwen")
        db_unique = cur.fetchone()[0]

    print(f"검증: JSONL={total}  DB={db_count}  DB(unique table_id)={db_unique}")
    if db_count != total or db_unique != total:
        print("*** 불일치 발견 -- 원인 확인 필요 ***")
    else:
        print("일치 확인됨 (정상)")

    conn.close()


if __name__ == "__main__":
    main()
