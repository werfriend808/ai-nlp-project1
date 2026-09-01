"""agent/kosis/backfill_version_meta.py — 표 버전(STAT_ID+표이름) 메타데이터를 별도
테이블(kosis_table_version_meta)로 적재하는 1회성 스크립트.

배경: build_vdb_index.py가 kosis_vdb_tables를 만들 때 tbl_id/org_id/text/embedding만
저장하고 STAT_ID/TBL_NM/SEND_DE는 버렸다. 이 세 값은 표 버전 중복(원인 C) 탐지에
그대로 쓸 수 있는데(agent/kosis/version_meta.py 참고), 다시 KOSIS API를 호출할 필요 없이
이미 크롤해둔 tables.jsonl에 다 들어있다.

⚠️ 처음엔 kosis_vdb_tables에 컬럼 3개를 ALTER TABLE로 추가하는 방식으로 짰다가 실측에서
막혔다(2026-08-25) — 이 테이블은 287,498건에 HNSW 벡터 인덱스가 걸려있어서, embedding과
무관한 컬럼만 UPDATE해도 Postgres가 각 행을 새 버전으로 다시 쓰면서 HNSW 인덱스까지
매번 갱신해야 해서 5,000행 배치 하나에 수 분씩 걸렸다(15,000/287,498에서 사실상 멈춤).
그래서 벡터 인덱스와 무관한 완전히 별도의 작은 테이블로 분리했다 — 인덱스 갱신 부담이
없어서 훨씬 빠르다.

사용법 (프로젝트 루트에서):
    python -m agent.kosis.backfill_version_meta            # 실제 반영
    python -m agent.kosis.backfill_version_meta --dry-run  # DB 안 바꾸고 매칭 통계만 출력
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

CRAWL_PATH = Path(__file__).parent / "crawl_output" / "tables.jsonl"
TABLE_NAME = "kosis_table_version_meta"
BATCH_SIZE = 10000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not CRAWL_PATH.exists():
        raise SystemExit(f"{CRAWL_PATH}가 없습니다.")

    rows = []
    with open(CRAWL_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            tbl_id = d.get("TBL_ID")
            if not tbl_id:
                continue
            rows.append((tbl_id, d.get("STAT_ID"), d.get("TBL_NM"), d.get("SEND_DE")))
    print(f"crawl_output/tables.jsonl에서 {len(rows)}건 로드")

    if args.dry_run:
        with_stat = sum(1 for r in rows if r[1])
        with_name = sum(1 for r in rows if r[2])
        with_date = sum(1 for r in rows if r[3])
        print(f"[dry-run] stat_id 있음: {with_stat}, tbl_nm 있음: {with_name}, send_de 있음: {with_date}")
        return

    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise SystemExit("SUPABASE_DB_URL이 없습니다.")
    conn = psycopg2.connect(db_url)
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                create table if not exists {TABLE_NAME} (
                    tbl_id text primary key,
                    stat_id text,
                    tbl_nm text,
                    send_de text
                );
                """
            )
        conn.commit()
        print(f"'{TABLE_NAME}' 테이블 준비 완료")

        total = len(rows)
        done = 0
        with conn.cursor() as cur:
            for start in range(0, total, BATCH_SIZE):
                batch = rows[start:start + BATCH_SIZE]
                execute_values(
                    cur,
                    f"""
                    insert into {TABLE_NAME} (tbl_id, stat_id, tbl_nm, send_de)
                    values %s
                    on conflict (tbl_id) do update set
                        stat_id = excluded.stat_id,
                        tbl_nm = excluded.tbl_nm,
                        send_de = excluded.send_de;
                    """,
                    batch,
                )
                conn.commit()
                done += len(batch)
                print(f"  백필 진행: {done}/{total}")

        with conn.cursor() as cur:
            cur.execute(f"select count(*) from {TABLE_NAME}")
            filled = cur.fetchone()[0]
        print(f"\n[완료] '{TABLE_NAME}' 총 행: {filled}/{total}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
