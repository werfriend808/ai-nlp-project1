"""
agent/kosis/build_vdb_index.py — 코랩에서 만든 KOSIS 표 28만여 개 임베딩을 Supabase(pgvector)에 적재한다.

vdb_embedding_colab.ipynb가 만든 두 파일(data/vdb_embeddings.npy, data/vdb_metadata.jsonl)을
읽어서 Postgres 테이블("kosis_vdb_tables")에 넣는다.

이 테이블은 기존 64개 수동 카탈로그(table_catalog.json, keyword_search가 씀)를
대체하는 게 아니라 보조하는 용도다 — keyword_search/embedding_search(64개 카탈로그)가
신뢰도 높은 후보를 못 찾았을 때, 이 VDB에서 추가로 후보를 찾아 리랭커에 같이 넘긴다
(전부 unverified로 취급).

2026-08-18: 저장소를 Chroma에서 Supabase(pgvector)로 옮겼다. 이유:
  1) Chroma는 로컬 서버(agent/kosis/chroma_db) 전용이라 코랩이 접근할 방법이 없어서,
     코랩 경유 배치 파이프라인에서는 VDB를 매번 건너뛰어야 했다 — Supabase는 클라우드라
     어디서든 접근 가능해서 이 문제가 사라진다.
  2) 임베딩 모델도 e5-large(1024차원, 28만7천 건 기준 약 1.1GB)에서 e5-small(384차원,
     약 420MB)로 낮췄다 — Supabase 무료 티어(500MB 안팎) 용량 제한 때문. 벡터 차원은
     로드하는 .npy 파일의 실제 shape을 그대로 읽어서 테이블을 만들므로, 모델을 바꾸면
     이 스크립트가 자동으로 맞는 차원의 테이블을 만든다(단, 기존에 다른 차원으로 이미
     테이블이 있으면 직접 DROP TABLE 하고 다시 실행해야 함 — pgvector는 컬럼 차원을
     사후에 못 바꾼다).

사전 준비물: .env에 SUPABASE_DB_URL(Postgres 연결 문자열) 필요.

사용법 (프로젝트 루트에서):
    python -m agent.kosis.build_vdb_index
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import psycopg2
from psycopg2.extras import execute_values

try:
    from dotenv import find_dotenv, load_dotenv

    load_dotenv(find_dotenv(usecwd=True))
except ImportError:
    pass

EMBEDDINGS_PATH = Path(__file__).parent.parent.parent / "data" / "vdb_embeddings.npy"
METADATA_PATH = Path(__file__).parent.parent.parent / "data" / "vdb_metadata.jsonl"
TABLE_NAME = "kosis_vdb_tables"
BATCH_SIZE = 4000


def _get_connection():
    db_url = os.environ.get("SUPABASE_DB_URL")
    if not db_url:
        raise SystemExit(
            "SUPABASE_DB_URL이 없습니다. .env에 Supabase 프로젝트의 Postgres 연결 문자열을 넣으세요."
        )
    return psycopg2.connect(db_url)


def main() -> None:
    if not EMBEDDINGS_PATH.exists() or not METADATA_PATH.exists():
        raise SystemExit(
            f"{EMBEDDINGS_PATH} 또는 {METADATA_PATH}가 없습니다. "
            "vdb_embedding_colab.ipynb를 먼저 실행하고 결과를 data/ 폴더로 받아오세요."
        )

    embeddings = np.load(EMBEDDINGS_PATH)
    rows = []
    with open(METADATA_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if len(rows) != embeddings.shape[0]:
        raise SystemExit(
            f"메타데이터({len(rows)}건)와 임베딩({embeddings.shape[0]}건) 개수가 안 맞습니다. "
            "코랩에서 다시 받아오세요."
        )

    dim = embeddings.shape[1]
    print(f"임베딩 {embeddings.shape[0]}건, 차원 {dim} 로드됨")

    conn = _get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("create extension if not exists vector;")
            cur.execute(
                f"""
                create table if not exists {TABLE_NAME} (
                    tbl_id text primary key,
                    org_id text,
                    text text,
                    embedding vector({dim})
                );
                """
            )
        conn.commit()

        total = len(rows)
        with conn.cursor() as cur:
            for start in range(0, total, BATCH_SIZE):
                end = min(start + BATCH_SIZE, total)
                batch_rows = rows[start:end]
                batch_vecs = embeddings[start:end]
                values = [
                    (r["tbl_id"], r.get("org_id") or "", r["text"], vec.tolist())
                    for r, vec in zip(batch_rows, batch_vecs)
                ]
                execute_values(
                    cur,
                    f"""
                    insert into {TABLE_NAME} (tbl_id, org_id, text, embedding)
                    values %s
                    on conflict (tbl_id) do update set
                        org_id = excluded.org_id,
                        text = excluded.text,
                        embedding = excluded.embedding;
                    """,
                    values,
                    template="(%s, %s, %s, %s::vector)",
                )
                conn.commit()
                print(f"  적재 진행: {end}/{total}")

        print(f"\n[완료] '{TABLE_NAME}'에 {total}건 적재됨")

        print("HNSW 인덱스 생성 중(이미 있으면 건너뜀, 28만여 건 기준 시간이 좀 걸릴 수 있음)...")
        with conn.cursor() as cur:
            cur.execute(
                f"""
                create index if not exists {TABLE_NAME}_embedding_hnsw_idx
                on {TABLE_NAME} using hnsw (embedding vector_cosine_ops);
                """
            )
        conn.commit()

        print("적재 검증 중...")
        with conn.cursor() as cur:
            cur.execute(f"select count(*) from {TABLE_NAME};")
            count = cur.fetchone()[0]
            if count != total:
                raise SystemExit(f"❌ 검증 실패: 기대 {total}건, 실제 {count}건 — 적재가 불완전합니다.")
            cur.execute(f"select tbl_id, org_id from {TABLE_NAME} limit 3;")
            sample = cur.fetchall()
        print(f"검증 통과: count={count}, 샘플 조회 성공 {sample}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
