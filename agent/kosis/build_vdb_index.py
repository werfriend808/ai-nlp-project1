"""
agent/kosis/build_vdb_index.py — 코랩에서 만든 KOSIS 표 28만여 개 임베딩을 Supabase(pgvector)에 적재한다.

vdb_embedding_colab.ipynb가 만든 두 파일(data/vdb_embeddings.npy, data/vdb_metadata.jsonl)을
읽어서 Postgres 테이블("kosis_vdb_tables")에 넣는다.

이 테이블은 기존 64개 수동 카탈로그(table_catalog.json, keyword_search가 씀)를
대체하는 게 아니라 보조하는 용도다 — keyword_search/embedding_search(64개 카탈로그)가
신뢰도 높은 후보(verified/64개 카탈로그의 unverified)를 못 찾았을 때, 이 VDB에서
추가로 후보를 찾아 리랭커에 같이 넘긴다(전부 unverified로 취급). 이 통합 로직 자체는
아직 별도 작업(reranker.py 쪽 수정)으로 남아있다 — 이 스크립트는 "인덱스를 만드는" 것까지만
담당한다.

2026-08-15 실측: 처음엔 PersistentClient(임베디드 모드)로 넣었는데, 28만7천 건 규모에서
매번 재현되는 버그가 확인됐다 — 적재+같은 프로세스 내 재조회(count/get)까지는 성공하는데,
새 프로세스로 다시 열면 "Error loading hnsw index"로 완전히 못 읽는다. client.close() +
대기를 추가해도 동일하게 재발했다. chromadb 깃허브 이슈로 검색해보니 대규모 컬렉션에서
HNSW 인덱스가 SQLite 임베딩 테이블과 개수가 안 맞아 깨지는 유형의 알려진 문제였다 —
백그라운드 compactor 스레드가 인덱스를 비동기로 디스크에 쓰는데, 이 컴퓨터 환경(RAM
7.4GB)에서 대량 upsert 도중 압축이 끝까지 못 끝나는 것으로 추정된다(구버전 chromadb는
Visual C++ 빌드 도구가 없어 설치 자체가 안 돼서 대안이 안 됐다).

그래서 PersistentClient(임베디드) 대신 `chroma run --path agent/kosis/chroma_db --port 8100`
으로 띄운 로컬 서버에 HttpClient로 접속해서 적재한다 — 서버 프로세스가 계속 살아있으니
"적재 프로세스가 압축 도중에 끝나버리는" 문제 자체가 구조적으로 없어진다.

사용법 (프로젝트 루트에서):
    python -m agent.kosis.build_vdb_index
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
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
CHROMA_HOST = "localhost"
CHROMA_PORT = 8100


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

    print(f"임베딩 {embeddings.shape[0]}건, 차원 {embeddings.shape[1]} 로드됨")

    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
    except Exception as e:
        raise SystemExit(
            f"Chroma 서버({CHROMA_HOST}:{CHROMA_PORT})에 연결 못 했습니다: {e}\n"
            f"먼저 'chroma run --path agent/kosis/chroma_db --port {CHROMA_PORT}'로 서버를 띄우세요."
        )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

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
