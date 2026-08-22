"""
notebooks/revert_golden_gold_tables.py — enrich_golden_gold_tables.py가 남긴 백업으로
kosis_vdb_tables의 text/embedding을 원상복구한다.

백업 파일엔 원본 text만 있고 embedding 벡터는 없다 — 임베딩 모델이 결정적(같은 입력엔
항상 같은 출력)이므로, 원본 text를 다시 인코딩하면 원래 있던 벡터와 동일한 값이 나온다.

사용법: python -m notebooks.revert_golden_gold_tables
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg2
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

BACKUP_PATH = Path(__file__).parent / "enrich_golden_gold_tables.bak.json"
TABLE_NAME = "kosis_vdb_tables"


def main() -> None:
    backup = json.loads(BACKUP_PATH.read_text(encoding="utf-8"))
    print(f"복구 대상 {len(backup)}개: {list(backup.keys())}")

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=1024)
    tbl_ids = list(backup.keys())
    texts = [backup[t] for t in tbl_ids]
    vectors = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)

    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor()
    for tbl_id, text, vec in zip(tbl_ids, texts, vectors):
        cur.execute(
            f"update {TABLE_NAME} set text = %s, embedding = %s::vector where tbl_id = %s",
            (text, vec.tolist(), tbl_id),
        )
    conn.commit()
    print(f"완료: {len(backup)}개 표 원복함")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
