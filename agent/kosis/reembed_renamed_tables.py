"""표명이 바뀐 표만 embedding_text 재생성 + 재임베딩한다.

fetch_korean_table_names.py가 영문 표명을 한글로 바꾼 뒤 쓰는 후속 단계.
reembed_v2_worker.py를 그대로 쓰지 않는 이유는 하나뿐이다 — 그 워커는 item/axis/value가
하나도 없는 표를 "텍스트가 어차피 안 바뀐다"고 보고 skipped 처리하는데, 지금은 표명 자체가
바뀌었으므로 그 표들도 반드시 다시 만들어야 한다(2,824건 중 802건이 여기 해당).

텍스트 생성 규칙·모델·차원은 운영 워커에서 그대로 가져다 쓴다.
되돌리려면 kosis_english_name_backup 테이블을 쓴다.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from agent.kosis.embedding_text import build_experimental_text
from agent.kosis.reembed_v2_worker import (
    EMBED_DIM, EMBED_MODEL_NAME, batch_update, encode_with_fallback, fetch_metadata_batch,
)

CHUNK = 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids-file", required=True)
    args = ap.parse_args()

    ids = json.loads(Path(args.ids_file).read_text(encoding="utf-8"))
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    print(f"재임베딩 대상 {len(ids):,}건")

    from sentence_transformers import SentenceTransformer

    print(f"{EMBED_MODEL_NAME} 로딩 중 (truncate_dim={EMBED_DIM})...", flush=True)
    model = SentenceTransformer(EMBED_MODEL_NAME, truncate_dim=EMBED_DIM, device="cuda")
    print("로딩 완료.", flush=True)

    t0 = time.time()
    done = 0
    for i in range(0, len(ids), CHUNK):
        chunk = ids[i:i + CHUNK]
        meta = fetch_metadata_batch(conn, chunk)
        texts, tids = [], []
        for tid in chunk:
            m = meta.get(tid)
            if not m:
                continue
            tids.append(tid)
            texts.append(build_experimental_text(m, "item_axis_value_capped", None))
        vecs, _ = encode_with_fallback(model, texts)
        batch_update(conn, [(t, x, v.tolist()) for t, x, v in zip(tids, texts, vecs)])
        done += len(tids)
        el = time.time() - t0
        print(f"  {done}/{len(ids)}  {done/el:.1f}행/s  ETA {(len(ids)-done)/max(done/el,0.01)/60:.1f}분",
              flush=True)

    print(f"\n완료: {done:,}건, {(time.time()-t0)/60:.1f}분")
    conn.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
