"""agent/kosis/reembed_v2_worker.py -- item_axis_value_capped 방식으로 TABLE
embedding_text/embedding만 재생성한다(ITEM/AXIS/AXIS VALUE 자체의 embedding은 안 만듦).

KOSIS API 호출 0회 -- 현재 DB(kosis_vdb_tables_qwen/items/axes/axis_values)에 이미
있는 데이터만 table_id로 JOIN해서 사용한다. embedding_text 생성 로직은
agent/kosis/embedding_text.py의 build_experimental_text(mode=
"item_axis_value_capped")를 그대로 재사용한다(새 포맷 만들지 않음).

resume 가능: kosis_reembed_v2_checkpoint_qwen(table_id, server_role, status)로
관리 -- 이미 status='success'인 table_id는 다시 처리하지 않는다.

DB UPDATE는 execute_values 기반 배치 UPDATE(FROM VALUES 조인)만 쓴다 -- 1건씩
UPDATE 금지.

사용법 (프로젝트 루트, venv 활성화 후):
    python -m agent.kosis.reembed_v2_worker SERVER_A [--limit N] [--batch-size 500]
"""
from __future__ import annotations

import argparse
import os
import time

import psycopg2
import psycopg2.extras

from agent.kosis.embedding_text import build_experimental_text

EMBED_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"
EMBED_DIM = 2560
GPU_BATCH_SIZE = 32
DB_BATCH_SIZE = 500


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


def load_pending(conn, role: str, limit: int | None):
    with conn.cursor() as cur:
        q = """
            select table_id from kosis_reembed_v2_checkpoint_qwen
            where server_role = %s and status not in ('success', 'skipped')
            order by table_id
        """
        if limit:
            q += " limit %s"
            cur.execute(q, (role, limit))
        else:
            cur.execute(q, (role,))
        return [r[0] for r in cur.fetchall()]


def fetch_metadata_batch(conn, table_ids: list[str]) -> dict[str, dict]:
    with conn.cursor() as cur:
        cur.execute(
            "select table_id, institution_name, table_name, embedding_text as old_text "
            "from kosis_vdb_tables_qwen where table_id = ANY(%s);",
            (table_ids,),
        )
        base = {r[0]: {"institution_name": r[1], "table_name": r[2], "old_text": r[3]} for r in cur.fetchall()}

        # 2026-08-27: ORDER BY 없이 읽으면 순서가 비결정적이고, 50개 cap이 엉뚱한 50개를
        # 고른다. id(bigserial)가 삽입 순서 = 원래 API 응답 순서를 보존하므로 그 순서로
        # 읽어야 나머지 27.9만 건(reembed_worker_fast가 만든 것)과 동일한 텍스트가 나온다.
        cur.execute("select table_id, item_name from kosis_vdb_items_qwen where table_id = ANY(%s) order by id;", (table_ids,))
        items: dict[str, list[str]] = {}
        for tid, name in cur.fetchall():
            if name:
                items.setdefault(tid, [])
                if name not in items[tid]:
                    items[tid].append(name)

        cur.execute(
            "select table_id, axis_name from kosis_vdb_axes_qwen where table_id = ANY(%s) order by table_id, axis_order;",
            (table_ids,),
        )
        axes: dict[str, list[str]] = {}
        for tid, name in cur.fetchall():
            if name:
                axes.setdefault(tid, [])
                if name not in axes[tid]:
                    axes[tid].append(name)

        # 분류값도 동일 -- id 순이면 축1 값들 -> 축2 값들 순으로 원래 그룹핑이 복원된다.
        cur.execute("select table_id, value_name from kosis_vdb_axis_values_qwen where table_id = ANY(%s) order by id;", (table_ids,))
        values_raw: dict[str, list[str]] = {}
        for tid, name in cur.fetchall():
            if name:
                values_raw.setdefault(tid, []).append(name)

    for tid in table_ids:
        base.setdefault(tid, {"institution_name": None, "table_name": None, "old_text": None})
        base[tid]["items"] = items.get(tid, [])
        base[tid]["axes"] = axes.get(tid, [])
        base[tid]["values_dedup"] = list(dict.fromkeys(values_raw.get(tid, [])))
    return base


def encode_with_fallback(model, texts: list[str]):
    """32 -> 16 -> 8 순서로 OOM 시 배치를 줄여 재시도(다른 GPU 작업과 동시 실행 대비)."""
    import torch

    for bs in (GPU_BATCH_SIZE, 16, 8):
        try:
            return model.encode(texts, batch_size=bs, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False), bs
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  [경고] batch_size={bs} OOM -- 더 작은 배치로 재시도", flush=True)
    raise RuntimeError("batch_size=8에서도 OOM -- GPU 여유 공간 부족")


def batch_update(conn, rows: list[tuple[str, str, list[float]]]):
    """rows: [(table_id, new_text, new_vec), ...] -- FROM VALUES 조인으로 한 번에 UPDATE."""
    if not rows:
        return
    with conn.cursor() as cur:
        psycopg2.extras.execute_values(
            cur,
            """
            update kosis_vdb_tables_qwen as t
            set embedding_text = v.embedding_text,
                embedding = v.embedding::vector,
                updated_at = now()
            from (values %s) as v(table_id, embedding_text, embedding)
            where t.table_id = v.table_id;
            """,
            rows,
            template="(%s, %s, %s::vector)",
        )
    conn.commit()


def mark_checkpoint_batch(conn, ids: list[str], status: str, error: str | None = None):
    if not ids:
        return
    with conn.cursor() as cur:
        cur.execute(
            "update kosis_reembed_v2_checkpoint_qwen set status=%s, error_message=%s, updated_at=now() "
            "where table_id = any(%s);",
            (status, error, ids),
        )
    conn.commit()


def process_chunk(conn, model, table_ids: list[str]) -> dict:
    """table_ids(최대 DB_BATCH_SIZE개)를 조회->텍스트 생성->GPU 인코딩->배치 UPDATE까지 처리."""
    meta = fetch_metadata_batch(conn, table_ids)

    unchanged_ids = []   # ITEM/AXIS/VALUE가 전혀 없어 텍스트가 기존과 동일 -> 재처리 불필요
    to_embed_ids, to_embed_texts = [], []
    for tid in table_ids:
        m = meta[tid]
        new_text = build_experimental_text(m, "item_axis_value_capped", None)
        if not m["items"] and not m["axes"] and not m["values_dedup"]:
            unchanged_ids.append(tid)
            continue
        to_embed_ids.append(tid)
        to_embed_texts.append(new_text)

    used_bs = None
    if to_embed_texts:
        vecs, used_bs = encode_with_fallback(model, to_embed_texts)
        update_rows = [(tid, txt, vec.tolist()) for tid, txt, vec in zip(to_embed_ids, to_embed_texts, vecs)]
        batch_update(conn, update_rows)
        mark_checkpoint_batch(conn, to_embed_ids, "success")

    mark_checkpoint_batch(conn, unchanged_ids, "skipped")
    return {"embedded": len(to_embed_ids), "skipped": len(unchanged_ids), "batch_size": used_bs}


def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["SERVER_A", "SERVER_B"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=DB_BATCH_SIZE)
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")
    from sentence_transformers import SentenceTransformer

    conn = get_connection()
    pending = load_pending(conn, args.role, args.limit)
    print(f"[{args.role}] 처리 대상 {len(pending)}건", flush=True)
    if not pending:
        print("처리할 표가 없습니다 (전부 success/skipped 이거나 partition이 비어 있음).")
        return

    print(f"{EMBED_MODEL_NAME} 로딩 중 (truncate_dim={EMBED_DIM})...", flush=True)
    model = SentenceTransformer(EMBED_MODEL_NAME, truncate_dim=EMBED_DIM, device="cuda")
    print("모델 로딩 완료.", flush=True)

    n_done = n_embedded = n_skipped = 0
    t0 = time.time()
    for i in range(0, len(pending), args.batch_size):
        chunk = pending[i:i + args.batch_size]
        try:
            r = process_chunk(conn, model, chunk)
        except Exception as e:
            print(f"  [오류] 청크 처리 실패({len(chunk)}건): {e}", flush=True)
            mark_checkpoint_batch(conn, chunk, "failed", str(e)[:500])
            continue

        n_done += len(chunk)
        n_embedded += r["embedded"]
        n_skipped += r["skipped"]
        elapsed = time.time() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        remaining = len(pending) - n_done
        eta_sec = remaining / rate if rate > 0 else float("inf")
        print(f"[{args.role}] {n_done}/{len(pending)} embedded={n_embedded} skipped={n_skipped} "
              f"batch_size={r['batch_size']} rate={rate:.2f}행/s ETA={eta_sec/3600:.2f}h", flush=True)

    conn.close()
    elapsed = time.time() - t0
    print(f"[{args.role}] 완료. 총 {n_done}건, embedded={n_embedded}, skipped={n_skipped}, "
          f"elapsed={elapsed/3600:.2f}h", flush=True)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
