"""agent/kosis/reembed_fix_errors.py — error_no_data/error_other로 남은 표를 최종
fix(n_axes 순서/err=20 처리/D·H·F·IR 주기 코드/1900년 확장)가 다 반영된 코드로
재검증한다.

kosis_reembed_errorfix_checkpoint_qwen(table_id, server_role, status)로 SERVER_A/
SERVER_B가 절반씩 나눠 처리, resume 가능. reembed_worker.py의 fetch_table_enrichment/
flush_batch를 그대로 재사용해서 ITEM/AXIS/AXIS VALUE/embedding 로직은 완전히 동일하게
유지한다 — 이 스크립트는 "무엇을 재검증할지"만 다르게 고른다.

사용법 (프로젝트 루트, venv 활성화 후):
    python -m agent.kosis.reembed_fix_errors SERVER_A [--limit N] [--concurrency 30] [--api-keys k1,k2]
"""
from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from agent.kosis.reembed_worker import (
    CHUNK_SIZE,
    EMBED_DIM,
    EMBED_MODEL_NAME,
    _load_env,
    flush_batch,
    get_connection,
    load_org_whitelist,
    process_one,
)


def load_targets(conn, role: str, limit: int | None):
    with conn.cursor() as cur:
        q = """
            select c.table_id, t.org_id, t.table_name, t.stat_id, t.send_date
            from kosis_reembed_errorfix_checkpoint_qwen c
            join kosis_vdb_tables_qwen t on t.table_id = c.table_id
            where c.server_role = %s and c.status not in ('success', 'still_error')
            order by c.table_id
        """
        if limit:
            q += " limit %s"
            cur.execute(q, (role, limit))
        else:
            cur.execute(q, (role,))
        return cur.fetchall()


def mark_errorfix(conn, table_id: str, status: str, error: str | None = None):
    with conn.cursor() as cur:
        cur.execute(
            """
            update kosis_reembed_errorfix_checkpoint_qwen
            set status = %s, error_message = %s, updated_at = now()
            where table_id = %s;
            """,
            (status, error, table_id),
        )


def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("role", choices=["SERVER_A", "SERVER_B"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=30)
    ap.add_argument("--api-keys", type=str, default=None)
    args = ap.parse_args()

    if args.api_keys:
        api_keys = [k.strip() for k in args.api_keys.split(",") if k.strip()]
    else:
        api_keys = [os.environ["KOSIS_API_KEY"]]
    print(f"KOSIS API 키 {len(api_keys)}개 사용", flush=True)

    org_whitelist = load_org_whitelist()

    print("모델 로딩 중 (Qwen3-Embedding-4B, dim=2560)...", flush=True)
    os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL_NAME, truncate_dim=EMBED_DIM, device="cuda")
    print("모델 로딩 완료.", flush=True)

    conn = get_connection()
    targets = load_targets(conn, args.role, args.limit)
    print(f"[{args.role}] 재처리 대상 {len(targets)}건", flush=True)
    if not targets:
        print("재처리할 표가 없습니다.")
        return

    with conn.cursor() as cur:
        ids = [t[0] for t in targets]
        for b in range(0, len(ids), 5000):
            cur.execute(
                "update kosis_reembed_errorfix_checkpoint_qwen set status='processing', updated_at=now() where table_id = any(%s)",
                (ids[b:b + 5000],),
            )
    conn.commit()

    t0 = time.time()
    n_done = n_recovered = n_still_error = n_failed = 0
    batch = []

    ex = ThreadPoolExecutor(max_workers=args.concurrency)
    aborted = False
    try:
        futs = {}
        for i, (table_id, org_id, table_name, stat_id, send_date) in enumerate(targets):
            key = api_keys[i % len(api_keys)]
            fut = ex.submit(process_one, 0, org_id, table_id, stat_id, table_name, send_date, key)
            futs[fut] = table_id

        for fut in as_completed(futs):
            table_id = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"tbl_id": table_id, "org_id": None, "tbl_nm": None, "stat_id": None,
                     "send_de": None, "enrichment": {"status": "error_other", "error": str(e)}}
            batch.append(r)

            if len(batch) >= CHUNK_SIZE:
                s, f = flush_batch(conn, model, org_whitelist, batch, args.role)
                for r2 in batch:
                    recovered = r2["enrichment"]["status"] == "success"
                    mark_errorfix(conn, r2["tbl_id"], "success" if recovered else "still_error",
                                  r2["enrichment"].get("error"))
                    if recovered:
                        n_recovered += 1
                    else:
                        n_still_error += 1
                conn.commit()
                n_failed += f
                n_done += len(batch)
                batch = []

                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = len(targets) - n_done
                eta = remaining / rate if rate > 0 else float("inf")
                print(
                    f"[{args.role}] {n_done}/{len(targets)} 복구={n_recovered} "
                    f"여전히실패={n_still_error} rate={rate:.2f}/s ETA={eta/3600:.1f}h",
                    flush=True,
                )
                if f > len(batch) * 0.5 and n_done < 1000:
                    print("DB insert 실패율이 높음 — 안전하게 중단합니다.", flush=True)
                    aborted = True
                    break

        if batch and not aborted:
            s, f = flush_batch(conn, model, org_whitelist, batch, args.role)
            for r2 in batch:
                recovered = r2["enrichment"]["status"] == "success"
                mark_errorfix(conn, r2["tbl_id"], "success" if recovered else "still_error",
                              r2["enrichment"].get("error"))
                if recovered:
                    n_recovered += 1
                else:
                    n_still_error += 1
            conn.commit()
            n_done += len(batch)
    finally:
        ex.shutdown(wait=False, cancel_futures=True)

    conn.close()
    elapsed = time.time() - t0
    print(f"[{args.role}] 완료. 총 {n_done}건, 복구={n_recovered}, 여전히실패={n_still_error}, "
          f"elapsed={elapsed/3600:.2f}h", flush=True)


if __name__ == "__main__":
    main()
