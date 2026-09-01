"""benchmark/metadata_embedding_experiment.py -- READ-ONLY. 28만 건 전체 재임베딩 전에,
70건 골든셋의 정답표(19개)만 대상으로 embedding_text에 ITEM/AXIS/AXIS VALUE/추가
메타데이터를 넣었을 때 Dense 검색 성능이 실제로 개선되는지 검증한다.

기존 production DB(kosis_vdb_tables_qwen)의 embedding은 절대 건드리지 않는다 -- 이
스크립트는 19개 정답표에 대해서만 "실험용" embedding_text를 새로 만들어 메모리/로컬
파일에서만 임베딩하고, production 코드/DB는 SELECT로만 조회한다.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

from agent.kosis.reembed_worker import EMBED_MODEL_NAME, EMBED_DIM
# 텍스트 생성 규칙은 운영 규격이라 agent/kosis/embedding_text.py가 단일 원본이다
# (예전엔 이 파일에 정의돼 있어서 운영 워커가 벤치마크를 import하는 역방향
#  의존 -> 순환 import가 났다, 2026-08-27).
from agent.kosis.embedding_text import build_experimental_text

GOLDEN_PATH = Path(__file__).parent.parent / "notebooks" / "골든셋_통합.xlsx"
EXP_DIR = Path(__file__).parent / "metadata_embedding_experiment_data"
RESULTS_DIR = Path(__file__).parent / "results"
NEW_TABLE = "kosis_vdb_tables_qwen"
_ORG_NORMALIZE = {"통계청": "국가데이터처", "KOSIS": None}

MODES = ["baseline", "item", "item_axis", "item_axis_value_full",
         "item_axis_value_capped", "full_metadata"]

_conn = None


def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
        _conn.autocommit = True
    return _conn


def _build_search_query(row: pd.Series) -> str:
    stat_parts = []
    stat_expr = row.get("statistic_expression")
    if isinstance(stat_expr, str) and stat_expr.strip():
        stat_parts.append(stat_expr.strip())
    for col in ("population", "region"):
        val = row.get(col)
        if isinstance(val, str) and val.strip():
            stat_parts.append(val.strip())
    org_parts = []
    org = row.get("source_org")
    if isinstance(org, str) and org.strip():
        normalized = _ORG_NORMALIZE.get(org.strip(), org.strip())
        if normalized:
            org_parts.append(normalized)
    parts = stat_parts + org_parts
    return " ".join(parts) if parts else str(row.get("sentence(원문 그대로)", ""))


def load_golden_set():
    df7 = pd.read_excel(GOLDEN_PATH, sheet_name="7단계_판정목록")
    df2 = pd.read_excel(GOLDEN_PATH, sheet_name="2단계_claim목록")
    df7 = df7.merge(
        df2[["claim_id", "statistic_expression", "population", "region", "source_org"]],
        on="claim_id", how="left",
    )
    ids_stripped = df7["matched_table_id(3단계)"].astype(str).str.strip()
    evalable = df7[ids_stripped != "없음"].reset_index(drop=True)
    claims = []
    for _, row in evalable.iterrows():
        gold_id = str(row["matched_table_id(3단계)"]).strip()
        sentence = str(row["sentence(원문 그대로)"])
        org = row.get("source_org")
        search_query = _build_search_query(row)
        claims.append({
            "sentence": sentence, "search_query": search_query,
            "gold_ids": [g.strip() for g in gold_id.split(",")],
        })
    return claims


def fetch_gold_metadata(gold_ids: list[str]) -> dict[str, dict]:
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            f"select table_id, institution_name, table_name from {NEW_TABLE} where table_id = ANY(%s);",
            (gold_ids,),
        )
        base = {r[0]: {"institution_name": r[1], "table_name": r[2]} for r in cur.fetchall()}

        cur.execute("select table_id, item_name from kosis_vdb_items_qwen where table_id = ANY(%s);", (gold_ids,))
        items: dict[str, list[str]] = {}
        for tid, name in cur.fetchall():
            if name:
                items.setdefault(tid, [])
                if name not in items[tid]:
                    items[tid].append(name)

        cur.execute("select table_id, axis_name from kosis_vdb_axes_qwen where table_id = ANY(%s) order by table_id, axis_order;", (gold_ids,))
        axes: dict[str, list[str]] = {}
        for tid, name in cur.fetchall():
            if name:
                axes.setdefault(tid, [])
                if name not in axes[tid]:
                    axes[tid].append(name)

        cur.execute("select table_id, value_name from kosis_vdb_axis_values_qwen where table_id = ANY(%s);", (gold_ids,))
        values: dict[str, list[str]] = {}
        for tid, name in cur.fetchall():
            if name:
                values.setdefault(tid, [])
                values[tid].append(name)

    for tid in gold_ids:
        base.setdefault(tid, {"institution_name": None, "table_name": None})
        base[tid]["items"] = items.get(tid, [])
        base[tid]["axes"] = axes.get(tid, [])
        base[tid]["values_raw"] = values.get(tid, [])
        base[tid]["values_dedup"] = list(dict.fromkeys(values.get(tid, [])))
    return base


def dense_pool_excluding(query_vec: list[float], exclude_ids: list[str], k: int = 100):
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select table_id, embedding::halfvec(2560) <=> %s::halfvec(2560) as dist
            from {NEW_TABLE}
            where table_id != ALL(%s)
            order by embedding::halfvec(2560) <=> %s::halfvec(2560)
            limit %s;
            """,
            (query_vec, exclude_ids, query_vec, k),
        )
        return cur.fetchall()


def approx_rank(gold_dist: float, pool_dists: list[float]) -> int | None:
    better = sum(1 for d in pool_dists if d < gold_dist)
    if better >= len(pool_dists):
        return None
    return better + 1


def main():
    os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")
    from sentence_transformers import SentenceTransformer

    EXP_DIR.mkdir(exist_ok=True)
    RESULTS_DIR.mkdir(exist_ok=True)

    claims = load_golden_set()
    all_gold_ids = sorted({g for c in claims for g in c["gold_ids"]})
    print(f"평가 대상 {len(claims)}건 claim, 고유 정답표 {len(all_gold_ids)}개", flush=True)

    meta = fetch_gold_metadata(all_gold_ids)

    survey_path = RESULTS_DIR / "gold_survey_names.json"
    survey_names = json.loads(survey_path.read_text(encoding="utf-8")) if survey_path.exists() else {}
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(f"select table_id, stat_id from {NEW_TABLE} where table_id = ANY(%s);", (all_gold_ids,))
        stat_id_by_table = dict(cur.fetchall())

    print(f"{EMBED_MODEL_NAME} 로딩 중 (truncate_dim={EMBED_DIM})...", flush=True)
    model = SentenceTransformer(EMBED_MODEL_NAME, truncate_dim=EMBED_DIM, device="cuda")
    vdb_instruction = (
        "Given a Korean news claim sentence, retrieve the KOSIS statistical table "
        "description that best matches it"
    )

    gold_vecs: dict[str, dict[str, np.ndarray]] = {mode: {} for mode in MODES}
    text_stats = []
    for mode in MODES:
        texts, ids = [], []
        for tid in all_gold_ids:
            survey = survey_names.get(stat_id_by_table.get(tid, ""))
            text = build_experimental_text(meta[tid], mode, survey)
            texts.append(text)
            ids.append(tid)
            tok_count = len(model.tokenizer.encode(text)) if hasattr(model, "tokenizer") else len(text.split())
            text_stats.append({
                "mode": mode, "table_id": tid, "embedding_text": text,
                "character_count": len(text), "token_count": tok_count,
                "item_count": len(meta[tid]["items"]), "axis_count": len(meta[tid]["axes"]),
                "axis_value_count_raw": len(meta[tid]["values_raw"]),
                "axis_value_count_dedup": len(meta[tid]["values_dedup"]),
            })
        vecs = model.encode(texts, batch_size=16, convert_to_numpy=True,
                             normalize_embeddings=True, show_progress_bar=False)
        for tid, vec in zip(ids, vecs):
            gold_vecs[mode][tid] = vec
        np.save(EXP_DIR / f"{mode}_embeddings.npy", vecs)
        (EXP_DIR / f"{mode}.jsonl").write_text(
            "\n".join(json.dumps({"table_id": tid, "embedding_text": t}, ensure_ascii=False)
                      for tid, t in zip(ids, texts)),
            encoding="utf-8",
        )
        print(f"[{mode}] {len(texts)}건 임베딩 완료, 평균 글자수={sum(len(t) for t in texts)/len(texts):.0f}", flush=True)

    (EXP_DIR / "text_stats.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in text_stats), encoding="utf-8"
    )

    print("\n쿼리별 dense pool 계산 중...", flush=True)
    per_query = []
    dense_recall = {mode: {10: 0, 50: 0, 100: 0} for mode in MODES}
    t0 = time.time()
    for i, c in enumerate(claims):
        text = f"Instruct: {vdb_instruction}\nQuery: {c['search_query'] or c['sentence']}"
        qvec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0]
        pool = dense_pool_excluding(qvec.tolist(), c["gold_ids"], k=100)
        pool_dists = [d for _, d in pool]

        row = {"idx": i, "sentence": c["sentence"], "gold_ids": c["gold_ids"], "by_mode": {}}
        for mode in MODES:
            best_rank = None
            for gid in c["gold_ids"]:
                if gid not in gold_vecs[mode]:
                    continue
                gold_dist = 1.0 - float(np.dot(qvec, gold_vecs[mode][gid]))
                r = approx_rank(gold_dist, pool_dists)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank = r
            row["by_mode"][mode] = best_rank
            if best_rank is not None:
                if best_rank <= 10:
                    dense_recall[mode][10] += 1
                if best_rank <= 50:
                    dense_recall[mode][50] += 1
                if best_rank <= 100:
                    dense_recall[mode][100] += 1

        row["baseline_pool_top1"] = pool[0][0] if pool else None
        per_query.append(row)

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(claims)}] elapsed={time.time()-t0:.0f}s", flush=True)

    n = len(claims)
    print("\n=== Dense Recall (모드별, HNSW 근사) ===")
    summary = {}
    for mode in MODES:
        r10 = dense_recall[mode][10] / n
        r50 = dense_recall[mode][50] / n
        r100 = dense_recall[mode][100] / n
        summary[mode] = {"recall@10": r10, "recall@50": r50, "recall@100": r100}
        print(f"{mode:24s} R@10={r10:.1%}  R@50={r50:.1%}  R@100={r100:.1%}")

    out = {"n": n, "unique_gold": len(all_gold_ids), "summary": summary, "per_query": per_query}
    out_path = RESULTS_DIR / "metadata_embedding_experiment.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장 -> {out_path}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
