"""1단계: 모든 기저 검색기 × 질의 변형을 한 번씩 돌려 순위 리스트를 캐시한다.

전략(융합)은 fuse.py가 이 캐시만 가지고 만든다 — 같은 검색 결과를 재사용하므로
전략을 아무리 많이 비교해도 DB를 다시 때리지 않는다. hard filter는 어디에도 없다.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import queries as Q
from retrievers import BM25, connect, dense_axes, dense_items, dense_tables, lexical_trgm, vec_literal

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
DEPTH = 200          # 모든 검색기가 200위까지 회수 -> Recall@200까지 측정 가능
INSTRUCTION = ("Given a Korean news claim sentence, retrieve the KOSIS statistical table "
               "description that best matches it")

# (검색기 이름) -> (종류, 질의변형)
PLAN: list[tuple[str, str, str]] = [
    ("dense_full",        "dense_table", "full"),
    ("dense_measurement", "dense_table", "measurement"),
    ("dense_population",  "dense_table", "population"),
    ("dense_region",      "dense_table", "region"),
    ("dense_condition",   "dense_table", "condition"),
    ("dense_struct",      "dense_table", "struct"),
    ("dense_ctxD2",       "dense_table", "ctx_D2"),
    ("dense_ctxD3",       "dense_table", "ctx_D3"),
    ("dense_ctxD4",       "dense_table", "ctx_D4"),
    ("dense_ctxD5",       "dense_table", "ctx_D5"),
    ("dense_expanded",    "dense_table", "expanded"),
    ("dense_expstruct",   "dense_table", "expanded_struct"),
    ("item_measurement",  "dense_item",  "measurement"),
    ("item_full",         "dense_item",  "full"),
    ("axis_condition",    "dense_axis",  "condition"),
    ("axis_full",         "dense_axis",  "full"),
    ("bm25_full",         "bm25",        "full"),
    ("bm25_struct",       "bm25",        "struct"),
    ("bm25_measurement",  "bm25",        "measurement"),
    ("bm25_expstruct",    "bm25",        "expanded_struct"),
    ("bm25_ctxD4",        "bm25",        "ctx_D4"),
    ("trgm_struct",       "trgm",        "struct"),      # 느린 기존 방식 — 비교 기준
]


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    slots = json.loads((ROOT / "benchmark/search_experiment/claim_slots.json").read_text(encoding="utf-8"))
    qmap = Q.build_all(ev, slots)
    (HERE / "queries.json").write_text(json.dumps(qmap, ensure_ascii=False, indent=1), encoding="utf-8")

    needed_dense = sorted({v for _, kind, v in PLAN if kind.startswith("dense") or kind == "dense_item"}
                          | {v for _, kind, v in PLAN if kind in ("dense_table", "dense_item", "dense_axis")})
    print(f"임베딩이 필요한 질의 변형 {len(needed_dense)}종 × {len(ev)}건", flush=True)

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560)

    vecs: dict[str, dict[str, str]] = {}      # variant -> claim_id -> vector literal
    for variant in needed_dense:
        cids, texts = [], []
        for r in ev:
            t = qmap[r["claim_id"]].get(variant)
            if t:
                cids.append(r["claim_id"])
                texts.append(f"Instruct: {INSTRUCTION}\nQuery: {t}")
        if not texts:
            continue
        enc = model.encode(texts, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
        vecs[variant] = {c: vec_literal(v) for c, v in zip(cids, enc)}
        print(f"  {variant}: {len(texts)}건 인코딩", flush=True)
    del model
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()

    bm = BM25()
    conn = connect(); cur = conn.cursor()

    runs: dict[str, dict[str, list[str]]] = {}
    lat: dict[str, list[float]] = {}
    for name, kind, variant in PLAN:
        runs[name] = {}; lat[name] = []
        t_all = time.time()
        for r in ev:
            cid = r["claim_id"]
            text = qmap[cid].get(variant, "")
            qv = vecs.get(variant, {}).get(cid)
            t0 = time.perf_counter()
            if kind == "dense_table":
                res = dense_tables(cur, qv, DEPTH) if qv else []
            elif kind == "dense_item":
                res = dense_items(cur, qv, DEPTH) if qv else []
            elif kind == "dense_axis":
                res = dense_axes(cur, qv, DEPTH) if qv else []
            elif kind == "bm25":
                res = bm.search(text, DEPTH)
            elif kind == "trgm":
                res = lexical_trgm(cur, text, DEPTH)
            else:
                res = []
            lat[name].append((time.perf_counter() - t0) * 1000)
            runs[name][cid] = res
        print(f"  [{name}] 완료 {time.time()-t_all:.0f}s  평균 {sum(lat[name])/len(lat[name]):.0f}ms", flush=True)

    (HERE / "runs.json").write_text(json.dumps(runs, ensure_ascii=False), encoding="utf-8")
    (HERE / "latency.json").write_text(json.dumps(lat), encoding="utf-8")
    print(f"\n저장 완료: runs.json ({len(runs)}개 검색기)")
    conn.close()


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
