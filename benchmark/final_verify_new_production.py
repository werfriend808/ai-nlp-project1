"""benchmark/final_verify_new_production.py -- READ-ONLY. 28.7만 건 item_axis_value_capped
재임베딩이 실제 production DB에 반영된 뒤, 70건 골든셋으로 전체 검색 파이프라인
(Keyword+Embedding+Dense+BM25 -> RRF -> Cross Encoder) 성능을 검증한다.

production 코드는 호출만 하고 수정하지 않는다. DB는 SELECT만 한다.

사용법: python -m benchmark.final_verify_new_production
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pandas as pd
import psycopg2

from agent.interfaces import Claim
from agent.mapping.keyword_search import keyword_search
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import (
    _merge_candidates, _apply_population_signal, _apply_institution_signal,
    _apply_gender_signal, _apply_region_signal, _apply_period_coverage_filter,
    rerank, expand_institution_query_aliases,
)
from agent.pipeline.batch_runner import _load_table_catalog_by_id
from agent.kosis.reembed_worker import EMBED_MODEL_NAME, EMBED_DIM

try:
    from agent.interfaces import TableCandidate
except ImportError:
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class TableCandidate:  # type: ignore[no-redef]
        table_id: str
        table_name: str
        score: float
        required_slots: list = field(default_factory=list)
        source_meta: Optional[str] = None
        org_id: Optional[str] = None

GOLDEN_PATH = Path(__file__).parent.parent / "notebooks" / "골든셋_통합.xlsx"
RESULTS_DIR = Path(__file__).parent / "results"
NEW_TABLE = "kosis_vdb_tables_qwen"
VDB_TOP_K = 100
BM25_TOP_K = 30
FINAL_TOP_K = 10
VDB_MIN_SIMILARITY = 0.5
_ORG_NORMALIZE = {"통계청": "국가데이터처", "KOSIS": None}

_conn = None


def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
        _conn.autocommit = True
        with _conn.cursor() as cur:
            cur.execute("set pg_trgm.similarity_threshold = 0.1;")
    return _conn


def batch_query_vdb_new(query_vec, *, top_k: int = VDB_TOP_K):
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select table_id, org_id, embedding_text,
                   embedding::halfvec(2560) <=> %s::halfvec(2560) as dist
            from {NEW_TABLE}
            order by embedding::halfvec(2560) <=> %s::halfvec(2560)
            limit %s;
            """,
            (query_vec, query_vec, top_k),
        )
        rows = cur.fetchall()
    out = []
    for tid, org_id, text, dist in rows:
        sim = 1.0 - float(dist)
        if sim < VDB_MIN_SIMILARITY:
            continue
        out.append(TableCandidate(table_id=tid, table_name=text, score=sim,
                                   required_slots=[], source_meta=f"kosis_vdb_qwen sim={sim:.3f}",
                                   org_id=org_id or None))
    return out


def lexical_query_vdb_new(query_text: str, *, top_k: int = BM25_TOP_K):
    if not query_text or not query_text.strip():
        return []
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select table_id, org_id, embedding_text, similarity(embedding_text, %s) as sim
            from {NEW_TABLE}
            where embedding_text %% %s
            order by sim desc
            limit %s;
            """,
            (query_text, query_text, top_k),
        )
        rows = cur.fetchall()
    return [TableCandidate(table_id=tid, table_name=text, score=float(sim), required_slots=[],
                            source_meta="kosis_vdb_qwen_lexical(pg_trgm)", org_id=org_id or None)
            for tid, org_id, text, sim in rows]


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
        period_val = row.get("period")
        claim = Claim(
            sentence=sentence,
            claim_type=str(row.get("claim_type") or "규모"),
            period=str(period_val) if pd.notna(period_val) else None,
            source_org=org.strip() if isinstance(org, str) and org.strip() else None,
            search_query=_build_search_query(row),
        )
        claims.append((claim, [g.strip() for g in gold_id.split(",")]))
    return claims


def rank_and_mrr(ordered_ids, gold_ids):
    ranks = [ordered_ids.index(g) + 1 for g in gold_ids if g in ordered_ids]
    rank = min(ranks) if ranks else None
    rr = (1.0 / rank) if rank else 0.0
    return rank, rr


def main():
    os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")
    from sentence_transformers import SentenceTransformer

    claims = load_golden_set()
    print(f"평가 대상 {len(claims)}건", flush=True)

    print(f"{EMBED_MODEL_NAME} 로딩 중 (truncate_dim={EMBED_DIM})...", flush=True)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, truncate_dim=EMBED_DIM, device="cuda")
    vdb_instruction = (
        "Given a Korean news claim sentence, retrieve the KOSIS statistical table "
        "description that best matches it"
    )

    print("64개 카탈로그 로딩...", flush=True)
    catalog_by_id = _load_table_catalog_by_id()
    catalog_document_texts = {tid: t["embedding_text"] for tid, t in catalog_by_id.items()}
    embedding_cache = build_table_embedding_cache()

    agg = {"cand": 0, "hit1": 0, "hit5": 0, "hit10": 0, "rr": []}
    per_query = []
    t0 = time.time()
    for i, (claim, gold_ids) in enumerate(claims):
        retrieval_q = expand_institution_query_aliases(claim.search_query or claim.sentence, claim.source_org)
        text = f"Instruct: {vdb_instruction}\nQuery: {retrieval_q}"
        vec = embed_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

        kw_results = keyword_search(claim)
        emb_results = embedding_search(claim, cache=embedding_cache)
        vdb_results = batch_query_vdb_new(vec, top_k=VDB_TOP_K)
        bm25_results = lexical_query_vdb_new(retrieval_q, top_k=BM25_TOP_K)

        merged = _merge_candidates(kw_results, emb_results, vdb_results, bm25_results)
        merged = _apply_population_signal(claim, merged, catalog_document_texts)
        merged = _apply_institution_signal(claim, merged, catalog_document_texts)
        merged = _apply_gender_signal(claim, merged, catalog_document_texts)
        merged = _apply_region_signal(claim, merged, catalog_document_texts)

        candidate_ids = [c.table_id for c in merged]
        candidate_recall = any(g in candidate_ids for g in gold_ids)

        doc_texts = dict(catalog_document_texts)
        result = rerank(claim, merged, top_k=FINAL_TOP_K, document_texts=doc_texts)
        result = _apply_period_coverage_filter(claim, result)
        result_ids = [c.table_id for c in result]

        rank, rr = rank_and_mrr(result_ids, gold_ids)
        agg["cand"] += int(candidate_recall)
        if rank == 1:
            agg["hit1"] += 1
        if rank is not None and rank <= 5:
            agg["hit5"] += 1
        if rank is not None and rank <= 10:
            agg["hit10"] += 1
        agg["rr"].append(rr)

        per_query.append({"idx": i, "sentence": claim.sentence, "gold_ids": gold_ids,
                           "candidate_recall": candidate_recall, "rank": rank,
                           "top3": result_ids[:3]})
        print(f"[{i+1}/{len(claims)}] cand={candidate_recall} rank={rank} top1={result_ids[0] if result_ids else None} | {claim.sentence[:35]}", flush=True)

    n = len(claims)
    elapsed = time.time() - t0
    cand_recall = agg["cand"] / n
    r1, r5, r10 = agg["hit1"] / n, agg["hit5"] / n, agg["hit10"] / n
    mrr = sum(agg["rr"]) / n

    print("\n=== NEW (item_axis_value_capped, production 실제 적용 후) ===")
    print(f"Candidate Recall: {cand_recall:.1%}")
    print(f"Recall@1: {r1:.1%}")
    print(f"Recall@5: {r5:.1%}")
    print(f"Recall@10: {r10:.1%}")
    print(f"MRR: {mrr:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"n": n, "elapsed_sec": elapsed,
           "summary": {"candidate_recall": cand_recall, "recall@1": r1, "recall@5": r5,
                       "recall@10": r10, "mrr": mrr},
           "per_query": per_query}
    out_path = RESULTS_DIR / "final_verify_new_production.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장 -> {out_path}")
    print(f"총 소요 {elapsed/60:.1f}분")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
