"""benchmark/additional_real_articles_verify.py -- READ-ONLY. 기존 70건 골든셋과 별개로,
notebooks/매핑 골든셋 ord 추가.xlsx(match_status="값 확인됨 (일치)")에서 확인된 18건의
실제 기사 claim으로 production retrieval(현재 재임베딩된 DB 그대로)을 검증한다.

KOSIS API 호출 없음, DB/embedding 변경 없음, production 코드는 호출만 한다.

사용법: python -m benchmark.additional_real_articles_verify
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

NOTEBOOKS = Path(__file__).parent.parent / "notebooks"
MAPPING_PATH = NOTEBOOKS / "매핑 골든셋 ord 추가.xlsx"
EXTRACT_PATHS = [
    NOTEBOOKS / "1.추출 골든셋(신영-최종).xlsx",
    NOTEBOOKS / "1.추출골든셋(은서).xlsx",
]
RESULTS_DIR = Path(__file__).parent / "results"
NEW_TABLE = "kosis_vdb_tables_qwen"
VDB_TOP_K = 100
BM25_TOP_K = 30
FINAL_TOP_K = 10

_conn = None


def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
        _conn.autocommit = True
        with _conn.cursor() as cur:
            cur.execute("set pg_trgm.similarity_threshold = 0.1;")
    return _conn


def dense_search_ranked(query_vec, k=VDB_TOP_K):
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
            (query_vec, query_vec, k),
        )
        return cur.fetchall()


def lexical_query_vdb_new(query_text, top_k=BM25_TOP_K):
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
        return cur.fetchall()


def load_18():
    df1 = pd.read_excel(EXTRACT_PATHS[0], sheet_name="추출_골든셋_단위분리")
    df2 = pd.read_excel(EXTRACT_PATHS[1], sheet_name="추출_골든셋_cleaned_1")
    sentences = pd.concat([df1, df2], ignore_index=True)
    mapping = pd.read_excel(MAPPING_PATH, sheet_name="KOSIS_매핑")
    merged = mapping.merge(sentences[["claim_id", "claim_sentence", "article_title", "topic"]],
                            on="claim_id", how="left")
    confirmed = merged[merged["match_status"].astype(str).str.strip().isin(
        ["값 확인됨 (일치)", "값 확인됨(일치)"])].reset_index(drop=True)
    out = []
    for _, row in confirmed.iterrows():
        out.append({
            "claim_id": row["claim_id"], "sentence": str(row["claim_sentence"]),
            "gold_id": str(row["kosis_table_id"]).strip(), "org_name": row.get("org_name"),
            "topic": row.get("topic"),
        })
    return out


def find_rank(ids, target):
    return (ids.index(target) + 1) if target in ids else None


def main():
    os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")
    from sentence_transformers import SentenceTransformer

    items = load_18()
    print(f"추가 검증 대상 {len(items)}건 (매핑 골든셋 ord 추가.xlsx, 값 확인됨(일치))", flush=True)

    embed_model = SentenceTransformer(EMBED_MODEL_NAME, truncate_dim=EMBED_DIM, device="cuda")
    vdb_instruction = (
        "Given a Korean news claim sentence, retrieve the KOSIS statistical table "
        "description that best matches it"
    )
    catalog_by_id = _load_table_catalog_by_id()
    catalog_document_texts = {tid: t["embedding_text"] for tid, t in catalog_by_id.items()}
    embedding_cache = build_table_embedding_cache()

    per_item = []
    t0 = time.time()
    for i, it in enumerate(items):
        sentence = it["sentence"]
        gold = it["gold_id"]
        claim = Claim(sentence=sentence, claim_type="규모", search_query=sentence)
        retrieval_q = expand_institution_query_aliases(sentence, None)

        text = f"Instruct: {vdb_instruction}\nQuery: {retrieval_q}"
        vec = embed_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()

        dense_rows = dense_search_ranked(vec, k=VDB_TOP_K)
        dense_ids = [r[0] for r in dense_rows]
        dense_rank = find_rank(dense_ids, gold)
        cand10 = dense_rank is not None and dense_rank <= 10
        cand50 = dense_rank is not None and dense_rank <= 50
        cand100 = dense_rank is not None and dense_rank <= 100

        bm25_rows = lexical_query_vdb_new(retrieval_q, top_k=BM25_TOP_K)
        bm25_ids = [r[0] for r in bm25_rows]
        bm25_rank = find_rank(bm25_ids, gold)

        kw_results = keyword_search(claim)
        emb_results = embedding_search(claim, cache=embedding_cache)
        kw_rank = find_rank([c.table_id for c in kw_results], gold)
        emb_rank = find_rank([c.table_id for c in emb_results], gold)

        vdb_results = [TableCandidate(table_id=r[0], table_name=r[2], score=1.0-float(r[3]),
                                       required_slots=[], source_meta=f"sim={1-float(r[3]):.3f}",
                                       org_id=r[1]) for r in dense_rows if (1.0-float(r[3])) >= 0.5]
        bm25_results = [TableCandidate(table_id=r[0], table_name=r[2], score=float(r[3]),
                                        required_slots=[], source_meta="lex", org_id=r[1])
                         for r in bm25_rows]

        merged = _merge_candidates(kw_results, emb_results, vdb_results, bm25_results)
        merged = _apply_population_signal(claim, merged, catalog_document_texts)
        merged = _apply_institution_signal(claim, merged, catalog_document_texts)
        merged = _apply_gender_signal(claim, merged, catalog_document_texts)
        merged = _apply_region_signal(claim, merged, catalog_document_texts)
        candidate_ids = [c.table_id for c in merged]
        candidate_recall = gold in candidate_ids

        doc_texts = dict(catalog_document_texts)
        result = rerank(claim, merged, top_k=FINAL_TOP_K, document_texts=doc_texts)
        result = _apply_period_coverage_filter(claim, result)
        result_ids = [c.table_id for c in result]
        final_rank = find_rank(result_ids, gold)

        top10_named = [(c.table_id, (c.table_name or "")[:35]) for c in result]

        case = "CASE1_in_top100" if cand100 else "CASE2_out_of_top100"
        row = {
            "claim_id": it["claim_id"], "sentence": sentence, "gold": gold,
            "org_name": it.get("org_name"), "dense_rank": dense_rank, "bm25_rank": bm25_rank,
            "keyword_rank": kw_rank, "embedding_rank": emb_rank,
            "candidate_pool_size": len(merged), "candidate_recall": candidate_recall,
            "final_rank": final_rank, "case": case,
            "cand10": cand10, "cand50": cand50, "cand100": cand100,
            "top10": top10_named,
        }
        per_item.append(row)
        print(f"[{i+1}/{len(items)}] {it['claim_id']} gold={gold} dense_rank={dense_rank} "
              f"final_rank={final_rank} case={case} | {sentence[:35]}", flush=True)

    n = len(items)
    elapsed = time.time() - t0
    cand_recall_10 = sum(1 for r in per_item if r["cand10"]) / n
    cand_recall_50 = sum(1 for r in per_item if r["cand50"]) / n
    cand_recall_100 = sum(1 for r in per_item if r["cand100"]) / n
    full_candidate_recall = sum(1 for r in per_item if r["candidate_recall"]) / n
    hit1 = sum(1 for r in per_item if r["final_rank"] == 1)
    hit5 = sum(1 for r in per_item if r["final_rank"] is not None and r["final_rank"] <= 5)
    hit10 = sum(1 for r in per_item if r["final_rank"] is not None and r["final_rank"] <= 10)
    mrr = sum((1.0 / r["final_rank"]) if r["final_rank"] else 0.0 for r in per_item) / n

    print("\n=== 추가 실제 기사 18건 결과 ===")
    print(f"Dense Candidate Recall@10: {cand_recall_10:.1%}")
    print(f"Dense Candidate Recall@50: {cand_recall_50:.1%}")
    print(f"Dense Candidate Recall@100: {cand_recall_100:.1%}")
    print(f"전체(4신호 병합) Candidate Recall: {full_candidate_recall:.1%}")
    print(f"Recall@1: {hit1/n:.1%}  Recall@5: {hit5/n:.1%}  Recall@10: {hit10/n:.1%}  MRR: {mrr:.3f}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"n": n, "elapsed_sec": elapsed,
           "summary": {"dense_candidate_recall@10": cand_recall_10,
                       "dense_candidate_recall@50": cand_recall_50,
                       "dense_candidate_recall@100": cand_recall_100,
                       "full_candidate_recall": full_candidate_recall,
                       "recall@1": hit1/n, "recall@5": hit5/n, "recall@10": hit10/n, "mrr": mrr},
           "per_item": per_item}
    out_path = RESULTS_DIR / "additional_real_articles_verify.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n결과 저장 -> {out_path}")
    print(f"총 소요 {elapsed/60:.1f}분")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
