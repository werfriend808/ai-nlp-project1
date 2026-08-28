"""benchmark/post_reembedding_bottleneck_analysis.py -- READ-ONLY. 재임베딩 완료 후
Recall@10=34.3%에서 막힌 원인을 CASE A(retrieval)/B(reranking)/C(성공)로 분해하고,
RRF vs Cross-Encoder-only, _NEW decoy, period filter, weighted blend를 전부 benchmark
내부에서만 시뮬레이션한다. production 코드/DB는 전혀 수정하지 않는다 -- reranker.py의
내부 함수(_parse_rrf_ranks, _rrf_fuse, _sigmoid, rerank_scores, _apply_period_coverage_filter)
를 그대로 호출만 해서 현재 production과 동일한 계산을 재현하되, 중간 상태를 전부 기록한다.

사용법: python -m benchmark.post_reembedding_bottleneck_analysis
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import psycopg2

from agent.interfaces import Claim
from agent.mapping.keyword_search import keyword_search
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.reranker import (
    _merge_candidates, _apply_population_signal, _apply_institution_signal,
    _apply_gender_signal, _apply_region_signal, _apply_period_coverage_filter,
    rerank_scores, expand_institution_query_aliases,
    _parse_rrf_ranks, _rrf_fuse, _sigmoid, RRF_K,
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


def find_rank(ids, target):
    return (ids.index(target) + 1) if target in ids else None


def main():
    os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")
    from sentence_transformers import SentenceTransformer

    claims = load_golden_set()
    print(f"평가 대상 {len(claims)}건", flush=True)

    print(f"{EMBED_MODEL_NAME} 로딩 중...", flush=True)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, truncate_dim=EMBED_DIM, device="cuda")
    vdb_instruction = (
        "Given a Korean news claim sentence, retrieve the KOSIS statistical table "
        "description that best matches it"
    )

    print("64개 카탈로그 로딩...", flush=True)
    catalog_by_id = _load_table_catalog_by_id()
    catalog_document_texts = {tid: t["embedding_text"] for tid, t in catalog_by_id.items()}
    embedding_cache = build_table_embedding_cache()

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
        gold_in_pool = [g for g in gold_ids if g in candidate_ids]
        gold = gold_in_pool[0] if gold_in_pool else (gold_ids[0] if gold_ids else None)
        candidate_recall = bool(gold_in_pool)

        # _NEW decoy 존재 여부(정답 자체거나, 후보 중에 gold_id+"_NEW"가 있는지)
        new_variant = f"{gold}_NEW"
        new_decoy_in_pool = new_variant in candidate_ids
        gold_is_new_variant = bool(gold) and gold.endswith("_NEW")

        doc_texts = dict(catalog_document_texts)
        documents = [doc_texts.get(c.table_id, c.table_name) for c in merged]
        ce_scores = rerank_scores(claim.sentence, documents)

        row = {
            "idx": i, "sentence": claim.sentence, "gold_ids": gold_ids, "gold": gold,
            "candidate_recall": candidate_recall, "candidate_count": len(merged),
            "new_decoy_in_pool": new_decoy_in_pool, "gold_is_new_variant": gold_is_new_variant,
        }

        if not merged or ce_scores is None:
            row["case"] = "A_no_candidate" if not candidate_recall else "ERROR_no_ce"
            per_query.append(row)
            print(f"[{i+1}/{len(claims)}] case={row['case']} | {claim.sentence[:35]}", flush=True)
            continue

        # ---- 방식 1: production 그대로(RRF融合) ----
        reranked = []
        for cand, score in zip(merged, ce_scores):
            reranked.append(replace(cand, score=_sigmoid(score),
                                     source_meta=f"{cand.source_meta} | rerank_raw={score:.3f}"))
        reranked.sort(key=lambda c: c.score, reverse=True)
        fused_all = _rrf_fuse(reranked)  # 전체(top_k 자르기 전)
        fused_all_filtered = _apply_period_coverage_filter(claim, fused_all)
        rrf_final_ids = [c.table_id for c in fused_all_filtered[:FINAL_TOP_K]]

        # ---- 방식 2: Cross Encoder score만(RRF 없이) ----
        ce_only_sorted = sorted(reranked, key=lambda c: c.score, reverse=True)
        ce_only_filtered = _apply_period_coverage_filter(claim, ce_only_sorted)
        ce_only_final_ids = [c.table_id for c in ce_only_filtered[:FINAL_TOP_K]]

        # ---- 방식 3: RRF/CE 가중 블렌드(둘 다 pool 내 min-max 정규화 후 가중합) ----
        rrf_scores_all = {c.table_id: c.score for c in fused_all}
        ce_scores_all = {c.table_id: c.score for c in reranked}
        rrf_vals = list(rrf_scores_all.values())
        ce_vals = list(ce_scores_all.values())
        rrf_min, rrf_max = min(rrf_vals), max(rrf_vals)
        ce_min, ce_max = min(ce_vals), max(ce_vals)

        def norm(v, lo, hi):
            return (v - lo) / (hi - lo) if hi > lo else 0.5

        blend_final = {}
        for w_rrf, w_ce, label in [(0.3, 0.7, "rrf30_ce70"), (0.5, 0.5, "rrf50_ce50"), (0.7, 0.3, "rrf70_ce30")]:
            blended = []
            for tid in candidate_ids:
                r = norm(rrf_scores_all.get(tid, rrf_min), rrf_min, rrf_max)
                c = norm(ce_scores_all.get(tid, ce_min), ce_min, ce_max)
                blended.append((tid, w_rrf * r + w_ce * c))
            blended.sort(key=lambda x: x[1], reverse=True)
            blend_final[label] = [tid for tid, _ in blended[:FINAL_TOP_K]]

        # gold의 각 신호별 순위/점수 추출
        gold_cand = next((c for c in merged if c.table_id == gold), None)
        gold_source_ranks = _parse_rrf_ranks(gold_cand.source_meta) if gold_cand else {}
        gold_ce_raw = None
        if gold_cand:
            gi = candidate_ids.index(gold)
            gold_ce_raw = ce_scores[gi]

        rrf_rank = find_rank([c.table_id for c in fused_all], gold) if gold else None
        rrf_final_rank = find_rank(rrf_final_ids, gold) if gold else None
        ce_only_rank = find_rank([c.table_id for c in ce_only_sorted], gold) if gold else None

        # CASE 분류
        if not candidate_recall:
            case = "A_retrieval_fail"
        elif rrf_final_rank is not None:
            case = "C_success"
        else:
            case = "B_rerank_fail"

        top10_detail = [{"rank": r + 1, "table_id": c.table_id,
                          "table_name": (c.table_name or "")[:40],
                          "rrf_score": round(c.score, 4)}
                         for r, c in enumerate(fused_all_filtered[:FINAL_TOP_K])]

        row.update({
            "case": case,
            "gold_source_ranks": gold_source_ranks,
            "gold_ce_raw": gold_ce_raw,
            "gold_rrf_rank_all": rrf_rank,
            "gold_rrf_final_rank": rrf_final_rank,
            "gold_ce_only_rank": ce_only_rank,
            "rrf_final_top10": rrf_final_ids,
            "ce_only_final_top10": ce_only_final_ids,
            "blend_final": blend_final,
            "top10_detail": top10_detail,
        })
        per_query.append(row)

        print(f"[{i+1}/{len(claims)}] case={case} rrf_rank={rrf_final_rank} ce_only_rank={ce_only_rank} "
              f"new_decoy={new_decoy_in_pool} | {claim.sentence[:35]}", flush=True)

    n = len(claims)
    elapsed = time.time() - t0

    def recall_mrr(final_key):
        hit1 = hit5 = hit10 = 0
        rrs = []
        for r in per_query:
            ids = r.get(final_key)
            gold = r.get("gold")
            if not ids or not gold:
                rrs.append(0.0)
                continue
            rank = find_rank(ids, gold)
            if rank == 1:
                hit1 += 1
            if rank is not None and rank <= 5:
                hit5 += 1
            if rank is not None and rank <= 10:
                hit10 += 1
            rrs.append(1.0 / rank if rank else 0.0)
        return {"recall@1": hit1 / n, "recall@5": hit5 / n, "recall@10": hit10 / n, "mrr": sum(rrs) / n}

    cand_recall = sum(1 for r in per_query if r["candidate_recall"]) / n

    summary = {"candidate_recall": cand_recall}
    summary["A_production_rrf"] = recall_mrr("rrf_final_top10")
    summary["B_ce_only"] = recall_mrr("ce_only_final_top10")

    for label in ["rrf30_ce70", "rrf50_ce50", "rrf70_ce30"]:
        def get_blend(key, r=None, label=label):
            return r["blend_final"].get(label) if r.get("blend_final") else None
        hit1 = hit5 = hit10 = 0
        rrs = []
        for r in per_query:
            ids = r["blend_final"].get(label) if r.get("blend_final") else None
            gold = r.get("gold")
            if not ids or not gold:
                rrs.append(0.0); continue
            rank = find_rank(ids, gold)
            if rank == 1: hit1 += 1
            if rank is not None and rank <= 5: hit5 += 1
            if rank is not None and rank <= 10: hit10 += 1
            rrs.append(1.0/rank if rank else 0.0)
        summary[label] = {"recall@1": hit1/n, "recall@5": hit5/n, "recall@10": hit10/n, "mrr": sum(rrs)/n}

    case_counts = {}
    for r in per_query:
        case_counts[r["case"]] = case_counts.get(r["case"], 0) + 1

    new_decoy_in_pool_count = sum(1 for r in per_query if r.get("new_decoy_in_pool"))
    gold_is_new_count = sum(1 for r in per_query if r.get("gold_is_new_variant"))

    print("\n=== CASE 분류 ===")
    print(case_counts)
    print("\n=== 방식별 최종 성능 ===")
    for k, v in summary.items():
        print(k, v)
    print(f"\n_NEW decoy가 candidate pool에 있는 query 수: {new_decoy_in_pool_count}")
    print(f"gold 자체가 _NEW 버전인 query 수: {gold_is_new_count}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = {"n": n, "elapsed_sec": elapsed, "case_counts": case_counts, "summary": summary,
           "new_decoy_in_pool_count": new_decoy_in_pool_count, "gold_is_new_count": gold_is_new_count,
           "per_query": per_query}
    out_path = RESULTS_DIR / "post_reembedding_bottleneck_analysis.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n결과 저장 -> {out_path}")
    print(f"총 소요 {elapsed/60:.1f}분")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
