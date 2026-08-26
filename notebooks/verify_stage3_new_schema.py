"""
notebooks/verify_stage3_new_schema.py — verify_stage3_on_golden_merged.py와 완전히 같은
골든셋(골든셋_통합.xlsx)·같은 검색+리랭크 방법론을, 새로 재임베딩한 스키마
(kosis_vdb_tables_qwen, Qwen3-Embedding-4B truncate_dim=2560)에 대해 재현한다.

OLD 스크립트를 건드리지 않고 그대로 둔 채(원본은 kosis_vdb_tables/1024d를 조회),
vdb_fn/bm25_fn 클로저만 새 스키마(kosis_vdb_tables_qwen, table_id/embedding_text 컬럼,
2560d)를 향하도록 새로 작성했다 — search_and_rerank()/keyword_search/embedding_search
등 파이프라인 코드는 원본과 동일하게 재사용(수정 없음), 64개 수동 카탈로그 경로도 VDB
재임베딩과 무관하므로 그대로다. TABLE 레벨만 비교한다(ITEM/AXIS 세분화는 이번 비교
범위 밖 — 사용자 확인됨).

사용법: python -m notebooks.verify_stage3_new_schema
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
import psycopg2

ORG_FIRST = os.environ.get("ORG_FIRST") == "1"

from agent.interfaces import Claim
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.keyword_search import keyword_search
from agent.mapping.reranker import search_and_rerank, expand_institution_query_aliases
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

GOLDEN_PATH = Path(__file__).parent / "골든셋_통합.xlsx"
REPORT_PATH = Path(__file__).parent / f"stage3_verify_report_new_schema{'_orgfirst' if ORG_FIRST else ''}.md"
TOP_K = 10

NEW_TABLE = "kosis_vdb_tables_qwen"
# OLD query_vdb.py와 동일 값(Qwen3-Embedding-4B 기준 실측 조정치) 그대로 이월.
VDB_TOP_K = int(os.environ.get("DENSE_TOP_K", "10"))
LEXICAL_TOP_K = int(os.environ.get("BM25_TOP_K", "30"))
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


def batch_query_vdb_new(query_vectors: list[list[float]], *, top_k: int = VDB_TOP_K) -> list[list[TableCandidate]]:
    if not query_vectors:
        return []
    conn = _get_conn()
    results: list[list[TableCandidate]] = []
    with conn.cursor() as cur:
        for query_vec in query_vectors:
            cur.execute(
                f"""
                select table_id, org_id, embedding_text, embedding <=> %s::vector as distance
                from {NEW_TABLE}
                order by embedding <=> %s::vector
                limit %s;
                """,
                (query_vec, query_vec, top_k),
            )
            rows = cur.fetchall()
            candidates = []
            for tbl_id, org_id, text, dist in rows:
                similarity = 1.0 - float(dist)
                if similarity < VDB_MIN_SIMILARITY:
                    continue
                candidates.append(
                    TableCandidate(
                        table_id=tbl_id,
                        table_name=text,
                        score=similarity,
                        required_slots=[],
                        source_meta=f"kosis_vdb_qwen model={EMBED_MODEL_NAME} dim={EMBED_DIM}",
                        org_id=org_id or None,
                    )
                )
            results.append(candidates)
    return results


def lexical_query_vdb_new(query_text: str, *, top_k: int = LEXICAL_TOP_K) -> list[TableCandidate]:
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
    return [
        TableCandidate(
            table_id=tbl_id,
            table_name=text,
            score=float(sim),
            required_slots=[],
            source_meta="kosis_vdb_qwen_lexical(pg_trgm)",
            org_id=org_id or None,
        )
        for tbl_id, org_id, text, sim in rows
    ]


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

    parts = (org_parts + stat_parts) if ORG_FIRST else (stat_parts + org_parts)
    return " ".join(parts) if parts else str(row.get("sentence(원문 그대로)", ""))


def _normalize(text: str) -> str:
    import re
    return re.sub(r"[\s·・]+", "", str(text))


def _core_terms(statistic_expression) -> list[str]:
    import re
    if not isinstance(statistic_expression, str):
        return []
    return re.findall(r"[가-힣A-Za-z0-9]{2,}", statistic_expression)


def main() -> None:
    df7 = pd.read_excel(GOLDEN_PATH, sheet_name="7단계_판정목록")
    df2 = pd.read_excel(GOLDEN_PATH, sheet_name="2단계_claim목록")
    df7 = df7.merge(
        df2[["claim_id", "statistic_expression", "population", "region", "source_org"]],
        on="claim_id", how="left",
    )
    ids_stripped = df7["matched_table_id(3단계)"].astype(str).str.strip()
    evalable = df7[ids_stripped != "없음"].reset_index(drop=True)
    print(f"[NEW SCHEMA] 정답표 있는 claim {len(evalable)}건 검증 시작 (전체 {len(df7)}건 중 판단불가 제외)")

    catalog_by_id = _load_table_catalog_by_id()
    document_texts = {tid: t["embedding_text"] for tid, t in catalog_by_id.items()}
    embedding_cache = build_table_embedding_cache()

    all_gold_ids: set[str] = set()
    for v in evalable["matched_table_id(3단계)"]:
        all_gold_ids.update(x.strip() for x in str(v).split(","))
    _vdb_conn = _get_conn()
    _vdb_cur = _vdb_conn.cursor()
    _vdb_cur.execute(
        f"select table_id, embedding_text from {NEW_TABLE} where table_id = ANY(%s)",
        (list(all_gold_ids),),
    )
    gold_table_text = {r[0]: r[1] for r in _vdb_cur.fetchall()}

    from sentence_transformers import SentenceTransformer

    print(f"{EMBED_MODEL_NAME} 로딩 중 (truncate_dim={EMBED_DIM})...")
    vdb_model = SentenceTransformer(EMBED_MODEL_NAME, truncate_dim=EMBED_DIM, device="cuda")
    vdb_instruction = (
        "Given a Korean news claim sentence, retrieve the KOSIS statistical table "
        "description that best matches it"
    )

    def _retrieval_query_text(claim) -> str:
        base = claim.search_query or claim.sentence
        return expand_institution_query_aliases(base, claim.source_org)

    def vdb_fn(claim):
        text = f"Instruct: {vdb_instruction}\nQuery: {_retrieval_query_text(claim)}"
        vec = vdb_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
        return batch_query_vdb_new([vec], top_k=VDB_TOP_K)[0]

    def bm25_fn(claim):
        return lexical_query_vdb_new(_retrieval_query_text(claim), top_k=LEXICAL_TOP_K)

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    misses = []
    cross = {"gap_hit": 0, "gap_miss": 0, "present_hit": 0, "present_miss": 0}
    report_lines = ["# 3단계(표 매칭) 골든셋 검증 결과 — NEW SCHEMA (Qwen3-Embedding-4B, 2560d)\n\n"]

    for i, row in evalable.iterrows():
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

        try:
            candidates = search_and_rerank(
                claim,
                keyword_fn=keyword_search,
                embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
                vdb_fn=vdb_fn,
                bm25_fn=bm25_fn,
                top_k=TOP_K,
                document_texts=document_texts,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(evalable)}] [FAIL] {sentence[:40]} -> {e}")
            misses.append((sentence, gold_id, None, str(e)))
            time.sleep(0.1)
            continue

        gold_ids = [g.strip() for g in gold_id.split(",")]
        result_ids = [c.table_id for c in candidates]
        ranks = [result_ids.index(g) + 1 for g in gold_ids if g in result_ids]
        rank = min(ranks) if ranks else None

        if rank == 1:
            hits_at_1 += 1
        if rank is not None and rank <= 5:
            hits_at_5 += 1
        if rank is not None and rank <= 10:
            hits_at_10 += 1
        reciprocal_ranks.append(1.0 / rank if rank else 0.0)

        terms = _core_terms(row.get("statistic_expression"))
        gold_texts_norm = [_normalize(gold_table_text[g]) for g in gold_ids if g in gold_table_text]
        has_term = any(
            any(_normalize(t) in gt for gt in gold_texts_norm) for t in terms
        ) if terms and gold_texts_norm else False
        is_hit10 = rank is not None
        if has_term:
            cross["present_hit" if is_hit10 else "present_miss"] += 1
        else:
            cross["gap_hit" if is_hit10 else "gap_miss"] += 1

        tag = f"rank={rank}" if rank else "MISS"
        vocab_tag = "용어있음" if has_term else "용어없음"
        print(f"[{i + 1}/{len(evalable)}] [{tag}][{vocab_tag}] gold={gold_id} | {sentence[:50]}")

        if rank is None:
            top3 = ", ".join(f"{c.table_id}({c.table_name})" for c in candidates[:3])
            misses.append((sentence, gold_id, top3, None))

    n = len(evalable)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    report_lines.append(f"- 평가 대상: {n}건\n")
    report_lines.append(f"- Recall@1: {hits_at_1}/{n} = {hits_at_1/n:.1%}\n")
    report_lines.append(f"- Recall@5: {hits_at_5}/{n} = {hits_at_5/n:.1%}\n")
    report_lines.append(f"- Recall@10: {hits_at_10}/{n} = {hits_at_10/n:.1%}\n")
    report_lines.append(f"- MRR: {mrr:.3f}\n\n")
    report_lines.append(f"## 미스 {len(misses)}건\n\n")
    for sentence, gold_id, top3, err in misses:
        if err:
            report_lines.append(f"- [예외] {sentence} (gold={gold_id}): {err}\n")
        else:
            report_lines.append(f"- {sentence} (gold={gold_id}) — top3: {top3}\n")
    REPORT_PATH.write_text("".join(report_lines), encoding="utf-8")

    gap_total = cross["gap_hit"] + cross["gap_miss"]
    present_total = cross["present_hit"] + cross["present_miss"]
    print(f"\n=== 최종 요약 (NEW SCHEMA) ===")
    print(f"평가 대상: {n}건")
    print(f"Recall@1: {hits_at_1}/{n} = {hits_at_1/n:.1%}")
    print(f"Recall@5: {hits_at_5}/{n} = {hits_at_5/n:.1%}")
    print(f"Recall@10: {hits_at_10}/{n} = {hits_at_10/n:.1%}")
    print(f"MRR: {mrr:.3f}")
    print(f"\n=== 원인별 교차표 ===")
    print(f"정답표 제목에 용어 없음(구조적 매칭 불가): {gap_total}건 중 그래도 top10 hit {cross['gap_hit']}건, miss {cross['gap_miss']}건")
    print(f"정답표 제목에 용어 있음: {present_total}건 중 hit {cross['present_hit']}건, miss {cross['present_miss']}건")
    print(f"리포트 -> {REPORT_PATH}")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
