"""재임베딩 스키마(kosis_vdb_tables_qwen, item+axis+분류값 포함된 완성본)에 우리 메인
파이프라인의 리랭킹 보정 신호 3개(AXIS 값 매칭·기간 커버리지 필터·버전 신선도)를 이식해서
얹었을 때 골든셋 70건에서 얼마나 더 개선되는지 검증한다.

원본 신호(agent/mapping/reranker.py의 _apply_axis_value_signal/_apply_period_coverage_filter/
_apply_version_freshness_signal)는 실시간 KOSIS API 호출 기반인데, 이 스키마는 axis_values/
period_start·end/stat_id·send_date가 이미 DB에 사전 인덱싱돼 있어서 API 호출 없이 DB
조회만으로 동일한 로직을 훨씬 빠르게/넓게(top-20 제한 없이) 적용할 수 있다.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import replace
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras

from agent.interfaces import Claim
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.keyword_search import keyword_search
from agent.mapping.reranker import search_and_rerank, expand_institution_query_aliases, _tag_source_rank, _parse_rrf_ranks
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
REPORT_PATH = Path(__file__).parent / "stage3_verify_report_qwen_full_signals.md"
TOP_K = 10
NEW_TABLE = "kosis_vdb_tables_qwen"
VDB_TOP_K = int(os.environ.get("DENSE_TOP_K", "10"))
LEXICAL_TOP_K = int(os.environ.get("BM25_TOP_K", "30"))
VDB_MIN_SIMILARITY = 0.5
AXIS_VALUE_TOP_N = 20
PERIOD_COVERAGE_TOP_N = 5

_ORG_NORMALIZE = {"통계청": "국가데이터처", "KOSIS": None}
_CORE_TERMS_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")
_PERIOD_YEAR_RE = re.compile(r"(19|20)\d{2}")

_conn = None


def _get_conn():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
        _conn.autocommit = True
        with _conn.cursor() as cur:
            cur.execute("set pg_trgm.similarity_threshold = 0.1;")
    return _conn


def batch_query_vdb_new(query_vectors, *, top_k=VDB_TOP_K):
    if not query_vectors:
        return []
    conn = _get_conn()
    results = []
    with conn.cursor() as cur:
        for query_vec in query_vectors:
            cur.execute(
                f"""
                select table_id, org_id, embedding_text, embedding <=> %s::vector as distance
                from {NEW_TABLE} order by embedding <=> %s::vector limit %s;
                """,
                (query_vec, query_vec, top_k),
            )
            candidates = []
            for tbl_id, org_id, text, dist in cur.fetchall():
                similarity = 1.0 - float(dist)
                if similarity < VDB_MIN_SIMILARITY:
                    continue
                candidates.append(TableCandidate(
                    table_id=tbl_id, table_name=text, score=similarity, required_slots=[],
                    source_meta=f"kosis_vdb_qwen model={EMBED_MODEL_NAME} dim={EMBED_DIM}",
                    org_id=org_id or None,
                ))
            results.append(candidates)
    return results


def lexical_query_vdb_new(query_text, *, top_k=LEXICAL_TOP_K):
    if not query_text or not query_text.strip():
        return []
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            f"""
            select table_id, org_id, embedding_text, similarity(embedding_text, %s) as sim
            from {NEW_TABLE} where embedding_text %% %s order by sim desc limit %s;
            """,
            (query_text, query_text, top_k),
        )
        rows = cur.fetchall()
    return [
        TableCandidate(table_id=t, table_name=txt, score=float(s), required_slots=[],
                        source_meta="kosis_vdb_qwen_lexical(pg_trgm)", org_id=o or None)
        for t, o, txt, s in rows
    ]


def _core_terms(statistic_expression) -> list[str]:
    if not isinstance(statistic_expression, str):
        return []
    return _CORE_TERMS_RE.findall(statistic_expression)


# ------------------------------------------------------------------
# 신호 1: AXIS 값 매칭 (kosis_vdb_axis_values_qwen 직접 조회, API 호출 없음)
# ------------------------------------------------------------------
def apply_axis_value_signal_qwen(claim, candidates: list[TableCandidate]) -> list[TableCandidate]:
    terms = [t.strip() for t in (claim.population, claim.statistic_expression) if t and t.strip()]
    if not terms or not candidates:
        return candidates
    head = candidates[:AXIS_VALUE_TOP_N]
    table_ids = [c.table_id for c in head]
    conn = _get_conn()
    matched_tables = set()
    with conn.cursor() as cur:
        for term in terms:
            if len(term) < 2:
                continue
            cur.execute(
                """
                select distinct table_id from kosis_vdb_axis_values_qwen
                where table_id = ANY(%s) and value_name ilike %s;
                """,
                (table_ids, f"%{term}%"),
            )
            matched_tables.update(r[0] for r in cur.fetchall())
    return [
        _tag_source_rank(c, "axis_value_rank", 1) if c.table_id in matched_tables else c
        for c in candidates
    ]


# ------------------------------------------------------------------
# 신호 2: 기간 커버리지 필터 (period_start/period_end 컬럼 직접 조회, API 호출 없음)
# ------------------------------------------------------------------
def _extract_target_year(claim) -> int | None:
    if not claim.period:
        return None
    m = _PERIOD_YEAR_RE.search(claim.period)
    return int(m.group(0)) if m else None


def apply_period_coverage_filter_qwen(claim, candidates: list[TableCandidate]) -> list[TableCandidate]:
    year = _extract_target_year(claim)
    if year is None or not candidates:
        return candidates

    axis_matched = [c for c in candidates if _parse_rrf_ranks(c.source_meta).get("axis_value_rank") == 1]
    if axis_matched:
        matched_ids = {c.table_id for c in axis_matched}
        head = [c for c in candidates if c.table_id in matched_ids]
        tail = [c for c in candidates if c.table_id not in matched_ids]
    else:
        head, tail = candidates[:PERIOD_COVERAGE_TOP_N], candidates[PERIOD_COVERAGE_TOP_N:]

    if not head:
        return candidates
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "select table_id, period_start, period_end from kosis_vdb_tables_qwen where table_id = ANY(%s);",
            ([c.table_id for c in head],),
        )
        ranges = {r[0]: (r[1], r[2]) for r in cur.fetchall()}

    kept, excluded = [], []
    for c in head:
        rng = ranges.get(c.table_id)
        if not rng or not rng[0] or not rng[1]:
            kept.append(c)
            continue
        start_y, end_y = rng[0][:4], rng[1][:4]
        try:
            covers = int(start_y) <= year <= int(end_y)
        except ValueError:
            kept.append(c)
            continue
        (kept if covers else excluded).append(c)
    return kept + excluded + tail


# ------------------------------------------------------------------
# 신호 3: 버전 신선도 (stat_id + table_name 클러스터, send_date로 최신판 우선)
# ------------------------------------------------------------------
def apply_version_freshness_signal_qwen(candidates: list[TableCandidate]) -> list[TableCandidate]:
    if not candidates:
        return candidates
    conn = _get_conn()
    with conn.cursor() as cur:
        cur.execute(
            "select table_id, stat_id, table_name, send_date from kosis_vdb_tables_qwen where table_id = ANY(%s);",
            ([c.table_id for c in candidates],),
        )
        meta = {r[0]: {"stat_id": r[1], "table_name": r[2], "send_date": r[3]} for r in cur.fetchall()}

    clusters: dict[tuple, list[TableCandidate]] = {}
    for c in candidates:
        m = meta.get(c.table_id)
        if not m or not m["stat_id"] or not m["table_name"]:
            continue
        key = (m["stat_id"], m["table_name"])
        clusters.setdefault(key, []).append(c)

    stale_ids, fresh_ids = set(), set()
    for members in clusters.values():
        with_date = [c for c in members if meta[c.table_id].get("send_date")]
        if len(with_date) < 2:
            continue
        freshest = max(with_date, key=lambda c: meta[c.table_id]["send_date"])
        fresh_ids.add(freshest.table_id)
        stale_ids.update(c.table_id for c in with_date if c.table_id != freshest.table_id)

    if not stale_ids:
        return candidates
    tagged = [
        _tag_source_rank(c, "version_fresh_rank", 1) if c.table_id in fresh_ids else c
        for c in candidates
    ]
    kept = [c for c in tagged if c.table_id not in stale_ids]
    demoted = [c for c in tagged if c.table_id in stale_ids]
    return kept + demoted


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


def main() -> None:
    df7 = pd.read_excel(GOLDEN_PATH, sheet_name="7단계_판정목록")
    df2 = pd.read_excel(GOLDEN_PATH, sheet_name="2단계_claim목록")
    df7 = df7.merge(
        df2[["claim_id", "statistic_expression", "population", "region", "source_org"]],
        on="claim_id", how="left",
    )
    ids_stripped = df7["matched_table_id(3단계)"].astype(str).str.strip()
    evalable = df7[ids_stripped != "없음"].reset_index(drop=True)
    print(f"[QWEN + 3신호] 정답표 있는 claim {len(evalable)}건 검증 시작")

    catalog_by_id = _load_table_catalog_by_id()
    document_texts = {tid: t["embedding_text"] for tid, t in catalog_by_id.items()}
    embedding_cache = build_table_embedding_cache()

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

    hits_at_1 = hits_at_5 = hits_at_10 = 0
    reciprocal_ranks = []
    misses = []
    report_lines = ["# 3단계 검증 — QWEN 재임베딩 + AXIS값/기간필터/버전신선도 3신호 적용\n\n"]

    for i, row in evalable.iterrows():
        gold_id = str(row["matched_table_id(3단계)"]).strip()
        sentence = str(row["sentence(원문 그대로)"])
        org = row.get("source_org")
        period_val = row.get("period")
        stat_expr = row.get("statistic_expression")
        population = row.get("population")
        claim = Claim(
            sentence=sentence,
            claim_type=str(row.get("claim_type") or "규모"),
            period=str(period_val) if pd.notna(period_val) else None,
            source_org=org.strip() if isinstance(org, str) and org.strip() else None,
            statistic_expression=stat_expr.strip() if isinstance(stat_expr, str) and stat_expr.strip() else None,
            population=population.strip() if isinstance(population, str) and population.strip() else None,
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
            candidates = apply_axis_value_signal_qwen(claim, candidates)
            candidates = apply_period_coverage_filter_qwen(claim, candidates)
            candidates = apply_version_freshness_signal_qwen(candidates)
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(evalable)}] [FAIL] {sentence[:40]} -> {e}")
            misses.append((sentence, gold_id, None, str(e)))
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

        tag = f"rank={rank}" if rank else "MISS"
        print(f"[{i + 1}/{len(evalable)}] [{tag}] gold={gold_id} | {sentence[:50]}")

        if rank is None:
            top3 = ", ".join(f"{c.table_id}({c.table_name[:30]})" for c in candidates[:3])
            misses.append((sentence, gold_id, top3, None))

    n = len(evalable)
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    summary = [
        f"평가 대상: {n}건",
        f"Recall@1: {hits_at_1}/{n} = {hits_at_1/n:.1%}",
        f"Recall@5: {hits_at_5}/{n} = {hits_at_5/n:.1%}",
        f"Recall@10: {hits_at_10}/{n} = {hits_at_10/n:.1%}",
        f"MRR: {mrr:.3f}",
    ]
    report_lines.extend(f"- {s}\n" for s in summary)
    report_lines.append(f"\n## 미스 {len(misses)}건\n\n")
    for sentence, gold_id, top3, err in misses:
        if err:
            report_lines.append(f"- [예외] {sentence} (gold={gold_id}): {err}\n")
        else:
            report_lines.append(f"- {sentence} (gold={gold_id}) — top3: {top3}\n")
    REPORT_PATH.write_text("".join(report_lines), encoding="utf-8")

    print(f"\n=== 최종 요약 (QWEN + 3신호) ===")
    for s in summary:
        print(s)
    print(f"리포트 -> {REPORT_PATH}")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
