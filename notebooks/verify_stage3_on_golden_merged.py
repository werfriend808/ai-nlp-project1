"""
notebooks/verify_stage3_on_golden_merged.py — 새 통합 골든셋(골든셋_통합.xlsx)의
7단계_판정목록 시트(matched_table_id(3단계) 컬럼)로 3단계 표 매칭을 검증.

agent/mapping/golden_set.py(예전 골든셋용)와 다르게 64개 수동 카탈로그로 제한하지
않는다 — 지금 파이프라인은 카탈로그+VDB(28.7만 개)를 다 쓰므로, search_and_rerank()를
실제 그대로(기관명 별칭 fix 포함) 호출해서 정답 tblId가 top-K 안에 몇 위로 들어오는지
Recall@K·MRR을 잰다. 정답표가 없는(판단불가) claim은 평가 대상에서 뺀다.

사용법: python -m notebooks.verify_stage3_on_golden_merged
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

# 2026-08-22: VDB 문서 텍스트는 "{기관명} ({연월}) {표제목}"으로 기관명이 맨 앞인데,
# claim_extractor_prompt.txt의 search_query 규칙은 반대로 "지표명 ... 기관명"(기관명이
# 맨 뒤)이다 — 쿼리·문서 순서가 서로 다른 게 dense 임베딩 유사도에 영향을 주는지
# 실측하려고 환경변수로 순서를 토글할 수 있게 한다. ORG_FIRST=1이면 문서 형식과 맞춰
# 기관명을 맨 앞으로 보낸다.
ORG_FIRST = os.environ.get("ORG_FIRST") == "1"

from agent.interfaces import Claim
from agent.kosis.query_vdb import batch_query_vdb, lexical_query_vdb, VDB_TOP_K, LEXICAL_TOP_K
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.keyword_search import keyword_search
from agent.mapping.reranker import search_and_rerank, expand_institution_query_aliases
from agent.pipeline.batch_runner import _load_table_catalog_by_id

GOLDEN_PATH = Path(__file__).parent / "골든셋_통합.xlsx"
REPORT_PATH = Path(__file__).parent / f"stage3_verify_report{'_orgfirst' if ORG_FIRST else ''}.md"
TOP_K = 10


# prompts/claim_extractor_prompt.txt의 search_query 정규화 규칙과 동일 — "통계청"은 구
# 명칭이라 반드시 "국가데이터처"로 바꿔 써야 한다(그 외 기관은 원문 그대로 정식 명칭 취급).
_ORG_NORMALIZE = {"통계청": "국가데이터처", "KOSIS": None}


def _build_search_query(row: pd.Series) -> str:
    """claim_extractor_prompt.txt의 search_query 규칙(지표명 + 있는 dimension만 +
    정규화 기관명, 숫자/연도 금지)을 골든셋의 구조화 필드로 재현한다. 처음에 claim
    문장 원문을 그대로 search_query로 썼더니(2단계_claim목록에 search_query 컬럼
    자체가 없어서) Recall@1이 18.6%까지 떨어졌다 — server.py 주석에 이미 "raw 문장
    vs HyDE 스타일 쿼리"로 Dense Recall 61.0%→29.3%, BM25 41.5%→2.4% 차이가
    실측 기록돼 있던 것과 정확히 같은 함정이었다. 반드시 이 형식으로 재구성해야
    실제 파이프라인과 동등한 조건으로 비교가 된다."""
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
    # 7단계 시트엔 statistic_expression/population/region/source_org가 없어서(claim_id,
    # sentence, claim_type, 값/판정 관련 컬럼만 있음) 검색 쿼리를 만들 재료가 2단계 시트에만
    # 있다 — claim_id로 조인해서 가져온다.
    df7 = df7.merge(
        df2[["claim_id", "statistic_expression", "population", "region", "source_org"]],
        on="claim_id", how="left",
    )
    # "없음"(문자열, NaN이 아님 — 실측 확인)이 판단불가 34건의 실제 값이라 notna()로는
    # 안 걸러진다. 일부 tblId엔 앞 공백도 섞여 있어(" DT_1DA7024S" 등) strip도 같이 한다.
    ids_stripped = df7["matched_table_id(3단계)"].astype(str).str.strip()
    evalable = df7[ids_stripped != "없음"].reset_index(drop=True)
    print(f"정답표 있는 claim {len(evalable)}건 검증 시작 (전체 {len(df7)}건 중 판단불가 제외)")

    catalog_by_id = _load_table_catalog_by_id()
    document_texts = {tid: t["embedding_text"] for tid, t in catalog_by_id.items()}
    embedding_cache = build_table_embedding_cache()

    # 정답표의 VDB 텍스트에 statistic_expression 핵심 용어가 아예 있는지 미리 조회 —
    # "세부분류 불일치"(제목에 용어 자체가 없어 구조적으로 못 찾음) vs "용어는 있는데
    # 못 찾음"(랭킹/임베딩 등 다른 원인)을 실제 hit/miss와 교차해서 우선순위를 정하기 위함.
    import psycopg2

    all_gold_ids: set[str] = set()
    for v in evalable["matched_table_id(3단계)"]:
        all_gold_ids.update(x.strip() for x in str(v).split(","))
    _vdb_conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    _vdb_cur = _vdb_conn.cursor()
    _vdb_cur.execute("select tbl_id, text from kosis_vdb_tables where tbl_id = ANY(%s)", (list(all_gold_ids),))
    gold_table_text = {r[0]: r[1] for r in _vdb_cur.fetchall()}

    from sentence_transformers import SentenceTransformer

    print("Qwen3-Embedding-4B 로딩 중...")
    vdb_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=1024)
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
        return batch_query_vdb([vec], top_k=VDB_TOP_K)[0]

    def bm25_fn(claim):
        return lexical_query_vdb(_retrieval_query_text(claim), top_k=LEXICAL_TOP_K)

    hits_at_1 = 0
    hits_at_5 = 0
    hits_at_10 = 0
    reciprocal_ranks = []
    misses = []
    # 교차표: vocab_gap(정답표 제목에 핵심 용어 없음) x hit@10 여부
    cross = {"gap_hit": 0, "gap_miss": 0, "present_hit": 0, "present_miss": 0}
    report_lines = ["# 3단계(표 매칭) 골든셋 검증 결과\n\n"]

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
            time.sleep(0.5)
            continue

        # 드물게(1건) 정답 tblId가 콤마로 여러 개 나열된 복합 claim이 있다 — 그중
        # 어느 하나라도 상위에 잡히면 맞은 걸로 본다(가장 좋은 순위 채택).
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

        # vocab_gap: 정답표(들) 중 하나라도 statistic_expression 핵심 용어를 포함하면
        # "용어 존재"로 본다(하나만 맞아도 검색 가능성이 있으므로).
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

        time.sleep(0.3)

    n = len(evalable)
    n_ok = n - len([m for m in misses if m[3] is not None])  # 실패(예외) 제외한 평가 성공 건수
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
    print(f"\n=== 최종 요약 ===")
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
