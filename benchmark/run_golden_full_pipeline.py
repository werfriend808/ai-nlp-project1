"""골든셋 50개 기사를 1~8단계 전체 파이프라인으로 돌리고 정답과 대조한다.

batch_runner에는 골든셋 모드가 없어서(하드코딩 시나리오나 CSV 샘플만 받는다) 여기서
골든셋 기사를 run_article()이 받는 형태로 바꿔 넣는다. 파이프라인 자체는 손대지 않는다.

대조 대상 (notebooks/골든셋_통합.xlsx):
  - 3단계  matched_table_id(3단계)   표 매칭이 맞았는지
  - 7단계  정답_verdict              최종 판정이 맞았는지

    python -m benchmark.run_golden_full_pipeline [--limit N] [--articles 5,19,30]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")

import pandas as pd

GOLDEN = ROOT / "notebooks/골든셋_통합.xlsx"
OUT = ROOT / "benchmark/golden_full_pipeline_results.json"


def _to_date(v):
    try:
        d = pd.to_datetime(v)
        return date(d.year, d.month, d.day)
    except Exception:
        return date(2025, 1, 1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="기사 N건만")
    ap.add_argument("--articles", help="기사 번호 콤마 구분 (예: 5,19,30)")
    ap.add_argument("--no-vdb", action="store_true")
    args = ap.parse_args()

    df1 = pd.read_excel(GOLDEN, sheet_name="1단계_기사목록")
    df7 = pd.read_excel(GOLDEN, sheet_name="7단계_판정목록")

    if args.articles:
        want = {s.strip() for s in args.articles.split(",")}
        df1 = df1[df1["번호"].astype(str).str.strip().isin(want)]
    if args.limit:
        df1 = df1.head(args.limit)

    articles = []
    for _, r in df1.iterrows():
        body = str(r["본문(정제됨)"] or "").strip()
        if not body:
            continue
        articles.append({
            "label": f"골든셋 기사 {str(r['번호']).strip()} — {str(r['기사제목'])[:50]}",
            "article_id": str(r["번호"]).strip(),
            "article_title": str(r["기사제목"] or "") or None,
            "published_date": _to_date(r["작성일"]),
            "article_text": body,
        })
    print(f"대상 기사 {len(articles)}건 / 정답 claim {len(df7)}건\n", flush=True)

    # ── 파이프라인 준비 (batch_runner.main과 동일한 방식) ────────────────────
    from agent.kosis.api_client import KosisApiClient
    from agent.kosis.calculator import KosisCalculator
    from agent.mapping.embedding_search import build_table_embedding_cache
    from agent.pipeline.batch_runner import (
        TABLE_PARAMS_PATH, _load_table_catalog_by_id, run_article,
    )

    client = KosisApiClient()
    calculator = KosisCalculator()
    with open(TABLE_PARAMS_PATH, encoding="utf-8") as f:
        table_params = json.load(f)
    catalog_by_id = _load_table_catalog_by_id()
    embedding_cache = build_table_embedding_cache()

    vdb_fn = bm25_fn = None
    if not args.no_vdb:
        from sentence_transformers import SentenceTransformer

        from agent.kosis.bm25_search import BM25_TOP_K, bm25_query_vdb
        from agent.kosis.query_vdb import VDB_TOP_K, VdbUnavailableError, batch_query_vdb
        from agent.mapping.reranker import build_lexical_query, build_retrieval_query

        print("Qwen3-Embedding-4B 로딩 중...", flush=True)
        model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560, device="cuda")
        instruction = ("Given a Korean news claim sentence, retrieve the KOSIS statistical "
                       "table description that best matches it")

        def vdb_fn(claim):  # noqa: F811
            text = f"Instruct: {instruction}\nQuery: {build_retrieval_query(claim)}"
            vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
            try:
                return batch_query_vdb([vec], top_k=VDB_TOP_K)[0]
            except VdbUnavailableError as e:
                print(f"[VDB] 실패({e})")
                return []

        def bm25_fn(claim):  # noqa: F811
            try:
                return bm25_query_vdb(build_lexical_query(claim), top_k=BM25_TOP_K)
            except VdbUnavailableError as e:
                print(f"[BM25] 실패({e})")
                return []

    # ── 실행 ────────────────────────────────────────────────────────────────
    all_records: list[dict] = []
    t_start = time.time()
    for i, art in enumerate(articles, 1):
        t0 = time.time()
        print(f"[{i}/{len(articles)}] 기사 {art['article_id']} 시작", flush=True)
        try:
            recs = run_article(art, client, calculator, table_params,
                               embedding_cache, catalog_by_id,
                               vdb_fn=vdb_fn, bm25_fn=bm25_fn)
        except Exception:  # noqa: BLE001 - 한 기사 실패로 배치를 멈추지 않는다
            print(f"    실패:\n{traceback.format_exc()[:600]}", flush=True)
            recs = []
        for r in recs:
            r["article_id"] = art["article_id"]
        all_records.extend(recs)
        print(f"    claim {len(recs)}건 / {time.time()-t0:.0f}s "
              f"(누적 {len(all_records)}건, {time.time()-t_start:.0f}s)", flush=True)
        OUT.write_text(json.dumps(all_records, ensure_ascii=False, indent=1, default=str),
                       encoding="utf-8")

    print(f"\n실행 완료 — claim {len(all_records)}건 / {time.time()-t_start:.0f}s")
    print(f"저장: {OUT}")

    # ── 정답 대조 ───────────────────────────────────────────────────────────
    ran_ids = {a["article_id"] for a in articles}
    truth = df7[df7["기사번호"].astype(str).str.strip().isin(ran_ids)]
    print(f"\n{'='*64}\n대조 (정답 claim {len(truth)}건 중 파이프라인이 처리한 것만)\n{'='*64}")

    def norm(s):
        return "".join(str(s).split())[:40]

    by_sent = {}
    for r in all_records:
        by_sent.setdefault(norm(r.get("claim_sentence", "")), r)

    matched = tbl_ok = verdict_ok = 0
    for _, t in truth.iterrows():
        r = by_sent.get(norm(t["sentence(원문 그대로)"]))
        if r is None:
            continue
        matched += 1
        gold_tbl = str(t["matched_table_id(3단계)"]).strip()
        got_tbl = str(r.get("table_id") or "").strip()
        if gold_tbl and gold_tbl != "없음" and got_tbl == gold_tbl:
            tbl_ok += 1
        gv, pv = str(t["정답_verdict"]).strip(), str(r.get("verdict") or "").strip()
        if gv and gv.lower() != "nan" and gv == pv:
            verdict_ok += 1

    print(f"정답셋 claim {len(truth)}건 중 파이프라인이 같은 문장을 뽑은 것: {matched}건 "
          f"({matched/max(len(truth),1):.1%})   <- 2단계 재현율")
    if matched:
        print(f"  그중 3단계 표 매칭 일치 : {tbl_ok}/{matched} ({tbl_ok/matched:.1%})")
        print(f"  그중 7단계 판정 일치   : {verdict_ok}/{matched} ({verdict_ok/matched:.1%})")

    from collections import Counter
    print(f"\n파이프라인 verdict 분포: {dict(Counter(str(r.get('verdict')) for r in all_records))}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
