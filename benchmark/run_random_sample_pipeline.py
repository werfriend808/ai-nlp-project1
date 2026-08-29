"""실제 기사 코퍼스에서 무작위 N건을 뽑아 1~8단계 전체를 돌린다.

골든셋은 "정답표가 있는 기사"만 골라 만든 표본이라 실제보다 유리하다. 또 정답표가
19종에 몰려 있어(상위 3개가 40%) 1~3건 차이를 판별할 수 없다는 게 2026-08-29에
확인됐다. 여기서는 정답 라벨 없이도 나오는 운영 지표를 뽑는다:

  - verdict 분포 (특히 표매칭_불충분 비율)
  - 기사당 claim 수, claim당 처리 시간
  - 3단계에서 탈락하는 비율

그리고 사람이 "이 표가 맞나"만 표시하면 바로 평가셋이 되도록 검토용 CSV를 같이 낸다 —
무작위 추출이라 골든셋과 정답표가 거의 안 겹쳐서 표 다양성 문제를 정면으로 푼다.

    python -m benchmark.run_random_sample_pipeline --n 10 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")

OUT_JSON = ROOT / "benchmark/random_sample_results.json"
OUT_CSV = ROOT / "benchmark/random_sample_review.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-vdb", action="store_true")
    args = ap.parse_args()

    from agent.kosis.api_client import KosisApiClient
    from agent.kosis.calculator import KosisCalculator
    from agent.mapping.embedding_search import build_table_embedding_cache
    from agent.pipeline.batch_runner import (
        TABLE_PARAMS_PATH, _load_table_catalog_by_id, load_articles_from_csv, run_article,
    )

    articles = load_articles_from_csv(n=args.n, seed=args.seed)
    print(f"무작위 추출 기사 {len(articles)}건 (seed={args.seed})\n", flush=True)
    for a in articles:
        print(f"  - {a['label'][:80]}", flush=True)

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

        print("\nQwen3-Embedding-4B 로딩 중...", flush=True)
        model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560, device="cuda")
        instruction = ("Given a Korean news claim sentence, retrieve the KOSIS statistical "
                       "table description that best matches it")

        def vdb_fn(claim):  # noqa: F811
            text = f"Instruct: {instruction}\nQuery: {build_retrieval_query(claim)}"
            vec = model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
            try:
                return batch_query_vdb([vec], top_k=VDB_TOP_K)[0]
            except VdbUnavailableError:
                return []

        def bm25_fn(claim):  # noqa: F811
            try:
                return bm25_query_vdb(build_lexical_query(claim), top_k=BM25_TOP_K)
            except VdbUnavailableError:
                return []

    records: list[dict] = []
    t_all = time.time()
    for i, art in enumerate(articles, 1):
        t0 = time.time()
        print(f"\n[{i}/{len(articles)}] {art['label'][:60]}", flush=True)
        try:
            recs = run_article(art, client, calculator, table_params,
                               embedding_cache, catalog_by_id,
                               vdb_fn=vdb_fn, bm25_fn=bm25_fn)
        except Exception:  # noqa: BLE001
            print(f"    실패:\n{traceback.format_exc()[:500]}", flush=True)
            recs = []
        for r in recs:
            r["article_label"] = art["label"]
        records.extend(recs)
        print(f"    claim {len(recs)}건 / {time.time()-t0:.0f}s", flush=True)
        OUT_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=1, default=str),
                            encoding="utf-8")

    n = len(records)
    print(f"\n{'='*70}")
    print(f"기사 {len(articles)}건 / claim {n}건 / {time.time()-t_all:.0f}s")
    if not n:
        print("claim이 하나도 안 나왔습니다.")
        return

    vc = Counter(str(r.get("verdict")) for r in records)
    print(f"\nverdict 분포")
    for k, v in vc.most_common():
        print(f"  {k:<24}{v:>4}건  {v/n:>6.1%}")

    matched = [r for r in records if r.get("table_id")]
    print(f"\n3단계 표 매칭 성공 : {len(matched)}/{n} ({len(matched)/n:.1%})")
    print(f"기사당 claim       : {n/len(articles):.1f}건")

    with OUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["검토(O/X)", "claim문장", "매칭표ID", "매칭표명", "verdict", "기사"])
        for r in records:
            w.writerow(["", r.get("claim_sentence", ""), r.get("table_id", ""),
                        str(r.get("table_name", ""))[:80], r.get("verdict", ""),
                        str(r.get("article_label", ""))[:50]])
    print(f"\n검토용 CSV: {OUT_CSV}")
    print("  '검토(O/X)' 열에 표가 맞으면 O, 틀리면 X를 적으면 그대로 평가셋이 됩니다.")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
