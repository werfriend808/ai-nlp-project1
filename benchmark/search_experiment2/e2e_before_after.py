"""3단계 전체(검색 4종 + 신호보정 + 리랭커 + RRF) end-to-end 교체 전후 비교.

오늘 리트리버를 크게 고쳤는데(R@200 77.1%->90.0%) 그게 최종 top-1까지 얼마나
전달되는지는 아직 안 쟀다. 리랭커를 손대기 전에 출발점을 확정하기 위한 측정.

  OLD = 2026-08-28 이전 운영 (문맥 없음 + trigram)
  NEW = 현재 (문맥 보강 + BM25)

두 설정 모두 search_and_rerank()를 실제 그대로 호출한다. document_texts도 운영과
같이 64개 카탈로그만 넘긴다(VDB 후보는 table_name 폴백) — 두 설정이 같은 조건이므로
비교는 성립한다.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")

from agent.interfaces import Claim
from agent.preprocessing.claim_extractor import attach_sentence_context
from agent.mapping.reranker import (
    search_and_rerank, build_retrieval_query, build_lexical_query,
)
from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
from agent.mapping.keyword_search import keyword_search
from agent.kosis.query_vdb import batch_query_vdb, lexical_query_vdb, VDB_TOP_K, LEXICAL_TOP_K, VdbUnavailableError
from agent.kosis.bm25_search import bm25_query_vdb, BM25_TOP_K
from agent.pipeline.batch_runner import _load_table_catalog_by_id
import queries as Q

TOP_K = 10
_STOP = {"전국", "전체", "국내", "없음", "nan", "", "-", "KOSIS"}


def clean(v):
    v = (v or "").strip()
    return "" if v.lower() in {s.lower() for s in _STOP} else v


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    slots = json.loads((ROOT / "benchmark/search_experiment/claim_slots.json").read_text(encoding="utf-8"))
    arts, cmap = Q.load_articles(), Q.load_claim_article_map()

    by_art: dict[str, list[dict]] = {}
    for r in ev:
        by_art.setdefault(cmap.get(r["claim_id"], ""), []).append(r)
    claims: dict[str, Claim] = {}
    for aid, rows in by_art.items():
        objs = []
        for r in rows:
            s = slots.get(r["claim_id"], {})
            objs.append(Claim(sentence=r["sentence"], claim_type="규모",
                              statistic_expression=clean(s.get("statistic_expression")) or None,
                              population=clean(s.get("population")) or None,
                              region=clean(s.get("region")) or None,
                              source_org=clean(s.get("source_org")) or None))
        attach_sentence_context(arts.get(aid, ""), objs)
        for r, c in zip(rows, objs):
            claims[r["claim_id"]] = c

    ids = [r["claim_id"] for r in ev]
    gold = {r["claim_id"]: set(r["gold"]) for r in ev}

    catalog_by_id = _load_table_catalog_by_id()
    document_texts = {tid: t["embedding_text"] for tid, t in catalog_by_id.items()}
    embedding_cache = build_table_embedding_cache()

    from sentence_transformers import SentenceTransformer
    print("Qwen3-Embedding-4B 로딩 중...", flush=True)
    vdb_model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560, device="cuda")
    instruction = ("Given a Korean news claim sentence, retrieve the KOSIS statistical table "
                   "description that best matches it")

    def make_vdb_fn():
        def vdb_fn(claim):
            text = f"Instruct: {instruction}\nQuery: {build_retrieval_query(claim)}"
            vec = vdb_model.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
            try:
                return batch_query_vdb([vec], top_k=VDB_TOP_K)[0]
            except VdbUnavailableError:
                return []
        return vdb_fn

    def trigram_fn(claim):
        try:
            return lexical_query_vdb(build_retrieval_query(claim), top_k=LEXICAL_TOP_K)
        except VdbUnavailableError:
            return []

    def bm25_fn(claim):
        try:
            return bm25_query_vdb(build_lexical_query(claim), top_k=BM25_TOP_K)
        except VdbUnavailableError:
            return []

    configs = [("OLD (문맥없음+trigram)", "0", trigram_fn), ("NEW (문맥+BM25)", "1", bm25_fn)]
    out: dict[str, dict] = {}

    for name, ctx_flag, lex_fn in configs:
        os.environ["KOSIS_QUERY_CONTEXT"] = ctx_flag
        print(f"\n=== {name} ===", flush=True)
        ranks: dict[str, int | None] = {}
        t0 = time.time()
        for n, cid in enumerate(ids, 1):
            try:
                cands = search_and_rerank(
                    claims[cid], keyword_fn=keyword_search,
                    embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
                    vdb_fn=make_vdb_fn(), bm25_fn=lex_fn,
                    top_k=TOP_K, document_texts=document_texts,
                )
                got = [c.table_id for c in cands]
                r = next((i + 1 for i, t in enumerate(got) if t in gold[cid]), None)
            except Exception as e:  # noqa: BLE001
                print(f"  {cid} 실패: {type(e).__name__}: {e}", flush=True)
                r = None
            ranks[cid] = r
            if n % 20 == 0:
                print(f"  {n}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
        out[name] = ranks
        print(f"  완료 {time.time()-t0:.0f}s", flush=True)

    def acc(ranks, k):
        return sum(1 for c in ids if ranks[c] and ranks[c] <= k) / len(ids)

    def mrr(ranks):
        return sum(1.0 / ranks[c] for c in ids if ranks[c]) / len(ids)

    print(f"\n{'설정':<26}{'top-1':>9}{'top-5':>9}{'top-10':>9}{'MRR':>9}")
    for name in out:
        print(f"{name:<26}" + "".join(f"{acc(out[name],k):>8.1%} " for k in (1, 5, 10))
              + f"{mrr(out[name]):>8.3f}")

    o, nw = out["OLD (문맥없음+trigram)"], out["NEW (문맥+BM25)"]
    for k in (1, 5, 10):
        win = [c for c in ids if not (o[c] and o[c] <= k) and (nw[c] and nw[c] <= k)]
        loss = [c for c in ids if (o[c] and o[c] <= k) and not (nw[c] and nw[c] <= k)]
        print(f"\ntop-{k}: +{len(win)} / -{len(loss)} → 순증 {len(win)-len(loss):+d}")
        if loss:
            print(f"  손해: {loss}")

    (Path(__file__).parent / "e2e_before_after.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
