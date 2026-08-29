"""리랭커 모델 비교 — 같은 후보 풀을 여러 cross-encoder로 재정렬해 최종 순위를 비교한다.

검색은 한 번만 돌려 후보 풀을 디스크에 저장하고(--build-pool), 모델마다 별도 프로세스로
재정렬한다(--model). 모델 하나가 CUDA device-side assert를 내면 그 프로세스의 CUDA
컨텍스트가 통째로 오염돼 이후 모든 연산이 실패하므로(2026-08-29 gte-multilingual에서
실측), 한 프로세스에서 연달아 돌리면 안 된다.

    python -m ... --build-pool
    python -m ... --model BAAI/bge-reranker-v2-m3 --label "bge-v2-m3"
    python -m ... --model none --label "리랭커 없음(항등)"
    python -m ... --report
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")

POOL = HERE / "reranker_pool.json"
RESULTS = HERE / "reranker_model_results"
TOP_K = 10
_STOP = {"전국", "전체", "국내", "없음", "nan", "", "-", "KOSIS"}


def _clean(v):
    v = (v or "").strip()
    return "" if v.lower() in {s.lower() for s in _STOP} else v


def _load_claims():
    from agent.interfaces import Claim
    from agent.preprocessing.claim_extractor import attach_sentence_context
    import queries as Q

    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    slots = json.loads((ROOT / "benchmark/search_experiment/claim_slots.json").read_text(encoding="utf-8"))
    arts, cmap = Q.load_articles(), Q.load_claim_article_map()
    by_art: dict[str, list[dict]] = {}
    for r in ev:
        by_art.setdefault(cmap.get(r["claim_id"], ""), []).append(r)
    claims = {}
    for aid, rows in by_art.items():
        objs = []
        for r in rows:
            s = slots.get(r["claim_id"], {})
            objs.append(Claim(sentence=r["sentence"], claim_type="규모",
                              statistic_expression=_clean(s.get("statistic_expression")) or None,
                              population=_clean(s.get("population")) or None,
                              region=_clean(s.get("region")) or None,
                              source_org=_clean(s.get("source_org")) or None))
        attach_sentence_context(arts.get(aid, ""), objs)
        for r, c in zip(rows, objs):
            claims[r["claim_id"]] = c
    return ev, claims


def build_pool() -> None:
    from agent.mapping import reranker as RK
    from agent.mapping.reranker import search_and_rerank, build_retrieval_query, build_lexical_query
    from agent.mapping.embedding_search import embedding_search, build_table_embedding_cache
    from agent.mapping.keyword_search import keyword_search
    from agent.kosis.query_vdb import batch_query_vdb, VDB_TOP_K, VdbUnavailableError
    from agent.kosis.bm25_search import bm25_query_vdb, BM25_TOP_K
    from agent.pipeline.batch_runner import _load_table_catalog_by_id

    ev, claims = _load_claims()
    ids = [r["claim_id"] for r in ev]
    catalog_by_id = _load_table_catalog_by_id()
    document_texts = {tid: t["embedding_text"] for tid, t in catalog_by_id.items()}
    embedding_cache = build_table_embedding_cache()

    from sentence_transformers import SentenceTransformer
    print("Qwen3-Embedding-4B 로딩...", flush=True)
    emb = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560, device="cuda")
    instruction = ("Given a Korean news claim sentence, retrieve the KOSIS statistical table "
                   "description that best matches it")
    os.environ["KOSIS_QUERY_CONTEXT"] = "1"

    def vdb_fn(claim):
        text = f"Instruct: {instruction}\nQuery: {build_retrieval_query(claim)}"
        vec = emb.encode([text], convert_to_numpy=True, normalize_embeddings=True)[0].tolist()
        try:
            return batch_query_vdb([vec], top_k=VDB_TOP_K)[0]
        except VdbUnavailableError:
            return []

    def bm25_fn(claim):
        try:
            return bm25_query_vdb(build_lexical_query(claim), top_k=BM25_TOP_K)
        except VdbUnavailableError:
            return []

    cur, captured = {"cid": None}, {}
    real = RK.rerank

    def capturing(claim, candidates, *, top_k=5, document_texts=None):
        captured[cur["cid"]] = [
            {"table_id": c.table_id, "table_name": c.table_name, "score": c.score,
             "source_meta": c.source_meta, "org_id": getattr(c, "org_id", None)}
            for c in candidates
        ]
        return real(claim, candidates, top_k=top_k, document_texts=document_texts)

    RK.rerank = capturing
    print("후보 풀 생성 중...", flush=True)
    t0 = time.time()
    for n, cid in enumerate(ids, 1):
        cur["cid"] = cid
        try:
            search_and_rerank(claims[cid], keyword_fn=keyword_search,
                              embedding_fn=lambda c: embedding_search(c, cache=embedding_cache),
                              vdb_fn=vdb_fn, bm25_fn=bm25_fn,
                              top_k=TOP_K, document_texts=document_texts)
        except Exception as e:  # noqa: BLE001
            print(f"  {cid} 검색 실패: {type(e).__name__}: {e}", flush=True)
            captured.setdefault(cid, [])
        if n % 35 == 0:
            print(f"  {n}/{len(ids)} ({time.time()-t0:.0f}s)", flush=True)
    RK.rerank = real

    gold = {r["claim_id"]: set(r["gold"]) for r in ev}
    sizes = [len(v) for v in captured.values()]
    inpool = [c for c in ids if gold[c] & {x["table_id"] for x in captured[c]}]
    POOL.write_text(json.dumps({"pool": captured, "document_texts": document_texts},
                               ensure_ascii=False), encoding="utf-8")
    print(f"완료 {time.time()-t0:.0f}s — 후보 평균 {sum(sizes)/len(sizes):.1f}개 "
          f"(최소 {min(sizes)}/최대 {max(sizes)})")
    print(f"후보 풀에 정답이 있는 claim: {len(inpool)}/{len(ids)} ({len(inpool)/len(ids):.1%})  <- 리랭커 상한")


def run_model(model_id: str, label: str, rerank_query: str | None = None) -> None:
    from agent.interfaces import TableCandidate
    from agent.mapping import reranker as RK

    data = json.loads(POOL.read_text(encoding="utf-8"))
    pool, document_texts = data["pool"], data["document_texts"]
    ev, claims = _load_claims()
    ids = [r["claim_id"] for r in ev]
    gold = {r["claim_id"]: set(r["gold"]) for r in ev}
    os.environ["KOSIS_RERANK_CONTEXT"] = "0"
    os.environ["KOSIS_RERANK_QUERY"] = rerank_query or "sentence"

    if model_id == "none":
        RK._DISABLE_RERANKER = True
    else:
        RK._DISABLE_RERANKER = False
        RK.RERANKER_MODEL = model_id
    RK._reranker_singleton = None

    ranks, t0 = {}, time.time()
    for cid in ids:
        cands = [TableCandidate(table_id=c["table_id"], table_name=c["table_name"],
                                score=c["score"], required_slots=[],
                                source_meta=c["source_meta"], org_id=c["org_id"])
                 for c in pool[cid]]
        out = RK.rerank(claims[cid], cands, top_k=TOP_K, document_texts=document_texts)
        got = [c.table_id for c in out]
        ranks[cid] = next((i + 1 for i, t in enumerate(got) if t in gold[cid]), None)
    sec = time.time() - t0

    RESULTS.mkdir(exist_ok=True)
    slug = label.replace("/", "_").replace(" ", "_")
    (RESULTS / f"{slug}.json").write_text(
        json.dumps({"label": label, "model": model_id, "ranks": ranks, "sec": sec},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    n1 = sum(1 for c in ids if ranks[c] == 1)
    print(f"  {label}: top-1 {n1}/{len(ids)} ({n1/len(ids):.1%}), {sec:.0f}s")


def report() -> None:
    data = json.loads(POOL.read_text(encoding="utf-8"))
    pool = data["pool"]
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    ids = [r["claim_id"] for r in ev]
    gold = {r["claim_id"]: set(r["gold"]) for r in ev}
    inpool = [c for c in ids if gold[c] & {x["table_id"] for x in pool[c]}]

    rows = []
    for f in sorted(RESULTS.glob("*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        rows.append(d)

    def acc(r, k):
        return sum(1 for c in ids if r.get(c) and r[c] <= k) / len(ids)

    def mrr(r):
        return sum(1.0 / r[c] for c in ids if r.get(c)) / len(ids)

    rows.sort(key=lambda d: -mrr(d["ranks"]))
    print(f"\n후보 풀에 정답이 있는 claim: {len(inpool)}/{len(ids)} ({len(inpool)/len(ids):.1%})  <- 리랭커 상한\n")
    print(f"{'모델':<26}{'top-1':>9}{'top-5':>9}{'top-10':>9}{'MRR':>9}{'claim당':>10}")
    for d in rows:
        r = d["ranks"]
        print(f"{d['label']:<26}" + "".join(f"{acc(r,k):>8.1%} " for k in (1, 5, 10))
              + f"{mrr(r):>8.3f}" + f"{d['sec']/len(ids):>9.2f}s")
    print(f"\n후보에 정답이 있던 {len(inpool)}건 중 1등으로 올린 비율")
    for d in rows:
        r = d["ranks"]
        got = sum(1 for c in inpool if r.get(c) == 1)
        print(f"  {d['label']:<26} {got:>3}/{len(inpool)}  ({got/len(inpool):.1%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-pool", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--label")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--rerank-query", help="sentence|context|struct|struct_sentence")
    a = ap.parse_args()
    if a.build_pool:
        build_pool()
    elif a.model:
        run_model(a.model, a.label or a.model, a.rerank_query)
    elif a.report:
        report()
