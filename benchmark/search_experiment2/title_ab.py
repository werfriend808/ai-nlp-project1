"""기사 제목을 검색 질의에 붙이면 나아지는가 — dense 검색 단계 측정.

claim 문장에 지표명이 없어 실패하는 사례(REPORT.md 유형 1)에서 제목에는 그 지표명이
있는 것이 확인됐다. 문맥(앞 문장)과 제목 중 무엇이, 혹은 둘 다가 효과적인지 잰다.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")

import pandas as pd
from agent.interfaces import Claim
from agent.preprocessing.claim_extractor import attach_sentence_context
from agent.mapping.reranker import build_retrieval_query
from retrievers import connect, dense_tables, vec_literal
import queries as Q

DEPTH = 200
INSTRUCTION = ("Given a Korean news claim sentence, retrieve the KOSIS statistical table "
               "description that best matches it")


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    arts, cmap = Q.load_articles(), Q.load_claim_article_map()
    df = pd.read_excel(ROOT / "notebooks/골든셋_통합.xlsx", sheet_name="1단계_기사목록")
    titles = {str(r["번호"]).strip(): str(r["기사제목"] or "") for _, r in df.iterrows()}

    by_art: dict[str, list[dict]] = {}
    for r in ev:
        by_art.setdefault(cmap.get(r["claim_id"], ""), []).append(r)
    claims: dict[str, Claim] = {}
    for aid, rows in by_art.items():
        objs = [Claim(sentence=r["sentence"], claim_type="규모",
                      article_title=titles.get(aid) or None) for r in rows]
        attach_sentence_context(arts.get(aid, ""), objs)
        for r, c in zip(rows, objs):
            claims[r["claim_id"]] = c

    ids = [r["claim_id"] for r in ev]
    gold = {r["claim_id"]: set(r["gold"]) for r in ev}
    n_t = sum(1 for c in claims.values() if c.article_title)
    n_c = sum(1 for c in claims.values() if c.context_before)
    print(f"claim {len(ids)}건 / 제목 있음 {n_t}건 / 문맥 있음 {n_c}건", flush=True)

    CONFIGS = [
        ("A 문장만",            "0", "0"),
        ("B 문맥+문장 (현재)",   "1", "0"),
        ("C 제목+문장",          "0", "1"),
        ("D 제목+문맥+문장",     "1", "1"),
    ]

    from sentence_transformers import SentenceTransformer
    print("모델 로딩...", flush=True)
    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560, device="cuda")
    conn = connect(); cur = conn.cursor()
    results: dict[str, dict[str, list[str]]] = {}
    samples: dict[str, str] = {}

    for name, ctx, title in CONFIGS:
        os.environ["KOSIS_QUERY_CONTEXT"] = ctx
        os.environ["KOSIS_QUERY_TITLE"] = title
        texts = [f"Instruct: {INSTRUCTION}\nQuery: {build_retrieval_query(claims[c])}" for c in ids]
        samples[name] = build_retrieval_query(claims["36-03"])
        t0 = time.time()
        enc = model.encode(texts, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
        results[name] = {c: dense_tables(cur, vec_literal(v), DEPTH) for c, v in zip(ids, enc)}
        print(f"  {name}: {time.time()-t0:.0f}s", flush=True)
    cur.close(); conn.close()

    def rec(name, k):
        return sum(1 for c in ids if gold[c] & set(results[name][c][:k])) / len(ids)

    print(f"\n{'질의 구성':<22}{'R@1':>9}{'R@10':>9}{'R@100':>9}{'R@200':>9}")
    for name, _, _ in CONFIGS:
        print(f"{name:<22}" + "".join(f"{rec(name,k):>8.1%} " for k in (1, 10, 100, 200)))

    base = "B 문맥+문장 (현재)"
    for name, _, _ in CONFIGS:
        if name == base:
            continue
        win = [c for c in ids if not gold[c] & set(results[base][c][:100]) and gold[c] & set(results[name][c][:100])]
        loss = [c for c in ids if gold[c] & set(results[base][c][:100]) and not gold[c] & set(results[name][c][:100])]
        print(f"\n{name} vs {base} (R@100): +{len(win)} / -{len(loss)} → 순증 {len(win)-len(loss):+d}")
        if win:  print(f"  이득: {win}")
        if loss: print(f"  손해: {loss}")

    print("\n질의 예시 (36-03):")
    for name, _, _ in CONFIGS:
        print(f"  [{name}] {samples[name][:110]}")

    (Path(__file__).parent / "title_ab.json").write_text(
        json.dumps(results, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
