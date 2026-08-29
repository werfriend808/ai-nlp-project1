"""문맥 보강 A/B — 운영 질의 조립(build_retrieval_query) 기준으로 재측정.

실험(REPORT.md)의 baseline은 claim 문장 원문(full)이었는데 운영은 search_query(HyDE 문구)를
우선 쓴다. "앞 문장 + search_query" 조합은 측정된 적이 없어서 여기서 확인한다.

A/B는 실험 재현(측정 하네스가 맞는지 검증), C/D가 실제로 알고 싶은 값.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")

from agent.interfaces import Claim
from agent.preprocessing.claim_extractor import attach_sentence_context
from agent.mapping.reranker import build_retrieval_query
from retrievers import connect, dense_tables, vec_literal
import queries as Q

DEPTH = 200
INSTRUCTION = ("Given a Korean news claim sentence, retrieve the KOSIS statistical table "
               "description that best matches it")
_STOP = {"전국", "전체", "국내", "없음", "nan", "", "-", "KOSIS"}


def clean(v):
    v = (v or "").strip()
    return "" if v.lower() in {s.lower() for s in _STOP} else v


def search_query_proxy(slot: dict) -> str:
    """운영 claim_extractor가 만드는 search_query 근사 —
    interfaces.py 주석의 정의 "{정규화 지표명} {있는 dimension만} {정규화 기관명}"."""
    parts = [clean(slot.get("statistic_expression")), clean(slot.get("population")),
             clean(slot.get("region")), clean(slot.get("source_org"))]
    return " ".join(p for p in parts if p).strip()


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    slots = json.loads((ROOT / "benchmark/search_experiment/claim_slots.json").read_text(encoding="utf-8"))
    arts, cmap = Q.load_articles(), Q.load_claim_article_map()

    # 기사 단위로 묶어 운영과 같은 방식으로 문맥을 채운다
    by_art: dict[str, list[dict]] = {}
    for r in ev:
        by_art.setdefault(cmap.get(r["claim_id"], ""), []).append(r)

    claims: dict[str, Claim] = {}
    for aid, rows in by_art.items():
        objs = []
        for r in rows:
            s = slots.get(r["claim_id"], {})
            objs.append(Claim(sentence=r["sentence"], claim_type="규모",
                              search_query=search_query_proxy(s) or None,
                              source_org=clean(s.get("source_org")) or None))
        attach_sentence_context(arts.get(aid, ""), objs)
        for r, c in zip(rows, objs):
            claims[r["claim_id"]] = c

    n_ctx = sum(1 for c in claims.values() if c.context_before)
    print(f"claim {len(claims)}건 / 문맥 확보 {n_ctx}건", flush=True)

    # 4종 질의 구성
    def variants(cid: str) -> dict[str, str]:
        c = claims[cid]
        ctx = c.context_before
        # A/B는 실험 재현 — 별칭 확장 없이 문장 원문 그대로
        a = c.sentence
        b = f"{ctx} {c.sentence}".strip() if ctx else c.sentence
        # C/D는 운영 조립 그대로 — build_retrieval_query()를 실제로 호출
        os.environ["KOSIS_QUERY_CONTEXT"] = "0"
        cc = build_retrieval_query(c)
        os.environ["KOSIS_QUERY_CONTEXT"] = "1"
        d = build_retrieval_query(c)
        return {"A_문장(실험baseline)": a, "B_문맥+문장(실험D2)": b,
                "C_운영현재(search_query)": cc, "D_운영+문맥": d}

    qmap = {cid: variants(cid) for cid in claims}
    names = list(next(iter(qmap.values())).keys())

    from sentence_transformers import SentenceTransformer
    print("모델 로딩...", flush=True)
    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=2560)

    ids = [r["claim_id"] for r in ev]
    results: dict[str, dict[str, list[str]]] = {}
    conn = connect(); cur = conn.cursor()
    for name in names:
        texts = [f"Instruct: {INSTRUCTION}\nQuery: {qmap[c][name]}" for c in ids]
        t0 = time.time()
        enc = model.encode(texts, batch_size=4, normalize_embeddings=True, show_progress_bar=False)
        results[name] = {}
        for cid, v in zip(ids, enc):
            results[name][cid] = dense_tables(cur, vec_literal(v), DEPTH)
        print(f"  {name}: {time.time()-t0:.0f}s", flush=True)
    cur.close(); conn.close()

    gold = {r["claim_id"]: set(r["gold"]) for r in ev}
    def recall(name, k):
        hit = sum(1 for c in ids if gold[c] & set(results[name][c][:k]))
        return hit / len(ids)

    print(f"\n{'질의':<26}{'R@1':>8}{'R@10':>8}{'R@100':>8}{'R@200':>8}")
    for name in names:
        print(f"{name:<26}" + "".join(f"{recall(name,k):>7.1%} " for k in (1, 10, 100, 200)))

    # D vs C 승패 (운영 기준 실제 효과)
    for base, new in [("C_운영현재(search_query)", "D_운영+문맥"), ("A_문장(실험baseline)", "B_문맥+문장(실험D2)")]:
        win = [c for c in ids if not gold[c] & set(results[base][c][:100]) and gold[c] & set(results[new][c][:100])]
        loss = [c for c in ids if gold[c] & set(results[base][c][:100]) and not gold[c] & set(results[new][c][:100])]
        print(f"\n{new} vs {base} (R@100): +{len(win)}건 / -{len(loss)}건 → 순증 {len(win)-len(loss):+d}")
        if loss:
            print(f"  손해: {loss}")

    (Path(__file__).parent / "ctx_prod_ab.json").write_text(
        json.dumps({"queries": qmap, "results": results}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
