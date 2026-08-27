"""검색기 조합을 그리디 전방선택으로 찾는다 — 선택은 dev에서만, 보고는 test로.

수작업 조합(COMBO-A~C)은 단독 1위 검색기를 빠뜨리는 등 임의성이 있었다. 여기서는
dev Recall@100을 최대화하는 검색기를 하나씩 추가하고, 확정된 조합을 test에 한 번만 적용한다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fuse import K_LIST, RRF_K, aggregate, per_claim_metrics, split_dev_test  # noqa: E402
from retrievers import rrf  # noqa: E402

HERE = Path(__file__).parent
ROOT = Path(__file__).resolve().parents[2]
MAX_RETRIEVERS = 7


def evaluate(runs, ev, gold, names, subset=None) -> dict:
    w = {n: 1.0 for n in names}
    per = []
    for r in ev:
        cid = r["claim_id"]
        if subset is not None and cid not in subset:
            continue
        f = rrf({n: runs[n].get(cid, []) for n in names}, k=RRF_K, weights=w, top_k=max(K_LIST))
        per.append(per_claim_metrics(f, gold[cid]))
    return aggregate(per, [0.0])


def select(runs, ev, gold, dev, test, lat_mean, pool, label):
    chosen: list[str] = []
    hist = []
    print(f"\n{label} (후보 {len(pool)}개, dev {len(dev)}건 기준)\n")
    for step in range(MAX_RETRIEVERS):
        best, best_score = None, -1.0
        for cand in pool:
            if cand in chosen:
                continue
            s = evaluate(runs, ev, gold, chosen + [cand], dev)["recall@100"]
            if s > best_score:
                best, best_score = cand, s
        prev = hist[-1]["dev_r100"] if hist else 0.0
        if best is None or best_score <= prev:
            print(f"  step {step+1}: dev 개선 없음 — 중단")
            break
        chosen.append(best)
        d = evaluate(runs, ev, gold, chosen, dev)
        t = evaluate(runs, ev, gold, chosen, test)
        a = evaluate(runs, ev, gold, chosen, None)
        ms = sum(lat_mean[n] for n in chosen)
        hist.append({"step": step + 1, "added": best, "chosen": list(chosen),
                     "dev_r100": d["recall@100"], "test_r100": t["recall@100"],
                     "all_r1": a["recall@1"], "all_r10": a["recall@10"],
                     "all_r50": a["recall@50"], "all_r100": a["recall@100"],
                     "all_r200": a["recall@200"], "mrr": a["mrr"],
                     "ndcg@10": a["ndcg@10"], "latency_ms": ms})
        print(f"  step {step+1}: +{best:<20} dev {d['recall@100']*100:5.1f}%  "
              f"test {t['recall@100']*100:5.1f}%  전체 {a['recall@100']*100:5.1f}%  {ms:.0f}ms")
    return hist


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    runs = json.loads((HERE / "runs.json").read_text(encoding="utf-8"))
    lat = json.loads((HERE / "latency.json").read_text(encoding="utf-8"))
    lat_mean = {k: sum(v) / len(v) for k, v in lat.items()}
    gold = {r["claim_id"]: set(r["gold"]) for r in ev}
    dev, test = split_dev_test(ev)

    pool = [n for n in runs if n != "trgm_struct"]   # 느리고 성능도 없어 후보에서 제외
    full = select(runs, ev, gold, dev, test, lat_mean, pool, "전체 검색기 그리디 선택")
    fast_pool = [n for n in pool if lat_mean[n] < 200]
    fast = select(runs, ev, gold, dev, test, lat_mean, fast_pool,
                  "빠른 검색기만(200ms 미만) 그리디 선택")

    (HERE / "greedy.json").write_text(json.dumps({"full": full, "fast": fast},
                                                 ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n저장: greedy.json")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
