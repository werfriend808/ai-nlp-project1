"""RRF 소스 가중치 A/B — 카탈로그 표가 소스 개수로 얻는 구조적 이점을 줄이면?

진단(2026-08-29): 1등을 뺏긴 정답 25건 중 19건이 "이긴 표가 소스를 더 많이 가진" 경우였다.
카탈로그(64개) 소속 표는 keyword_search와 embedding_search 둘 다에서 자동으로 순위를
얻으므로 RRF 합산에서 구조적으로 유리하다. 이걸 줄이면 VDB 단독 정답이 올라오는지 잰다.

주의: 소스가 적은 게 "카탈로그에 없어서"일 수도 있고 "실제로 관련성이 낮아서"일 수도 있다.
RRF는 원래 후자를 걸러내는 장치이므로, 가중치를 건드리면 지금 맞히는 것도 깨질 수 있다.
그래서 순증과 함께 "표 몇 개가 움직였는지"를 같이 본다 — 골든셋 70건이 정답표 19개에
몰려 있어(상위 3개가 40%) claim 수만 보면 표 한두 개 차이를 일반 결론으로 착각하게 된다.
"""
from __future__ import annotations
import json, os, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")

from agent.interfaces import Claim, TableCandidate
from agent.preprocessing.claim_extractor import attach_sentence_context
from agent.mapping import reranker as RK
from agent.mapping.reranker import RRF_K, _parse_rrf_ranks, _sigmoid, _RERANK_RAW_RE, rerank_scores
import queries as Q

_S = {"전국", "전체", "국내", "없음", "nan", "", "-", "KOSIS"}


def cl(v):
    v = (v or "").strip()
    return "" if v.lower() in {s.lower() for s in _S} else v


def fuse(cands, ranks_list, mode: str, k: int = RRF_K):
    """mode별 RRF 점수 계산. ranks_list[i] = {source: rank} for cands[i]."""
    scores = []
    for ranks in ranks_list:
        if not ranks:
            scores.append(0.0); continue
        vals = [1.0 / (k + r) for r in ranks.values()]
        if mode == "sum":            # 현재 방식 — 소스마다 더함
            s = sum(vals)
        elif mode == "mean":         # 소스 개수로 정규화
            s = sum(vals) / len(vals)
        elif mode == "max":          # 가장 강한 신호 하나만
            s = max(vals)
        elif mode == "sqrt":         # 절충 — 개수의 제곱근으로 나눔
            s = sum(vals) / (len(vals) ** 0.5)
        else:
            raise ValueError(mode)
        scores.append(s)
    order = sorted(range(len(cands)), key=lambda i: -scores[i])
    return [cands[i] for i in order]


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    slots = json.loads((ROOT / "benchmark/search_experiment/claim_slots.json").read_text(encoding="utf-8"))
    data = json.loads((HERE / "reranker_pool.json").read_text(encoding="utf-8"))
    pool, dt = data["pool"], data["document_texts"]
    arts, cmap = Q.load_articles(), Q.load_claim_article_map()

    by = {}
    for r in ev:
        by.setdefault(cmap.get(r["claim_id"], ""), []).append(r)
    claims = {}
    for aid, rows in by.items():
        o = [Claim(sentence=r["sentence"], claim_type="규모",
                   statistic_expression=cl(slots.get(r["claim_id"], {}).get("statistic_expression")) or None,
                   population=cl(slots.get(r["claim_id"], {}).get("population")) or None,
                   region=cl(slots.get(r["claim_id"], {}).get("region")) or None,
                   source_org=cl(slots.get(r["claim_id"], {}).get("source_org")) or None) for r in rows]
        attach_sentence_context(arts.get(aid, ""), o)
        for r, c in zip(rows, o):
            claims[r["claim_id"]] = c

    ids = [r["claim_id"] for r in ev]
    gold = {r["claim_id"]: set(r["gold"]) for r in ev}
    gtbl = {r["claim_id"]: r["gold"][0] for r in ev}
    os.environ["KOSIS_RERANK_QUERY"] = "sentence"
    RK._reranker_singleton = None

    # 리랭커 점수는 mode와 무관하므로 한 번만 계산해 재사용한다.
    cache = {}
    for cid in ids:
        cands = [TableCandidate(table_id=c["table_id"], table_name=c["table_name"], score=c["score"],
                                required_slots=[], source_meta=c["source_meta"], org_id=c["org_id"])
                 for c in pool[cid]]
        docs = [(dt or {}).get(c.table_id, c.table_name) for c in cands]
        sc = rerank_scores(claims[cid].sentence, docs)
        if sc is None:
            sc = [0.0] * len(cands)
        order = sorted(range(len(cands)), key=lambda i: -sc[i])
        rr = {}
        for pos, i in enumerate(order, 1):
            rr[i] = pos
        ranks_list = []
        for i, c in enumerate(cands):
            r = dict(_parse_rrf_ranks(c.source_meta))
            r["reranker_rank"] = rr[i]
            ranks_list.append(r)
        cache[cid] = (cands, ranks_list)

    MODES = [("sum (현재)", "sum"), ("sqrt", "sqrt"), ("mean", "mean"), ("max", "max")]
    out = {}
    for label, mode in MODES:
        ranks = {}
        for cid in ids:
            cands, rl = cache[cid]
            fused = fuse(cands, rl, mode)
            ranks[cid] = next((i + 1 for i, c in enumerate(fused) if c.table_id in gold[cid]), None)
        out[label] = ranks

    def acc(r, k):
        return sum(1 for c in ids if r[c] and r[c] <= k) / len(ids)

    def mrr(r):
        return sum(1.0 / r[c] for c in ids if r[c]) / len(ids)

    print(f"{'RRF 방식':<16}{'top-1':>9}{'top-5':>9}{'top-10':>9}{'MRR':>9}")
    for label, _ in MODES:
        r = out[label]
        print(f"{label:<16}" + "".join(f"{acc(r,k):>8.1%} " for k in (1, 5, 10)) + f"{mrr(r):>8.3f}")

    base = out["sum (현재)"]
    for label, _ in MODES[1:]:
        r = out[label]
        win = [c for c in ids if base[c] != 1 and r[c] == 1]
        loss = [c for c in ids if base[c] == 1 and r[c] != 1]
        wt, lt = Counter(gtbl[c] for c in win), Counter(gtbl[c] for c in loss)
        print(f"\n{label} vs 현재 (top-1): +{len(win)} / -{len(loss)} → 순증 {len(win)-len(loss):+d}")
        print(f"   이득 claim {len(win)}건이 걸친 표 {len(wt)}종: {dict(wt)}")
        print(f"   손해 claim {len(loss)}건이 걸친 표 {len(lt)}종: {dict(lt)}")


if __name__ == "__main__":
    main()
