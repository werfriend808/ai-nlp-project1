"""확신도 게이트(is_rrf_trusted)가 무엇을 걸러내고 무엇을 버리는지.

3단계 최상위 후보에 게이트를 걸었을 때와 안 걸었을 때를 비교한다. 게이트를 통과 못 한
claim은 파이프라인이 "표매칭_불충분"으로 버려 4~8단계로 가지 못한다.

게이트는 오답을 막으려고 2026-08-20에 넣었는데, 그 대가로 정답을 몇 건 버리는지는
측정된 적이 없다. 그 교환비를 잰다.
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")
os.environ.setdefault("HF_HOME", "/home/ubuntu/data/hf_cache")

from agent.interfaces import Claim, TableCandidate
from agent.preprocessing.claim_extractor import attach_sentence_context
from agent.mapping import reranker as RK
from agent.mapping.reranker import is_rrf_trusted, MIN_RERANKER_CONFIDENCE, _sigmoid, _RERANK_RAW_RE
import queries as Q

POOL = HERE / "reranker_pool.json"
_STOP = {"전국", "전체", "국내", "없음", "nan", "", "-", "KOSIS"}


def clean(v):
    v = (v or "").strip()
    return "" if v.lower() in {s.lower() for s in _STOP} else v


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    slots = json.loads((ROOT / "benchmark/search_experiment/claim_slots.json").read_text(encoding="utf-8"))
    arts, cmap = Q.load_articles(), Q.load_claim_article_map()
    data = json.loads(POOL.read_text(encoding="utf-8"))
    pool, document_texts = data["pool"], data["document_texts"]

    by_art = {}
    for r in ev:
        by_art.setdefault(cmap.get(r["claim_id"], ""), []).append(r)
    claims = {}
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
    os.environ["KOSIS_RERANK_QUERY"] = "sentence"
    RK._reranker_singleton = None

    rows = []
    for cid in ids:
        cands = [TableCandidate(table_id=c["table_id"], table_name=c["table_name"],
                                score=c["score"], required_slots=[],
                                source_meta=c["source_meta"], org_id=c["org_id"])
                 for c in pool[cid]]
        out = RK.rerank(claims[cid], cands, top_k=5, document_texts=document_texts)
        if not out:
            rows.append({"cid": cid, "top": None, "correct": False, "passed": False, "conf": None})
            continue
        top = out[0]
        m = _RERANK_RAW_RE.search(top.source_meta or "")
        conf = _sigmoid(float(m.group(1))) if m else None
        rows.append({
            "cid": cid, "top": top.table_id,
            "correct": top.table_id in gold[cid],
            "passed": bool(top.source_meta and is_rrf_trusted(top.source_meta)),
            "conf": conf, "meta": top.source_meta,
        })

    n = len(rows)
    off_ok = sum(1 for r in rows if r["correct"])
    passed = [r for r in rows if r["passed"]]
    dropped = [r for r in rows if not r["passed"]]
    on_ok = sum(1 for r in passed if r["correct"])
    lost = sum(1 for r in dropped if r["correct"])
    blocked_wrong = sum(1 for r in dropped if not r["correct"])

    print(f"골든셋 {n}건 / 확신도 임계값 MIN_RERANKER_CONFIDENCE = {MIN_RERANKER_CONFIDENCE}\n")
    print(f"{'':<26}{'매칭 시도':>10}{'정답':>8}{'오답':>8}{'정밀도':>10}")
    print(f"{'게이트 OFF (전부 통과)':<26}{n:>10}{off_ok:>8}{n-off_ok:>8}{off_ok/n:>9.1%}")
    print(f"{'게이트 ON (현재)':<26}{len(passed):>10}{on_ok:>8}{len(passed)-on_ok:>8}"
          f"{(on_ok/len(passed) if passed else 0):>9.1%}")
    print(f"\n게이트가 버린 {len(dropped)}건의 내역")
    print(f"  오답을 막음(의도한 효과) : {blocked_wrong}건")
    print(f"  정답을 버림(대가)        : {lost}건   {[r['cid'] for r in dropped if r['correct']]}")
    print(f"\n최종 정답 수:  게이트 OFF {off_ok}건  →  게이트 ON {on_ok}건  ({on_ok-off_ok:+d})")

    print(f"\n통과 사유 분해 (통과 {len(passed)}건)")
    from collections import Counter
    why = Counter()
    for r in passed:
        meta = r["meta"] or ""
        if "keyword_rank" in meta: why["keyword_rank"] += 1
        elif any(k in meta for k in ("population_rank","institution_rank","gender_rank","region_rank")): why["신호_rank"] += 1
        else: why["리랭커 1위+확신도"] += 1
    for k, v in why.most_common():
        print(f"  {k:<22} {v}건")

    confs = [r["conf"] for r in rows if r["conf"] is not None]
    if confs:
        confs_sorted = sorted(confs)
        print(f"\n최상위 후보 확신도 분포 (n={len(confs)})")
        print(f"  최소 {min(confs):.3f} / 중앙 {confs_sorted[len(confs)//2]:.3f} / 최대 {max(confs):.3f}")
        for th in (0.1, 0.3, 0.5, 0.7, 0.9):
            keep = [r for r in rows if r["conf"] is not None and r["conf"] >= th]
            ok = sum(1 for r in keep if r["correct"])
            print(f"  임계값 {th:.1f} → 통과 {len(keep):>2}건, 그중 정답 {ok:>2}건 "
                  f"(정밀도 {ok/len(keep) if keep else 0:.1%}, 전체 대비 회수 {ok/n:.1%})")


if __name__ == "__main__":
    main()
