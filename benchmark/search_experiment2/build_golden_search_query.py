"""골든셋 70건에 대한 진짜 search_query를 claim_extractor로 받아 저장한다.

지금까지 골든셋에는 search_query가 없어서, 운영과 같은 조건(운영은 search_query로 검색한다)
으로는 어떤 실험도 할 수 없었다. 여기서 한 번 만들어 두면 이후 실험이 재사용한다.

주의: extract_claims는 비결정적이라(claim_extractor.py 주석 참고) 여기서 얻는 값은
"가능한 값 중 한 표본"이다. 또 추출기가 골든셋 70문장을 전부 다시 뽑아준다는 보장이 없어
매칭에 실패한 건은 search_query가 비게 된다 — 커버리지를 같이 기록한다.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(Path(__file__).parent))
from dotenv import load_dotenv; load_dotenv(ROOT / ".env")

from agent.preprocessing.claim_extractor import (
    ClaimExtractorError, extract_claims, _locate_sentence, split_sentences,
)
from agent.preprocessing.claim_candidate_scanner import dedupe_repeated_sentences
import queries as Q

OUT = Path(__file__).parent / "golden_search_query.json"


def main() -> None:
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    arts, cmap = Q.load_articles(), Q.load_claim_article_map()

    by_art: dict[str, list[dict]] = {}
    for r in ev:
        by_art.setdefault(cmap.get(r["claim_id"], ""), []).append(r)
    print(f"기사 {len(by_art)}건 / claim {len(ev)}건", flush=True)

    out: dict[str, dict] = {}
    for n, (aid, rows) in enumerate(sorted(by_art.items()), 1):
        body = arts.get(aid, "")
        t0 = time.time()
        try:
            extracted = extract_claims(body)
        except ClaimExtractorError as e:
            print(f"  [{n}/{len(by_art)}] 기사 {aid}: 추출 실패 {e}", flush=True)
            extracted = []
        # 추출된 claim을 정제 본문 기준 문장 위치로 색인 -> 골든 문장과 같은 자리면 매칭
        sents = split_sentences(dedupe_repeated_sentences(body))
        pos: dict[int, object] = {}
        for c in extracted:
            i = _locate_sentence(sents, c.sentence)
            if i >= 0 and i not in pos:
                pos[i] = c

        hit = 0
        for r in rows:
            gi = _locate_sentence(sents, r["sentence"])
            c = pos.get(gi) if gi >= 0 else None
            sq = (c.search_query if c else None) or None
            if sq:
                hit += 1
            out[r["claim_id"]] = {
                "sentence": r["sentence"],
                "search_query": sq,
                "source_org": (c.source_org if c else None),
                "statistic_expression": (c.statistic_expression if c else None),
                "matched": c is not None,
            }
        print(f"  [{n}/{len(by_art)}] 기사 {aid}: 추출 {len(extracted)}건, "
              f"골든 {len(rows)}건 중 search_query 확보 {hit}건 ({time.time()-t0:.0f}s)", flush=True)

    got = sum(1 for v in out.values() if v["search_query"])
    print(f"\n총 {len(out)}건 중 search_query 확보 {got}건 ({got/len(out):.1%})")
    lens = [len(v["search_query"]) for v in out.values() if v["search_query"]]
    if lens:
        print(f"길이: 평균 {sum(lens)/len(lens):.1f}자, 최소 {min(lens)}, 최대 {max(lens)}")
        for cid, v in list(out.items())[:8]:
            print(f"  {cid}: {v['search_query']!r}")
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
