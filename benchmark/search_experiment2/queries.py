"""질의 변형 생성 — claim 필드 분해(실험 C), 문맥 보강(실험 D), 질의 확장(실험 F).

원칙: LLM을 새로 호출하지 않는다. 필드는 골든셋 2단계 시트(운영에서는 claim_extractor가
채우는 것과 같은 필드)를 쓰고, 확장은 규칙 사전 + PRF(의사 적합성 피드백)만 쓴다.
PRF는 실제 코퍼스에서 가져온 표명을 붙이는 것이라 없는 통계 개념을 지어내지 않는다.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

# ── 문맥(실험 D) ──────────────────────────────────────────────────────────
_SENT_END = re.compile(r"(?<=[.!?다])\s+")


def load_articles() -> dict[str, str]:
    df = pd.read_excel(ROOT / "notebooks/골든셋_통합.xlsx", sheet_name="1단계_기사목록")
    return {str(r["번호"]).strip(): str(r["본문(정제됨)"] or "") for _, r in df.iterrows()}


def load_claim_article_map() -> dict[str, str]:
    df = pd.read_excel(ROOT / "notebooks/골든셋_통합.xlsx", sheet_name="2단계_claim목록")
    return {str(r["claim_id"]).strip(): str(r["기사번호"]).strip() for _, r in df.iterrows()}


def split_sentences(body: str) -> list[str]:
    return [s.strip() for s in _SENT_END.split(body) if s.strip()]


def locate(sentences: list[str], claim: str) -> int:
    """claim 문장이 기사에서 몇 번째인지. 원문이 살짝 다를 수 있어 접두어로 찾는다."""
    head = claim.strip()[:20]
    for i, s in enumerate(sentences):
        if head and head in s:
            return i
    return -1


def context_variants(body: str, claim: str) -> dict[str, str]:
    """D-1 ~ D-5. 못 찾으면 claim 원문으로 대체(불이익 없이 baseline과 동일해짐)."""
    sents = split_sentences(body)
    i = locate(sents, claim)
    if i < 0:
        return {k: claim for k in ("D1", "D2", "D3", "D4", "D5")}
    prev = sents[i - 1] if i > 0 else ""
    nxt = sents[i + 1] if i + 1 < len(sents) else ""
    para = " ".join(sents[max(0, i - 3): i + 4])      # 문단 근사 — 앞뒤 3문장
    return {
        "D1": claim,
        "D2": f"{prev} {claim}".strip(),
        "D3": f"{claim} {nxt}".strip(),
        "D4": f"{prev} {claim} {nxt}".strip(),
        "D5": para,
    }


# ── 필드 분해(실험 C) ─────────────────────────────────────────────────────
_AGE = re.compile(r"(\d+\s*세\s*(?:이상|이하|미만)?|\d+\s*[~-]\s*\d+\s*세|\d0대|[가-힣]+세대)")
_STOP = {"전국", "전체", "국내", "없음", "nan", "", "-"}


def fields(slot: dict, sentence: str) -> dict[str, str]:
    def clean(v):
        v = (v or "").strip()
        return "" if v.lower() in _STOP else v

    ages = _AGE.findall(sentence)
    gender = " ".join(g for g in ("여성", "남성") if g in sentence)
    cond = " ".join(x for x in [clean(slot.get("population")), " ".join(ages), gender] if x)
    return {
        "measurement": clean(slot.get("statistic_expression")) or sentence,
        "population": clean(slot.get("population")),
        "region": clean(slot.get("region")),
        "condition": cond,
        "org": clean(slot.get("source_org")),
        "struct": " ".join(x for x in [clean(slot.get("statistic_expression")),
                                       clean(slot.get("population")),
                                       clean(slot.get("region"))] if x),
    }


# ── 질의 확장(실험 F) ─────────────────────────────────────────────────────
# KOSIS 표명에서 실제로 쓰이는 표현으로만 구성한 규칙 사전. 새 통계 개념을 만들지 않는다.
SYNONYMS = {
    "취업자": ["경제활동인구", "고용"],
    "실업": ["실업률", "실업자"],
    "쉬었음": ["비경제활동인구"],
    "소비자물가": ["물가지수"],
    "소매판매": ["소매판매액지수", "서비스업동향"],
    "산업생산": ["광공업생산지수", "전산업생산지수"],
    "설비투자": ["설비투자지수"],
    "건설기성": ["건설업"],
    "자살률": ["사망원인", "사망률"],
    "출생": ["출생아수", "인구동향"],
    "혼인": ["혼인건수"],
    "주택": ["주택가격", "주택보급률"],
    "전세": ["전세가격지수", "주택가격동향"],
    "수출": ["수출입", "무역"],
    "경기": ["경기종합지수"],
    "고령": ["고령자", "연령별"],
    "가구": ["가구원수", "가구주"],
    "소득": ["가계동향", "가계소득"],
}


def expand(text: str) -> str:
    extra = []
    for k, vs in SYNONYMS.items():
        if k in text:
            extra.extend(vs)
    return (text + " " + " ".join(dict.fromkeys(extra))).strip() if extra else text


def prf(base_query: str, top_table_names: list[str], n: int = 3) -> str:
    """의사 적합성 피드백 — 1차 검색 상위 표명을 질의에 덧붙인다(코퍼스에 실재하는 문자열)."""
    add = " ".join(top_table_names[:n])
    return f"{base_query} {add}".strip()


def build_all(eval_set: list[dict], slots: dict) -> dict[str, dict[str, str]]:
    """claim_id -> {질의이름: 텍스트}"""
    arts = load_articles()
    cmap = load_claim_article_map()
    out: dict[str, dict[str, str]] = {}
    for r in eval_set:
        cid, sent = r["claim_id"], r["sentence"]
        s = slots.get(cid, {})
        f = fields(s, sent)
        ctx = context_variants(arts.get(cmap.get(cid, ""), ""), sent)
        q = {"full": sent, **{f"ctx_{k}": v for k, v in ctx.items()},
             "measurement": f["measurement"], "population": f["population"],
             "region": f["region"], "condition": f["condition"], "struct": f["struct"],
             "expanded": expand(sent), "expanded_struct": expand(f["struct"])}
        out[cid] = {k: v for k, v in q.items() if v}
    return out


if __name__ == "__main__":
    ev = json.loads((ROOT / "benchmark/search_experiment/eval_set.json").read_text(encoding="utf-8"))
    sl = json.loads((ROOT / "benchmark/search_experiment/claim_slots.json").read_text(encoding="utf-8"))
    qs = build_all(ev, sl)
    print(f"claim {len(qs)}건, 질의 변형 {len(next(iter(qs.values())))}종")
    cid = ev[10]["claim_id"]
    for k, v in qs[cid].items():
        print(f"  [{k:<16}] {v[:96]}")
