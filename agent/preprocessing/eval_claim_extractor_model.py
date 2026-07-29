"""
agent/preprocessing/eval_claim_extractor_model.py — 2단계 claim_extractor 모델 교체 비교 (개발 보조 도구)

역할: 파이프라인 코드가 아니라, data.csv에서 관련도 True인 기사 몇 건을 뽑아 extract_claims()를
HCX-003과 HCX-DASH-002 두 모델로 각각 돌려서 지연시간/추출량/필드 완성도/형식 안정성을 비교하는
1회성 점검 스크립트.

※ data.csv에는 "정답 Claim 리스트"가 없어서(classifier의 TRUE/FALSE 라벨 같은 게 없음) 정밀도/
  재현율 같은 정답 대비 정확도는 잴 수 없습니다. 대신 자동으로 셀 수 있는 대리 지표(proxy metric)
  4개로 비교합니다 — 절대적인 "정확도"가 아니라 두 모델 사이의 상대 비교 용도입니다.

사용법 (프로젝트 루트에서):
    python -m agent.preprocessing.eval_claim_extractor_model
"""

from __future__ import annotations

import csv
import random
import re
import time
from pathlib import Path

from agent.interfaces import Claim
from agent.preprocessing.claim_extractor import ClaimExtractorError, extract_claims

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "data.csv"
MODELS = ["HCX-003", "HCX-DASH-002"]

# claim_extractor_prompt.txt의 few-shot 3개(쌀 소비량/KDI 성장률/외환보유액)에 이미 쓰인 기사는
# 샘플에서 제외 (모델이 "본 적 있는" 예시로 테스트하면 일반화 성능을 과대평가하게 됨).
FEWSHOT_TITLES = {
    "국민 1인당 쌀 소비량 40년 연속 줄어들어",
    "KDI 올해 2% 성장 전망, 석 달 만에 0.4%포인트 낮춰",
    "4000억달러대 지켰지만… 외환 보유액 5년 만에 최소",
}

VALID_CLAIM_TYPES = {"규모", "증감률", "비교", "전망"}
_DIGIT_RE = re.compile(r"\d")


def load_true_rows() -> list[tuple[str, str]]:
    with open(DATA_PATH, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [
            (row[0], row[3])
            for row in reader
            if len(row) == 5 and row[4].strip() == "True" and row[0] not in FEWSHOT_TITLES
        ]


def _quality(claims: list[Claim]) -> dict:
    """claim 리스트 하나를 4개 대리 지표로 요약.

    - n: 추출된 claim 개수
    - numeric_ratio: sentence에 숫자가 실제로 포함된 claim 비율
      (2단계 정의상 "수치 기반 주장"만 뽑아야 하므로, 숫자 없는 문장이 섞이면 오탐)
    - field_fill_ratio: period/unit/population 3개 필드 중 채워진 비율
      (구조화 정보를 얼마나 상세히 뽑는지 — 4단계 slot_filler가 활용할 수 있는 정보량)
    - type_valid_ratio: claim_type이 interfaces.py에 정의된 4종(규모/증감률/비교/전망)
      안에 드는 비율 (형식 지시 준수도)
    """
    n = len(claims)
    if n == 0:
        return {"n": 0, "numeric_ratio": 0.0, "field_fill_ratio": 0.0, "type_valid_ratio": 0.0}
    numeric = sum(1 for c in claims if _DIGIT_RE.search(c.sentence))
    fields = sum(sum(1 for v in (c.period, c.unit, c.population) if v) for c in claims)
    type_valid = sum(1 for c in claims if c.claim_type in VALID_CLAIM_TYPES)
    return {
        "n": n,
        "numeric_ratio": numeric / n,
        "field_fill_ratio": fields / (n * 3),
        "type_valid_ratio": type_valid / n,
    }


def run_one(body: str, model: str) -> dict:
    """extract_claims() 호출 하나를 실행하고 결과를 요약 dict로 돌려줌.

    "ok"는 JSON 파싱 성공 여부만 뜻하지 않음 — HCX-003이 긴 기사에서 종종 내는
    "40003 Context length exceeded"처럼 요청 자체가 거부되는 경우(requests.HTTPError)도
    여기서 실패로 잡히므로, "호출이 끝까지 살아남아 Claim 리스트를 돌려줬는지"에 가까움.
    claims 원본도 같이 담아서, 실제로 어떤 문장을 뽑았는지 사람이 눈으로 대조할 수 있게 함.
    """
    start = time.perf_counter()
    try:
        claims = extract_claims(body, model=model)
        elapsed = time.perf_counter() - start
        return {"ok": True, "elapsed": elapsed, "claims": claims, **_quality(claims)}
    except (ClaimExtractorError, Exception) as e:  # noqa: BLE001 - 점검 스크립트라 실패해도 계속 진행
        elapsed = time.perf_counter() - start
        return {
            "ok": False,
            "elapsed": elapsed,
            "claims": [],
            "n": 0,
            "numeric_ratio": 0.0,
            "field_fill_ratio": 0.0,
            "type_valid_ratio": 0.0,
            "error": str(e)[:80],
        }


def _format_claim(c: Claim) -> str:
    return (
        f'"{c.sentence}" '
        f"(type={c.claim_type}, period={c.period or '-'}, unit={c.unit or '-'}, "
        f"population={c.population or '-'})"
    )


def _print_claim_diff(per_model_claims: dict[str, list[Claim]]) -> None:
    """모델별로 뽑은 claim 문장을 나란히 출력하고, 상대 모델엔 없는 문장(완전 일치 기준)에
    표시를 붙임. sentence는 모델마다 표현이 달라 완전 일치가 아니면 안 걸릴 수 있음 —
    그런 애매한 경우는 사람이 직접 눈으로 대조해야 함(자동 판별 한계)."""
    sentence_sets = {m: {c.sentence for c in claims} for m, claims in per_model_claims.items()}
    for model, claims in per_model_claims.items():
        other_models = [m for m in per_model_claims if m != model]
        print(f"  [{model}] {len(claims)}개")
        if not claims:
            print("      (없음)")
            continue
        for c in claims:
            in_others = all(c.sentence in sentence_sets[m] for m in other_models)
            tag = "" if in_others else "  ← 다른 모델엔 없음(문자열 완전일치 기준, 표현 차이일 수도 있음)"
            print(f"    - {_format_claim(c)}{tag}")


def main(n_articles: int = 8, seed: int = 42) -> None:
    rows = load_true_rows()
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n_articles, len(rows)))

    per_model: dict[str, list[dict]] = {m: [] for m in MODELS}

    for title, body in sample:
        print(f"\n{'=' * 70}\n{title[:50]}")
        per_model_claims: dict[str, list[Claim]] = {}
        for model in MODELS:
            r = run_one(body, model)
            per_model[model].append(r)
            per_model_claims[model] = r["claims"]
            status = "OK" if r["ok"] else f"FAIL({r.get('error')})"
            print(
                f"  [{model:14}] {status:16} {r['elapsed']:5.2f}s  claims={r['n']:2}  "
                f"수치포함={r['numeric_ratio']:.0%}  필드채움={r['field_fill_ratio']:.0%}  "
                f"claim_type유효={r['type_valid_ratio']:.0%}"
            )
        print("  --- 실제 추출 문장 (직접 비교용) ---")
        _print_claim_diff(per_model_claims)

    print(f"\n{'=' * 70}\n=== 요약 (n={len(sample)}개 기사, 모델당 {len(sample)}회 호출) ===")
    for model in MODELS:
        results = per_model[model]
        ok = [r for r in results if r["ok"]]
        fail_n = len(results) - len(ok)

        def avg(key: str) -> float:
            return sum(r[key] for r in ok) / len(ok) if ok else 0.0

        print(f"\n[{model}]")
        print(f"  호출 성공(요청 거부/파싱 실패 없이 끝까지 완료): {len(ok)}/{len(results)}건 (실패 {fail_n}건)")
        print(f"  평균 응답 시간: {avg('elapsed'):.2f}초")
        print(f"  기사당 평균 추출 claim 수: {avg('n'):.2f}개")
        print(f"  claim 문장에 수치 포함 비율: {avg('numeric_ratio'):.1%}")
        print(f"  period/unit/population 필드 채움 비율: {avg('field_fill_ratio'):.1%}")
        print(f"  claim_type 4종 값 준수 비율: {avg('type_valid_ratio'):.1%}")


if __name__ == "__main__":
    main()
