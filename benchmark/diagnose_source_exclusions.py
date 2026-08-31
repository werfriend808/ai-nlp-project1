"""benchmark/diagnose_source_exclusions.py — 2단계 출처 필터가 뭘, 왜 걸러내는지 실측 집계.

배경: "파이프라인 실패 해부" 보고서(무작위 50건 샘플)에서 출처 필터가 168건(67.7%)을
걸러낸다고 나왔는데, 이 숫자 자체는 "버그"가 아니다 — 기사에 나오는 claim 중 상당수는
정말로 KOSIS 국가승인통계가 아닌 해외기관/민간기업/개인 발언이라 걸러지는 게 맞다.
문제는 그 168건 중 "진짜 걸러져야 할 것"과 "화이트리스트 기관인데 source_org를
못 채워서/못 매칭해서 억울하게 걸러진 것"이 섞여 있다는 것 — 이 스크립트는 그 둘을
구분해서 실제로 손댈 가치가 있는 버그가 몇 건인지 숫자로 보여준다.

agent.pipeline.batch_runner를 그대로 import하지 않는다 — 그 모듈은 3단계 이후
(psycopg2/임베딩 모델 등 GPU 서버 전용 의존성)까지 한 파일에 다 물려 있어서, 로컬
PC에서 1~2단계만 쓰고 싶어도 import 자체가 막힌다(2026-08-31 실측: ModuleNotFoundError:
psycopg2). 그래서 1~2단계에 필요한 순수 로직만(CSV 로더/기사 정제/재시도 래퍼) 이
파일 안에 그대로 복제해 온다 — batch_runner.py의 동명 함수와 로직이 갈라지지 않게
주의(있는 그대로 옮김, 수정 없음).

용법:
    python -m benchmark.diagnose_source_exclusions --n 50 --seed 42

--n/--seed는 batch_runner.load_articles_from_csv와 같은 기본값(42)을 써서, 보고서가
썼을 법한 표본과 최대한 비슷한 무작위 50건을 재현한다(완전히 같은 표본이라는 보장은
없다 — 보고서 생성 당시 정확히 어떤 --csv-n/--csv-seed였는지 기록이 없으면 실측
재현일 뿐이라는 점을 결과 해석 시 감안할 것).

버킷 정의 (filter_verifiable_claims의 5개 조건과 1:1 대응):
    no_source_org        : resolve_claim_sources 이후에도 source_org가 빈 채로 남음
    unmatched_org         : source_org는 있는데 classify_source가 kosis_verified가 아님
                             (여기가 "화이트리스트 기관인데 못 알아본" 진짜 버그가 숨을 곳)
    hallucinated_value     : 문장에 숫자가 아예 없는데 value가 채워짐
    value_mismatch         : 문장에 숫자는 있는데 value/comparison_value와 다름
    missing_value          : claim_type=규모/증감률인데 value가 None
    enforcement_or_eval    : 단속/제재/평가 등 통계 아닌 행정조치 키워드
    passed                 : 전부 통과 (3단계로 넘어감)

unmatched_org 버킷은 예시를 전부 출력한다 — 눈으로 훑어서 "어? 이거 화이트리스트에
있어야 하는 기관 아냐?"인 것과 "이건 원래 못 걸러야 정상"인 것을 사람이 직접 나눠야
한다(자동 판별은 못 한다 — 화이트리스트 자체를 고치는 게 이 스크립트의 목적이므로).
"""

from __future__ import annotations

import argparse
import csv
import io
import random
import re
import time
from dataclasses import dataclass
from collections import Counter
from pathlib import Path
from typing import Optional

from agent.preprocessing.claim_candidate_scanner import _normalize_quotes
from agent.preprocessing.claim_extractor import (
    extract_claims,
    recover_missed_claims,
    strip_title_prefix_from_claims,
    correct_scale_errors,
)
from agent.preprocessing.classifier import classify
from agent.preprocessing.source_filter import (
    _has_hallucinated_value,
    _looks_like_enforcement_or_evaluation,
    _missing_value_for_magnitude_claim,
    _value_mismatches_sentence,
    classify_source,
    resolve_claim_sources,
)

# ── 아래 세 개는 agent/pipeline/batch_runner.py에서 그대로 복제 (수정 없이) ──────

DATA_CSV_PATH = Path(__file__).parent.parent / "data" / "data_set.csv"

_BYLINE_RE = re.compile(
    r"[가-힣]+\s*기자\s*입력\s*\d{4}\d{2}\d{2}\s*\d{2}:\d{2}"
    r"(?:\s*업데이트\s*\d{4}\.\d{2}\.\d{2}\.\s*\d{2}:\d{2})?"
    r"\s*\d+\s*"
)

_JUNK_SECTION_RE = re.compile(
    r"By\s*Taboola|많이\s*본\s*뉴스|오늘의\s*멤버십|AI\s*추천"
    r"|#\S+(?:\s+#\S+){1,}"
    r"|구독수\s*\d+"
)


def _clean_scraped_article_text(title: str, raw_text: str, max_len: int = 3000) -> str:
    normalized_title = _normalize_quotes(title)
    normalized_text = _normalize_quotes(raw_text)
    anchor = normalized_title[:12].strip()
    idx = normalized_text.find(anchor) if anchor else -1
    start = idx if idx >= 0 else 0
    trimmed = raw_text[start : start + max_len]

    if idx >= 0 and title and _normalize_quotes(trimmed).startswith(normalized_title):
        trimmed = title + "\n" + trimmed[len(title) :]

    m = _BYLINE_RE.search(trimmed[: len(title) + 100])
    if m:
        trimmed = trimmed[: m.start()] + trimmed[m.end() :]

    junk_match = _JUNK_SECTION_RE.search(trimmed)
    if junk_match:
        trimmed = trimmed[: junk_match.start()]

    return trimmed


def _load_csv_rows(path: Path = DATA_CSV_PATH) -> list[dict]:
    with open(path, encoding="utf-8-sig") as f:
        text = f.read().replace("﻿", "")
    return [
        r for r in csv.DictReader(io.StringIO(text))
        if r.get("검색 구분 레이블", "").strip().lower() == "true"
    ]


def _row_to_article(row: dict) -> dict:
    title = row.get("기사제목", "")
    return {
        "label": f"[data_set.csv] {title[:40]}",
        "article_title": title,
        "article_text": _clean_scraped_article_text(title, row["기사 본문 전체"]),
    }


def load_articles_from_csv(path: Path = DATA_CSV_PATH, n: int = 15, seed: int = 42) -> list[dict]:
    rows = _load_csv_rows(path)
    random.Random(seed).shuffle(rows)
    return [_row_to_article(row) for row in rows[:n]]


def _call_with_retry(fn, attempts: int = 3, delay_seconds: float = 5.0):
    last_error: Optional[Exception] = None
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except Exception as e:
            last_error = e
            if attempt < attempts:
                time.sleep(delay_seconds)
    raise last_error


def _dedup_claims_by_sentence(claims: list) -> list:
    seen: set[str] = set()
    out = []
    for c in claims:
        key = re.sub(r"\s+", "", c.sentence)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


# ── 여기부터 이 스크립트 고유 로직 ───────────────────────────────────────


def _classify_exclusion(claim) -> str:
    """filter_verifiable_claims와 완전히 같은 순서/조건으로 검사해서, 어떤 조건에
    걸렸는지(맨 먼저 걸린 것 하나) 이름으로 돌려준다. 전부 통과하면 "passed"."""
    if classify_source(claim.source_org) != "kosis_verified":
        return "no_source_org" if not claim.source_org else "unmatched_org"
    if _has_hallucinated_value(claim):
        return "hallucinated_value"
    if _value_mismatches_sentence(getattr(claim, "value", None), claim.sentence):
        return "value_mismatch"
    if _value_mismatches_sentence(getattr(claim, "comparison_value", None), claim.sentence):
        return "value_mismatch"
    if _missing_value_for_magnitude_claim(claim):
        return "missing_value"
    if _looks_like_enforcement_or_evaluation(claim.sentence):
        return "enforcement_or_eval"
    return "passed"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    articles = load_articles_from_csv(n=args.n, seed=args.seed)
    print(f"[표본] data_set.csv에서 {len(articles)}건 무작위 추출 (seed={args.seed})\n")

    buckets: Counter[str] = Counter()
    unmatched_examples: list[tuple[str, str]] = []  # (source_org, sentence)
    no_source_examples: list[tuple[str, str]] = []  # (article_label, sentence) — 기사 단위로 몰려있는지 보려고
    value_mismatch_examples: list[tuple[str, object, object, str]] = []  # (source_org, value, comparison_value, sentence)
    skipped_articles = 0

    for i, article in enumerate(articles, 1):
        print(f"[{i}/{len(articles)}] {article['label']}", end=" ")
        try:
            cls_result = _call_with_retry(lambda: classify(article["article_text"]))
        except Exception as e:
            print(f"→ 1단계 실패({type(e).__name__}), 스킵")
            skipped_articles += 1
            continue
        if not cls_result.label:
            print("→ 1단계에서 무관 판정, 스킵")
            continue

        try:
            claims = _call_with_retry(lambda: extract_claims(article["article_text"]))
            claims = recover_missed_claims(article["article_text"], claims)
            claims = strip_title_prefix_from_claims(claims, article.get("article_title"))
            claims = _dedup_claims_by_sentence(claims)
        except Exception as e:
            print(f"→ 2단계 실패({type(e).__name__}), 스킵")
            skipped_articles += 1
            continue

        claims = correct_scale_errors(claims)
        claims = resolve_claim_sources(claims, cls_result.reason)
        print(f"→ claim {len(claims)}개")

        for c in claims:
            reason = _classify_exclusion(c)
            buckets[reason] += 1
            if reason == "unmatched_org":
                unmatched_examples.append((c.source_org, c.sentence))
            elif reason == "no_source_org":
                no_source_examples.append((article["label"], c.sentence))
            elif reason == "value_mismatch":
                value_mismatch_examples.append(
                    (c.source_org, getattr(c, "value", None), getattr(c, "comparison_value", None), c.sentence)
                )

    total = sum(buckets.values())
    print(f"\n{'=' * 70}")
    print(f"기사 {len(articles)}건 중 {skipped_articles}건 1~2단계 자체 실패로 스킵")
    print(f"총 claim {total}개 중 버킷별 분포:")
    print(f"{'=' * 70}")
    for name, count in buckets.most_common():
        pct = count / total * 100 if total else 0
        print(f"  {name:22s} {count:4d}건  ({pct:5.1f}%)")

    print(f"\n{'=' * 70}")
    print(f"unmatched_org 예시 (source_org는 채워졌는데 화이트리스트 불일치, {len(unmatched_examples)}건) —")
    print("이 중 '얘는 진짜 KOSIS 통계 기관 맞는데?' 싶은 게 있으면 source_filter.py의")
    print("KOSIS_VERIFIED_ORGS/_org_appears_standalone을 고칠 근거가 됩니다:")
    print(f"{'=' * 70}")
    for org, sentence in unmatched_examples:
        print(f"  [{org!r}] {sentence[:80]}")

    print(f"\n{'=' * 70}")
    print(f"no_source_org 예시 (source_org 완전히 못 채움, {len(no_source_examples)}건) — 기사별로 묶음.")
    print("한 기사에서 여러 건이 몰려 나오면, 그 기사 claim 전체가 source_org=None이라")
    print("backfill_source_org 자체가 채울 소스가 없었다는 뜻(→ infer_org_from_reason이나")
    print("claim_extractor 프롬프트 쪽을 봐야 함). 문장 내용상 통계청류 통계가 명백한데도")
    print("여기 있으면 그게 진짜 버그입니다:")
    print(f"{'=' * 70}")
    no_source_by_article: dict[str, list[str]] = {}
    for label, sentence in no_source_examples:
        no_source_by_article.setdefault(label, []).append(sentence)
    for label, sentences in sorted(no_source_by_article.items(), key=lambda kv: -len(kv[1])):
        print(f"\n  {label}  ({len(sentences)}건)")
        for s in sentences:
            print(f"    - {s[:90]}")

    print(f"\n{'=' * 70}")
    print(f"value_mismatch 예시 (문장 속 숫자와 value/comparison_value가 안 맞음, {len(value_mismatch_examples)}건) —")
    print("정말 오귀속(다른 문장 값이 잘못 붙음)인지, 아니면 한글 표기 변형을 필터가 못 알아본")
    print("오탐(false positive)인지 눈으로 확인이 필요합니다. 오탐이면 _korean_number_variants를")
    print("보강해야 하고, 진짜 오귀속이면 필터가 맞게 일하고 있는 겁니다:")
    print(f"{'=' * 70}")
    for org, value, comparison_value, sentence in value_mismatch_examples:
        print(f"  [{org}] value={value!r} comparison_value={comparison_value!r}")
        print(f"    {sentence[:100]}")


if __name__ == "__main__":
    main()
