"""
notebooks/verify_stage2_on_golden_merged.py — 새 통합 골든셋(골든셋_통합.xlsx)의
2단계_claim목록 시트로 claim_extractor를 검증하는 1회성 스크립트.

agent/preprocessing/verify_claim_extractor_on_golden.py와 같은 매칭 방식(1차 문장
텍스트 매칭 + 2차 핵심 숫자 겹침 재매칭)을 그대로 재사용한다 — 그 스크립트가 이미
실측으로 검증해둔 방식(recall 53.4%→80.8% 개선 확인)이라 새로 설계하지 않는다.
다만 대상 파일이 다르므로(claim이 있는 기사가 50개 중 10개뿐인 새 통합 골든셋)
데이터 로딩 부분만 새로 짠다.

사용법: python -m notebooks.verify_stage2_on_golden_merged
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from agent.pipeline.batch_runner import _clean_scraped_article_text
from agent.preprocessing.claim_extractor import ClaimExtractorError, extract_claims, recover_missed_claims
from agent.preprocessing.verify_claim_extractor_on_golden import (
    _extract_distinctive_numbers,
    _has_digit,
    _match_gold_to_extracted,
    _rematch_by_number_overlap,
)

GOLDEN_PATH = Path(__file__).parent / "골든셋_통합.xlsx"
REPORT_PATH = Path(__file__).parent / "stage2_verify_report.md"

MAX_RETRIES = 4
RETRY_WAIT_SECONDS = (5, 10, 15, 20)
DELAY_BETWEEN_CALLS = 2.5


def _extract_with_retry(body: str):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            claims = extract_claims(body)
            return recover_missed_claims(body, claims)
        except (ClaimExtractorError, Exception) as e:  # noqa: BLE001
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT_SECONDS[attempt])
    raise last_err  # type: ignore[misc]


def main() -> None:
    df1 = pd.read_excel(GOLDEN_PATH, sheet_name="1단계_기사목록")
    df2 = pd.read_excel(GOLDEN_PATH, sheet_name="2단계_claim목록")

    articles_with_claims = sorted(df2["기사번호"].unique())
    print(f"claim이 있는 기사 {len(articles_with_claims)}건 검증 시작")

    total_gold = 0
    total_found_raw = 0
    total_found_adjusted = 0
    total_extracted = 0
    total_extra_legit = 0
    total_extra_bad = 0
    failed = []
    report_lines = ["# 2단계(claim_extractor) 골든셋 검증 결과\n\n"]

    for i, article_no in enumerate(articles_with_claims):
        art_row = df1[df1["번호"] == article_no].iloc[0]
        title = art_row["기사제목"]
        raw_text = art_row["본문(정제됨)"]
        body = _clean_scraped_article_text(title, raw_text)

        gold_sentences = df2[df2["기사번호"] == article_no]["sentence(원문 그대로)"].tolist()

        try:
            claims = _extract_with_retry(body)
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(articles_with_claims)}] [FAIL] {title[:40]} -> {e}")
            failed.append(title)
            time.sleep(DELAY_BETWEEN_CALLS)
            continue

        extracted_sentences = [c.sentence for c in claims]
        matched_extracted: set[str] = set()
        missed_gold: list[str] = []

        for gs in gold_sentences:
            found = _match_gold_to_extracted(gs, extracted_sentences)
            if found:
                matched_extracted.add(found)
            else:
                missed_gold.append(gs)
        n_found_raw = len(gold_sentences) - len(missed_gold)

        missed_gold = _rematch_by_number_overlap(missed_gold, claims, matched_extracted)
        n_found_adjusted = len(gold_sentences) - len(missed_gold)

        extra_claims = [c for c in claims if c.sentence not in matched_extracted]
        extra_legit = [c for c in extra_claims if _has_digit(c.sentence)]
        extra_bad = [c for c in extra_claims if not _has_digit(c.sentence)]

        n_gold = len(gold_sentences)
        total_gold += n_gold
        total_found_raw += n_found_raw
        total_found_adjusted += n_found_adjusted
        total_extracted += len(extracted_sentences)
        total_extra_legit += len(extra_legit)
        total_extra_bad += len(extra_bad)

        tag = "OK" if n_found_adjusted == n_gold else "MISS"
        print(
            f"[{i + 1}/{len(articles_with_claims)}] [{tag}] 기사{article_no} gold={n_gold} "
            f"발견(1차)={n_found_raw} 발견(보정)={n_found_adjusted} 추출총={len(extracted_sentences)} "
            f"추가(정당추정)={len(extra_legit)} 추가(규칙위반추정)={len(extra_bad)} | {title[:30]}"
        )

        report_lines.append(f"## [기사{article_no}] {title}\n\n")
        report_lines.append(
            f"- gold={n_gold}, 발견(1차)={n_found_raw}, 발견(보정)={n_found_adjusted}, "
            f"추출총={len(extracted_sentences)}, 추가(정당추정)={len(extra_legit)}, "
            f"추가(규칙위반추정)={len(extra_bad)}\n\n"
        )
        if missed_gold:
            report_lines.append("**진짜 누락:**\n")
            for ms in missed_gold:
                report_lines.append(f"- {ms}\n")
            report_lines.append("\n")
        if extra_bad:
            report_lines.append("**추가(규칙위반추정):**\n")
            for ec in extra_bad:
                report_lines.append(f"- [{ec.claim_type}] {ec.sentence}\n")
            report_lines.append("\n")
        if extra_legit:
            report_lines.append("**추가(정당추정):**\n")
            for ec in extra_legit:
                report_lines.append(f"- [{ec.claim_type}] value={ec.value} {ec.sentence}\n")
            report_lines.append("\n")

        time.sleep(DELAY_BETWEEN_CALLS)

    report_lines.append("## 최종 요약\n\n")
    report_lines.append(f"- 검증한 기사: {len(articles_with_claims) - len(failed)}건 (실패 {len(failed)}건)\n")
    if total_gold:
        report_lines.append(f"- recall(1차, 문장 매칭만): {total_found_raw}/{total_gold} = {total_found_raw/total_gold:.1%}\n")
        report_lines.append(f"- recall(보정, 숫자 재매칭 포함): {total_found_adjusted}/{total_gold} = {total_found_adjusted/total_gold:.1%}\n")
    if total_extracted:
        precision_est = (total_found_adjusted + total_extra_legit) / total_extracted
        report_lines.append(f"- precision 추정치: {precision_est:.1%} (전체 추출 {total_extracted}건 중 정당 추정 {total_found_adjusted + total_extra_legit}건)\n")

    REPORT_PATH.write_text("".join(report_lines), encoding="utf-8")

    print(f"\n=== 최종 요약 ===")
    print(f"검증한 기사: {len(articles_with_claims) - len(failed)}건 (실패 {len(failed)}건)")
    if total_gold:
        print(f"recall (1차): {total_found_raw}/{total_gold} = {total_found_raw/total_gold:.1%}")
        print(f"recall (보정): {total_found_adjusted}/{total_gold} = {total_found_adjusted/total_gold:.1%}")
    print(f"리포트 -> {REPORT_PATH}")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
