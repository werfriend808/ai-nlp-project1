"""
agent/preprocessing/verify_claim_extractor_on_golden.py — 2단계 claim_extractor를 골든셋 기준으로 검증

verify_classifier_on_golden.py와 같은 골든셋(notebooks/추출 골든셋 단위 분리.xlsx)을 쓰지만,
비교 축이 다르다. 1단계는 article 단위 True/False였고, 2단계는 "그 기사에서 사람이 뽑아둔
claim_sentence들을 우리 claim_extractor가 실제로 찾아내는가"를 확인해야 한다.

⚠️ claim_numeric_value로 직접 비교하면 안 되는 이유:
    골든셋의 claim_numeric_value 표기가 일관적이지 않다. 예:
      - "0.38%의 스트레스 DSR 금리"        -> 0.38 (문장에 쓰인 숫자 그대로)
      - "R&D 성공률 97%"                  -> 0.97 (100으로 나눈 소수로 환산해서 기록)
      - "51.18%(1만7400원) 오른 5만1400원" -> 51400 (퍼센트가 아니라 원화 가격을 핵심값으로 선택)
    사람이 케이스마다 "핵심 수치"를 다르게 해석해서 적어놔서, 우리 claim_extractor의 value
    필드(문장에 쓰인 숫자를 그대로 뽑도록 설계됨, prompts/claim_extractor_prompt.txt 참고)와
    숫자로 직접 비교하면 표기 관례 차이 때문에 진짜 오류가 아닌데도 대량으로 불일치가 난다.

    그래서 이 스크립트는 claim_sentence(원문 문장) 텍스트 매칭을 1차 기준으로 삼는다 —
    골든셋도 우리 claim_extractor도 둘 다 "기사 원문 문장을 그대로" 뽑는 게 설계 원칙이라
    (같은 이유로 claim_extractor_prompt.txt few-shot에도 "절대 요약하지 말고 원문 그대로"
    규칙이 있음), 문장 텍스트 겹침이 숫자 비교보다 훨씬 신뢰도 높은 축이다.

⚠️ 매칭 보정 (2026-08-05, 실측 후 반영):
    1차 문장 매칭만 쓰면 recall을 상당히 저평가한다는 걸 실측으로 확인했다. 예:
    gold="15일 통계청이 발표한 '2024년 12월 및 연간 고용 동향'에 따르면, 지난달 취업자
    수는 2804만1000명으로..." vs 추출="'2024년 12월 및 연간 고용 동향'에 따르면, 지난달
    취업자 수는 2804만1000명으로..." — 앞의 "15일 통계청이 발표한"만 빠졌을 뿐 사실상
    같은 claim인데 정확 부분포함이 아니라서 "누락"이자 "추가"로 이중 오카운트됐다.

    그래서 1차 매칭에 실패한 gold는 2차로 "핵심 숫자가 겹치는 미매칭 추출 claim이
    있는가"로 한 번 더 확인한다(같은 사실을 다른 문장으로 재진술했거나 사소한 어구
    차이로 문자열 매칭만 실패한 경우를 구제). 30개 기사 표본 재계산 결과 recall이
    53.4%→80.8%로 크게 올라갔다 — 문자열 매칭의 한계가 성능을 상당히 저평가하고
    있었다는 뜻.

    또한 최종까지 안 매칭된 "추가" claim은 golden set이 애초에 기사당 일부 claim만
    골라 적어둔 것일 수 있어(전수조사 아님, 팀 확인 완료) 전부 오탐으로 볼 수 없다.
    그래서 문장에 숫자가 있는지로 1차 분류한다: 숫자가 있으면 "정당한 추가 claim일
    가능성 높음", 숫자가 아예 없으면 "수치 없는 배경설명 문장을 뽑은 것"이라 프롬프트
    규칙(수치 없는 문장은 추출 금지) 위반으로 본다. 표본 15건을 직접 읽고 대조한 결과와
    방향이 일치했다(정당 다수, 진짜 규칙위반은 소수).

    이 구분은 완전한 사람 검수가 아니라 자동 추정치다 — 정확한 수치가 필요하면
    "정당 추정"/"규칙위반 추정"으로 분류된 문장을 사람이 직접 다시 봐야 한다.

사용법 (프로젝트 루트에서):
    python -m agent.preprocessing.verify_claim_extractor_on_golden
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd

from agent.pipeline.batch_runner import _clean_scraped_article_text
from agent.preprocessing.claim_extractor import ClaimExtractorError, extract_claims

NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "notebooks"
CLAIMS_XLSX = NOTEBOOKS_DIR / "추출 골든셋 단위 분리.xlsx"
# 팀원의 verify_classifier_on_golden.py는 "data.csv"를 가리키는데, data/*.csv가
# .gitignore돼 있어 로컬마다 원본 파일명이 다르다(우리 로컬엔 data_set.csv로 있음).
DATA_CSV = Path(__file__).parent.parent.parent / "data" / "data_set.csv"

# ⚠️ 정제 로직 실험 결과 (2026-08-05):
#
# 정제 없이 원문 그대로 넣으면 recall 28.8%. batch_runner.py의 _clean_scraped_article_text
# (제목 위치부터 고정 3000자)를 적용하니 53.4%로 개선 — 지금까지 시도한 것 중 최고 결과.
#
# 이후 "제목 위치별로 실제 본문 시작점이 다르고(0~4600자), 끝나는 지점도 기사마다
# 다르다"는 걸 확인하고, 바이라인("입력 20YY.MM.DD") 패턴이 두 번째로 등장하는 지점을
# 실제 본문 끝으로 보는 가변 경계 로직을 시도했다. 골든셋 30개 기사로 텍스트 위치만
# 대조했을 땐 gold claim 커버리지가 97.2%까지 올라 훨씬 정확해 보였지만, 실제로
# extract_claims()에 넣어보니 recall이 오히려 49.3%로 떨어졌다.
#
# 원인: 가변 경계의 평균 길이가 4754자(최대 6853자)로 고정 3000자보다 60% 커서, 30개 중
# 29개가 자동으로 청킹(2회 이상 분할 호출)됐다. 텍스트 커버리지는 좋아졌지만 모델에게
# 한 번에/한 조각에 넘기는 절대적인 글자 수 자체가 커지면서, 이전에 확인했던 "긴 입력에서
# 일부만 뽑는" 성향이 다시 발동해 오히려 더 산만해진 것으로 판단(추가 추출도 106→157건
# 증가). 즉 "잡음 제거"와 "모델이 다 뽑게 만들기"는 서로 다른 문제였고, 가변 경계는
# 앞의 문제만 풀고 뒤의 문제를 악화시켰다.
#
# 결론: 고정 3000자 방식이 지금까지 실측된 것 중 최선이라 이걸로 확정한다. 더 개선하려면
# "정확한 경계 + 작은 청크(1000~1500자) 조합"을 시도해볼 수 있는데, 이건 별도 검증이
# 더 필요해서 이번 라운드에서는 보류.
CLEAN_MAX_LEN = 3000

# verify_classifier_on_golden.py와 동일한 이유로 제외 (article_url 오타로 본문 조회 불가).
KNOWN_BAD_ARTICLE_IDS = {"A001"}

MAX_RETRIES = 4
RETRY_WAIT_SECONDS = (5, 10, 15, 20)
DELAY_BETWEEN_CALLS = 1.2


def _normalize_url(url: object) -> str:
    return str(url).strip().rstrip("/")


def _normalize_text(text: str) -> str:
    """공백 차이만으로 매칭 실패하지 않도록 공백을 전부 제거하고 비교."""
    return re.sub(r"\s+", "", str(text))


def build_article_gold_claims() -> pd.DataFrame:
    """article_id별로 gold claim_sentence 리스트를 모은다 (claim_sentence 없는 행/기사는 제외 —
    2단계는 애초에 claim이 있는 기사에서만 의미가 있다, 1단계 True/False 판정은 별개)."""
    df = pd.read_excel(CLAIMS_XLSX)
    df = df[~df["article_id"].isin(KNOWN_BAD_ARTICLE_IDS)]
    df = df[df["claim_sentence"].notna()]

    rows = []
    for article_id, g in df.groupby("article_id"):
        rows.append(
            {
                "article_id": article_id,
                "article_title": g.iloc[0]["article_title"],
                "article_url": g.iloc[0]["article_url"],
                "gold_sentences": list(g["claim_sentence"]),
            }
        )
    return pd.DataFrame(rows)


def _extract_with_retry(body: str):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return extract_claims(body)
        except (ClaimExtractorError, Exception) as e:  # noqa: BLE001 - 점검 스크립트, 계속 진행해야 함
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT_SECONDS[attempt])
    raise last_err  # type: ignore[misc]


def _match_gold_to_extracted(gold_sentence: str, extracted_sentences: list[str]) -> str | None:
    """gold_sentence 하나를 찾아준 extracted 문장이 있으면 그 문장을, 없으면 None을 반환.

    완전 동일할 필요는 없음 — claim_extractor가 지시어("이는" 등)로 이어붙인 두 문장을
    합쳐서 반환하는 경우가 있어서(prompts/claim_extractor_prompt.txt 규칙 참고), 부분
    포함(어느 한쪽이 다른 쪽을 포함)이면 매칭으로 인정한다.
    """
    gold_norm = _normalize_text(gold_sentence)
    for ext in extracted_sentences:
        ext_norm = _normalize_text(ext)
        if gold_norm in ext_norm or ext_norm in gold_norm:
            return ext
    return None


def _extract_distinctive_numbers(text: str) -> set[str]:
    """비교용 숫자 집합. 한 자리 숫자(0~9)는 우연히 겹칠 확률이 높아서 제외한다
    (예: 서로 무관한 두 claim이 그냥 "3일"의 "3"으로 겹치는 경우 방지)."""
    nums = re.findall(r"\d+\.?\d*", text)
    return {n for n in nums if len(n.replace(".", "")) >= 2}


def _has_digit(text: str) -> bool:
    return bool(re.search(r"\d", text))


def _rematch_by_number_overlap(missed_gold: list[str], claims: list, matched_extracted: set[str]) -> list[str]:
    """1차 문장 매칭에서 놓친 gold를, 아직 안 쓰인 추출 claim 중 핵심 숫자가 겹치는 것으로
    한 번 더 구제한다(매칭됐으면 matched_extracted에 추가하고 반환값에서 뺀다).

    같은 gold 숫자로 여러 claim이 겹칠 수 있어서, claim 하나는 gold 하나에만 쓰도록
    matched_extracted에 즉시 등록해 중복 사용을 막는다.
    """
    still_missed = []
    for gs in missed_gold:
        gs_nums = _extract_distinctive_numbers(gs)
        hit = None
        if gs_nums:
            for c in claims:
                if c.sentence in matched_extracted:
                    continue
                if gs_nums & _extract_distinctive_numbers(c.sentence):
                    hit = c.sentence
                    break
        if hit:
            matched_extracted.add(hit)
        else:
            still_missed.append(gs)
    return still_missed


def main() -> None:
    gold = build_article_gold_claims()

    data = pd.read_csv(DATA_CSV)
    data["_url_norm"] = data["URL"].map(_normalize_url)
    gold["_url_norm"] = gold["article_url"].map(_normalize_url)

    merged = gold.merge(data[["_url_norm", "기사 본문 전체"]], on="_url_norm", how="left")
    missing_body = merged[merged["기사 본문 전체"].isna()]
    merged = merged.dropna(subset=["기사 본문 전체"])

    print(f"검증 대상 {len(merged)}개 기사 (data.csv에서 본문 못 찾음 {len(missing_body)}건 제외)")
    if len(missing_body):
        for _, r in missing_body.iterrows():
            print(f"  [본문없음] {r['article_id']} {r['article_title'][:30]}")

    total_gold = 0
    total_found_raw = 0
    total_found_adjusted = 0
    total_extracted = 0
    total_extra_legit = 0
    total_extra_bad = 0
    failed = []
    per_article_report = []

    for i, row in merged.reset_index(drop=True).iterrows():
        title = row["article_title"]
        body = _clean_scraped_article_text(title, row["기사 본문 전체"], max_len=CLEAN_MAX_LEN)
        gold_sentences = row["gold_sentences"]

        try:
            claims = _extract_with_retry(body)
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(merged)}] [FAIL] {title[:40]} -> {e}")
            failed.append(title)
            time.sleep(DELAY_BETWEEN_CALLS)
            continue

        extracted_sentences = [c.sentence for c in claims]
        matched_extracted: set[str] = set()
        missed_gold: list[str] = []

        # 1차: 문장 텍스트 매칭
        for gs in gold_sentences:
            found = _match_gold_to_extracted(gs, extracted_sentences)
            if found:
                matched_extracted.add(found)
            else:
                missed_gold.append(gs)
        n_found_raw = len(gold_sentences) - len(missed_gold)

        # 2차: 1차에서 놓친 gold를 핵심 숫자 겹침으로 재매칭 (같은 사실, 다른 어구)
        missed_gold = _rematch_by_number_overlap(missed_gold, claims, matched_extracted)
        n_found_adjusted = len(gold_sentences) - len(missed_gold)

        # 최종까지 안 매칭된 추출 claim: 숫자 유무로 "정당 추정" vs "규칙위반 추정" 1차 분류
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
            f"[{i + 1}/{len(merged)}] [{tag}] gold={n_gold} 발견(1차)={n_found_raw} "
            f"발견(보정)={n_found_adjusted} 추출총={len(extracted_sentences)} "
            f"추가(정당추정)={len(extra_legit)} 추가(규칙위반추정)={len(extra_bad)} | {title[:40]}"
        )
        if missed_gold:
            for ms in missed_gold:
                print(f"    [진짜누락] {ms[:80]}")
        for ec in extra_bad:
            print(f"    [추가-규칙위반추정] [{ec.claim_type}] {ec.sentence[:100]}")
        for ec in extra_legit:
            print(f"    [추가-정당추정] [{ec.claim_type}] value={ec.value} {ec.sentence[:100]}")

        per_article_report.append((title, n_gold, n_found_adjusted, missed_gold, extra_legit, extra_bad))
        time.sleep(DELAY_BETWEEN_CALLS)

    n_articles = len(merged) - len(failed)
    print(f"\n=== 최종 요약 ===")
    print(f"검증한 기사: {n_articles}건 (실패 {len(failed)}건)")
    print(f"recall (1차, 문장 매칭만): {total_found_raw}/{total_gold} = {total_found_raw/total_gold:.1%}")
    print(f"recall (보정, 숫자 재매칭 포함): {total_found_adjusted}/{total_gold} = {total_found_adjusted/total_gold:.1%}")
    print(f"전체 추출 {total_extracted}건 중 정당 추정 {total_found_adjusted + total_extra_legit}건, "
          f"규칙위반 추정 {total_extra_bad}건")
    if total_extracted:
        precision_est = (total_found_adjusted + total_extra_legit) / total_extracted
        print(f"precision 추정치: {precision_est:.1%}")
    print("(주의: '보정 recall'과 'precision 추정치'는 자동 휴리스틱 기반 추정값이며, "
          "정확한 수치가 필요하면 [추가-정당추정]/[추가-규칙위반추정] 항목을 사람이 직접 재확인해야 함)")


if __name__ == "__main__":
    main()
