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

    그래서 이 스크립트는 claim_sentence(원문 문장) 텍스트 매칭으로 "찾았는지"만 확인한다 —
    골든셋도 우리 claim_extractor도 둘 다 "기사 원문 문장을 그대로" 뽑는 게 설계 원칙이라
    (같은 이유로 claim_extractor_prompt.txt few-shot에도 "절대 요약하지 말고 원문 그대로"
    규칙이 있음), 문장 텍스트 겹침이 숫자 비교보다 훨씬 신뢰도 높은 축이다.

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
    total_found = 0
    total_extra = 0
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

        for gs in gold_sentences:
            found = _match_gold_to_extracted(gs, extracted_sentences)
            if found:
                matched_extracted.add(found)
            else:
                missed_gold.append(gs)

        n_gold = len(gold_sentences)
        n_found = n_gold - len(missed_gold)
        n_extra = len(extracted_sentences) - len(matched_extracted)

        total_gold += n_gold
        total_found += n_found
        total_extra += n_extra

        tag = "OK" if n_found == n_gold else "MISS"
        print(
            f"[{i + 1}/{len(merged)}] [{tag}] gold={n_gold} 발견={n_found} "
            f"추출총={len(extracted_sentences)} 추가={n_extra} | {title[:40]}"
        )
        if missed_gold:
            for ms in missed_gold:
                print(f"    [누락] {ms[:80]}")

        per_article_report.append((title, n_gold, n_found, n_extra, missed_gold))
        time.sleep(DELAY_BETWEEN_CALLS)

    print(f"\n=== 최종 요약 ===")
    print(f"검증한 기사: {len(merged) - len(failed)}건 (실패 {len(failed)}건)")
    print(f"gold claim 총 {total_gold}건 중 {total_found}건 발견 (recall {total_found/total_gold:.1%})")
    print(f"gold와 매칭 안 된 추가 추출: {total_extra}건 (기사당 평균 {total_extra/max(1,len(merged)-len(failed)):.2f}건)")


if __name__ == "__main__":
    main()
