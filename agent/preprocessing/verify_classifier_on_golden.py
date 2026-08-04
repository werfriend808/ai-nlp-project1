"""
agent/preprocessing/verify_classifier_on_golden.py — 1단계 classifier를 골든셋 기준으로 검증

골든셋(notebooks/추출 골든셋 단위 분리.xlsx)의 진짜 용도: 사람이 기사 하나하나를 검토해서
"이 기사에서 뽑을 수 있는 수치 claim이 있는가"를 판단해둔 것. article_id로 묶었을 때
- claim_sentence가 하나라도 채워진 기사 -> 정답 True (통계 claim이 실제로 존재)
- 그 기사의 모든 행이 claim_sentence 빈값 -> 정답 False (검토했지만 claim 없음)

즉 "골든셋에 있으면 무조건 True"가 아니라, **골든셋 자체가 이미 True/False 정답 라벨을
담고 있고**, 이 스크립트는 같은 기사 본문(data.csv에서 article_url로 조회)을 우리
classifier()에 넣어서 그 정답과 일치하는지 확인한다.

알려진 데이터 버그: article_id=A001은 두 행(A001-01 서울 인구 / A001-02 배스킨라빈스)이
같은 article_url을 공유하는데 실제로는 서로 다른 기사다(원본 골든셋 오타로 추정,
notebooks/매핑 골든셋 ord 추가.xlsx의 A001-01(중복) 행 note에도 기록됨). data.csv에서
이 URL로는 "배스킨라빈스" 본문만 가져올 수 있어 "서울 인구" claim의 실제 본문을 확인할
방법이 없으므로 이 기사는 검증에서 제외한다.

HCX API 429(rate limit) 대응으로 재시도(백오프)와 호출 간 딜레이를 둔다.

사용법 (프로젝트 루트에서):
    python -m agent.preprocessing.verify_classifier_on_golden
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from agent.preprocessing.classifier import ClassifierError, classify

NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "notebooks"
CLAIMS_XLSX = NOTEBOOKS_DIR / "추출 골든셋 단위 분리.xlsx"
DATA_CSV = Path(__file__).parent.parent.parent / "data" / "data.csv"

# article_id=A001: 서울 인구/배스킨라빈스 두 기사가 같은 article_url을 공유하는 알려진 오타.
# data.csv에서 실제 본문을 신뢰성 있게 조회할 수 없어 검증 대상에서 제외.
KNOWN_BAD_ARTICLE_IDS = {"A001"}

MAX_RETRIES = 4
RETRY_WAIT_SECONDS = (5, 10, 15, 20)
DELAY_BETWEEN_CALLS = 1.2


def _normalize_url(url: object) -> str:
    return str(url).strip().rstrip("/")


def build_article_gold_labels() -> pd.DataFrame:
    df = pd.read_excel(CLAIMS_XLSX)
    df = df[~df["article_id"].isin(KNOWN_BAD_ARTICLE_IDS)]

    rows = []
    for article_id, g in df.groupby("article_id"):
        has_claim = g["claim_sentence"].notna().any()
        # claim이 있으면 그 claim이 달린 title을, 없으면 첫 title을 대표로 사용
        rep = g[g["claim_sentence"].notna()].iloc[0] if has_claim else g.iloc[0]
        rows.append(
            {
                "article_id": article_id,
                "article_title": rep["article_title"],
                "article_url": rep["article_url"],
                "gold_label": bool(has_claim),
            }
        )
    return pd.DataFrame(rows)


def _classify_with_retry(body: str):
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return classify(body)
        except (ClassifierError, Exception) as e:  # noqa: BLE001 - 점검 스크립트, 계속 진행해야 함
            last_err = e
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_WAIT_SECONDS[attempt])
    raise last_err  # type: ignore[misc]


def main() -> None:
    gold = build_article_gold_labels()

    data = pd.read_csv(DATA_CSV)
    data["_url_norm"] = data["URL"].map(_normalize_url)
    gold["_url_norm"] = gold["article_url"].map(_normalize_url)

    merged = gold.merge(data[["_url_norm", "기사 본문 전체"]], on="_url_norm", how="left")
    missing_body = merged[merged["기사 본문 전체"].isna()]
    merged = merged.dropna(subset=["기사 본문 전체"])

    print(f"검증 대상 {len(merged)}건 (data.csv에서 본문 못 찾음 {len(missing_body)}건 제외)")
    if len(missing_body):
        for _, r in missing_body.iterrows():
            print(f"  [본문없음] {r['article_id']} {r['article_title'][:30]}")

    mismatches = []
    failed = []
    n_ok = 0

    for i, row in merged.reset_index(drop=True).iterrows():
        title = row["article_title"]
        body = row["기사 본문 전체"]
        gold_label = row["gold_label"]

        try:
            result = _classify_with_retry(body)
        except Exception as e:  # noqa: BLE001
            print(f"[{i + 1}/{len(merged)}] [FAIL] {title[:40]} -> {e}")
            failed.append(title)
            time.sleep(DELAY_BETWEEN_CALLS)
            continue

        ok = result.label == gold_label
        tag = "OK" if ok else "MISMATCH"
        print(
            f"[{i + 1}/{len(merged)}] [{tag}] 골든셋정답={gold_label} 우리={result.label} "
            f"score={result.score:.2f} {title[:40]}"
        )
        if ok:
            n_ok += 1
        else:
            mismatches.append((title, gold_label, result.label, result.score, result.reason))

        time.sleep(DELAY_BETWEEN_CALLS)

    total = n_ok + len(mismatches)
    print(f"\n=== 최종 요약: {total}건 중 정답 {n_ok}건 ({n_ok/total:.1%}), "
          f"실패 {len(failed)}건, 불일치 {len(mismatches)}건 ===")
    for title, gold_label, pred, score, reason in mismatches:
        kind = "오탐(FALSE인데 TRUE)" if pred and not gold_label else "누락(TRUE인데 FALSE)"
        print(f"  [{kind}] 골든셋={gold_label} 우리={pred} score={score:.2f} | {title}")
        print(f"    근거: {reason}")


if __name__ == "__main__":
    main()
