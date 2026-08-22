"""
notebooks/verify_stage1_on_golden_merged.py — 새 통합 골든셋(golden_merged.xlsx)의
1단계_기사목록 시트로 classify()를 검증하는 1회성 스크립트.

기존 verify_classifier_on_golden.py는 예전 골든셋 파일(추출 골든셋 단위 분리.xlsx)에
고정돼 있어서 이 새 통합 파일 구조(1단계_기사목록/2단계_claim목록/7단계_판정목록)에는
안 맞는다. 이 스크립트는 딱 1단계(classifier)만 검증한다.

골든셋의 "본문(정제됨)" 컬럼이 실제로는 완전히 정제된 게 아니라(내비게이션 메뉴/구독수
등 위젯 잡음이 그대로 남아있는 경우가 실측 확인됨, 2026-08-22) 그대로 쓰지 않고,
agent.pipeline.batch_runner._clean_scraped_article_text()(방금 구독수 필터 추가로
갱신됨)를 다시 적용해서 "지금 실제 파이프라인이 볼 텍스트"와 동일한 입력으로 테스트한다.

사용법: python -m notebooks.verify_stage1_on_golden_merged
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from agent.preprocessing.classifier import ClassifierError, classify
from agent.pipeline.batch_runner import _clean_scraped_article_text

GOLDEN_PATH = Path(__file__).parent / "골든셋_통합.xlsx"
REPORT_PATH = Path(__file__).parent / "stage1_verify_report_hcx007.md"

# 2026-08-22: HCX-DASH-002가 golden set 50건 중 34%(17건)를 틀려서(거의 다 민간기업/
# 해외통계/1회성 행정조치를 True로 잘못 통과시킴), 프로젝트에서 이미 더 무거운 추론에
# 쓰고 있는 상위 등급(judge/explain이 쓰는 HCX-007)으로 같은 벤치마크를 재실행해
# 정확도가 실제로 개선되는지 확인한다. classify()가 이미 model= 오버라이드를 받으므로
# 코드 변경 없이 이 상수만 바꾸면 됨.
CLASSIFY_MODEL = "HCX-007"


def main() -> None:
    df = pd.read_excel(GOLDEN_PATH, sheet_name="1단계_기사목록")

    rows = []
    for _, r in df.iterrows():
        title = r["기사제목"]
        raw_text = r["본문(정제됨)"]
        gold = bool(r["1단계_판정(True/False)"])
        cleaned = _clean_scraped_article_text(title, raw_text)

        for attempt in range(3):
            try:
                result = classify(cleaned, model=CLASSIFY_MODEL)
                pred = result.label
                reason = result.reason
                score = result.score
                break
            except ClassifierError as e:
                print(f"  [재시도 {attempt + 1}/3] {r['번호']}번 실패: {e}")
                time.sleep(3)
        else:
            pred, reason, score = None, "classify() 3회 실패", None

        match = pred == gold
        rows.append(
            {
                "번호": r["번호"],
                "기사제목": title,
                "gold": gold,
                "pred": pred,
                "match": match,
                "score": score,
                "reason": reason,
            }
        )
        print(f"[{r['번호']:>2}] gold={gold!s:<5} pred={pred!s:<5} {'OK' if match else 'MISMATCH'}  {title[:30]}")
        time.sleep(0.3)

    result_df = pd.DataFrame(rows)
    n = len(result_df)
    correct = result_df["match"].sum()
    acc = correct / n * 100

    mismatches = result_df[~result_df["match"]]

    report = [f"# 1단계(classifier) 골든셋 검증 결과\n\n", f"- 전체 {n}건 중 {correct}건 일치, 정답률 {acc:.1f}%\n\n"]
    report.append(f"## 불일치 {len(mismatches)}건\n\n")
    for _, r in mismatches.iterrows():
        report.append(
            f"### [{r['번호']}] {r['기사제목']}\n"
            f"- gold={r['gold']}, pred={r['pred']} (score={r['score']})\n"
            f"- reason: {r['reason']}\n\n"
        )
    REPORT_PATH.write_text("".join(report), encoding="utf-8")

    print(f"\n정답률: {correct}/{n} ({acc:.1f}%)")
    print(f"불일치 {len(mismatches)}건 -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
