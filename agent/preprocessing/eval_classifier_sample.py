"""
agent/preprocessing/eval_classifier_sample.py — 1단계 classifier 소규모 샘플 테스트 (개발 보조 도구)

역할: 파이프라인 코드가 아니라, data_set.csv에서 20~30건을 뽑아 classify()를 실제로 돌려보고
사람 라벨과 비교 + score 0.4~0.6 애매 구간 분포를 확인하기 위한 1회성 점검 스크립트.

사용법 (프로젝트 루트에서):
    python -m agent.preprocessing.eval_classifier_sample
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from agent.preprocessing.classifier import classify

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "data_set.csv"

# few-shot 프롬프트(classifier_prompt.txt)에 이미 예시로 쓰인 기사는 샘플에서 제외
# (모델이 "본 적 있는" 예시로 테스트하면 실제 일반화 성능을 과대평가하게 됨)
FEWSHOT_TITLES = {
    "9명 가족 잃고, 하염없이 기다리던 푸딩이... 동물단체가 구조",
    "4000억달러대 지켰지만… 외환 보유액 5년 만에 최소",
    "국민 1인당 쌀 소비량 40년 연속 줄어들어",
    "국정자원 전산 시스템 8개 추가 정상화… 복구율 71.7%",
    "KDI 올해 2% 성장 전망, 석 달 만에 0.4%포인트 낮춰",
    "강남·서초보다 비싸네…원룸 평균 월세 102만원,이 동네 어디?",
    "美 4월 소비자물가 상승률 2.3%... 4년여 만에 최저",
    "전남경찰, 여객기 참사 관련 김이배 제주항공 대표 소환조사",
}


def load_rows() -> list[list[str]]:
    with open(DATA_PATH, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # header
        return [row for row in reader if len(row) == 5 and row[4].strip() in ("TRUE", "FALSE")]


def main(n_true: int = 15, n_false: int = 10, seed: int = 42) -> None:
    rows = load_rows()
    true_rows = [r for r in rows if r[4].strip() == "TRUE" and r[0] not in FEWSHOT_TITLES]
    false_rows = [r for r in rows if r[4].strip() == "FALSE" and r[0] not in FEWSHOT_TITLES]

    rng = random.Random(seed)
    sample = rng.sample(true_rows, n_true) + rng.sample(false_rows, n_false)
    rng.shuffle(sample)

    results = []
    for title, date, url, body, gold_label in sample:
        gold = gold_label.strip() == "TRUE"
        try:
            result = classify(body)
        except Exception as e:  # noqa: BLE001 - 점검 스크립트라 실패해도 계속 진행
            print(f"[에러] {title[:30]}... -> {e}")
            continue
        correct = result.label == gold
        results.append((title, gold, result.label, result.score, correct))
        mark = "O" if correct else "X"
        print(f"[{mark}] gold={gold!s:5} pred={result.label!s:5} score={result.score:.2f}  {title[:40]}")

    total = len(results)
    if total == 0:
        print("결과 없음 (전부 실패)")
        return

    n_correct = sum(1 for r in results if r[4])
    ambiguous = [r for r in results if 0.4 <= r[3] <= 0.6]
    fp = [r for r in results if not r[4] and r[2] and not r[1]]  # gold FALSE인데 pred TRUE
    fn = [r for r in results if not r[4] and not r[2] and r[1]]  # gold TRUE인데 pred FALSE

    print("\n=== 요약 ===")
    print(f"총 {total}건 중 정답 {n_correct}건 (정확도 {n_correct/total:.1%})")
    print(f"score 0.4~0.6 애매 구간: {len(ambiguous)}건 / {total}건")
    for r in ambiguous:
        print(f"  - score={r[3]:.2f} gold={r[1]} pred={r[2]}  {r[0][:40]}")
    print(f"오탐(FALSE인데 TRUE로 분류): {len(fp)}건")
    for r in fp:
        print(f"  - score={r[3]:.2f}  {r[0][:40]}")
    print(f"누락(TRUE인데 FALSE로 분류): {len(fn)}건")
    for r in fn:
        print(f"  - score={r[3]:.2f}  {r[0][:40]}")


if __name__ == "__main__":
    main()
