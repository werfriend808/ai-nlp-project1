"""
agent/mapping/measure_catalog_coverage.py — table_catalog.json이 data_set.csv를
실제로 얼마나 커버하는지 실측하는 개발 보조 도구.

방법: data_set.csv에서 검색 구분 레이블=True인 기사 중 표본을 뽑아
  (1) 페이지 네비게이션/광고 텍스트 제거
  (2) classifier.py로 "진짜 국가 공식 통계 기반 주장" 기사만 필터
  (3) claim_extractor.py로 개별 claim 문장 추출
  (4) keyword_search.py(실제 프로덕션 매칭 함수)로 카탈로그 매칭 여부 판정
을 거쳐, claim 문장 단위 커버율과 미커버 기관(source_org) 빈도를 집계한다.

주의: classify()/extract_claims()는 실제 HCX API를 호출한다 (표본 크기만큼 과금/시간 소요).

사용법 (프로젝트 루트에서):
    python -m agent.mapping.measure_catalog_coverage --n 150 --seed 99
    python -m agent.mapping.measure_catalog_coverage --n 20 --seed 1 --gov-only  # 한국 정부기관 출처만
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
from collections import Counter
from pathlib import Path

from agent.preprocessing.classifier import classify
from agent.preprocessing.claim_extractor import extract_claims
from agent.mapping.keyword_search import keyword_search

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "data_set.csv"

# 한국 정부/공공기관만 걸러내기 위한 화이트리스트. UBS/무디스/미 상무부 같은 해외기관이나
# 한국통신사업자연합회/증권사 리서치 같은 민간기관은 여기 없으면 --gov-only에서 제외된다.
# (접미사 "부"만으로 거르면 "미 상무부"처럼 해외 정부 부처도 걸려버려서, 명시적 화이트리스트로 관리)
KOREAN_GOV_ORGS = {
    "통계청", "국세청", "관세청", "한국은행", "기획재정부", "기획예산처",
    "산업통상자원부", "산업통상부", "산업통상부(산업통상자원부)",
    "금융위원회", "금융위", "금융감독원", "금감원", "국가데이터처",
    "고용노동부", "교육부", "행정안전부", "국토교통부", "해양수산부",
    "농림축산식품부", "보건복지부", "환경부", "중소벤처기업부",
    "과학기술정보통신부", "과학기술통신부", "문화체육관광부", "여성가족부",
    "통일부", "외교부", "국방부", "법무부", "공정거래위원회", "공정위",
    "방송통신위원회", "개인정보보호위원회", "원자력안전위원회", "국민권익위원회",
    "한국부동산원", "특허청", "기상청", "산림청", "병무청", "조달청",
    "소방청", "경찰청", "해양경찰청", "식품의약품안전처", "인사혁신처",
    "법제처", "국무조정실", "감사원", "정부", "질병관리청",
    "국가정보자원관리원", "행정중심복합도시건설청", "새만금개발청",
}


def is_korean_gov_org(org: str) -> bool:
    return org.strip() in KOREAN_GOV_ORGS

# data_set.csv 본문에 조선일보 페이지 네비게이션/광고/관련기사 텍스트가 그대로 섞여 있는
# 경우가 많아(2,507건 중 약 45%), byline 뒤~다음 노이즈 마커 전까지만 잘라서 쓴다.
BYLINE_RE = re.compile(r"입력\s*20\d{2}\.\d{2}\.\d{2}\.")
STOP_MARKERS = ["구독수", "Video Player", "By Taboola", "많이 본 뉴스", "AI 추천", "100자평"]


def clean_body(body: str) -> str:
    m = BYLINE_RE.search(body)
    if not m:
        return body
    tail = body[m.end():]
    stop_positions = [tail.find(marker) for marker in STOP_MARKERS if tail.find(marker) != -1]
    end = min(stop_positions) if stop_positions else len(tail)
    cleaned = tail[:end].strip()
    return cleaned if len(cleaned) > 50 else body


def load_true_rows(path: Path = DATA_PATH) -> list[list[str]]:
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        return [r for r in reader if len(r) == 5 and r[4].strip() == "True"]


def main(n: int = 150, seed: int = 99, sleep_sec: float = 0.35, gov_only: bool = False) -> None:
    rows = load_true_rows()
    rng = random.Random(seed)
    sample = rng.sample(rows, min(n, len(rows)))

    n_classify_true = 0
    n_covered = 0
    n_uncovered = 0
    uncovered_orgs: Counter = Counter()
    uncovered_examples: dict[str, list[str]] = {}

    for i, (title, date, url, body, label) in enumerate(sample, 1):
        cleaned = clean_body(body)
        try:
            cls = classify(cleaned)
        except Exception as e:  # noqa: BLE001 - 점검 스크립트라 실패해도 계속 진행
            print(f"[{i}/{len(sample)}] classify 에러 -> {e}")
            continue
        time.sleep(sleep_sec)
        if not cls.label:
            continue
        n_classify_true += 1

        try:
            claims = extract_claims(cleaned)
        except Exception as e:  # noqa: BLE001
            print(f"[{i}/{len(sample)}] extract 에러 -> {e}")
            continue

        # claim_extractor는 보통 기사 첫 문장에만 기관명을 붙이고(예: "통계청은... 밝혔다"),
        # 같은 통계를 부연하는 뒤 문장들은 source_org가 비어있는 경우가 많다. 문장 단위로만
        # 걸러내면 같은 정부 통계 기사의 후속 문장들이 부당하게 빠지므로, "기사 안에 정부기관
        # 출처 문장이 하나라도 있으면 그 기사의 모든 claim을 정부통계로 간주"하는 방식으로 판단.
        article_is_gov = any(is_korean_gov_org((c.source_org or "").strip()) for c in claims)

        for c in claims:
            org = (c.source_org or "").strip()
            if gov_only and not article_is_gov:
                continue  # 정부기관 출처 문장이 기사 전체에 하나도 없으면 --gov-only에서 제외
            results = keyword_search(c, top_k=1)
            is_covered = bool(results) and results[0].score > 0
            if is_covered:
                n_covered += 1
            else:
                n_uncovered += 1
                if org:
                    uncovered_orgs[org] += 1
                    uncovered_examples.setdefault(org, []).append(c.sentence[:80])
            mark = "O" if is_covered else "X"
            print(f"[{mark}] {title[:30]} :: {c.sentence[:50]}")
        time.sleep(sleep_sec)

    total_claims = n_covered + n_uncovered
    print("\n=== 요약 ===")
    print(f"표본 {len(sample)}건 중 classifier='관련' {n_classify_true}건 "
          f"({n_classify_true/len(sample)*100:.1f}%)")
    if gov_only:
        print("(--gov-only: 한국 정부/공공기관 출처 claim만 집계)")
    if total_claims:
        print(f"claim 문장 {total_claims}건 중 카탈로그 매칭(keyword_search) {n_covered}건 "
              f"({n_covered/total_claims*100:.1f}%)")
    else:
        print("집계할 claim이 없습니다 (gov-only 필터에 걸린 게 없거나 표본이 너무 작음).")
    print("\n=== 미커버 기관 빈도 (상위 20) ===")
    for org, cnt in uncovered_orgs.most_common(20):
        print(f"  {org}: {cnt}")
        for s in uncovered_examples[org][:2]:
            print(f"    - {s}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=150, help="표본 크기 (기본 150)")
    parser.add_argument("--seed", type=int, default=99, help="랜덤 시드")
    parser.add_argument("--gov-only", action="store_true",
                         help="한국 정부/공공기관(KOREAN_GOV_ORGS) 출처 claim만 집계·출력")
    args = parser.parse_args()
    main(n=args.n, seed=args.seed, gov_only=args.gov_only)
