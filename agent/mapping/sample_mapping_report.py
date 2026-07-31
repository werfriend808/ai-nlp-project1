"""
agent/mapping/sample_mapping_report.py — data_set_true.csv(이미 "검색 구분 레이블"=True인
기사만 모아둔 파일)에서 무작위 n건을 뽑아, 그 n건에 곧바로 2~3단계(수치 주장 추출 ->
통계표 매핑)를 돌려서 claim 문장별로 "통계표에 맵핑됐는가"를 사람이 바로 훑어볼 수 있게
표시하는 점검 스크립트.

CSV 라벨을 그대로 신뢰하고 뽑기 때문에 1단계(classify()) 재분류는 하지 않는다 — 예전엔
data_set.csv 전체(True/False 섞임)에서 뽑아 classify()로 다시 걸러가며 label=True n건이
모일 때까지 반복했지만, 데이터 소스 자체가 이미 True만 담고 있으니 그 재검증 단계가
불필요해졌다. 대신 CSV 라벨과 classify()의 판단 기준(prompts/classifier_prompt.txt —
정부 공식·반복 통계만 True, 민간기업/해외/1회성 수치는 False)이 달라서 이 n건 안에도
classify() 기준으로는 False일 기사가 섞여 있을 수 있다는 점은 감안해야 한다.

agent/mapping/measure_catalog_coverage.py(대규모 표본, keyword_search로 커버율 집계)와
같은 방식으로 "해당 claim에 맞는 통계표가 table_catalog.json에 있는가"만 keyword_search
결과 유무로 판정한다 (찾음/못찾음 2단계).

⚠️ search_and_rerank(embedding_search 포함)는 일부러 안 씀: .env에
KOSIS_DISABLE_EMBEDDING=1이 걸려 있으면 embedding_search가 해시 기반 더미 벡터로
폴백해서 의미 없는 유사도 점수라도 항상 후보를 하나 반환한다 — 그래서 "찾음/못찾음"
판정에 embedding 후보를 섞으면 진짜 카탈로그 커버 여부와 무관하게 항상 "찾음"처럼
보이는 문제가 생긴다. keyword_search는 table_catalog.json의 keywords 필드가 실제로
문장에 등장했을 때만 후보를 반환하므로, "카탈로그에 있는지 없는지"를 판정하기엔
이쪽이 훨씬 신뢰할 수 있는 신호다.

주의: extract_claims()는 실제 HCX API를 호출한다 (표본 크기만큼 과금/시간 소요).

사용법 (프로젝트 루트에서):
    python -m agent.mapping.sample_mapping_report            # 실행마다 무작위 10건
    python -m agent.mapping.sample_mapping_report --n 20
    python -m agent.mapping.sample_mapping_report --seed 1    # 특정 표본 재현하고 싶을 때만
    python -m agent.mapping.sample_mapping_report --out report.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import time
from pathlib import Path

from agent.preprocessing.claim_extractor import extract_claims
from agent.mapping.keyword_search import keyword_search

DATA_PATH = Path(__file__).parent.parent.parent / "data" / "data_set_true.csv"

# data_set.csv 본문에 조선일보 페이지 네비게이션/광고/관련기사 텍스트가 그대로 섞여 있는
# 경우가 많아서(measure_catalog_coverage.py에서 이미 확인된 문제), byline 뒤~다음 노이즈
# 마커 전까지만 잘라서 쓴다.
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
    """data_set_true.csv는 이미 "검색 구분 레이블"=True인 기사만 담고 있으므로
    (agent/mapping/sample_mapping_report.py 실행 시 별도 필터링 불필요), 형식만 확인한다."""
    with open(path, encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)
        return [r for r in reader if len(r) == 5]


def judge_mapping(candidates) -> tuple[str, str]:
    """keyword_search 결과만으로 "table_catalog.json에 해당 통계표가 있는가"를 판정한다.

    반환: (상태 라벨, 상세 설명)
    """
    if not candidates:
        return "못찾음", "keyword_search 매칭 후보 없음"
    top = candidates[0]
    return "찾음", f"{top.table_name}({top.table_id}) score={top.score:.2f} | {top.source_meta}"


def main(n: int = 10, seed: int | None = None, sleep_sec: float = 0.35, out: str | None = None) -> None:
    if seed is None:
        # --seed를 안 주면 실행할 때마다 다른 무작위 표본이 뽑히도록 시스템 난수로 시드를 생성.
        # 나중에 같은 표본을 재현하고 싶으면 여기 출력되는 시드값을 --seed로 넘기면 됨.
        seed = random.SystemRandom().randint(0, 2**31 - 1)
        print(f"(무작위 시드 사용: {seed} — 재현하려면 --seed {seed})")

    rows = load_true_rows()
    rng = random.Random(seed)
    pool = rows[:]
    rng.shuffle(pool)

    # data_set_true.csv는 이미 전부 label=True라서, 1단계 재분류 없이 셔플한 순서
    # 그대로 앞에서 n건을 뽑는다 (그래서 "확인한 기사 수"와 "표본 크기"가 항상 같다).
    sample: list[list[str]] = pool[:n]

    print(f"표본 확정: data_set_true.csv에서 무작위 {len(sample)}건 선정 (1단계 재분류 생략)")

    report: list[dict] = []
    n_claims = n_found = n_not_found = 0

    for i, (title, date, url, body, label) in enumerate(sample, 1):
        cleaned = clean_body(body)
        print(f"\n[{i}/{len(sample)}] {title}")

        try:
            claims = extract_claims(cleaned)
        except Exception as e:  # noqa: BLE001
            print(f"  2단계 에러 -> {e}")
            continue
        time.sleep(sleep_sec)

        if not claims:
            print("  2단계: 추출된 수치 주장 없음")
            continue

        for c in claims:
            candidates = keyword_search(c, top_k=3)
            status, detail = judge_mapping(candidates)
            n_claims += 1
            if status == "찾음":
                n_found += 1
            else:
                n_not_found += 1

            print(f"  [{status}] {c.sentence[:60]}")
            print(f"      -> {detail}")

            report.append({
                "title": title,
                "date": date,
                "claim_sentence": c.sentence,
                "claim_type": c.claim_type,
                "status": status,
                "detail": detail,
                "top3_candidates": [
                    {"table_id": cand.table_id, "table_name": cand.table_name, "score": round(cand.score, 3)}
                    for cand in candidates[:3]
                ],
            })

    print(f"\n{'=' * 60}")
    print(f"기사 {len(sample)}건 표본, 수치 주장(claim) 총 {n_claims}건")
    if n_claims:
        print(f"  찾음(카탈로그에 있음): {n_found}건 ({n_found/n_claims*100:.1f}%)")
        print(f"  못찾음(카탈로그에 없음): {n_not_found}건 ({n_not_found/n_claims*100:.1f}%)")

    if out:
        out_path = Path(out)
        out_path.write_text(
            json.dumps(
                {
                    "summary": {
                        "n_articles": len(sample),
                        "n_claims": n_claims,
                        "n_found": n_found,
                        "n_not_found": n_not_found,
                        "seed": seed,
                    },
                    "claims": report,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="무작위 기사 표본 1~3단계 맵핑 여부 점검")
    parser.add_argument("--n", type=int, default=10, help="표본 기사 수 (기본 10)")
    parser.add_argument("--seed", type=int, default=None, help="랜덤 시드 (생략하면 실행마다 무작위)")
    parser.add_argument("--out", type=str, default=None, help="결과 JSON 저장 경로")
    args = parser.parse_args()
    main(n=args.n, seed=args.seed, out=args.out)
