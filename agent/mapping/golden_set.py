"""
agent/mapping/golden_set.py — 임베딩/리랭커 모델 비교용 골든셋 로더.

두 파일을 claim_id 기준으로 join해서 (claim_sentence, gold_table_id) 쌍을 만든다:
  - notebooks/추출 골든셋 단위 분리.xlsx  : claim_id -> claim_sentence
  - notebooks/매핑 골든셋 ord 추가.xlsx   : claim_id -> kosis_table_id, match_status

⚠️ 그대로 쓰면 안 되는 이유 두 가지, 그래서 이 모듈이 필요하다:

1. match_status="매칭 실패"인 claim(19건, 2026-07-31 기준)은 애초에 KOSIS에 정식
   등재된 통계가 없는 케이스(관세청 잠정치 등)다. 어떤 임베딩/리랭커를 써도 못 맞히는 게
   정상이라 Recall@K/MRR 계산에서 반드시 제외해야 한다 — 안 그러면 카탈로그 공백을
   "모델이 못 찾음"으로 착각하게 된다.

2. 정답 kosis_table_id 40건 중 31건은 골든셋 작성 시점엔 table_catalog.json에 없던
   tblId였다. 그중 5개는 실제로 새 표로 추가했지만(DT_1J22112 등), 나머지는 이미
   카탈로그에 "다른 tblId"로 동일한 통계가 들어있는 경우였다(예: 골든셋의 소매판매
   DT_1K41013 -> 카탈로그엔 이미 DT_1K41017로 API 검증까지 끝난 표가 있음). 이런
   케이스를 override 안 하면 카탈로그에 진짜 정답이 있는데도 "못 찾음"으로 집계된다.
   TABLE_ID_OVERRIDES가 이 매핑을 담당한다 (2026-07-31 확인 기준, 근거는 대화 로그 참고).

   DT_1DA7002S(골든셋) vs DT_1DA7102S(카탈로그 기존값)는 설명·수치(청년실업률 5.9%)가
   똑같아 보이는데, 어느 쪽 tblId가 실제 KOSIS 코드로 맞는지 라이브 API로 검증하지
   않았다. 일단 카탈로그 기존값(DT_1DA7102S)을 신뢰하기로 하고 override에 넣었다 —
   나중에 검증 결과가 다르면 여기만 고치면 된다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "notebooks"
CLAIMS_XLSX = NOTEBOOKS_DIR / "추출 골든셋 단위 분리.xlsx"
MAPPING_XLSX = NOTEBOOKS_DIR / "매핑 골든셋 ord 추가.xlsx"
CATALOG_PATH = Path(__file__).parent / "table_catalog.json"

# 골든셋의 kosis_table_id -> table_catalog.json에 실제로 존재하는 tblId.
# 2026-07-31 대조 결과 확정된 값. 근거는 모듈 docstring 참고.
TABLE_ID_OVERRIDES: dict[str, str] = {
    "DT_1K41013": "DT_1K41017",     # 소매판매: 총계 항목 없는 버전 -> 총지수 검증된 버전
    "DT_1G18011": "DT_1G18007",     # 건설기성: 불변지수 -> 경상지수(카탈로그 검증판)
    "DT_1L9U001": "DT_1L9U103",     # 가계동향 월평균소득 -> 소득5분위별 가계수지로 흡수
    "DT_1HDALF05": "DT_1L9U103",    # 5분위 배율 -> 동일 표에서 파생 가능
    "DT_1DA7E33S": "DT_1DA7E06S_NEW",  # 시도x산업별 취업자 -> 산업별 취업자(전국)
    "DT_1DA7002S": "DT_1DA7102S",   # 연령별 실업률: 카탈로그 기존값 신뢰(미검증, 위 docstring 참고)
}

# 매칭 자체가 안 됐거나(정답 없음) 미완료인 claim은 평가에서 제외.
_NO_ANSWER_STATUSES = {"매칭 실패", "미완료"}


@dataclass
class GoldExample:
    claim_id: str
    sentence: str
    gold_table_id: str          # override 적용 후 값 (table_catalog.json 기준)
    raw_table_id: str           # 골든셋 원본 값 (override 전)
    match_status: str
    confidence: str              # "high"(값 확인됨) / "low"(표 후보 확인 등)
    overridden: bool             # raw_table_id != gold_table_id 였는지


def _confidence_of(match_status: str) -> str:
    return "high" if match_status.startswith("값 확인됨") else "low"


def load_golden_set(
    *,
    claims_path: Path = CLAIMS_XLSX,
    mapping_path: Path = MAPPING_XLSX,
    catalog_path: Path = CATALOG_PATH,
    warn_missing: bool = True,
) -> list[GoldExample]:
    """골든셋을 로드해서 (claim_sentence, gold_table_id) 평가용 리스트로 정제한다.

    반환되는 gold_table_id는 전부 table_catalog.json에 실제로 존재하는 tblId임을
    보장한다(override 적용 후에도 카탈로그에 없는 건 warn_missing=True일 때 출력하고
    제외한다).
    """
    claims_df = pd.read_excel(claims_path)
    mapping_df = pd.read_excel(mapping_path)
    merged = claims_df.merge(mapping_df, on="claim_id", how="left")

    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog_ids = {t["tblId"] for t in catalog["tables"]}

    examples: list[GoldExample] = []
    skipped_no_answer = 0
    skipped_missing_in_catalog: list[tuple[str, str]] = []

    for _, row in merged.iterrows():
        status = row.get("match_status")
        raw_id = row.get("kosis_table_id")

        if pd.isna(raw_id) or status in _NO_ANSWER_STATUSES:
            skipped_no_answer += 1
            continue

        gold_id = TABLE_ID_OVERRIDES.get(raw_id, raw_id)
        if gold_id not in catalog_ids:
            skipped_missing_in_catalog.append((row["claim_id"], gold_id))
            continue

        examples.append(
            GoldExample(
                claim_id=row["claim_id"],
                sentence=row["claim_sentence"],
                gold_table_id=gold_id,
                raw_table_id=raw_id,
                match_status=status,
                confidence=_confidence_of(status),
                overridden=(gold_id != raw_id),
            )
        )

    if warn_missing:
        print(
            f"[golden_set] 총 {len(merged)}건 중 정답 없음/미완료 {skipped_no_answer}건 제외, "
            f"카탈로그에 없는 gold_table_id {len(skipped_missing_in_catalog)}건 제외 "
            f"-> 평가 가능 {len(examples)}건"
        )
        if skipped_missing_in_catalog:
            print(f"  카탈로그에 없는 항목: {skipped_missing_in_catalog}")

    return examples


if __name__ == "__main__":
    # python -m agent.mapping.golden_set
    examples = load_golden_set()
    n_high = sum(1 for e in examples if e.confidence == "high")
    print(f"\nhigh-confidence(값 확인됨): {n_high}건, low-confidence(표 후보 확인 등): {len(examples) - n_high}건")
    for e in examples[:5]:
        print(f"  [{e.claim_id}] ({e.confidence}) {e.gold_table_id} | {e.sentence[:50]}")
