"""
agent/mapping/keyword_search.py — 3단계: 규칙 기반 키워드 매칭

팀 계약(interfaces.py) 기준:
입력: Claim 1건 (sentence, claim_type, period, unit, population)
출력: TableCandidate의 리스트 (top-k)

모델 불필요 — 동의어 사전 기반 규칙 매칭.
table_catalog.json의 keywords 필드와 SYNONYMS 사전을 이용해
Claim.sentence 안에 등장하는 키워드를 찾아 매칭 점수를 계산한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

try:
    from agent.interfaces import Claim, TableCandidate
except ImportError:
    from dataclasses import dataclass, field
    from typing import Optional

    @dataclass
    class Claim:  # type: ignore[no-redef]
        sentence: str
        claim_type: str
        period: Optional[str] = None
        unit: Optional[str] = None
        population: Optional[str] = None

    @dataclass
    class TableCandidate:  # type: ignore[no-redef]
        table_id: str
        table_name: str
        score: float
        required_slots: list = field(default_factory=list)
        source_meta: Optional[str] = None

CATALOG_PATH = Path(__file__).parent / "table_catalog.json"

# ---------------------------------------------------------------------------
# 동의어 사전 — "표현이 다른" 사례(실전1 EDA에서 발견된 갭 유형) 보완용.
# key: 기사에 자주 등장하는 표현 → value: table_catalog.json의 keywords/title에
# 실제로 쓰이는 정규 표현. 여러 정규 표현으로 확장될 수 있어 리스트로 관리.
# 필요할 때마다 이 사전에 계속 추가해나가면 됨.
# ---------------------------------------------------------------------------
SYNONYMS: dict[str, list[str]] = {
    "취업자": ["취업자수", "고용률"],
    "실업자": ["실업률"],
    "고용": ["고용률", "고용동향"],
    "일자리": ["고용률", "취업자수"],
    "물가": ["소비자물가지수", "소비자물가", "물가지수"],
    "장바구니": ["장바구니 물가", "소비자물가지수"],
    "집값": ["집값", "아파트 매매가격", "주택매매가격지수"],
    "부동산": ["주택매매가격지수", "주택가격동향"],
    "출산": ["합계출산율", "출생률", "저출산", "출생아수"],
    "저출생": ["저출산", "출생아수", "출생률"],
    "출생아": ["출생아수", "합계출산율", "출생률"],
    "인구감소": ["인구감소", "주민등록인구"],
    "수출": ["수출액", "수출 증가율"],
    "수입": ["수입액"],
    "무역흑자": ["무역수지", "무역흑자"],
    "성장률": ["경제성장률", "GDP", "국내총생산"],
    "GDP": ["국내총생산", "경제성장률"],
}


def _load_catalog(path: Path = CATALOG_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. table_catalog.json이 먼저 있어야 합니다."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tables"]


def _expand_query_terms(sentence: str) -> set[str]:
    """문장에서 동의어 사전을 거쳐 검색에 쓸 정규화된 키워드 집합을 만든다."""
    terms: set[str] = set()
    for raw_term, mapped_terms in SYNONYMS.items():
        if raw_term in sentence:
            terms.update(mapped_terms)
    return terms


def _score_table(sentence: str, expanded_terms: set[str], table: dict) -> tuple[float, list[str]]:
    """표 하나에 대해 (매칭 점수, 매칭된 키워드 목록)을 계산한다."""
    matched: list[str] = []
    keywords = table.get("keywords", [])

    for kw in keywords:
        if kw in sentence:
            matched.append(kw)
        elif kw in expanded_terms:
            matched.append(kw)

    if table.get("title", "") and re.search(re.escape(table["title"][:6]), sentence):
        # 표 제목 앞부분이 문장에 그대로 등장하면 강한 신호로 취급
        matched.append(f"[title]{table['title']}")

    if not matched:
        return 0.0, []

    # 매칭 개수를 0~1 사이로 정규화 (키워드 3개 이상 매칭되면 만점 취급)
    score = min(1.0, len(matched) / 3)
    return score, matched


def keyword_search(
    claim: Claim,
    *,
    top_k: int = 5,
    catalog: list[dict] | None = None,
) -> list[TableCandidate]:
    """Claim 1건을 받아 키워드 규칙 매칭으로 후보 통계표를 반환한다."""
    tables = catalog if catalog is not None else _load_catalog()
    expanded_terms = _expand_query_terms(claim.sentence)

    candidates: list[TableCandidate] = []
    for table in tables:
        score, matched = _score_table(claim.sentence, expanded_terms, table)
        if score <= 0:
            continue
        candidates.append(
            TableCandidate(
                table_id=table["tblId"],
                table_name=table["title"],
                score=score,
                required_slots=table.get("required_slots", []),
                source_meta=f"keyword_search matched={matched}",
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[:top_k]


if __name__ == "__main__":
    # python -m agent.mapping.keyword_search
    test_claims = [
        Claim(sentence="지난달 청년 실업률이 6%에 육박했다", claim_type="규모"),
        Claim(sentence="취업자 수가 46개월 만에 감소 전환했다", claim_type="증감률"),
        Claim(sentence="지난달 소비자물가가 전년 동월 대비 2.2% 올랐다", claim_type="증감률"),
        Claim(sentence="장바구니 물가 부담이 커지고 있다", claim_type="규모"),
        Claim(sentence="전국 집값이 하락세를 보였다", claim_type="비교"),
    ]
    for c in test_claims:
        results = keyword_search(c)
        print(f"\n[{c.sentence}]")
        for r in results:
            print(f"  - {r.table_name} ({r.table_id}) score={r.score:.2f} | {r.source_meta}")