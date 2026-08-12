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
    "취업자": ["취업자수"],
    "실업자": ["실업률"],
    "고용": ["고용률", "고용동향"],
    "일자리": ["고용률", "취업자수"],
    "장바구니": ["장바구니 물가", "소비자물가지수"],
    "집값": ["집값", "아파트 매매가격", "주택매매가격지수"],
    "부동산": ["주택매매가격지수", "주택가격동향"],
    "출산": ["합계출산율", "출생률", "저출산", "출생아수"],
    "저출생": ["저출산", "출생아수", "출생률"],
    "출생아": ["출생아수", "합계출산율", "출생률"],
    "인구감소": ["인구감소", "주민등록인구"],
    "수출이": ["수출액", "수출 증가율"],
    "수출은": ["수출액", "수출 증가율"],
    "수입": ["수입액"],
    "무역흑자": ["무역수지", "무역흑자"],
    "성장률": ["경제성장률", "GDP", "국내총생산"],
    "GDP": ["국내총생산", "경제성장률"],
    "이자율": ["대출금리", "예금은행 금리"],
    "대출이자": ["대출금리", "여신금리"],
    # "기준금리"(중앙은행 정책금리)를 "대출금리"(시중은행 대출금리)의 동의어로 잘못 등록해뒀던
    # 항목을 제거함 — 2026-08-09, "일본의 기준금리는 0.5%다"가 국내 예금은행 대출금리 표로
    # 잘못 매칭되던 실제 사례 발견(2단계 개념 자체가 다르고, 국가도 다름). 기준금리는
    # docs/PENDING_TABLES.md에 이미 기록된 대로 KOSIS Open API로 애초에 검증 불가능한
    # 통계라서, 억지로 다른 표에 연결하지 않고 그냥 매칭 안 되게(정직하게 미해결) 둔다.
    "주가": ["코스피지수", "증시", "주가지수"],
    "코스피지수": ["코스피", "KOSPI"],
    "가계빚": ["가계신용", "가계부채"],
    "가계대출": ["가계신용", "가계부채"],
    "전셋값": ["전세가격", "전세지수"],
    "전세값": ["전세가격", "전세지수"],
    "이사": ["국내이동", "인구이동"],
    "학원비용": ["사교육비", "학원비"],
    "임금격차": ["시간당임금", "임금격차"],
}


def _load_catalog(path: Path = CATALOG_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. table_catalog.json이 먼저 있어야 합니다."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tables"]


def _normalize(text: str) -> str:
    """공백 제거 비교용 정규화. 기사 본문은 "혼인 건수"처럼 카탈로그 키워드
    "혼인건수"와 띄어쓰기가 달라지는 경우가 흔해서, 공백을 무시하고 비교한다."""
    return re.sub(r"\s+", "", text)


# SYNONYMS의 raw_term(취업자/이사/주가 등)은 전부 내부 공백이 없는 단일 단어라서, 원래
# _normalize()처럼 문장 전체의 공백을 지우고 부분 문자열로 비교할 이유가 없다 — 오히려
# 문장에서 서로 무관한 두 단어가 공백 하나만 사이에 두고 붙어있을 때 우연히 다른 raw_term과
# 겹치는 오탐이 생긴다. 2026-08-06 300개 배치 실측에서 "청년이 사상 처음으로"가 공백 제거
# 시 "청년이사상..."이 되어 "이사"(-> "국내이동"/"인구이동" 동의어)와 우연히 겹쳐 매칭되는
# 걸 발견("쉬었음" 관련 claim이 엉뚱하게 "시군구별 이동자수" 표로 매칭된 원인). raw_term은
# 공백 없는 단일 단어뿐이므로, 원문 그대로(공백 제거 없이) 부분 문자열 검사만 하면 이런
# 단어 경계를 넘나드는 오탐 없이 정상적인 경우("이사하는")는 그대로 잡힌다.
def _expand_query_terms(sentence: str) -> set[str]:
    """문장에서 동의어 사전을 거쳐 검색에 쓸 정규화된 키워드 집합을 만든다."""
    terms: set[str] = set()
    for raw_term, mapped_terms in SYNONYMS.items():
        if raw_term in sentence:
            terms.update(mapped_terms)
    return terms


# ⚠️ 2026-08-05 시도했다가 되돌린 것: "코스피" 같은 흔한 단어 1개 매칭만으로 무관한
# claim이 매칭되는 문제(예: "연기금 순매수 규모"가 "코스피 지수" 표로 잘못 매칭)를
# 막으려고 최소 매칭 키워드 개수를 2개로 올렸었다. 근데 골든셋 38건으로 직접
# 비교해보니 keyword_search top-1 정답률이 50.0%(19/38)→23.7%(9/38)로 오히려
# 크게 나빠졌다 — 이 카탈로그(표 59개)는 claim 하나당 키워드가 보통 1개만 걸리는
# 경우가 대부분이라, "2개 이상"이 대다수의 정상 매칭까지 같이 걸러버렸다. 그래서
# 되돌렸다. "코스피" 오매칭 문제는 여전히 남아있는데, 원인이 이 표(코스피 지수)의
# keywords 리스트 자체가 ['코스피','코스피지수','KOSPI','종합주가지수','증시',
# '주가지수']처럼 사실상 동의어를 여러 항목으로 나눠놔서 단어 하나로도 매칭 개수가
# 부풀려지는 것이라, 전체 문턱값을 올리는 방식으로는 (다른 표의 정상 매칭까지 다치지
# 않고는) 못 고친다. 카탈로그의 keywords 중복 정리, 또는 claim.population 같은
# 2단계 메타데이터까지 매칭에 활용하는 설계 변경이 필요해 보임 — B와 상의 필요.
# 실제 배치(claim_extractor가 뽑은 자연스러운 문장)에서 keyword_search가 claim의 92%에서
# 아무 표도 못 찾는 문제가 실측 확인됨(2026-08-11). 원인은 카탈로그 키워드가 "소비자물가지수"
# 처럼 명사형 접미사(지수/지표/총액 등)가 붙어있는데, 문장은 "소비자물가가 올랐다"처럼 어간만
# 쓰는 경우가 흔해서 정확한 부분 문자열 매칭에 실패하기 때문. 골든셋(claim_extractor가 아니라
# 사람이 다듬어 쓴 문장)에서는 53.7%로 훨씬 잘 맞아서 이 가설과 일치한다.
# 접두 N자 이상이 겹치면 어간 매칭으로 인정한다 — 최소 길이를 4자로 둔 이유는 "0세"가 "60세"에
# 우연히 부분 문자열로 걸리던 버그(table_catalog.json DT_1B04006)처럼 짧은 키워드의 우연한
# 충돌을 막기 위함. 4자 이상 접두 일치는 우연히 겹칠 확률이 훨씬 낮다.
_MIN_STEM_MATCH_LEN = 4


def _stem_prefix_match(kw_norm: str, sentence_norm: str) -> bool:
    """kw_norm과 sentence_norm 안의 어떤 부분 문자열이 앞에서부터 최소
    _MIN_STEM_MATCH_LEN자 이상 일치하면 True (명사형 접미사만 다른 경우를 잡기 위함)."""
    if len(kw_norm) < _MIN_STEM_MATCH_LEN:
        return False
    prefix = kw_norm[:_MIN_STEM_MATCH_LEN]
    start = sentence_norm.find(prefix)
    while start != -1:
        max_len = min(len(kw_norm), len(sentence_norm) - start)
        match_len = 0
        while match_len < max_len and kw_norm[match_len] == sentence_norm[start + match_len]:
            match_len += 1
        if match_len >= _MIN_STEM_MATCH_LEN:
            return True
        start = sentence_norm.find(prefix, start + 1)
    return False


def _score_table(sentence: str, expanded_terms: set[str], table: dict) -> tuple[float, list[str]]:
    """표 하나에 대해 (매칭 점수, 매칭된 키워드 목록)을 계산한다."""
    matched: list[str] = []
    keywords = table.get("keywords", [])
    normalized_sentence = _normalize(sentence)

    for kw in keywords:
        kw_norm = _normalize(kw)
        if kw_norm in normalized_sentence:
            matched.append(kw)
        elif kw in expanded_terms:
            matched.append(kw)
        elif _stem_prefix_match(kw_norm, normalized_sentence):
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