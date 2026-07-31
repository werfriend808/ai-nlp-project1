"""
agent/mapping/keyword_search.py — 3단계: 규칙 기반 키워드 매칭

팀 계약(interfaces.py) 기준:
입력: Claim 1건 (sentence, claim_type, period, unit, population)
출력: TableCandidate의 리스트 (top-k)

모델 불필요 — 동의어 사전 기반 규칙 매칭.
table_catalog.json의 keywords 필드와 SYNONYMS 사전을 이용해
Claim.sentence(+ statistic_expression/population, _build_search_text 참고) 안에
등장하는 키워드를 찾아 매칭 점수를 계산한다.
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
        statistic_expression: Optional[str] = None

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
    "물가": ["소비자물가지수", "소비자물가", "물가지수"],
    "장바구니": ["장바구니 물가", "소비자물가지수"],
    "집값": ["집값", "아파트 매매가격", "주택매매가격지수"],
    "부동산": ["주택매매가격지수", "주택가격동향"],
    "출산": ["합계출산율", "출생률", "저출산", "출생아수"],
    "저출생": ["저출산", "출생아수", "출생률"],
    "출생아": ["출생아수", "합계출산율", "출생률"],
    "인구감소": ["인구감소", "주민등록인구"],
    "수출": ["수출액", "수출 증가율"],
    "무역흑자": ["무역수지", "무역흑자"],
    "성장률": ["경제성장률", "GDP", "국내총생산"],
    "GDP": ["국내총생산", "경제성장률"],
    "이자율": ["대출금리", "예금은행 금리"],
    "기준금리": ["대출금리", "예금은행 금리"],
    "대출이자": ["대출금리", "여신금리"],
    "주가": ["코스피지수", "증시", "주가지수"],
    "코스피지수": ["코스피", "KOSPI"],
    "가계빚": ["가계신용", "가계부채"],
    "가계대출": ["가계신용", "가계부채"],
    "전셋값": ["전세가격", "전세지수"],
    "전세값": ["전세가격", "전세지수"],
    "이사": ["국내이동", "인구이동"],
    "학원비용": ["사교육비", "학원비"],
    "임금격차": ["시간당임금", "임금격차"],
    "소매판매": ["전산업생산지수", "산업생산", "산업활동동향"],
    "생산지표": ["전산업생산지수", "산업생산"],
    "나라살림": ["통합재정수지", "재정수지"],
    "나라 살림": ["통합재정수지", "재정수지"],
    "재정적자": ["통합재정수지", "재정수지"],
    "국민연금": ["국민연금 가입자", "국민연금 수급자"],
    "소득 격차": ["소득 5분위", "소득분위"],
    "소득 양극화": ["소득 5분위", "소득분위"],
    "저소득층": ["소득 하위", "소득 5분위"],
    "고소득층": ["소득 상위", "소득 5분위"],
}

# "소매판매"/"생산지표"(동의어)와 "경제성장률"/"GDP"/"국내총생산"/"성장률"(DT_200Y102의
# 리터럴 keywords)는 전부 국가명 없이 무조건 매칭되면 "미국의 소매판매"/"세계 경제성장률"
# 같은 해외 claim까지 한국 표로 잘못 연결하는 오탐이 생긴다(전자는 sample_mapping_report.py
# 실행 중 실제 확인됨, 후자는 같은 유형이라 함께 처리). 문장에 외국 표시어가 있으면
# 이 표현들만 매칭에서 제외한다 — 나머지 동의어/키워드는 기존대로 동작.
_FOREIGN_MARKERS = (
    "미국", "중국", "일본", "유럽", "EU", "독일", "영국", "프랑스", "인도",
    "사우디", "베트남", "브라질", "러시아", "대만", "홍콩", "캐나다", "호주",
    "인도네시아", "멕시코", "싱가포르", "태국", "필리핀",
)
# "세계"/"글로벌"/"전세계"는 한때 마커에 포함시켰다가 뺐음: "세계 경제성장률"(진짜 해외
# GDP 얘기) 오탐은 잡지만, "세계 금융위기"/"세계 최대"처럼 국내 claim에도 흔히 섞여
# 쓰이는 단어라 오히려 국내 경제성장률 claim을 광범위하게 못 찾게 만드는 회귀가 발생함
# (실제 확인됨). 국가명 단독보다 오탐/오답 트레이드오프가 나빠서 제외.
_LOCALE_SENSITIVE_TERMS = {
    "소매판매", "생산지표",
    "경제성장률", "국내총생산", "GDP", "성장률",
    # 고용/실업(2026-07-30 sample_mapping_report.py 실행 중 실제 확인됨): "미국 ADP 고용
    # 지표가 예상을 밑도는 7.7만명을 기록"과 "미국 중서부 공장 근로자 900명 해고"가 둘 다
    # DT_1DA7001S(성별 경제활동인구총괄)로 잘못 매칭됨 — 고용률/취업자수 같은 키워드가
    # 국가명 없이도 매칭돼서 생긴 문제.
    "경제활동인구", "고용률", "경제활동참가율", "취업자수", "고용동향",
    "청년실업률", "연령별실업률", "세대별실업률", "고령층실업률", "실업률",
    # 물가(같은 실행에서 확인 대신, classifier_prompt.txt 예시7의 "4월 미국 소비자물가"가
    # 바로 이 유형 — 1단계에서 걸러지긴 하지만 keyword_search 자체도 오매칭 위험이 같음):
    "소비자물가지수", "소비자물가", "CPI", "물가지수", "물가상승률",
    # 무역/수출입("2015년 대만의 대중국 수출은 전년 대비 12.4% 감소했다"가 국가별
    # 수출액·수입액 표로 잘못 매칭된 사례 — 이 표는 한국의 대상국별 무역만 다루므로
    # 대만-중국 양자 무역과는 무관함):
    "수출액", "수입액", "무역수지", "무역흑자", "무역적자", "수출 증가율",
}


def _has_foreign_marker(sentence: str) -> bool:
    return any(marker in sentence for marker in _FOREIGN_MARKERS)


# 짧은 리터럴 keyword가 무관한 합성어 안에 우연히 들어있어서 오탐이 나는 경우 보완용
# (예: "유동인구"는 상권분석 빅데이터 용어라 KOSIS 주민등록인구(DT_1B04005N)와 무관한데,
# keyword "인구"가 그 글자를 그대로 포함하고 있어서 매칭돼버림 — sample_mapping_report.py
# 실행 중 실제 확인됨). "수입"/"수입차"와 같은 종류의 부분 문자열 충돌.
_COMPOUND_EXCLUSIONS: dict[str, tuple[str, ...]] = {
    "인구": ("유동인구", "유동 인구", "생산가능인구", "생산가능 인구", "인구밀도", "인구 밀도"),
}


def _is_compound_excluded(kw: str, sentence: str) -> bool:
    return any(excl in sentence for excl in _COMPOUND_EXCLUSIONS.get(kw, ()))


def _load_catalog(path: Path = CATALOG_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 가 없습니다. table_catalog.json이 먼저 있어야 합니다."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tables"]


def _build_search_text(claim: Claim) -> str:
    """claim.sentence만으로는 못 잡는 케이스 보완용 검색 텍스트를 만든다.

    claim_extractor(2단계)가 "이는"/"전달보다" 같은 지시어로 앞 문장에 기대는 후속
    문장에는, 원문에 키워드가 없어도 statistic_expression/population 필드에 실제
    지표명("외환 보유액", "취업자수" 등)을 이미 채워서 넘겨준다(팀 계약 필드,
    prompts/claim_extractor_prompt.txt 예시 참고). sentence만 보고 매칭하면 이런
    후속 문장이 keyword_search에서 전부 누락되므로, 두 필드도 검색 대상에 포함한다.
    """
    parts = [claim.sentence]
    stat_expr = getattr(claim, "statistic_expression", None)
    if stat_expr:
        parts.append(stat_expr)
    if claim.population:
        parts.append(claim.population)
    return " ".join(parts)


def _expand_query_terms(search_text: str) -> set[str]:
    """검색 텍스트에서 동의어 사전을 거쳐 검색에 쓸 정규화된 키워드 집합을 만든다."""
    terms: set[str] = set()
    has_foreign_marker = _has_foreign_marker(search_text)
    for raw_term, mapped_terms in SYNONYMS.items():
        if raw_term not in search_text:
            continue
        if raw_term in _LOCALE_SENSITIVE_TERMS and has_foreign_marker:
            continue
        terms.update(mapped_terms)
    return terms


def _score_table(search_text: str, expanded_terms: set[str], table: dict) -> tuple[float, list[str]]:
    """표 하나에 대해 (매칭 점수, 매칭된 키워드 목록)을 계산한다."""
    matched: list[str] = []
    keywords = table.get("keywords", [])
    has_foreign_marker = _has_foreign_marker(search_text)

    for kw in keywords:
        if kw in _LOCALE_SENSITIVE_TERMS and has_foreign_marker:
            continue
        if _is_compound_excluded(kw, search_text):
            continue
        if kw in search_text:
            matched.append(kw)
        elif kw in expanded_terms:
            matched.append(kw)

    if table.get("title", "") and re.search(re.escape(table["title"][:6]), search_text):
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
    search_text = _build_search_text(claim)
    expanded_terms = _expand_query_terms(search_text)

    candidates: list[TableCandidate] = []
    for table in tables:
        score, matched = _score_table(search_text, expanded_terms, table)
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