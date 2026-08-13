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
from functools import lru_cache
from pathlib import Path
from typing import Optional

try:
    from agent.interfaces import Claim, TableCandidate
except ImportError:
    from dataclasses import dataclass, field

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
#
# 2026-08-13 실측 발견(4→5로 상향): "서비스업"/"국민연금"처럼 업종·기관명이 정확히 4글자인
# 경우, 그 뒤에 붙는 진짜 핵심 단어가 완전히 달라도(취업자 vs 생산, 가입자 vs 연기금) 앞
# 4글자만 같으면 매칭돼버리는 오탐이 실제 배치에서 확인됨 — "서비스업 생산은 전달보다 0.1%
# 감소했다"가 "산업별 취업자"(서비스업 취업자) 표로, "한국거래소에 따르면 국민연금 등
# 연기금은... 순매수했다"가 "국민연금 가입자 현황" 표로 오매칭됨. 5자로 늘려도 원래 이 기능을
# 만든 이유였던 "소비자물가지수" vs "소비자물가가"(5자 "소비자물가"까지 겹침) 케이스는 그대로
# 살아있어 recall 손실 없이 이 두 오매칭만 해결된다(직접 계산으로 확인).
_MIN_STEM_MATCH_LEN = 5


def _stem_prefix_match(kw_norm: str, sentence_norm: str) -> bool:
    """kw_norm과 sentence_norm 안의 어떤 부분 문자열이 앞에서부터 최소
    _MIN_STEM_MATCH_LEN자 이상 일치하면 True (명사형 접미사만 다른 경우를 잡기 위함).

    2026-08-13: kiwipiepy(형태소 분석기)를 못 쓰는 환경(미설치 등)에서만 쓰는 폴백으로
    격하됨 — 아래 _morph_match() 참고. 글자수 임계값 방식은 "서비스업"/"국민연금"처럼
    업종·기관명이 정확히 임계값 길이인 경우 우연히 걸리는 오탐 위험이 구조적으로 남아있다."""
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


# ---------------------------------------------------------------------------
# 2026-08-13: 형태소 분석 기반 매칭 (kiwipiepy) — _stem_prefix_match의 "글자수만 세는"
# 한계(오탐)를 해결하기 위해 도입. 카탈로그 키워드 쪽만 명사로 쪼개서 통계 접미사(지수/율 등)
# 를 뗀 "핵심 명사"를 뽑고, 그 각각이 문장 원문에 부분 문자열로 있는지 확인한다.
#
# 왜 "명사 집합이 완전히 겹쳐야 매칭"(순수 형태소 비교)이 아니라 "핵심 명사를 문장 원문에
# 부분 문자열로 검색"인가: 처음엔 문장도 형태소로 쪼개서 명사 집합끼리 비교했는데, 복합명사가
# 카탈로그 키워드 쪽과 문장 쪽에서 다르게 쪼개지는 경우(예: "판매액"은 kiwi가 한 토큰으로 묶는데
# 문장엔 "판매"만 있어서 실패, "전산업"도 붙여 쓰면 한 토큰인데 띄어 쓰면 "전"이 분리돼 버려서
# 실패) 정상 매칭 2건이 오히려 깨지는 게 실측 확인됨. 그래서 문장 쪽은 형태소로 안 쪼개고
# 원문 그대로 부분 문자열 검사만 해서, 분절 방식 차이에 영향을 안 받게 했다.
try:
    from kiwipiepy import Kiwi

    _kiwi: "Kiwi | None" = Kiwi()
except ImportError:
    _kiwi = None

# 카탈로그 키워드에 흔히 붙는 통계 접미사 — 명사 끝에 이게 붙어있으면 떼어내고 비교한다
# (독립 토큰으로 나오든, "판매액"처럼 다른 명사에 붙어 하나의 토큰으로 나오든 둘 다 처리).
#
# "율"은 일부러 뺐다 — 2026-08-13 실측: kiwi가 "환율"을 [환,율]로 안 쪼개고 통째로 하나의
# 명사 토큰("환율")으로 인식하는데, 여기서 "율"을 접미사로 떼면 "환" 1글자만 남아 "전환했다"의
# "환"에 우연히 걸리는 오탐이 재현됐다. "환율"/"이자율" 같은 단어는 "율"이 뒤에 붙는 게 아니라
# 단어 자체에 융합된 개념이라, "지수"/"현황"처럼 안전하게 뗄 수 있는 접미사가 아니다.
#
# 2026-08-13 추가(건수/수지/총괄/비율): table_catalog.json 전체 키워드·제목 462개를 훑어서
# "명사가 2개 이상일 때 마지막 토큰" 빈도를 실측하고, 후보마다 실제 카탈로그 예시로 안전성을
# 검증한 뒤 추가함. "수지"는 "국제수지"/"경상수지"/"무역수지"처럼 가끔 한 토큰으로 붙지만, 떼어낸
# 뒤에도 "국제"/"경상"/"무역"처럼 그 자체로 멀쩡한 단어가 남아서 "환율"과 달리 안전함을 확인.
# "수"(count)는 후보였으나 뺐다 — "세수"(稅收, 세금 수입)가 한 토큰으로 붙는데 "수"를 떼면
# "세" 1글자만 남아, "환율"과 똑같은 유형의 오탐 위험이 재현되는 게 확인됨.
_DROPPABLE_NOUN_SUFFIXES = (
    "지수", "지표", "현황", "동향", "통계", "조사", "총액", "액", "등",
    "건수", "수지", "총괄", "비율",
)


def _strip_droppable_suffix(token: str) -> Optional[str]:
    for suf in _DROPPABLE_NOUN_SUFFIXES:
        if token == suf:
            return None
        if token.endswith(suf) and len(token) > len(suf):
            return token[: -len(suf)]
    return token


@lru_cache(maxsize=None)
def _core_noun_tokens(keyword: str) -> tuple[str, ...]:
    """카탈로그 키워드 하나를 명사로 쪼개고 통계 접미사를 뗀 "핵심 명사" 튜플을 반환한다.
    표 개수·키워드 개수가 적어서(카탈로그 전체를 다 합쳐도 수백 개) 전부 캐싱해도 부담 없다.

    2026-08-13: 품사 태그를 NNG/NNP(일반/고유명사)로만 한정한다 — 원래는 "NN"으로 시작하는
    태그를 전부 인정했는데, NNB(의존명사, 혼자 못 쓰는 말)까지 포함되면서 "0세"의 "세"처럼
    숫자가 명사로 안 잡혀 떨어져 나가고 의존명사 "세" 한 글자만 남아 "하락세"에 우연히
    걸리는 오탐이 있었다("0세"가 "60세"에 우연히 걸리던 원래 버그가 다른 경로로 재발). 글자수
    임계값 대신 품사로 걸러내면 "빵"처럼 진짜 의미 있는 1글자 일반명사는 그대로 살릴 수 있다."""
    if _kiwi is None:
        return ()
    nouns = [t.form for t in _kiwi.tokenize(keyword) if t.tag in ("NNG", "NNP")]
    stripped = [_strip_droppable_suffix(n) for n in nouns]
    return tuple(s for s in stripped if s)


def _morph_match(keyword: str, sentence_norm: str) -> bool:
    """keyword의 핵심 명사 전부가 (형태소 분석 없이) 문장 원문에 부분 문자열로 있으면 True."""
    tokens = _core_noun_tokens(keyword)
    if not tokens:
        return False
    return all(tok in sentence_norm for tok in tokens)


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
        elif _kiwi is not None:
            if _morph_match(kw, normalized_sentence):
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

    print("\n=== 2026-08-13 회귀 테스트: _MIN_STEM_MATCH_LEN 4→5 (업종/기관명 4글자 오탐 방지) ===")
    # (a) 원래 stem match를 만든 이유였던 케이스는 그대로 살아있어야 함
    assert _stem_prefix_match("소비자물가지수", "소비자물가가올랐다")
    print("[유지] '소비자물가지수' vs '소비자물가가' 여전히 매칭됨")

    # (b) "서비스업"(4글자) 뒤에 다른 단어가 붙으면 더 이상 오매칭되면 안 됨
    #     (실측: "서비스업 생산은 전달보다 0.1% 감소했다"가 "산업별 취업자" 표로 잘못 매칭됨)
    assert not _stem_prefix_match("서비스업취업자", "서비스업생산은전달보다")
    service_claim_results = keyword_search(Claim(sentence="서비스업 생산은 전달보다 0.1% 감소했다.", claim_type="규모"))
    assert all("취업자" not in r.table_name for r in service_claim_results), (
        f"❌ '서비스업 생산' claim이 여전히 취업자 표로 오매칭됨: {service_claim_results}"
    )
    print("[수정 확인] '서비스업 생산' claim이 더 이상 '취업자' 표로 오매칭 안 됨")

    # (c) "국민연금"(4글자) 뒤에 다른 단어가 붙으면 더 이상 오매칭되면 안 됨
    #     (실측: "국민연금 등 연기금은... 순매수했다"가 "국민연금 가입자/수급자 현황" 표로 잘못 매칭됨)
    assert not _stem_prefix_match("국민연금가입자", "국민연금등연기금은")
    pension_claim_results = keyword_search(
        Claim(
            sentence="한국거래소에 따르면 국민연금 등 연기금은 연초 이후 유가증권시장에서 1조6132억원을 순매수했다.",
            claim_type="규모",
        )
    )
    assert all("가입자" not in r.table_name and "수급자" not in r.table_name for r in pension_claim_results), (
        f"❌ '국민연금 등 연기금' claim이 여전히 가입자/수급자 표로 오매칭됨: {pension_claim_results}"
    )
    print("[수정 확인] '국민연금 등 연기금' claim이 더 이상 가입자/수급자 표로 오매칭 안 됨")
    print("\n[전체 통과] keyword_search 어간매칭 오탐 회귀 테스트 3건 모두 통과")

    if _kiwi is not None:
        print("\n=== 2026-08-13 회귀 테스트: 형태소 매칭(_morph_match) 도입 후 재발한 오탐 3건 ===")
        # (d) "환율"을 접미사 "율" 제거 대상으로 잘못 취급해 "환" 1글자만 남았던 문제
        #     (실측: "환" 1글자가 "전환했다"의 "환"에 우연히 걸림)
        assert _core_noun_tokens("환율") == ("환율",), f"❌ '환율'이 통째로 안 남음: {_core_noun_tokens('환율')}"
        exchange_rate_results = keyword_search(
            Claim(sentence="취업자 수가 46개월 만에 감소 전환했다", claim_type="증감률")
        )
        assert all("환율" not in r.table_name for r in exchange_rate_results), (
            f"❌ '전환했다'가 여전히 환율 표로 오매칭됨: {exchange_rate_results}"
        )
        print("[수정 확인] '전환했다'가 더 이상 환율 표로 오매칭 안 됨")

        # (e) "0세"에서 숫자 "0"이 명사로 안 잡혀 의존명사 "세" 1글자만 남았던 문제
        #     (실측: "세" 1글자가 "하락세"의 "세"에 우연히 걸림 — NNB 태그 제외로 해결)
        assert _core_noun_tokens("0세") == (), f"❌ '0세'가 빈 튜플이 아님(NNB 필터 실패): {_core_noun_tokens('0세')}"
        price_drop_results = keyword_search(Claim(sentence="전국 집값이 하락세를 보였다", claim_type="비교"))
        assert all("주민등록인구" not in r.table_name for r in price_drop_results), (
            f"❌ '하락세'가 여전히 0세 인구 표로 오매칭됨: {price_drop_results}"
        )
        print("[수정 확인] '하락세'가 더 이상 0세 인구 표로 오매칭 안 됨")

        # (f) 2글자 미만 토큰을 전부 잘라내던 예전 방식이 "빵"(진짜 1글자 명사) 같은 정상
        #     토큰까지 지워서, "빵 물가"가 "물가"만 남아 아무 물가 claim에나 걸리던 문제
        assert "빵" in _core_noun_tokens("빵 물가"), f"❌ '빵'이 핵심 명사에서 빠짐: {_core_noun_tokens('빵 물가')}"
        generic_price_results = keyword_search(
            Claim(sentence="지난달 소비자물가가 전년 동월 대비 2.2% 올랐다", claim_type="증감률")
        )
        assert all("빵" not in (r.source_meta or "") for r in generic_price_results), (
            f"❌ '빵'이 안 나온 물가 claim이 '빵 물가' 키워드로 오매칭됨: {generic_price_results}"
        )
        print("[수정 확인] '빵' 언급 없는 물가 claim이 더 이상 '빵 물가' 키워드로 오매칭 안 됨")

        print("\n[전체 통과] 형태소 매칭 오탐 회귀 테스트 3건 모두 통과")
    else:
        print("\n[건너뜀] kiwipiepy 미설치 환경 — 형태소 매칭 회귀 테스트는 _stem_prefix_match 폴백만 사용됨")