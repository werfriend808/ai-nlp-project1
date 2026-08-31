"""
agent/preprocessing/source_filter.py — 2단계 이후: claim 단위 출처기관 검증 필터

배경: 1단계(classifier)는 "기사 전체"가 국가 공식 통계를 인용하는지만 판단한다. 하지만
2단계(claim_extractor)는 그 기사 안의 숫자 있는 문장을 전부 뽑아버려서, 같은 기사 안에
진짜 KOSIS로 검증 가능한 claim과 해외기관/민간기업/1회성 사건류 claim이 섞여서 나온다
(2026-08-05 실측: data_set.csv 100건 샘플에서 확인 — "OECD 한국 성장률 전망" 기사가
1단계를 통과하면, 그 기사 안의 진짜 통계청 claim과 OECD 인용 claim이 구분 없이 같이
추출됨). 이 필터는 claim.source_org 하나하나를 보고, 실제 KOSIS 국가승인통계를
생산하는 기관인지 판단해서 3단계(표매칭)로 넘기기 전에 걸러낸다.

한계: 이 whitelist/blacklist는 KOSIS API로 매번 실시간 조회해서 검증한 게 아니라
도메인 지식 기반 휴리스틱이다(우리 table_catalog.json에 실제 등장하는 기관명 + 상식적으로
확실한 기관/비기관을 정리). UNCERTAIN으로 분류된 기관은 실제 KOSIS 등재 여부가
불확실하므로, 새로 자주 나오는 게 있으면 이 목록에 추가해나가야 한다.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

# agent/kosis/build_org_whitelist.py가 만드는 캐시 — table_params.json에 실제 등장하는
# orgId를 KOSIS API(getMeta type=ORG)로 직접 조회해서 만든, 카탈로그와 항상 동기화되는
# 기관명 목록이다(2026-08-13 도입). 아래 KOSIS_VERIFIED_ORGS(수동 목록)와 합쳐서 쓴다 —
# 수동 목록만으로는 새 표를 추가할 때 발행 기관을 깜빡 빠뜨리는 문제가 실측 확인됐다
# (과학기술정보통신부/기획예산처 누락 사례). 파일이 아직 없으면(스크립트 실행 전) 빈 채로
# 두고 수동 목록만 쓴다 — 배치 실행 자체가 막히면 안 되므로 조용히 넘어간다.
_ORG_WHITELIST_CACHE_PATH = Path(__file__).parent / "kosis_org_whitelist.json"


def _load_cached_org_names() -> set[str]:
    if not _ORG_WHITELIST_CACHE_PATH.exists():
        return set()
    try:
        data = json.loads(_ORG_WHITELIST_CACHE_PATH.read_text(encoding="utf-8"))
        return set(data.values())
    except (json.JSONDecodeError, OSError):
        return set()


def _normalize_whitespace(text: str) -> str:
    """공백 제거 비교용 정규화. agent_chat.py의 동명 함수와 동일한 이유 — "국가데이터처"가
    classifier의 reason 텍스트엔 "국가 데이터처"처럼 띄어써서 나오는 경우가 실제로 확인됨."""
    return re.sub(r"\s+", "", text)


def _org_appears_standalone(org: str, normalized_text: str) -> bool:
    """org(부처 약칭 등)가 normalized_text에 실제로 그 기관을 가리키며 등장하는지 확인한다.

    2026-08-18: "복지부"/"산업부"/"고용부"/"국토부"처럼 "부"로 끝나는 부처 약칭이
    "복지부문"/"산업부문"/"고용부문"/"국토부문"(예산 "부문"·"분야" 뜻, 해당 부처와 무관)의
    앞부분과 우연히 겹쳐서, 국내 예산 통계 기사가 엉뚱하게 그 부처 발표로 오인되는 위험이
    실측 확인됐다("달"이 "달러"의 일부로 오탐되던 문제와 같은 부류).
    처음엔 kiwi 형태소 분석으로 "org가 독립 토큰인지"를 확인하려 했으나, "중기부"/
    "과기정통부"처럼 kiwi 사전에 없어 진짜 부처를 가리킬 때도 "중기"+"부"/"과기"+"정통부"로
    쪼개지는 경우가 실측 확인돼서(kiwi 의존 시 오히려 정상 매칭까지 깨짐), 대신
    _is_false_friend_only와 같은 원리(알려진 오탐 패턴만 지우고 남는지 확인)를 "OO부문"
    형태에 일반화해서 적용한다 — org가 "부"로 끝날 때만 "org+문" 등장을 지운 뒤에도 org가
    남는지 본다. kiwi 사전 등재 여부와 무관하게 항상 정확하게 동작한다."""
    if not org.endswith("부"):
        return org in normalized_text
    remaining = normalized_text.replace(org + "문", "")
    return org in remaining

try:
    from agent.interfaces import Claim
except ImportError:
    from dataclasses import dataclass

    @dataclass
    class Claim:  # type: ignore[no-redef]
        sentence: str
        source_org: Optional[str] = None


# 우리 table_catalog.json에 실제로 등장하거나, 정부 부처/청 단위로 KOSIS 국가승인통계를
# 생산하는 게 상식적으로 확실한 기관들. orgId 기준으로 이미 검증된 것(통계청/국가데이터처=101,
# 한국은행=301, 관세청=134, 한국무역협회=360, 보건복지부=117 등)과 나머지 정부 부처를 포함.
#
# 2026-08-13: table_params.json에 실제로 등장하는 orgId 14개 전부를 KOSIS getMeta(type=ORG)
# API로 직접 조회해서 대조한 결과, "과학기술정보통신부"(orgId=127, 연구개발비 표)와
# "기획예산처"(orgId=184, 통합재정수지 표 — 2008년 기획재정부로 통합되기 전 옛 명칭인데
# KOSIS 등록엔 그 이름 그대로 남아있음)가 화이트리스트에 빠져있는 걸 발견해서 추가함.
#
# 2026-08-31: 무작위 50건 실측(benchmark/diagnose_source_exclusions.py)에서 정식명만
# 있고 약칭이 없어서 걸러진 사례 2건 발견 후 추가. "국민연금"(연기금 순매수 관련 기사에서
# "국민연금공단"이 아니라 "국민연금"으로만 언급되는 게 더 흔함 — _org_appears_standalone은
# 화이트리스트 단어가 기사 문구 "안에" 있는지만 보므로, 화이트리스트 쪽이 기사 표현보다
# 길면 매칭이 안 됨), "우정사업본부"(과학기술정보통신부 산하 기관인데 기사에서 상위
# 부처명이 "과학기술통신부"로 오타 나는 경우가 있어, 오타 여부와 무관하게 통과하도록
# 기관명 자체를 화이트리스트에 직접 추가).
_MANUAL_KOSIS_VERIFIED_ORGS = {
    "통계청", "국가데이터처", "한국은행", "한은", "국세청", "관세청", "농림축산식품부", "농식품부",
    "식품의약품안전처", "식약처", "기획재정부", "기재부", "기획예산처", "고용노동부", "고용부",
    "보건복지부", "복지부",
    "국토교통부", "국토부", "행정안전부", "행안부", "여성가족부", "여가부",
    "중소벤처기업부", "중기부", "교육부",
    "산업통상자원부", "산업통상부", "산업부", "문화체육관광부", "문체부", "환경부", "해양수산부", "해수부",
    "과학기술정보통신부", "과기정통부",
    "국민연금공단", "국민연금", "한국거래소", "한국부동산원", "한국무역협회", "산림청", "기상청",
    "우정사업본부",
}

# 수동 목록 + 카탈로그 기반 자동 캐시를 합친 최종 화이트리스트.
KOSIS_VERIFIED_ORGS = _MANUAL_KOSIS_VERIFIED_ORGS | _load_cached_org_names()

# 확실히 KOSIS로 검증 불가능한 출처 — 해외기관/정부, 민간기업, 신용평가사, 정당, 개인 등.
KNOWN_NOT_KOSIS = {
    "연방준비제도", "연방준비제도(연준)", "연준", "미국 노동부", "OECD", "IMF",
    "세계은행", "S&P", "무디스", "피치", "S&P, 무디스, 피치", "JP모건",
    "한국신용데이터(KCD)", "한국신용데이터", "KCD", "이동통신 3사", "국민의힘",
    "더불어민주당", "정부",
}


def classify_source(source_org: Optional[str]) -> str:
    """source_org 문자열 하나를 세 등급 중 하나로 분류한다.

    반환값: "kosis_verified" / "not_kosis" / "uncertain"
    (source_org가 None/빈 문자열이면 "uncertain" — 판단할 근거 자체가 없다는 뜻)
    """
    if not source_org:
        return "uncertain"

    # "(추정)" 같은 접미사나 개인 이름 뒤에 붙는 소속 기관명("김용현 한은 자금순환팀장"의
    # "한은"처럼) 매칭을 놓치지 않도록, 정확히 일치 우선 확인 후 부분 포함도 검사.
    # 공백 무시 비교("국가 데이터처" vs "국가데이터처") 적용 — 2026-08-05 실측 확인.
    stripped = source_org.replace("(추정)", "").strip()
    normalized = _normalize_whitespace(stripped)

    if stripped in KOSIS_VERIFIED_ORGS:
        return "kosis_verified"
    if stripped in KNOWN_NOT_KOSIS:
        return "not_kosis"

    for org in KOSIS_VERIFIED_ORGS:
        if _org_appears_standalone(_normalize_whitespace(org), normalized):
            return "kosis_verified"
    for org in KNOWN_NOT_KOSIS:
        if _normalize_whitespace(org) in normalized:
            return "not_kosis"

    return "uncertain"


# "발표/집계/공표/통계/조사"류 표현이 reason에 함께 있을 때만 infer_org_from_reason을
# 발동시키는 안전장치(2026-08-13). "밝히다"/"말하다"류 범용 인용 동사는 일부러 뺐다 —
# "OO 관계자는 ~라고 말했다"처럼 스포크스퍼슨 발언 어디에나 붙어서 실제 통계 발표와
# 구분이 안 된다(유튜버 수퍼챗 수입 기사에서 "국세청 관계자는 ~라고 말했다"의 국세청이
# 엉뚱하게 전체 claim의 출처로 오인된 사례, 2026-08-13 실측).
_STATS_ATTRIBUTION_RE = re.compile(r"(발표|집계|공표|통계|조사)")


def infer_org_from_reason(reason: Optional[str]) -> Optional[str]:
    """1단계(classifier)의 ClassificationResult.reason 텍스트에서 KOSIS 검증된 기관명을
    찾아낸다 — claim_extractor가 특정 claim의 source_org를 못 채웠을 때(같은 기사 내
    다른 claim에도 org가 하나도 없는 경우, backfill_source_org로도 못 살리는 케이스),
    이미 1단계에서 계산해둔 reason에 기관명이 언급돼 있으면 그걸 fallback으로 쓰기 위함
    (claim_extractor.py 자체는 건드리지 않는 비침습적 보완 — 2026-08-05).

    실제 검증(2026-08-05): source_org가 아예 없던 기사 22건 중 6건에서 이 방식으로
    기관명 복구 확인(행정안전부/통계청 x2/기획재정부 x2/농식품부).

    2026-08-13: reason에 통계 발표를 나타내는 키워드가 없으면 기관명이 있어도 폴백을
    발동하지 않는다(_STATS_ATTRIBUTION_RE 참고) — 일반 인용문에 기관명이 스쳐 지나가는
    오탐을 막기 위함."""
    if not reason or not _STATS_ATTRIBUTION_RE.search(reason):
        return None
    normalized = _normalize_whitespace(reason)
    for org in sorted(KOSIS_VERIFIED_ORGS, key=len, reverse=True):
        if _org_appears_standalone(_normalize_whitespace(org), normalized):
            return org
    return None


def resolve_claim_sources(claims: list[Claim], classifier_reason: Optional[str] = None) -> list[Claim]:
    """한 기사에서 나온 claim들의 source_org를 최대한 채운다 (backfill → reason fallback 순).

    1) 같은 기사 내 다른 claim에 채워진 org로 역채움(backfill_source_org)
    2) 그래도 비어있으면(같은 기사의 모든 claim이 원래 다 비어있던 경우), 1단계
       classifier의 reason 텍스트에서 기관명을 찾아 채움(infer_org_from_reason)
    """
    filled = backfill_source_org(claims)
    if classifier_reason:
        inferred = infer_org_from_reason(classifier_reason)
        if inferred:
            from dataclasses import replace
            filled = [c if c.source_org else replace(c, source_org=inferred) for c in filled]
    return filled


def _has_hallucinated_value(claim: Claim) -> bool:
    """claim.value가 채워져 있는데 원문 문장(sentence)에 숫자가 단 하나도 없으면 True
    (claim_extractor의 순수 환각으로 본다).

    2026-08-14 실측: "최상목... 승선원 확인을 통해 실종자 수색에 만전을 기해달라"고
    했다" 문장에서 claim_extractor(LLM)가 value=11.0, statistic_expression="승선원 수"를
    만들어냈는데, 원문 어디에도 숫자가 없었다 — 문장에 있는 숫자를 잘못 해석한 게 아니라
    아예 없는 숫자를 지어낸 것. 표기법 차이("7만" vs 70000처럼) 때문에 값이 정확히
    일치하는지 확인하기는 어렵지만, "숫자가 문장에 하나도 없는데 value가 채워졌다"는
    해석의 여지가 없는 신호라 안전하게 걸러낼 수 있다. (드물게 "칠만 명"처럼 순한글
    숫자 표현을 쓰는 문장은 이 필터에 걸릴 수 있음 — 실측 데이터에서는 관측되지 않았지만
    알려진 한계로 남겨둠.)"""
    value = getattr(claim, "value", None)
    if value is None:
        return False
    return not any(ch.isdigit() for ch in claim.sentence)


def _to_korean_scale(n: int) -> str:
    """정수를 기사에서 흔히 쓰는 조/억/만 복합 표기로 변환한다. 예: 415600000000 -> "4156억",
    183000 -> "18만3000". "천"은 일부러 안 넣는다 — "3000"이 "3천"인지 그냥 "3000"인지
    애매해서 넣으면 오탐(허위 불일치)만 늘어난다."""
    if n == 0:
        return "0"
    parts: list[str] = []
    remainder = n
    for unit_value, unit_name in ((10**12, "조"), (10**8, "억"), (10**4, "만")):
        if remainder >= unit_value:
            count = remainder // unit_value
            parts.append(f"{count}{unit_name}")
            remainder %= unit_value
    if remainder > 0 or not parts:
        parts.append(str(remainder if parts else n))
    return "".join(parts)


def _korean_number_variants(value: float) -> set[str]:
    """value가 원문 문장에 어떤 형태로 적혀 있을 수 있는지 후보 문자열들을 만든다.

    2026-08-13 골든셋 QA 스크립트(verify_all_fields.py)에서 int(round(60.8))처럼 소수점을
    반올림해서 "61"로 만들어버리는 바람에, 정상 값(60.8)을 오탐으로 잘못 걸러낸 버그가
    있었다 — 그 교훈으로 여기서는 소수점을 절대 반올림하지 않는다."""
    is_int = float(value).is_integer()
    variants: set[str] = set()
    if is_int:
        int_value = int(value)
        variants.add(str(int_value))
        variants.add(f"{int_value:,}")
        variants.add(_to_korean_scale(int_value))
    else:
        variants.add(str(value))
    return variants


def _value_mismatches_sentence(value: Optional[float], sentence: str) -> bool:
    """value가 채워져 있는데, 그 값에 해당하는 숫자가 문장 어디에도 없으면 True.

    _has_hallucinated_value("문장에 숫자가 아예 없음")보다 넓은 개념 — "문장에 숫자는
    있지만 이 value랑은 다른 숫자"인 경우까지 잡는다. 실측(2026-08-20,
    verifications_export.json) 사례: "울릉군은 2014년 상반기부터 10년간... 왕좌를
    차지했다"라는 문장에 value=83.5가 붙어있었는데, 이 83.5는 같은 기사의 다른 문장
    (고용률 83.5%)에서 온 값이 잘못 옮겨붙은 것이었다 — 문장 자체엔 2014/10/2023만
    있고 83.5는 없었음.

    완벽한 탐지는 아니다(한글 숫자 표기 경우의 수가 다양해서 놓치는 패턴이 있을 수 있음) —
    그래서 filter_verifiable_claims에서 정상 claim까지 잘못 거를 위험을 감안해 보수적으로
    쓴다(의심스러우면 제외하되, 애매한 표기 패턴 자체를 넓게 인정)."""
    if value is None:
        return False
    variants = _korean_number_variants(value)
    return not any(v in sentence for v in variants)


_VALUE_REQUIRED_CLAIM_TYPES = ("규모", "증감률")


def _missing_value_for_magnitude_claim(claim: Claim) -> bool:
    """claim_type이 "규모"/"증감률"인데 value가 None이면 True.

    2026-08-20 실측(오늘 배치 47건 중 5건, verifications.db): "KDDX 개념설계는 2012년...
    기본설계는 2020년... 수주했다"처럼 문장에 연도 등 숫자는 있어도 정작 비교할 통계값
    자체는 없는 경우, "경기 침체로 음식점·유원지 손님 '뚝'..."처럼 기사 제목이 그대로
    claim으로 뽑힌 경우, "...지정돼 혜택을 받을 것으로 보인다"처럼 미래 전망 문장인데
    claim_type이 '전망'이 아니라 '규모'로 잘못 분류된 경우가 모두 이 패턴이었다.
    claim_type='규모'라는 라벨 자체가 "이 문장은 수치 규모를 주장한다"는 뜻인데 value가
    없으면 그 라벨과 내용이 모순이라, 3단계(표매칭)로 보낼 근거가 없다고 본다.

    _has_hallucinated_value/_value_mismatches_sentence는 반대로 "value가 있는데 문장과
    안 맞는" 경우만 잡아서, "value 자체가 없는" 이 경우는 못 잡았다(둘 다 value is None이면
    바로 False를 반환하고 넘어감) — 이 함수가 그 빈틈을 메운다."""
    if claim.claim_type not in _VALUE_REQUIRED_CLAIM_TYPES:
        return False
    return getattr(claim, "value", None) is None


# 통계 발표가 아니라 단속/제재/평가 등 1회성 행정조치 결과를 나타내는 표현. 발행 기관이
# KOSIS_VERIFIED_ORGS에 있어도(그 기관이 진짜 KOSIS 통계도 내는 곳이어도) 이 claim 자체가
# 그 통계가 아니라 이번 건 개별 발표/조치 결과일 가능성이 높다는 신호다 — 기관 단위
# 화이트리스트만으로는 이 구분이 안 된다.
#
# 2026-08-20 실측 확인된 2개 사례: (1) 기획재정부(진짜 KOSIS 통계도 냄)의 "공공기관 경영
# 평가에서 13곳이 성과급을 받지 못하는 낙제점을 받았다" — 평가결과지 통계가 아님. (2)
# 금융감독원(진짜 KOSIS 통계도 냄)의 "금융투자 사이트 1428건을 적발해... 차단을 의뢰했다"/
# "60건에 대해 경찰청에 수사를 의뢰했다" — 단속실적이지 통계가 아님. 둘 다 표매칭 단계까지
# 넘어가서 무관한 표(예산현황/국제수지)에 억지로 매칭되거나 "애매"로 낭비됐다.
#
# 한계: 이 방식은 블랙리스트라 완전할 수 없다(행정조치를 나타내는 한국어 표현은 사실상
# 무한함) — 알려진 패턴만 값싸게 걸러내는 보조 장치일 뿐, 진짜 안전망은 하류의 RRF 신뢰
# 게이트(agent/mapping/reranker.py의 is_rrf_trusted, 확신도 낮으면 애매 처리)다.
_ENFORCEMENT_OR_EVALUATION_KEYWORDS = (
    "적발", "단속", "낙제점", "성과급", "경영평가", "감사결과", "행정처분",
    "과태료", "과징금", "제재", "의뢰",
)


def _looks_like_enforcement_or_evaluation(sentence: str) -> bool:
    """통계 발표가 아니라 단속/제재/평가 등 1회성 행정조치 결과 문장인지 확인한다."""
    normalized = _normalize_whitespace(sentence)
    return any(kw in normalized for kw in _ENFORCEMENT_OR_EVALUATION_KEYWORDS)


def filter_verifiable_claims(claims: list[Claim]) -> list[Claim]:
    """claim 리스트에서 source_org가 "kosis_verified"이고, value 환각/오귀속이 의심되지
    않고, claim_type="규모"/"증감률"인데 value가 비어있지 않고, 단속/제재/평가 등 통계가
    아닌 행정조치 결과로 보이지 않는 것만 남긴다.

    "uncertain"(source_org 없음/휴리스틱에 없는 기관)과 "not_kosis"는 전부 제외한다 —
    분류기(1단계)와 같은 원칙: 확실하지 않으면 3단계(표매칭)로 넘기지 않는다. source_org가
    비어있는 claim을 같은 기사의 다른 claim 출처로 채워 넣는(article-level backfill) 로직은
    이 함수의 책임이 아니다 — Claim에 기사 단위 식별자가 없어서, 호출하는 쪽(기사 하나를
    처리하는 오케스트레이션 코드)이 같은 기사 내 claim들을 모아 backfill한 뒤 이 함수에
    넘겨야 한다."""
    return [
        c
        for c in claims
        if classify_source(c.source_org) == "kosis_verified"
        and not _has_hallucinated_value(c)
        and not _value_mismatches_sentence(getattr(c, "value", None), c.sentence)
        and not _value_mismatches_sentence(getattr(c, "comparison_value", None), c.sentence)
        and not _missing_value_for_magnitude_claim(c)
        and not _looks_like_enforcement_or_evaluation(c.sentence)
    ]


def backfill_source_org(claims: list[Claim]) -> list[Claim]:
    """같은 기사에서 나온 claim 리스트(claims)를 받아, source_org가 비어있는 claim에
    그 기사의 지배적인(가장 많이 등장한) source_org를 채워 넣는다.

    실제 데이터에서 확인된 패턴(2026-08-05): 한 기사 안 첫 문장엔 "한국은행에 따르면"처럼
    출처가 명시되고, 뒤따르는 문장들은 "전월 대비...올랐다"처럼 출처를 반복 안 함 —
    claim_extractor가 문장 단위로 source_org를 채우다 보니 뒤쪽 claim들이 비어버린다.
    같은 기사 안에서는 보통 주 출처가 하나이므로, 다수결로 역채움한다.

    ⚠️ 2026-08-10 실측 버그: 순수 빈도 다수결만 쓰면, 한 기사에 "정부 관계자 발언"과
    "공식 통계 발표"처럼 서로 다른 출처가 섞여 있을 때(예: "최상목 부총리" 인용문 2건 +
    "통계청 3월 소비자물가 동향" 인용 1건) 더 자주 언급된 개인 발언(최상목)이 다수결로
    이겨버려서, 실제로는 통계청이 출처인 뒤쪽 claim들(오징어채 40.3% 등 품목별 물가
    상승률)까지 "최상목"으로 잘못 채워지는 문제를 발견했다. classify_source("최상목")도
    "통계청"이 채워졌어야 할 claim도 결과적으로 최소 uncertain 처리라 필터링 자체는
    맞게 되지만, 원래는 kosis_verified로 통과해야 할 claim이 잘못 걸러진 것.
    그래서 단순 빈도가 아니라, "검증된 기관(classify_source==kosis_verified)이 하나라도
    있으면 그걸 우선"하도록 바꾼다 — 인용된 개인 발언보다 실제 통계 발표기관이 수치의
    진짜 출처일 가능성이 높다는 판단."""
    orgs_in_article = [c.source_org for c in claims if c.source_org]
    if not orgs_in_article:
        return claims

    from collections import Counter

    counts = Counter(orgs_in_article)
    verified_orgs = [org for org in counts if classify_source(org) == "kosis_verified"]
    if verified_orgs:
        dominant = max(verified_orgs, key=lambda o: counts[o])
    else:
        dominant = counts.most_common(1)[0][0]

    filled: list[Claim] = []
    for c in claims:
        if c.source_org:
            filled.append(c)
        else:
            from dataclasses import replace
            filled.append(replace(c, source_org=dominant))
    return filled


if __name__ == "__main__":
    #   python -m agent.preprocessing.source_filter
    samples = [
        Claim(sentence="통계청 발표...", claim_type="규모", source_org="통계청"),
        Claim(sentence="OECD 전망...", claim_type="규모", source_org="OECD"),
        Claim(sentence="출처 불명 claim...", claim_type="규모", source_org=None),
        Claim(sentence="김용현 한은 자금순환팀장 발언...", claim_type="규모", source_org="김용현 한은 자금순환팀장"),
    ]
    for s in samples:
        print(s.source_org, "->", classify_source(s.source_org))

    filled = backfill_source_org(samples)
    print("\nbackfill 후:")
    for s in filled:
        print(s.source_org)

    # 2026-08-10 실측 회귀 테스트: "최상목 부총리" 기사 재현 —
    # 개인 발언(최상목, 2건)이 공식 통계 발표(통계청, 1건)보다 더 자주 언급돼도
    # 통계청이 backfill 우선순위를 가져야 한다.
    mixed_source_samples = [
        Claim(sentence="정부가 요금 동결", claim_type="규모", source_org="최상목"),
        Claim(sentence="최상목 인용문", claim_type="규모", source_org="최상목"),
        Claim(sentence="가공식품 외식 물가 인상", claim_type="규모", source_org=None),
        Claim(sentence="오징어채 40.3% 초콜릿 15.5%", claim_type="증감률", source_org=None),
        Claim(sentence="통계청 2.1% 상승", claim_type="증감률", source_org="통계청"),
    ]
    mixed_filled = backfill_source_org(mixed_source_samples)
    assert mixed_filled[2].source_org == "통계청", (
        f"기대: 통계청, 실제: {mixed_filled[2].source_org} — 다수결이 개인 발언을 "
        f"잘못 우선시키는 회귀가 재발했습니다."
    )
    assert mixed_filled[3].source_org == "통계청"
    print("\n[회귀 테스트 통과] 혼합 출처(개인 발언 다수 + 공식 통계 소수)에서 "
          "공식 통계가 backfill 우선순위를 가져감 확인")

    # 2026-08-13 회귀 테스트: infer_org_from_reason 안전장치.
    # (a) "~가 공식적으로 발표하였다"류 실제 통계 발표 reason은 여전히 복구돼야 한다
    #     (SKT 유심보호서비스 기사, 과기정통부 — claim_extractor가 source_org를 전부
    #     None으로 뽑았던 실제 케이스).
    good_reason = (
        "과학기술정보통신부가 SK텔레콤의 유심 정보 해킹 사건과 관련하여 신규 가입자 모집 "
        "중단 및 유심 물량 공급 안정화를 위한 조치를 취했다는 내용을 공식적으로 발표하였다."
    )
    assert infer_org_from_reason(good_reason) == "과학기술정보통신부", (
        f"기대: 과학기술정보통신부, 실제: {infer_org_from_reason(good_reason)!r} — "
        f"통계 발표 reason 복구가 깨졌습니다."
    )
    # (b) "OO 관계자는 ~라고 말했다"류 일반 인용문은 기관명이 스쳐도 폴백이 발동하면 안
    #     된다(유튜버 수퍼챗 수입 기사, 국세청 오탐 재현 — 2026-08-13 실측).
    bad_reason = "국세청 관계자는 유튜버들의 수퍼챗 등 후원금이 과세 대상이라고 말했다."
    assert infer_org_from_reason(bad_reason) is None, (
        f"기대: None, 실제: {infer_org_from_reason(bad_reason)!r} — 일반 인용문 오탐이 "
        f"재발했습니다."
    )
    print(
        "[회귀 테스트 통과] infer_org_from_reason 안전장치 — 통계 발표 reason은 복구되고, "
        "일반 인용문 오탐은 계속 차단됨 확인"
    )

    # 2026-08-14 회귀 테스트: _has_hallucinated_value — 원문에 숫자가 없는데 value가
    # 채워진 claim(claim_extractor 환각)을 걸러내는지, 정상 claim은 안 건드리는지 확인.
    # 실제 오탐 재현 사례("부안 어선 화재" 기사, value=11.0인데 원문에 숫자 없음).
    hallucinated_claim = Claim(
        sentence=(
            "최상목 대통령 권한대행 부총리 겸 기획재정부 장관이 전북 부안군 해상에서 "
            "발생한 어선 화재 사고에 대해 \"최우선적으로 인명을 구조하고, 정확한 승선원 "
            "확인을 통해 실종자 수색에 만전을 기해달라\"고 했다."
        ),
        claim_type="규모",
        source_org="기획재정부",
        value=11.0,
    )
    real_claim = Claim(
        sentence="지난해 국내 1인 가구가 800만 가구를 넘어서 역대 최대 규모를 기록했다.",
        claim_type="규모",
        source_org="통계청",
        value=8000000.0,
    )
    filtered = filter_verifiable_claims([hallucinated_claim, real_claim])
    assert hallucinated_claim not in filtered, "❌ 숫자 환각 claim이 안 걸러짐"
    assert real_claim in filtered, "❌ 정상 claim이 잘못 걸러짐"
    print(
        "[수정 확인] 숫자 환각 claim(부안 어선 화재, value=11.0)은 걸러지고, "
        "정상 claim(1인가구 800만)은 그대로 통과됨"
    )

    # 2026-08-20 회귀 테스트: _value_mismatches_sentence — 문장에 숫자는 있지만 value랑
    # 다른 숫자인 "오귀속" claim을 걸러내는지 확인. 실제 재현 사례(울릉군 고용률 기사):
    # value=83.5가 "10년간 왕좌를 차지했다"는 문장에 잘못 붙어있었음(진짜 83.5%는 다른
    # 문장에서 나온 값).
    misattributed_claim = Claim(
        sentence=(
            "울릉군은 2014년 상반기부터 10년간 단 한 차례(2023년 상반기) 경북 청송군에 "
            "밀렸던 것을 제외하고 꾸준히 왕좌를 차지했다."
        ),
        claim_type="규모",
        source_org="통계청",
        value=83.5,
    )
    # 정상 claim들 — 한글 만/억 복합 표기, 천단위 콤마, 소수점 값이 전부 안 걸리는지 확인
    # (소수점을 반올림해서 오탐 냈던 예전 QA 스크립트 버그 재발 방지 포함).
    korean_scale_claim = Claim(
        sentence="작년 12월 말 우리나라 외환 보유액은 4156억달러로 집계됐다.",
        claim_type="규모", source_org="한국은행", value=415600000000.0,
    )
    decimal_claim = Claim(
        sentence="한국경영자총협회의 올해 초 조사에 따르면, 신규 채용 계획이 있다고 응답한 기업은 60.8%였다.",
        claim_type="규모", source_org="통계청", value=60.8,
    )
    mismatch_filtered = filter_verifiable_claims(
        [misattributed_claim, korean_scale_claim, decimal_claim]
    )
    assert misattributed_claim not in mismatch_filtered, "❌ 값 오귀속 claim이 안 걸러짐"
    assert korean_scale_claim in mismatch_filtered, "❌ 한글 억 단위 표기 정상 claim이 잘못 걸러짐"
    assert decimal_claim in mismatch_filtered, "❌ 소수점 정상 claim이 잘못 걸러짐(반올림 버그 재발?)"
    print(
        "[수정 확인] 값 오귀속 claim(울릉군, value=83.5가 다른 문장 것)은 걸러지고, "
        "한글 억 단위·소수점 정상 claim은 그대로 통과됨"
    )

    # 2026-08-20 회귀 테스트: _missing_value_for_magnitude_claim — claim_type='규모'인데
    # value가 None인 claim을 걸러내는지 확인. 실제 재현 사례(오늘 배치 47건 중 5건):
    # 연도만 있고 진짜 통계값이 없는 문장, 기사 제목이 그대로 뽑힌 경우, 미래 전망 문장이
    # '전망' 대신 '규모'로 잘못 분류된 경우가 전부 이 패턴이었음(agent/kosis/verify_api_client.py
    # 아님, verifications.db 2026-08-20 실행분 직접 확인).
    no_value_claim = Claim(
        sentence="KDDX 개념설계는 2012년 대우조선해양(현 한화오션), 기본설계는 2020년 HD현대중공업이 수주했다.",
        claim_type="규모", source_org="산업통상자원부", value=None,
    )
    title_only_claim = Claim(
        sentence="경기 침체로 음식점·유원지 손님 '뚝'... 매출 급감",
        claim_type="규모", source_org="통계청", value=None,
    )
    real_value_claim = Claim(
        sentence="지난해 국내 1인 가구가 800만 가구를 넘어서 역대 최대 규모를 기록했다.",
        claim_type="규모", source_org="통계청", value=8000000.0,
    )
    forecast_claim = Claim(
        sentence="내년에는 물가가 더 오를 것으로 전망된다.",
        claim_type="전망", source_org="통계청", value=None,
    )
    missing_value_filtered = filter_verifiable_claims(
        [no_value_claim, title_only_claim, real_value_claim, forecast_claim]
    )
    assert no_value_claim not in missing_value_filtered, "❌ 값 없는 '규모' claim(KDDX)이 안 걸러짐"
    assert title_only_claim not in missing_value_filtered, "❌ 값 없는 '규모' claim(제목)이 안 걸러짐"
    assert real_value_claim in missing_value_filtered, "❌ 값 있는 정상 '규모' claim이 잘못 걸러짐"
    assert forecast_claim in missing_value_filtered, "❌ claim_type='전망'은 이 필터 대상 밖이라 안 걸려야 하는데 잘못 걸러짐"
    print(
        "[수정 확인] claim_type='규모'인데 value=None인 claim(KDDX, 기사제목)은 걸러지고, "
        "값 있는 정상 claim은 그대로 통과됨 (전망 claim_type은 이 필터 범위 밖이라 별개로 정상 제외)"
    )

    print("\n최종 통과:", [c.source_org for c in filter_verifiable_claims(backfill_source_org(samples))])