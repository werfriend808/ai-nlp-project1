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

import re
from typing import Optional


def _normalize_whitespace(text: str) -> str:
    """공백 제거 비교용 정규화. agent_chat.py의 동명 함수와 동일한 이유 — "국가데이터처"가
    classifier의 reason 텍스트엔 "국가 데이터처"처럼 띄어써서 나오는 경우가 실제로 확인됨."""
    return re.sub(r"\s+", "", text)

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
KOSIS_VERIFIED_ORGS = {
    "통계청", "국가데이터처", "한국은행", "한은", "국세청", "관세청", "농림축산식품부", "농식품부",
    "식품의약품안전처", "식약처", "기획재정부", "기재부", "고용노동부", "고용부", "보건복지부", "복지부",
    "국토교통부", "국토부", "행정안전부", "행안부", "여성가족부", "여가부",
    "중소벤처기업부", "중기부", "교육부",
    "산업통상자원부", "산업통상부", "산업부", "문화체육관광부", "문체부", "환경부", "해양수산부", "해수부",
    "국민연금공단", "한국거래소", "한국부동산원", "한국무역협회", "산림청", "기상청",
}

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
        if _normalize_whitespace(org) in normalized:
            return "kosis_verified"
    for org in KNOWN_NOT_KOSIS:
        if _normalize_whitespace(org) in normalized:
            return "not_kosis"

    return "uncertain"


def infer_org_from_reason(reason: Optional[str]) -> Optional[str]:
    """1단계(classifier)의 ClassificationResult.reason 텍스트에서 KOSIS 검증된 기관명을
    찾아낸다 — claim_extractor가 특정 claim의 source_org를 못 채웠을 때(같은 기사 내
    다른 claim에도 org가 하나도 없는 경우, backfill_source_org로도 못 살리는 케이스),
    이미 1단계에서 계산해둔 reason에 기관명이 언급돼 있으면 그걸 fallback으로 쓰기 위함
    (claim_extractor.py 자체는 건드리지 않는 비침습적 보완 — 2026-08-05).

    실제 검증(2026-08-05): source_org가 아예 없던 기사 22건 중 6건에서 이 방식으로
    기관명 복구 확인(행정안전부/통계청 x2/기획재정부 x2/농식품부)."""
    if not reason:
        return None
    normalized = _normalize_whitespace(reason)
    for org in sorted(KOSIS_VERIFIED_ORGS, key=len, reverse=True):
        if _normalize_whitespace(org) in normalized:
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


def filter_verifiable_claims(claims: list[Claim]) -> list[Claim]:
    """claim 리스트에서 source_org가 "kosis_verified"인 것만 남긴다.

    "uncertain"(source_org 없음/휴리스틱에 없는 기관)과 "not_kosis"는 전부 제외한다 —
    분류기(1단계)와 같은 원칙: 확실하지 않으면 3단계(표매칭)로 넘기지 않는다. source_org가
    비어있는 claim을 같은 기사의 다른 claim 출처로 채워 넣는(article-level backfill) 로직은
    이 함수의 책임이 아니다 — Claim에 기사 단위 식별자가 없어서, 호출하는 쪽(기사 하나를
    처리하는 오케스트레이션 코드)이 같은 기사 내 claim들을 모아 backfill한 뒤 이 함수에
    넘겨야 한다."""
    return [c for c in claims if classify_source(c.source_org) == "kosis_verified"]


def backfill_source_org(claims: list[Claim]) -> list[Claim]:
    """같은 기사에서 나온 claim 리스트(claims)를 받아, source_org가 비어있는 claim에
    그 기사의 지배적인(가장 많이 등장한) source_org를 채워 넣는다.

    실제 데이터에서 확인된 패턴(2026-08-05): 한 기사 안 첫 문장엔 "한국은행에 따르면"처럼
    출처가 명시되고, 뒤따르는 문장들은 "전월 대비...올랐다"처럼 출처를 반복 안 함 —
    claim_extractor가 문장 단위로 source_org를 채우다 보니 뒤쪽 claim들이 비어버린다.
    같은 기사 안에서는 보통 주 출처가 하나이므로, 다수결로 역채움한다."""
    orgs_in_article = [c.source_org for c in claims if c.source_org]
    if not orgs_in_article:
        return claims

    from collections import Counter
    dominant = Counter(orgs_in_article).most_common(1)[0][0]

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

    print("\n최종 통과:", [c.source_org for c in filter_verifiable_claims(backfill_source_org(samples))])