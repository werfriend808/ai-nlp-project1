"""
agent/preprocessing/claim_candidate_scanner.py — 2단계 보조: 규칙 기반 숫자 문장 후보 스캐너

배경: claim_extractor(LLM 단독 추출)가 원문에 있는 숫자 포함 문장을 놓치는 사례가 실측
확인됨(2026-08-12, 실데이터 배치 33개 기사 분석). "기사가 길어서 놓친다"는 기존 가설과 달리,
짧은 기사에서도 한 문단에 비슷한 숫자가 여러 개 몰려있으면 1~2개만 뽑고 멈추는 경향이 확인됨
(예: "코스피 3000..." 기사는 1447자로 청킹 기준(3000자)에 안 걸리는데도 종목별 시총 수치
15개 이상 중 대부분을 놓침).

이 모듈은 LLM을 대체하지 않는다 — 원문을 정규식으로 스캔해서 "검증 가치 있을 수 있는 숫자
문장" 후보를 뽑아, claim_extractor 결과와 대조했을 때 놓친 게 있는지 감지하는 안전망
(하이브리드 설계의 1단계, 규칙 기반 recall 보강용)이다. 정밀도가 낮아서(날짜·제품 스펙 등
숫자는 있지만 통계가 아닌 문장은 애초에 아래 패턴에 안 걸리게 설계했지만, 개별 기업/상품
가격처럼 국가 통계가 아닌 것도 일부 섞인다) 이 스캐너의 후보를 그대로 claim으로 확정하면 안
되고, find_missed_candidates()로 claim_extractor 결과와 대조한 뒤 사람 검토 또는 LLM
재확인을 거쳐야 한다.
"""

from __future__ import annotations

import re

# 검증 가치 있는 숫자로 볼 단위들. "일/시/분/년"처럼 날짜·시각에 흔히 쓰이는 단위는 일부러
# 뺐다 — 넣으면 "13일", "오전 11시 22분"처럼 순수 날짜/시각 표기까지 전부 후보로 잡혀서
# 오탐이 급증한다(실측: 33개 기사 원문에 이런 날짜 표기가 문장마다 있음). 법령상 기준값
# ("30일 전", "1년 이내")도 "일/년" 단독이라 이 화이트리스트로는 애초에 안 걸린다 — 이는
# claim_extractor 프롬프트의 기존 제외 규칙과 방향이 같다.
_CURRENCY_UNITS = r"원|달러|엔|유로|위안"
_COUNT_UNITS = r"명|건|개사|개|톤|t|kg|㎏|CGT|척"
_PERCENT_UNITS = r"%p|%|퍼센트포인트|퍼센트"

# ⚠️ "개"는 실데이터 검증(2026-08-12, 33개 기사)에서 제품 스펙 문장("2개와 3개의 좌석을
# 배치한 차량")까지 후보로 잡는 오탐 원인으로 확인됐다. 그런데 claim_extractor_prompt.txt의
# 기존 few-shot 예시("전체 40개 중 30개가 정상화돼 복구율은 75.0%")처럼 "개"가 진짜 통계
# 단위인 정상 케이스도 있어서, "개"를 빼거나 문맥 조건을 더 걸면 keyword_search.py에서 이미
# 한 번 겪은 것과 같은 함정(임계값을 조이다가 정상 매칭까지 같이 걸러지는 문제)에 빠질 수
# 있다. 그래서 일부러 안 좁힌다 — 이 스캐너는 recall 우선 안전망이라 정밀도 손실은
# find_missed_candidates() 이후의 검토 단계(사람 또는 LLM 재확인)에서 흡수하는 게 맞는
# 설계다.

_NUMBER = r"[0-9][0-9,]*(?:\.[0-9]+)?"

# "3만6195달러"/"4156억달러"처럼 조/억/만/천 단위가 숫자 사이에 끼는 한글 복합 표기를
# 하나의 수치로 인식하기 위한 패턴(judge.py의 _find_compound_numbers와 같은 문제의식 —
# 단, 여기서는 값을 계산할 필요 없이 "숫자+단위 조합이 존재하는지"만 보면 되므로 훨씬 단순함).
_KOREAN_SCALE = r"조|억|만|천"
_COMPOUND_NUMBER = rf"{_NUMBER}(?:{_KOREAN_SCALE})?(?:{_NUMBER}(?:{_KOREAN_SCALE})?)*"

# 2026-08-17 추가: 배수·분수 표현 — "10년새 2배", "무려 두 배 가까이 올랐다"(judge_prompt.txt
# 실제 예시), "3분의 1로 줄었다", "절반 수준" 등. 숫자 뒤에 곧바로 "배"가 오는 경우(아라비아
# 숫자든 한글 숫자든)와, "N분의 M"/"절반" 분수 표현 둘 다 잡는다. 한글 숫자는 "한 배"(=1배,
# 사실상 안 씀)부터 "열 배"까지만 등록 — 뉴스에서 실제로 관측된 범위(대부분 두세 배 이내)를
# 넉넉히 덮으면서, 무한정 아무 한글 단어나 "배"에 결합하는 오탐을 막기 위함.
_MULTIPLIER_KOREAN_NUMS = r"한|두|세|네|다섯|여섯|일곱|여덟|아홉|열"
_MULTIPLIER_RE = rf"(?:{_NUMBER}|{_MULTIPLIER_KOREAN_NUMS})\s*배"
_FRACTION_RE = r"절반|[0-9]+\s*분의\s*[0-9]+"

# 2026-08-17 추가: 단위 없는 순수 지수/포인트 값 — "코스피지수가 2497.09로 마감했다",
# "소비자물가 지수는 116.38로 나타났다"처럼 원/달러/%/명 같은 단위가 안 붙고 소수점 숫자만
# 있는 지수 표현(오늘 하루 종일 다룬 "2020=100 기준" 지수류가 전형적으로 이 모양). 아무
# 소수점 숫자나 다 잡으면 오탐이 심해지므로, "N.NN을 기록/마감/집계/보였다" 같은 보고
# 동사가 근처(6자 이내, 조사 허용)에 붙어 있을 때만, 또는 "포인트"/"P"가 바로 붙어있을 때만
# 후보로 인정한다.
_REPORT_VERBS = r"기록했|기록되었|마감했|나타났|보였|집계됐|집계되었"
_INDEX_VALUE_RE = rf"[0-9]+\.[0-9]+\s*(?:포인트|P\b)|[0-9]+\.[0-9]+(?=[^0-9]{{0,6}}(?:{_REPORT_VERBS}))"

# 정규식은 아래 6가지 유형을 후보로 잡는다:
#   1) 숫자 + 화폐/수량/퍼센트 단위 (예: "3만6195달러", "52%", "883만명")
#   2) "N년째/개월째/주째" 또는 "N년/개월 연속" — 추세 지속 기간 표현(2026-08-12 발견 gap)
#   3) 숫자 + "위" — 순위 (예: "5위", "전국 4위")
#   4) "역대 최고/최저"·"최대치"·"최저치" — 극값 표현(숫자가 문장 다른 곳에 있는 경우가
#      많아 국소 패턴만으로 못 잡는 case 보강용)
#   5) 배수·분수 표현 (예: "2배", "두 배", "절반", "3분의 1") — 2026-08-17 발견 gap
#   6) 단위 없는 지수/포인트 값 (예: "2497.09로 마감했다") — 2026-08-17 발견 gap
_INCLUDE_PATTERN = re.compile(
    rf"{_COMPOUND_NUMBER}\s*(?:{_CURRENCY_UNITS}|{_COUNT_UNITS})"
    rf"|{_NUMBER}\s*(?:{_PERCENT_UNITS})"
    rf"|[0-9]+\s*(?:년째|개월째|주째)"
    rf"|[0-9]+\s*(?:년|개월)\s*연속"
    rf"|[0-9]+\s*위\b"
    rf"|역대\s*최(?:고|대|저)치?"
    rf"|{_MULTIPLIER_RE}"
    rf"|{_FRACTION_RE}"
    rf"|{_INDEX_VALUE_RE}"
)

# 문장 끝 경계 — claim_extractor.py의 청크 분할과 동일한 "다./요."-종결 기준을 재사용해서
# 두 모듈이 문장을 다르게 나누지 않게 한다.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[다요]\.)\s+")


def split_sentences(article_text: str) -> list[str]:
    """기사 본문을 문장 단위로 쪼갠다. 정확한 언어학적 분리기가 아니라, "다./요." 종결
    기준의 근사치 — claim_candidate와 claim_extractor 출력을 나중에 부분 문자열로 대조할
    때 쓸 정도의 정밀도면 충분하다."""
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(article_text) if s.strip()]


def scan_numeric_candidates(article_text: str) -> list[str]:
    """기사 본문에서 검증 가치 있을 수 있는 숫자 포함 문장을 전부 후보로 뽑는다.

    LLM 호출 없이 정규식만으로 동작하며, claim_extractor보다 recall은 높지만 precision은
    낮다(개별 기업 주가/제품 가격처럼 국가 통계가 아닌 숫자도 일부 섞임) — 최종 판단은
    이 함수의 책임이 아니다."""
    return [s for s in split_sentences(article_text) if _INCLUDE_PATTERN.search(s)]


def _count_numeric_expressions(sentence: str) -> int:
    """문장 안에 있는 개별 수치 표현 개수를 센다(_INCLUDE_PATTERN 매치 수) — 이 문장에서
    claim이 최소 몇 개 나와야 하는지 추정하는 용도.

    완벽한 값은 아니다 — "58.2%에서 90.4%로 상승" 같은 비교(comparison) claim은 숫자가
    2개(58.2%, 90.4%)지만 실제로는 claim 1개가 맞다. 그래도 "부분 추출"을 잡아내는
    보수적인 하한값으로는 충분하고, 이 스캐너 자체가 recall 우선·precision은 다음
    단계(사람/LLM 재확인)에서 흡수하는 설계라 이런 과탐지는 허용 범위로 본다."""
    return len(_INCLUDE_PATTERN.findall(sentence))


def find_missed_candidates(article_text: str, extracted_sentences: list[str]) -> list[str]:
    """스캐너 후보 중 claim_extractor가 "충분히" 뽑지 않은 것만 골라낸다.

    claim_extractor의 sentence는 원문 그대로이거나(대부분) 지시어로 앞 문장과 이어붙인
    경우(claim_extractor_prompt.txt 규칙)라서, "후보 문장이 추출된 문장에 부분 문자열로
    포함되는지"로 대조한다 — 완전 일치를 요구하면 이어붙이기 케이스를 다 놓친다.

    2026-08-18 수정: 예전엔 "후보 문장이 extracted_sentences 어디에라도 한 번 포함되면
    끝"이었는데, 이러면 한 문장에서 claim이 여러 개 나와야 하는 경우(예: "건설업 취업자도
    9만7000명 줄어, 작년 5월 이후 1년 2개월째 감소세다" — 증감량 claim + 지속기간 claim
    2개가 나와야 함) claim_extractor가 1개만 뽑고 멈춰도 "이미 커버됐다"고 오판해서 두
    번째 claim이 영원히 복구되지 않는 문제가 있었다(골든셋 라벨링 중 5-06a/5-06b 같은
    실제 케이스로 확인). 이제 "몇 번 커버됐는지"(covered_count)를 "이 문장에 수치 표현이
    몇 개 있는지"(_count_numeric_expressions)와 비교해서, 커버 횟수가 모자라면 여전히
    missed로 본다.
    """
    missed: list[str] = []
    for candidate in scan_numeric_candidates(article_text):
        covered_count = sum(1 for extracted in extracted_sentences if candidate in extracted)
        expected_count = _count_numeric_expressions(candidate)
        if covered_count >= expected_count:
            continue
        missed.append(candidate)
    return missed


if __name__ == "__main__":
    #   python -m agent.preprocessing.claim_candidate_scanner
    # 2026-08-12 실측 회귀 테스트: 오늘 실제로 발견한 케이스들을 그대로 재현한다.

    # 케이스 1 — "N년째 증가" 트렌드 표현은 후보로 잡혀야 함(claim_extractor가 놓쳤던 문장)
    assert scan_numeric_candidates("우리나라 1인당 국민소득은 2022년부터 3년째 증가 추세이다.")
    print("[케이스1] 'N년째 증가' 추세 표현 후보 인식 확인")

    # 케이스 2 — 순수 날짜/시각 표기는 후보로 잡히면 안 됨(오탐 방지)
    assert not scan_numeric_candidates("13일 최 대행은 어선 화재 사고 보고를 받고 지시했다.")
    assert not scan_numeric_candidates("오전 11시 22분 헌재가 대통령 파면을 결정했다.")
    print("[케이스2] 순수 날짜/시각 표기는 후보에서 제외됨 확인")

    # 케이스 3 — 법령상 기준값("30일 전", "1년 이내")은 후보로 잡히면 안 됨
    assert not scan_numeric_candidates("시행 30일 전에 소비자 동의를 다시 받아야 한다.")
    print("[케이스3] 법령상 기준값 표기는 후보에서 제외됨 확인")

    # 케이스 3-1, 3-2 — 실데이터 33개 기사 검증(2026-08-12)에서 확인: 순수 날짜 발표
    # 문장·기간 단축 표현도 "일/주" 단위를 화이트리스트에서 뺀 설계 덕에 이미 제외됨
    # (211개 후보 전수 검사 결과 이런 유형의 오탐이 0건이었음 — 별도 제외 규칙 불필요).
    assert not scan_numeric_candidates("20일 정부는 이 같은 내용의 정책을 발표했다.")
    assert not scan_numeric_candidates("사전 심사 기간을 5주에서 10일 이내로 줄였다.")
    print("[케이스3-1/2] 실데이터 기준 날짜 발표·기간 단축 표현도 제외됨 재확인")

    # 케이스 4 — 한글 복합 큰 수(조/억/만 단위) + 화폐단위는 후보로 잡혀야 함
    assert scan_numeric_candidates("작년 12월 말 우리나라 외환 보유액은 4156억달러로 집계됐다.")
    print("[케이스4] 한글 복합 큰 수(억/만 단위) + 화폐단위 인식 확인")

    # 케이스 5 — find_missed_candidates: 실제로 claim_extractor가 놓쳤던 문장이 감지되는지
    article = (
        "우리나라 작년 1인당 국민총소득(GNI)이 2년 연속 일본을 앞선 것으로 집계됐다. "
        "우리나라 1인당 국민소득은 2022년부터 3년째 증가 추세이다."
    )
    extracted = ["우리나라 작년 1인당 국민총소득(GNI)이 2년 연속 일본을 앞선 것으로 집계됐다."]
    missed = find_missed_candidates(article, extracted)
    assert missed == ["우리나라 1인당 국민소득은 2022년부터 3년째 증가 추세이다."], missed
    print("[케이스5] claim_extractor 결과와 대조해서 놓친 문장 1건 정확히 감지 확인")

    # 케이스 6 — 2026-08-18 수정 확인: 한 문장에서 claim이 2개 나와야 하는데 1개만 뽑힌
    # "부분 추출" 상황을 이제 놓치지 않고 잡아내는지 (골든셋 5-06a/5-06b 실제 사례 재현).
    partial_sentence = "건설 경기 불황으로 건설업 취업자도 9만7000명 줄어, 작년 5월 이후 1년 2개월째 감소세다."
    partial_article = "제조업도 어려운 상황이다. " + partial_sentence
    # claim_extractor가 증감량 claim(9만7000명) 하나만 뽑고 지속기간 claim(1년 2개월째)을
    # 놓친 상황을 재현 — extracted_sentences에 이 문장이 딱 1번만 들어있음.
    only_one_extracted = [partial_sentence]
    missed6 = find_missed_candidates(partial_article, only_one_extracted)
    assert partial_sentence in missed6, (
        f"❌ 부분 추출(claim 1개만 뽑힘)이 감지 안 됨 — 수정 전 버그가 재현됨: {missed6}"
    )
    print("[케이스6] 한 문장에 claim 2개가 나와야 하는데 1개만 뽑힌 부분 추출 상황 감지 확인")

    # 케이스 7 — 케이스 6과 같은 문장인데, 이번엔 claim이 실제로 2개 다 뽑힌(정상) 상황.
    # covered_count(2) >= expected_count(2)라 더 이상 missed로 잡히면 안 됨(과잉 재요청 방지).
    both_extracted = [partial_sentence, partial_sentence]
    missed7 = find_missed_candidates(partial_article, both_extracted)
    assert partial_sentence not in missed7, (
        f"❌ 이미 claim 2개 다 뽑힌 문장을 여전히 missed로 잘못 판단함: {missed7}"
    )
    print("[케이스7] claim 2개 다 정상적으로 뽑힌 경우엔 missed로 안 잡히는지 확인(회귀 방지)")

    print("\n[전체 통과] 규칙 기반 후보 스캐너 회귀 테스트 7건 모두 통과")
