"""
agent/preprocessing/test_preprocessing.py — 1,2단계(classifier/claim_extractor) 단위 테스트

Day2 체크리스트: "각자 담당 모듈 단위 테스트 마무리 (샘플 10건 이상)" 대응.
agent/mapping/test_mapping.py와 동일한 패턴(TEST_CASES + 자동 채점)을 따른다.

classifier 케이스는 few-shot 프롬프트에 쓰인 판단 기준("정부/공공기관이 발표한 공식
통계·수치 기반 주장만 TRUE")과 실제 data_set.csv 라벨이 명확히 일치하는 것만 골랐다.
(라벨은 TRUE인데 본문에 숫자가 전혀 없는 애매한 기사들은 기준 자체가 안 맞는 케이스라 제외)

실행 (프로젝트 루트에서, 실제 HCX API를 호출함):
    python -m agent.preprocessing.test_preprocessing
"""

from __future__ import annotations

from agent.preprocessing.classifier import classify
from agent.preprocessing.claim_extractor import extract_claims

# (기사 본문 발췌, 정답 label, 카테고리) — data_set.csv에서 발췌, few-shot 예시와는 겹치지 않음
CLASSIFIER_CASES: list[tuple[str, bool, str]] = [
    (
        "중앙재난안전대책본부는 11일 오후 6시 기준 국가정보자원관리원 화재로 중단된 행정정보시스템 "
        "709개 중 242개가 정상화됐다고 밝혔다. 이에 따라 시스템 복구율은 34.1%가 됐다. 중요도가 큰 "
        "1등급 시스템은 전체 40개 중 30개가 정상화돼 복구율은 75.0%다.",
        True,
        "정부부처 공식 발표(수치)",
    ),
    (
        "유엔 식량농업기구(FAO)가 집계한 지난달 세계 식량 가격지수가 124.9로 전달보다 1.6% 하락했다고 "
        "농림축산식품부가 8일 밝혔다. 작년 11월 127.7까지 올랐던 이 지수는 12월 127.0, 지난달 124.9로 "
        "두 달 연속 하락했다.",
        True,
        "정부부처 공식 발표(수치)",
    ),
    (
        "3일 내란 특별검사팀에 소환된 김주현 전 민정수석이 오후 9시 57분 조사를 마치고 귀가했다. "
        "이날 오전 9시 47분 출석한 지 12시간 10분 만이다.",
        False,
        "수치·통계 없는 사건 기사",
    ),
    (
        "충북 제천의 한 주택에서 태어난 신생아가 숨져 경찰이 수사 중이다. 22일 경찰에 따르면 이날 "
        "오후 2시쯤 제천시 백운면 한 단독주택에서 갓 태어난 아기가 숨진 것 같다는 112 신고가 접수됐다.",
        False,
        "수치·통계 없는 사건 기사",
    ),
    (
        "[오늘의 날씨] 2025년 7월 8일. 전국 가끔 구름 많음.",
        False,
        "수치·통계 없는 생활 정보",
    ),
    (
        "'순직 해병 수사 외압' 의혹을 수사하는 해병대원 특검팀의 특별검사보로 류관석·이금규·김숙정·"
        "정민영 변호사가 20일 임명됐다. 해병대원 특검팀을 마지막으로 내란·김건희 등 3대 특검이 열흘 "
        "만에 특검보 인선을 모두 매듭지었다.",
        False,
        "수치·통계 없는 인사 기사 (이름 나열을 수치로 오판하기 쉬운 회귀 테스트)",
    ),
    (
        "방학 중에 집에 혼자 있다가 화재 사고로 숨진 초등학생 문모(12)양의 가정에 시민들의 위로금이 "
        "답지하고 있다. 4일 인천사회복지공동모금회에는 문양의 가정을 위해 기부하겠다는 후원금 800여 "
        "만원이 모였다. 인천 서구는 긴급 생계비 154만원을 3개월간 지원할 예정이다.",
        False,
        "숫자는 있지만 국가 통계가 아닌 기부금·지원금 (오탐 회귀 테스트)",
    ),
    (
        "충남소방본부에 따르면 이날 오후 9시 33분쯤 서산시 동문동 한 모텔 2층에서 불이 났다. 이 불로 "
        "1명이 숨지고 17명이 다친 것으로 파악됐다.",
        False,
        "숫자는 있지만 국가 통계가 아닌 사고 피해 규모 (few-shot 9번 예시 회귀 테스트)",
    ),
]

# (기사 본문, 최소한 이 claim_type들은 추출돼야 함, 카테고리)
CLAIM_EXTRACTOR_CASES: list[tuple[str, set[str], str]] = [
    (
        "통계청이 23일 발표한 '2024년 양곡소비량조사 결과'에 따르면, 작년 국민 1인당 쌀 소비량은 "
        "1년 전보다 1.1%(0.6kg) 감소한 55.8kg을 기록했다. 작년 소비량은 30년 전인 1994년(108.3kg)의 "
        "절반 수준이다.",
        {"규모", "증감률", "비교"},
        "규모+증감률+비교 복합",
    ),
    (
        "한국은행이 6일 발표한 외환 보유액 통계에 따르면, 작년 12월 말 우리나라 외환 보유액은 "
        "4156억달러로 집계됐다.",
        {"규모"},
        "단순 규모",
    ),
    (
        "중앙재난안전대책본부는 11일 오후 6시 기준 행정정보시스템 복구율은 34.1%가 됐다고 밝혔다. "
        "1등급 시스템은 전체 40개 중 30개가 정상화돼 복구율은 75.0%다.",
        {"규모"},
        "비율(%) 규모",
    ),
    (
        "국책연구기관인 한국개발연구원(KDI)이 올해 한국 경제 성장률 전망치를 2%에서 1.6%로 낮췄다.",
        {"전망"},
        "전망치",
    ),
    (
        "유엔 식량농업기구(FAO)가 집계한 지난달 세계 식량 가격지수가 124.9로 전달보다 1.6% 하락했다. "
        "이 지수는 2014~2016년 평균 가격을 100으로 본 상대적 수치다.",
        {"증감률"},
        "증감률",
    ),
]


def run_classifier_tests() -> int:
    total = len(CLASSIFIER_CASES)
    passed = 0
    print(f"=== classifier 테스트 ({total}건) ===\n")
    for article_text, expected_label, category in CLASSIFIER_CASES:
        result = classify(article_text)
        ok = result.label == expected_label
        passed += ok
        mark = "O" if ok else "X"
        print(f"[{mark}] [{category}] 기대={expected_label} 실제={result.label} score={result.score:.2f}")
        print(f"    reason: {result.reason}")
        if not ok:
            print(f"    기사: {article_text[:60]}...")
    print(f"\nclassifier 정답률: {passed}/{total}\n")
    return passed


def run_claim_extractor_tests() -> int:
    total = len(CLAIM_EXTRACTOR_CASES)
    passed = 0
    print(f"=== claim_extractor 테스트 ({total}건) ===\n")
    for article_text, expected_types, category in CLAIM_EXTRACTOR_CASES:
        claims = extract_claims(article_text)
        extracted_types = {c.claim_type for c in claims}
        ok = expected_types.issubset(extracted_types)
        passed += ok
        mark = "O" if ok else "X"
        print(f"[{mark}] [{category}] 기대 claim_type⊆{expected_types} 실제={extracted_types} ({len(claims)}건 추출)")
        for c in claims:
            print(f"    - [{c.claim_type}] {c.sentence}")
        print()
    print(f"claim_extractor 정답률: {passed}/{total}\n")
    return passed


def run_tests() -> None:
    cls_passed = run_classifier_tests()
    claim_passed = run_claim_extractor_tests()
    print("=== 결과 요약 ===")
    print(f"classifier       : {cls_passed}/{len(CLASSIFIER_CASES)}")
    print(f"claim_extractor  : {claim_passed}/{len(CLAIM_EXTRACTOR_CASES)}")


if __name__ == "__main__":
    run_tests()
