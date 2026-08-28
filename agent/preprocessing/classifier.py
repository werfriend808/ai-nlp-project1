"""
agent/preprocessing/classifier.py — 1단계 기사 관련도 분류

팀 계약(interfaces.py) 기준:
    입력: 기사 본문(str)
    출력: ClassificationResult (label, score, reason)

모델: HCX-007
프롬프트: prompts/classifier_prompt.txt (few-shot 8개 포함, {article_text} 자리에 본문을 채워 넣음)

2026-08-22 실측: HCX-DASH-002가 골든셋 50건 중 34%(17건)를 틀려서(민간기업/해외통계/
1회성 행정조치를 True로 오판) HCX-007로 재검증했더니 정답률이 66.0% -> 90.0%로 개선됨
(notebooks/stage1_verify_report.md vs notebooks/stage1_verify_report_hcx007.md 참고).
그래서 기본 모델을 HCX-007로 교체함.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .hcx_client import call_hcx
from .claim_candidate_scanner import scan_numeric_candidates

try:
    from interfaces import ClassificationResult
except ImportError:
    try:
        from agent.interfaces import ClassificationResult
    except ImportError:  # 단독 실행/테스트용 폴백
        from dataclasses import dataclass

        @dataclass
        class ClassificationResult:  # type: ignore[no-redef]
            label: bool
            score: float
            reason: str


MODEL = "HCX-007"
PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "classifier_prompt.txt"
SYSTEM_PROMPT = "아래 지시사항을 정확히 따르고, 반드시 지정된 JSON 형식으로만 응답하세요."


class ClassifierError(RuntimeError):
    """분류 응답을 JSON으로 파싱하지 못한 경우."""


# 2026-08-14 실측: classifier_prompt.txt에 이미 "1회성 사건 수치는 FALSE" 규칙과 근접한
# 예시(예시 9/13/15/21 — 개별 사고 인명피해, 정부 관계자 발언만으로는 TRUE 아님 등)가
# 있는데도, 모델(HCX-DASH-002)이 "정부 관계자가 공식적으로 발표했다"는 이유만으로
# label=True를 낸 사례가 확인됐다("부안 어선 화재" 기사 — 개별 사고의 구조 상황일 뿐
# 통계가 아닌데 score=0.92로 통과). 정상적으로 TRUE 판정된 사례들의 reason은 예외 없이
# "통계"/"조사"/"수치"/"증가율"처럼 실제 통계 어휘를 담고 있었던 반면, 이 오탐 사례의
# reason은 "신뢰성이 높다"류의 공식성/구체성 언급뿐 통계 어휘가 전혀 없었다 — 즉 모델이
# 자기 판단 근거를 설명하면서 스스로 "이건 통계가 아니다"라는 신호를 흘리고 있었다.
#
# score만 낮추는 방식은 검토했으나 버렸다 — batch_runner.py/export_for_rerank.py 둘 다
# cls_result.label만 보고 분기하고 score는 기록용일 뿐이라, score를 낮춰도 파이프라인
# 동작에 실제 영향이 없다(실측 확인). 그래서 label 자체를 뒤집는다. False→True로는
# 절대 안 바뀌므로(label=True인 경우만 검사) 새로운 오탐(원래 False였는데 True가 되는
# 경우)을 만들 위험은 없고, 기존에 부정확하게 True였던 것 중 일부를 되돌리는 방향으로만
# 작동한다.
_STATISTICAL_REASON_RE = re.compile(
    r"(통계|조사|집계|지수|비율|동향|증감률|증가율|감소율|수치|증가|감소|상승|하락|"
    r"성장률|늘었|줄었|올랐|내렸|떨어졌)"
)


def _looks_like_real_statistic(reason: str) -> bool:
    return bool(_STATISTICAL_REASON_RE.search(reason))


# 2026-08-20 실측: 위 안전장치가 reason 문자열에 "통계"/"수치" 같은 단어가 있는지만
# 보는데, 모델이 실제로는 통계가 아닌 기사(예: "한전이 정전 복구를 마쳤다고 밝혔다" — 숫자
# 전혀 없음)를 두고도 "공식 통계나 조사 결과에 준하는 수치 기반 사실이다"처럼 그 단어
# 자체를 근거 설명에 끌어써버리면 안전장치를 그대로 통과해버린다(자체 점검 중 재현,
# score=0.9로 오탐). "부안 어선 화재" 사례와 달리 이번엔 reason 자체가 통계 어휘를
# 포함하고 있어서 키워드 매칭만으로는 못 잡는다.
#
# 진짜로 검증 가치 있는 수치 주장 기사라면 원문(article_text)에 통계로 볼 만한 수치
# 표현이 최소 하나는 있어야 한다는 게 훨씬 강한 신호다 — reason은 모델이 사후에 지어낼
# 수 있지만, 원문 자체는 그렇지 않다. 다만 "원문에 숫자가 있는가"를 단순히 아무 자릿수
# 문자(\d) 존재 여부로만 보면 안 된다 — 뉴스 기사는 "2일 부산 ~에서 사고가 발생해"처럼
# 거의 항상 날짜로 시작해서, 순수 날짜/시각 표기만 있어도 "숫자 있음"으로 잘못 통과된다
# (직접 확인: 정전 사고 기사도 "2일"이라는 숫자 자체는 있음). claim_candidate_scanner.py의
# scan_numeric_candidates()는 정확히 이 구분(날짜/시각 등은 제외, 통화/수량/퍼센트/순위/
# 극값 표현 등만 인정)을 위해 이미 만들어져 있으므로 새로 만들지 않고 그대로 재사용한다.
def _article_has_real_statistic_candidate(article_text: str) -> bool:
    return bool(scan_numeric_candidates(article_text))


def _load_prompt_template(path: Path = PROMPT_PATH) -> str:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다. A가 few-shot 프롬프트를 먼저 작성해야 합니다.")
    return path.read_text(encoding="utf-8")


def _extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ClassifierError(f"응답에서 JSON 객체를 찾지 못했습니다: {text!r}")
    return json.loads(match.group(0))


def classify(article_text: str, *, model: str = MODEL, temperature: float = 0.0) -> ClassificationResult:
    """기사 본문 하나를 받아 1단계 분류 결과(label/score/reason)를 돌려줍니다.

    temperature=0.0: 기본값(call_hcx의 0.2)을 두면 같은 기사를 넣어도 판정이 흔들리거나
    reason이 원문에 없는 내용을 지어내는 경우가 있어(2026-08-10 실기사 검증에서 발견 —
    "빚 못 갚은 자영업자" 기사의 reason에 원문에 없는 "멕시코"가 등장) 결정적으로 고정한다.
    """
    template = _load_prompt_template()
    prompt = template.replace("{article_text}", article_text)

    reply = call_hcx(
        model=model, system_prompt=SYSTEM_PROMPT, user_content=prompt, temperature=temperature
    )

    try:
        parsed = _extract_json_object(reply)
        label = bool(parsed["label"])
        score = float(parsed["score"])
        reason = str(parsed["reason"])
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        raise ClassifierError(f"응답 파싱 실패: {reply!r}") from e

    if label and not _looks_like_real_statistic(reason):
        reason = (
            f"{reason} [안전장치: label=True인데 reason에 통계 어휘가 없어 "
            "1회성 사건/공식 발언 오판정 가능성으로 보고 False로 정정함]"
        )
        label = False
        score = min(score, 0.15)
    elif label and not _article_has_real_statistic_candidate(article_text):
        reason = (
            f"{reason} [안전장치: label=True인데 원문에 통계로 볼 만한 수치 표현이 "
            "전혀 없어(reason이 통계 어휘를 언급해도) 검증 가능한 수치 주장이 있을 수 "
            "없다고 보고 False로 정정함]"
        )
        label = False
        score = min(score, 0.15)

    return ClassificationResult(label=label, score=score, reason=reason)


if __name__ == "__main__":
    # 2026-08-14 회귀 테스트: _looks_like_real_statistic() — LLM 호출 없이 순수 텍스트만
    # 검사하므로 API 비용 없이 실행 가능. classifier_prompt.txt의 실제 TRUE 예시 reason들이
    # 전부 안 걸리는지(오탐 방지 안전장치가 정상 케이스를 잘못 뒤집지 않는지) + 실제 오탐
    # 사례(부안 어선 화재)의 reason은 걸리는지 확인.
    true_example_reasons = [
        "통계청이 발표한 공식 조사 결과를 수치와 함께 인용하고 있다.",
        "한국은행이 발표한 공식 통계 수치를 그대로 인용하고 있다.",
        "정부 부처가 발표한 수치(복구율)를 인용하고 있다.",
        "국책연구기관이 발표한 수치 기반 전망치를 인용하고 있다.",
        "기사 대부분은 기업간 배송 경쟁 서술이지만, '작년 국내 유통 전체 매출에서 온라인 "
        "비율이 역대 최대인 50.6%'는 국가 통계(온라인쇼핑 동향 등)로 검증 가능한 수치이므로 "
        "TRUE로 판단한다.",
    ]
    for r in true_example_reasons:
        assert _looks_like_real_statistic(r), f"❌ 정상 TRUE reason이 안전장치에 걸림: {r!r}"
    print("[회귀 확인] classifier_prompt.txt의 TRUE 예시 reason 5건 모두 안전장치 통과(안 걸림)")

    bad_reason = (
        "정부 관계자(대통령 권한대행 부총리 겸 기획재정부 장관)가 공식적으로 발표한 내용을 "
        "담고 있으며, 구체적인 사고 상황과 대응 방안, 그리고 인명 구조 현황 등이 포함되어 "
        "있어 신뢰성이 높다."
    )
    assert not _looks_like_real_statistic(bad_reason), (
        f"❌ 실제 오탐 사례(부안 어선 화재)의 reason이 안전장치에 안 걸림: {bad_reason!r}"
    )
    print("[수정 확인] '부안 어선 화재' 실제 오탐 사례의 reason은 안전장치에 정상적으로 걸림")

    print("\n[전체 통과] _looks_like_real_statistic 회귀 테스트 모두 통과")

    # 2026-08-20 회귀 테스트: _article_has_real_statistic_candidate() — 자체 점검 중
    # 재현된 오탐(원문에 순수 날짜 숫자("2일")밖에 없는 "정전 사고 복구" 기사인데, reason이
    # "통계"/"수치" 단어를 스스로 끌어써서 위 _looks_like_real_statistic 안전장치를 그대로
    # 통과해버린 사례).
    no_stat_article = (
        "2일 부산 해운대구 한 아파트 단지에서 정전 사고가 발생해 주민들이 불편을 겪었다. "
        "한국전력은 복구 작업을 마쳤다고 밝혔다."
    )
    sneaky_reason = (
        "한국전력이라는 공공기관이 정전 사고 복구 작업 완료 사실을 밝히고 있으므로, "
        "이는 공식 통계나 조사 결과에 준하는 수치 기반 사실이다."
    )
    assert _looks_like_real_statistic(sneaky_reason), (
        "이 테스트의 전제(reason이 통계 어휘를 포함해서 기존 안전장치를 통과함)가 깨짐"
    )
    assert not _article_has_real_statistic_candidate(no_stat_article), (
        f"❌ 테스트 전제 오류: {no_stat_article!r}"
    )
    print(
        "[회귀 확인] 원문에 통계로 볼 만한 수치 표현이 없으면(날짜 숫자만 있어도), "
        "reason이 통계 어휘를 포함해도 기존 안전장치를 우회함(수정 전 재현)"
    )

    # 정상 케이스 회귀 확인: 실제 통계 수치가 있는 원문은 이 안전장치에 걸리면 안 됨.
    stat_article = "5일 통계청이 발표한 소비자물가동향에 따르면 지난달 소비자물가가 전년 동월 대비 2.2% 올랐다."
    assert _article_has_real_statistic_candidate(stat_article), (
        f"❌ 정상 원문이 통계 수치 없음으로 잘못 판정됨: {stat_article!r}"
    )
    print("[회귀 확인] 통계 수치가 있는 정상 원문은 안전장치에 안 걸림(회귀 방지)")

    #   python -m agent.preprocessing.classifier
    sample = (
        "통계청이 23일 발표한 '2024년 양곡소비량조사 결과'에 따르면, "
        "작년 국민 1인당 쌀 소비량은 1년 전보다 1.1%(0.6kg) 감소한 55.8kg을 기록했다."
    )
    print(classify(sample))
