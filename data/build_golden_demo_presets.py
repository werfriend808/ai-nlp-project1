"""data/build_golden_demo_presets.py — notebooks/골든셋_통합.xlsx를 데모 프리셋으로
verifications.db(로컬 SQLite, production Supabase 아님)에 저장한다.

2026-08-31 전면 재작성: 원래 notebooks/claims_golden_merged.csv 등(merge_golden_set.py가
예전 A조/B조 개별 파일들을 병합한 것)을 소스로 썼는데, 사용자가 지적해서 확인해보니
이건 옛날/불완전한 골든셋이었다(기사 31개, claim 74개, claim_type도 없어서 추정해야 했음).
실제로 팀이 최종적으로 쓰는 골든셋은 notebooks/골든셋_통합.xlsx다(기사 50개, claim 104개,
claim_type·comparison_operator·매칭 표·KOSIS 실측값·정답_verdict까지 전부 이미 정리되어
있어 추정할 필요가 없다). 이 스크립트는 그 파일을 소스로 완전히 새로 만든 버전이다.

시트 구성:
  1단계_기사목록   : 번호, 기사제목, 작성일, URL, 본문(정제됨— 광고/내비 잡음 섞여 있어
                     화면 표시용으로는 안 씀, db.fetch_article_text로 라이브 재수집)
  2단계_claim목록  : claim_id, 기사번호, sentence(원문 그대로), claim_type, period, unit,
                     value, comparison_operator, comparison_target, comparison_value 등
                     — Claim() 생성에 필요한 필드가 이미 다 정리되어 있음
  7단계_판정목록   : claim_id, matched_table_id(3단계), matched_table_name,
                     kosis_value(실측), kosis_period, 정답_verdict, reason(판정 근거)

claim마다 세 갈래로 처리한다:
  1) matched_table_id가 "없음"(KOSIS 미등재 등) → judge() 호출 불가. 정답_verdict +
     reason을 그대로 근거로 직접 기록한다(사람이 KOSIS를 직접 찾아본 뒤 내린 결론).
  2) 표는 매칭됐지만 kosis_value(실측)가 없음(예: "12개월 연속 감소" 같은 시계열 전체를
     봐야 하는 claim — 7단계 judge()는 단일 Claim vs 단일 ComputedResult 비교만 하므로
     이런 "연속성" 판단은 애초에 이 아키텍처로 자동화할 수 없다) → 마찬가지로 정답_verdict
     + reason을 직접 기록한다(자동화 불가 영역이라는 걸 정직하게 반영, 지어내지 않음).
  3) 표 + 실측값 둘 다 있음 → Claim/ComputedResult를 만들어 실제 judge()(규칙 1차 필터 +
     애매하면 HCX-003)를 호출한다. comparison_operator가 있으면(증감/증감률) kosis_value가
     부호 없는 절대 크기로 적혀 있으므로(예: "97000", 근거: "2,057-1,960=97") judge.py의
     계산기 규약(감소=음수)에 맞춰 여기서 부호를 붙여 ComputedResult.raw_value를 만든다
     (수치를 지어내는 게 아니라 이미 사람이 실측한 값에 부호만 맞추는 것).

사용법 (프로젝트 루트에서, 실제 HCX 판정 LLM 호출 발생):
    python -m data.build_golden_demo_presets
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.interfaces import Claim, ComputedResult  # noqa: E402
from agent.verdict.judge import JudgeError, judge  # noqa: E402
from db.fetch_article_text import fetch_clean_article_text  # noqa: E402
from db.store import insert_verification, make_result_id  # noqa: E402

XLSX_PATH = ROOT / "notebooks" / "골든셋_통합.xlsx"

# judge() 호출 결과가 재현 안 될 정도로 흔들리는 사례가 실제로 나오면(2026-08-31, 예전
# claims_golden_merged.csv 빌드 때 4건 재현됨 — 명백히 다른 수치인데 LLM이 "차이 작다"며
# 오판정) 여기 claim_id를 추가해서 데모 프리셋에서 제외한다. 지금은 비어 있고, 실제
# 실행 로그를 보고 필요할 때만 채운다(미리 추측해서 넣지 않음).
UNSTABLE_JUDGE_CLAIM_IDS: set[str] = set()

# 사용자가 데모 데이터에서 빼달라고 명시적으로 요청한 기사(2026-08-31) — 표에 있는 claim이
# 전부 "판단불가"뿐이라 데모 가치가 낮다는 이유. "우선" 제외라고 했으니 나중에 다시 필요하면
# 여기서 번호만 지우면 된다.
EXCLUDED_ARTICLE_NOS: set[int] = {43}  # "국내 서비스업 생산성… 제조업의 40%에 그쳐"

_ARTICLE_TITLE_CLEAN_RE = re.compile(r"^[“\"]|[”\"]$")


def _clean_title(t: str) -> str:
    # 기사목록 시트 제목에 스마트 따옴표(“ ”)가 붙어 있는 게 있어서(예: '"그냥 쉬었다"는
    # 20대...') article_title로 그대로 쓰면 프론트/자체 매칭 표시에서 어색해 보일 순 있지만
    # 실제 화면 표시 title로 굳이 바꿀 필요는 없다 — 여기서는 원본 그대로 둔다(임의 변형 금지).
    return t


# 지원하는 ISO류 표기: "2025-06-01 00:00:00"(엑셀이 날짜로 읽은 셀), "2025-06-01",
# "2025-06", "2025.10", "25.05"(2자리 연도), "2024"(연도만).
_ISO_YMD_TS_RE = re.compile(r"^(\d{4})-(\d{2})-\d{2}(?:\s+00:00:00)?$")
_ISO_YM_RE = re.compile(r"^(\d{4})[-.](\d{1,2})$")
_ISO_YY_M_RE = re.compile(r"^(\d{2})\.(\d{2})$")
_ISO_Y_RE = re.compile(r"^(\d{4})$")


def _normalize_period_core(p: str) -> str:
    m = _ISO_YMD_TS_RE.match(p)
    if m:
        return f"{m.group(1)}년 {int(m.group(2))}월"
    m = _ISO_YM_RE.match(p)
    if m:
        return f"{m.group(1)}년 {int(m.group(2))}월"
    m = _ISO_YY_M_RE.match(p)
    if m:
        return f"20{m.group(1)}년 {int(m.group(2))}월"
    m = _ISO_Y_RE.match(p)
    if m:
        return f"{m.group(1)}년"
    return p


def _normalize_period(p: Optional[str]) -> Optional[str]:
    """골든셋_통합.xlsx의 period가 ISO 스타일("2025-06", "2024", 엑셀이 날짜 셀로 읽어
    버린 "2025-06-01 00:00:00" 등)로 적혀 있는데, judge.py의 _period_granularity()는
    실제 운영 2단계가 늘 만들어내는 한글 표기("2024년", "2025년 6월", "202506" 6자리
    연속 숫자 등)를 전제로 짜여 있다. "2025-06"은 대시로 끊겨서 6자리 연속 숫자로도 안
    잡히고 "월" 글자도 없어서 "년"으로(틀리게) 인식되는 반면, 같은 걸
    "2025-06(전년동월대비)"처럼 괄호 설명이 붙으면 그 안의 "월" 글자 때문에 "월"로
    인식되어 — 사실 둘 다 월 단위인데 서로 다른 걸로 오판정되는 버그를 실측 확인했다
    (claim 5-12b, 2026-08-31: 실제로는 -1.0%p vs -1.0%p로 정확히 일치인데 LLM이 "통계는
    연간 평균이라 비교 기준이 다르다"는 근거 없는 이유를 대며 불일치로 판정). 그래서
    여기서 미리 한글 표기로 바꿔서 넘긴다(값 자체는 안 바꾸고 표기만 정규화 — 지어내는
    게 아님). "(전년동월대비)"처럼 괄호 설명이 붙어 있으면 그 앞부분만 정규화하고
    설명은 그대로 붙여 돌려준다. 못 알아보는 형식(범위 "~"/화살표 "→" 등)은 원본
    그대로 반환 — _period_granularity가 못 알아들으면 그냥 None을 주는 안전한 폴백이라
    괜히 잘못 변환하는 것보다 낫다."""
    if not p:
        return p
    p = p.strip()
    m = re.match(r"^(.*?)(\(.*\))$", p)
    if m:
        return _normalize_period_core(m.group(1).strip()) + m.group(2)
    return _normalize_period_core(p)


def load_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    xl = pd.ExcelFile(XLSX_PATH)
    articles = xl.parse("1단계_기사목록")
    claims = xl.parse("2단계_claim목록")
    verdicts = xl.parse("7단계_판정목록")
    return articles, claims, verdicts


def _nz(v) -> Optional[str]:
    """NaN/빈 문자열을 None으로. pandas가 빈 셀을 float('nan')으로 주는 경우가 많아서
    문자열 필드 전반에 필요."""
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    s = str(v).strip()
    return s or None


def _to_float(v) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    articles, claims, verdicts = load_dataset()
    articles_by_no = articles.set_index("번호")

    merged = claims.merge(verdicts[[
        "claim_id", "matched_table_id(3단계)", "matched_table_name", "kosis_value(실측)",
        "kosis_period", "정답_verdict", "gap_type", "reason(판정 근거)",
    ]], on="claim_id", how="left")

    print(f"골든셋_통합: 기사 {len(articles)}건, claim {len(merged)}건")

    # URL별로 한 번만 라이브 fetch (같은 기사에 claim이 여러 개라 반복 요청 방지)
    text_cache: dict[str, Optional[str]] = {}

    n_judged = n_direct = n_skipped = n_failed = n_url_fail = 0

    for _, row in merged.iterrows():
        claim_id = row["claim_id"]
        article_no = row["기사번호"]
        if article_no in EXCLUDED_ARTICLE_NOS:
            continue
        art = articles_by_no.loc[article_no]
        article_title = _clean_title(str(art["기사제목"]))
        article_url = str(art["URL"]).strip()
        published_date = _nz(art["작성일"])
        if isinstance(published_date, str) and " " in published_date:
            published_date = published_date.split(" ")[0]

        sentence = str(row["sentence(원문 그대로)"]).strip()
        claim_type = row["claim_type"]
        claim_unit = _nz(row["unit"])
        claim_period = _nz(row["period"])
        claim_value = _to_float(row["value"])
        comparison_operator = _nz(row["comparison_operator"])
        comparison_target = _nz(row["comparison_target"])
        comparison_value = _to_float(row["comparison_value"])
        population = _nz(row["population"])
        region = _nz(row["region"])
        source_org = _nz(row["source_org"])
        statistic_expression = _nz(row["statistic_expression"])

        if article_url not in text_cache:
            try:
                text_cache[article_url] = fetch_clean_article_text(article_url)
            except Exception as e:  # noqa: BLE001
                print(f"[URL FETCH FAIL] {article_url}: {e}")
                text_cache[article_url] = None
        if text_cache[article_url] is None:
            print(f"[스킵·본문없음] {claim_id}: {article_url}")
            n_url_fail += 1
            continue

        matched_table_id = _nz(row["matched_table_id(3단계)"])
        matched_table_name = _nz(row["matched_table_name"])
        kosis_value_raw = _nz(row["kosis_value(실측)"])  # "청양군 60.3"처럼 텍스트가 섞인 것도 있어 원문 문자열도 따로 보관
        kosis_value = _to_float(row["kosis_value(실측)"])
        kosis_period = _nz(row["kosis_period"])
        gold_verdict = _nz(row["정답_verdict"]) or "판단불가"
        reason_text = _nz(row["reason(판정 근거)"])

        record_base = {
            # 골든셋_통합.xlsx는 한 문장에서 여러 사실을 따로 뽑아낸 claim이 많아(예: 5-01a/
            # 5-01b가 같은 문장에서 각각 "취업자 수 수준값"과 "전년 대비 증감폭"을 검증) 같은
            # 기사 안에 claim_sentence가 완전히 동일한 행이 24그룹·29건 있다(실측 확인,
            # 2026-08-31). make_result_id(title, sentence)만 쓰면 이런 동일 문장 claim들이
            # 전부 같은 result_id로 뭉개져서(INSERT OR REPLACE) 마지막 것만 남고 나머지가
            # 조용히 사라진다 — 사용자가 지적한 "클레임이 하나도 제대로 안 뽑힌 것 같다"의
            # 실제 원인이었다. claim_id를 해시 입력에 더해 각 claim이 별개 행으로 남게 한다.
            "result_id": make_result_id(article_title, f"{sentence}|{claim_id}"),
            "article_title": article_title,
            "article_url": article_url,
            "published_date": published_date,
            "claim_sentence": sentence,
            "claim_type": claim_type,
            "statistic_expression": statistic_expression,
            "normalized_statistic_name": statistic_expression,
            "statistic_category": None,
            "value": claim_value,
            "unit": claim_unit,
            "comparison_operator": comparison_operator,
            "comparison_target": comparison_target,
            "comparison_value": comparison_value,
            "time_expression": claim_period,
            "reference_time": kosis_period,
            "population": population,
            "region": region,
            "source_org": source_org,
            "source_report": None,
            "kosis_table_id": matched_table_id,
            "kosis_table": matched_table_name,
            "kosis_item": None,
            "kosis_dimension": None,
            "calculation_required": comparison_operator is not None,
            "calculation_type": None,
            "verification_possible": None,
            "ambiguity_reason": None,
            "verification_result": None,
            "mismatch_reason": None,
            "evidence": None,
            "classifier_score": 0.98,
            "reviewer_agrees": None,
            "reviewer_corrected_verdict": None,
        }

        if claim_id in UNSTABLE_JUDGE_CLAIM_IDS:
            print(f"[스킵·불안정] {claim_id}: judge() 재현성 문제로 제외")
            n_skipped += 1
            continue

        if matched_table_id is None or kosis_value is None or claim_value is None:
            # claim_value가 없으면(예: "역대 최대치를 기록했다"처럼 단일 수치가 아니라
            # 시계열 전체를 봐야 확인되는 주장 — 36-01 실측 확인, 2026-08-31) 표+실측값이
            # 있어도 judge()에 넘길 단일 비교 기준 자체가 없다. 이런 것도 "연속 개월"류와
            # 같은 이유로 직접기록으로 보낸다.
            # 표 자체가 없거나(KOSIS 미등재), 표는 있어도 단일 수치로 못 뽑는 경우
            # (연속 개월 수 판단 등 — judge()가 애초에 처리 못 하는 영역) — 사람이 이미
            # 낸 결론(정답_verdict)과 근거(reason)를 그대로 기록한다.
            #
            # reason(판정 근거)이 비어 있는데(예: 30-06a~e, 30-08a/b) matched_table_id·
            # kosis_value(실측)는 있는 행이 7건 있다(2026-08-31 실측 확인) — 이걸 그냥 빈
            # 채로 두고 나중에 "LLM이 쓴 것처럼" 다시 쓰는 스크립트에 넘겼더니, 근거가 아예
            # 없으니 LLM이 그럴듯한 내용을 통째로 지어내는 사고가 실제로 났다(예: "자살률"
            # claim인데 "암 발생률"이라고 지어냄, 전혀 다른 수치를 지어냄). 그래서 reason이
            # 없어도 matched_table_id/kosis_value가 있으면 그 실측값으로 최소한의 근거
            # 문장을 여기서 직접 만든다(지어내는 게 아니라 이미 있는 실측값을 문장으로
            # 옮기는 것) — evidence가 완전히 빈 채로 다음 단계에 넘어가는 일이 없게 한다.
            evidence_text = reason_text
            if evidence_text is None and matched_table_id is not None and kosis_value_raw is not None:
                table_label = matched_table_name or matched_table_id
                evidence_text = (
                    f"KOSIS {table_label}({matched_table_id}) 실측값: {kosis_value_raw}"
                    + (f" ({kosis_period})" if kosis_period else "")
                )
            record_base.update(
                verification_possible="불가능",
                ambiguity_reason=evidence_text,
                verification_result=gold_verdict,
                evidence=evidence_text,
            )
            insert_verification(record_base)
            n_direct += 1
            print(f"[직접기록={gold_verdict}] {claim_id}: {sentence[:40]}")
            continue

        # judge() 호출 대상 — comparison_operator가 있으면(증감/증감률) kosis_value가
        # 부호 없는 절대 크기로 적혀 있어(예: "97000") judge.py 계산기 규약(감소=음수)에
        # 맞춰 부호를 붙인다. 없으면(규모, 순수 수준값) 그대로 쓴다.
        signed_kosis_value = kosis_value
        if comparison_operator == "감소":
            signed_kosis_value = -abs(kosis_value)
        elif comparison_operator == "증가":
            signed_kosis_value = abs(kosis_value)

        value_type = "증감폭" if (claim_type == "규모" and comparison_operator is not None) else None
        if comparison_operator is not None:
            calc_type = "증감률" if claim_unit == "%" else "증감"
        else:
            calc_type = "단순조회"

        claim_obj = Claim(
            sentence=sentence,
            claim_type=claim_type,
            period=_normalize_period(claim_period),
            unit=claim_unit,
            population=population,
            statistic_expression=statistic_expression,
            value=abs(claim_value) if claim_value is not None else None,
            value_type=value_type,
            comparison_operator=comparison_operator,
            comparison_target=comparison_target,
            comparison_value=comparison_value,
            region=region,
            source_org=source_org,
        )
        computed_obj = ComputedResult(
            calc_type=calc_type,
            raw_value=signed_kosis_value,
            unit=claim_unit or "",
            period=_normalize_period(kosis_period) or _normalize_period(claim_period) or "",
        )

        try:
            verdict = judge(
                claim_obj, computed_obj,
                matched_table_name=matched_table_name,
                article_date=published_date,
            )
        except JudgeError as e:
            print(f"[FAIL] {claim_id} judge() 실패: {e}")
            n_failed += 1
            continue

        record_base.update(
            calculation_type=calc_type,
            verification_possible="가능",
            verification_result=verdict.verdict,
            mismatch_reason=verdict.gap_type,
            evidence=verdict.reason,
        )
        insert_verification(record_base)
        n_judged += 1
        tag = "OK" if verdict.verdict == gold_verdict else f"MISMATCH(정답={gold_verdict})"
        print(f"[judge={verdict.verdict} {tag}] {claim_id}: {sentence[:40]}")

    print(
        f"\n총 {n_judged}건 judge() 호출·저장, {n_direct}건 직접기록, "
        f"{n_skipped}건 스킵(불안정), {n_url_fail}건 본문 수집 실패, {n_failed}건 judge() 실패"
    )


if __name__ == "__main__":
    main()
