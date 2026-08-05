"""
agent/verdict/verify_judge_on_golden.py — 7단계(judge.py)를 판정 골든셋으로 단독 검증

7단계 judge()는 Claim + ComputedResult 두 객체만 있으면 되고 1~6단계를 실제로 돌릴
필요가 없다. 판정 골든셋(notebooks/판정 골든셋.xlsx 등, merge_golden_set.py가 병합한
verdict_golden_merged.csv)에 사람이 직접 매긴 정답(verification_result: 일치/불일치/확인불가)이
있으므로, 이걸 claims_golden_merged.csv(주장)·mapping_golden_merged.csv(KOSIS 조회값)와
조인해서 Claim/ComputedResult를 만들어 judge()에 넣고 정답과 비교한다.

⚠️ 근사치 파싱 — 완전 자동화가 불가능한 이유:
    mapping_golden_merged.csv의 kosis_value 컬럼은 사람이 자유 형식으로 적은 텍스트라
    ("9,384,325명 (2024.1)", "116.29(지수)/+2.1%", "축산물121.52/돼지고기125.73/..." 등)
    100% 정확한 파싱이 불가능하다. 이 스크립트는 claim의 단위(claim_numeric_unit)에 맞는
    세그먼트를 슬래시(/)로 분리해서 고르는 best-effort 파서(_parse_kosis_value)를 쓴다.
    claim_type/comparison_operator도 골든셋에 없는 필드라 claim_comparison_target 유무와
    단위로 추정한다. 그래서 이 스크립트의 결과는 "7단계 로직이 대략 얼마나 맞는지"를 보는
    용도이지, 완전히 정밀한 성능 수치는 아니다 — 리포트에 파싱된 Claim/ComputedResult 값을
    그대로 남겨서 사람이 검증할 수 있게 한다.

    kosis_value가 없는 행(match_status="매칭 실패"/"표 후보 확인 *"/"미완료")은 애초에
    5단계가 값을 못 가져온 것과 같아서 judge()를 부를 수 없다 — 골든셋 정답도 대부분
    "확인불가"이므로 스킵하고 별도 집계한다.

사용법 (프로젝트 루트에서):
    python -m agent.verdict.verify_judge_on_golden
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from agent.interfaces import Claim, ComputedResult
from agent.verdict.judge import JudgeError, judge

NOTEBOOKS_DIR = Path(__file__).parent.parent.parent / "notebooks"
CLAIMS_CSV = NOTEBOOKS_DIR / "claims_golden_merged.csv"
MAPPING_CSV = NOTEBOOKS_DIR / "mapping_golden_merged.csv"
VERDICT_CSV = NOTEBOOKS_DIR / "verdict_golden_merged.csv"
REPORT_PATH = Path(__file__).parent.parent.parent / "tests" / "judge_golden_scoring_report.md"

_NUM_UNIT_RE = re.compile(r"^([+\-]?)([\d,]+\.?\d*)\s*\(?([%가-힣]*)\)?")
_PERIOD_RE = re.compile(r"\((\d{4}(?:\.\d{1,2})?)")


def _parse_kosis_value(raw: object, claim_unit: Optional[str]) -> Optional[tuple[float, str, Optional[str]]]:
    """kosis_value 자유 텍스트를 (raw_value, unit, period)로 best-effort 파싱."""
    if not isinstance(raw, str) or not raw.strip():
        return None

    period_match = _PERIOD_RE.search(raw)
    period = period_match.group(1) if period_match else None

    segments = raw.split("/")

    def _extract(seg: str) -> Optional[tuple[float, str]]:
        m = _NUM_UNIT_RE.match(seg.strip())
        if not m or not m.group(2):
            return None
        value = float(m.group(2).replace(",", ""))
        if m.group(1) == "-":
            value = -value
        unit = m.group(3) or ""
        return value, unit

    if claim_unit == "%":
        for seg in segments:
            if "%" in seg:
                parsed = _extract(seg)
                if parsed:
                    return parsed[0], "%", period

    parsed = _extract(segments[0])
    if parsed:
        return parsed[0], parsed[1], period
    return None


def _infer_claim_type(comparison_target: object, unit: Optional[str]) -> str:
    """골든셋에 claim_type 컬럼이 없어 comparison_target 유무·단위로 추정한다."""
    has_target = isinstance(comparison_target, str) and comparison_target.strip()
    if has_target and unit == "%":
        return "증감률"
    if has_target:
        return "비교"
    return "규모"


def _infer_calc_type(unit: str) -> str:
    return "증감률" if unit == "%" else "단순조회"


def _to_optional_float(value: object) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_dataset() -> pd.DataFrame:
    claims = pd.read_csv(CLAIMS_CSV)
    mapping = pd.read_csv(MAPPING_CSV)
    verdict = pd.read_csv(VERDICT_CSV)

    merged = claims.merge(mapping, on="claim_id", how="inner").merge(verdict, on="claim_id", how="inner")
    return merged


def main() -> None:
    df = build_dataset()
    print(f"골든셋 조인 결과: {len(df)}건 (claims ∩ mapping ∩ verdict)")

    rows_out = []
    skipped_no_value = 0
    failed = []

    for _, row in df.iterrows():
        claim_unit = row.get("claim_numeric_unit") if isinstance(row.get("claim_numeric_unit"), str) else None
        parsed = _parse_kosis_value(row.get("kosis_value"), claim_unit)
        if parsed is None:
            skipped_no_value += 1
            continue

        computed_value, computed_unit, computed_period = parsed
        claim_type = _infer_claim_type(row.get("claim_comparison_target"), claim_unit)
        calc_type = _infer_calc_type(computed_unit)

        claim = Claim(
            sentence=row["claim_sentence"],
            claim_type=claim_type,
            period=str(row.get("claim_period")) if pd.notna(row.get("claim_period")) else None,
            unit=claim_unit,
            value=_to_optional_float(row.get("claim_numeric_value")),
        )
        computed = ComputedResult(
            calc_type=calc_type,
            raw_value=computed_value,
            unit=computed_unit,
            period=computed_period or claim.period or "",
        )

        gold = row["verification_result"]
        try:
            verdict = judge(claim, computed)
        except JudgeError as e:
            print(f"[FAIL] {row['claim_id']} -> {e}")
            failed.append(row["claim_id"])
            continue

        # 골든셋은 "확인불가"라는 단어를 쓰고 judge.py의 Verdict는 같은 개념을
        # "판단불가"라고 부른다 — 서로 다른 값이 아니라 용어 차이이므로 동치로 채점한다.
        pred_normalized = "확인불가" if verdict.verdict == "판단불가" else verdict.verdict
        ok = pred_normalized == gold
        rows_out.append(
            {
                "claim_id": row["claim_id"],
                "gold": gold,
                "pred": verdict.verdict,
                "ok": ok,
                "claim_sentence": row["claim_sentence"][:50],
                "claim_value": claim.value,
                "claim_type(추정)": claim_type,
                "computed_value": computed_value,
                "computed_unit": computed_unit,
                "reason": verdict.reason[:80],
            }
        )
        tag = "OK" if ok else "MISMATCH"
        print(f"[{tag}] {row['claim_id']} gold={gold} pred={verdict.verdict} | {row['claim_sentence'][:40]}")

    result_df = pd.DataFrame(rows_out)
    n_total = len(result_df)
    n_correct = result_df["ok"].sum() if n_total else 0

    print(f"\n=== 최종 요약 ===")
    print(f"KOSIS 값 없어서 제외(확인불가 대상): {skipped_no_value}건")
    print(f"judge() 호출 실패: {len(failed)}건")
    print(f"채점 대상: {n_total}건, 정답: {n_correct}건 ({n_correct/n_total:.1%})" if n_total else "채점 대상 없음")

    if n_total:
        print("\n[혼동행렬]")
        confusion = pd.crosstab(result_df["gold"], result_df["pred"], rownames=["정답"], colnames=["예측"])
        print(confusion)

    _write_report(result_df, skipped_no_value, failed)
    print(f"\n리포트 -> {REPORT_PATH}")


def _write_report(df: pd.DataFrame, skipped_no_value: int, failed: list) -> None:
    lines = ["# 7단계(judge.py) 판정 골든셋 채점 리포트\n\n"]
    lines.append(
        "⚠️ Claim/ComputedResult를 골든셋 텍스트에서 근사치로 파싱했습니다 — 정밀한 수치가 "
        "아니라 대략적인 로직 검증용입니다. 상세 근거는 스크립트 docstring 참고.\n\n"
    )
    lines.append(f"- KOSIS 값 파싱 불가로 제외(대부분 골든셋 정답=확인불가): {skipped_no_value}건\n")
    lines.append(f"- judge() 호출 실패: {len(failed)}건\n")

    if len(df):
        n_correct = df["ok"].sum()
        lines.append(f"- **채점 대상 {len(df)}건 중 정답 {n_correct}건 ({n_correct/len(df):.1%})**\n\n")
        lines.append("## claim별 상세\n\n")
        lines.append(df.to_markdown(index=False))
        lines.append("\n")

    REPORT_PATH.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
