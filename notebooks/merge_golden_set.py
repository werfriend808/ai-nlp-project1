"""
notebooks/merge_golden_set.py — 골든셋 병합 + 무결성 체크 + (필요시) verdict 유도

원래는 추출/매핑/판정 A조+B조 6개 파일을 병합하는 스크립트였는데, 판정(verdict) 골든셋과
B조(신영) 파일들이 깃허브에 커밋된 적 없는 로컬 전용 파일이라 유실된 상태다(2026-08-03).
지금 남아있는 건 A조 추출·매핑(89건, 깃허브에 있음)뿐이라, 이 스크립트는 두 경우를 다
처리한다:
  - 판정 골든셋 xlsx가 있으면: 예전처럼 그걸 정답으로 그대로 씀
  - 없으면: 매핑 골든셋의 match_status(+note에 적힌 "오차 X%" 실측값)에서
    verification_result를 유도한다. 사람이 최종적으로 일치/불일치를 명시적으로 라벨링한
    건 아니라서 판정 골든셋보다 신뢰도가 약간 낮다 — 아래 _derive_verdict()의 규칙 참고.

사용법 (프로젝트 루트에서):
    python -m notebooks.merge_golden_set
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

NOTEBOOKS_DIR = Path(__file__).parent

CLAIMS_A = NOTEBOOKS_DIR / "추출 골든셋 단위 분리.xlsx"
CLAIMS_B = NOTEBOOKS_DIR / "추출 골든셋(신영-단위분리).xlsx"
MAPPING_A = NOTEBOOKS_DIR / "매핑 골든셋 ord 추가.xlsx"
MAPPING_B = NOTEBOOKS_DIR / "매핑 골든셋(신영).xlsx"
VERDICT_A = NOTEBOOKS_DIR / "판정 골든셋.xlsx"
VERDICT_B = NOTEBOOKS_DIR / "판정_골든셋_신영.xlsx"

CLAIMS_OUT = NOTEBOOKS_DIR / "claims_golden_merged.csv"
MAPPING_OUT = NOTEBOOKS_DIR / "mapping_golden_merged.csv"
VERDICT_OUT = NOTEBOOKS_DIR / "verdict_golden_merged.csv"
ISSUES_OUT = NOTEBOOKS_DIR / "golden_merge_issues.md"

DATA_CSV = NOTEBOOKS_DIR.parent / "data" / "data.csv"
DATA_GOLDEN_OUT = NOTEBOOKS_DIR.parent / "data" / "data_set_golden.csv"

# match_status -> verification_result 직접 매핑 (사람이 이미 일치/불일치를 판단한 것들)
_MATCH_STATUS_DIRECT = {
    "값 확인됨 (일치)": "일치",
    "값 확인됨 (근접, 일치)": "일치",
    "값 확인됨 (불일치)": "불일치",
    "값 확인됨 (근접, 불일치)": "불일치",
    "값 확인됨 (유사, 완전일치 아님)": "불일치",
}
_ERROR_PCT_THRESHOLD = 2.0  # note의 "오차 X%"가 이 이하면 일치로 유도
_ERROR_PCT_RE = re.compile(r"오차\s*([\d.]+)\s*%")


def _derive_verdict(match_status: object, note: object) -> str:
    """match_status를 verification_result로 유도. "값 확인됨(자동 API 검증)"은 note에
    적힌 실측 오차(%)로 일치/불일치를 가른다. 나머지(매칭 실패/표 후보 확인/미완료)는
    KOSIS 값 자체가 불명확하거나 없는 경우라 확인불가로 둔다."""
    status = str(match_status)
    if status in _MATCH_STATUS_DIRECT:
        return _MATCH_STATUS_DIRECT[status]
    if status == "값 확인됨(자동 API 검증)":
        m = _ERROR_PCT_RE.search(str(note))
        if m:
            return "일치" if float(m.group(1)) <= _ERROR_PCT_THRESHOLD else "불일치"
        return "일치" if "일치" in str(note) else "확인불가"
    return "확인불가"  # 매칭 실패 / 표 후보 확인(*) / 미완료


def _load(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def _concat_if_exists(a_path: Path, b_path: Path) -> tuple[pd.DataFrame, bool]:
    """b_path가 있으면 A+B 병합, 없으면 A만. (있었는지 여부)도 같이 반환."""
    a = _load(a_path)
    if b_path.exists():
        return pd.concat([a, _load(b_path)], ignore_index=True), True
    return a, False


def merge_claims() -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    merged, had_b = _concat_if_exists(CLAIMS_A, CLAIMS_B)
    empty_mask = merged["claim_sentence"].isna() | (merged["claim_sentence"].astype(str).str.strip() == "")
    return merged[~empty_mask].copy(), merged[empty_mask].copy(), had_b


def merge_mapping(valid_claim_ids: set) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    merged, had_b = _concat_if_exists(MAPPING_A, MAPPING_B)
    bad_mask = ~merged["claim_id"].isin(valid_claim_ids)
    return merged[~bad_mask].copy(), merged[bad_mask].copy(), had_b


def build_verdict(valid_claim_ids: set, mapping_valid: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """판정 골든셋이 있으면 그대로 쓰고, 없으면 매핑 골든셋에서 유도한다."""
    if VERDICT_A.exists():
        merged, _ = _concat_if_exists(VERDICT_A, VERDICT_B)
        if "result_id" in merged.columns:
            merged = merged.drop(columns=["result_id"])
        merged = merged[merged["claim_id"].isin(valid_claim_ids)].copy()
        return merged, "판정 골든셋 원본 사용"

    derived = mapping_valid[["claim_id", "match_status", "note"]].copy()
    derived["verification_result"] = derived.apply(
        lambda r: _derive_verdict(r["match_status"], r["note"]), axis=1
    )
    derived["mismatch_type"] = None
    derived["judge_reasoning"] = derived["note"]
    return derived[["claim_id", "verification_result", "mismatch_type", "judge_reasoning"]], (
        "판정 골든셋 없음 -> 매핑 골든셋 match_status/note에서 유도(사람이 직접 매긴 라벨 아님)"
    )


def _normalize_url(url: object) -> str:
    return str(url).strip().rstrip("/")


def build_data_set_golden(claims_valid: pd.DataFrame) -> tuple[int, int]:
    golden_urls = {_normalize_url(u) for u in claims_valid["article_url"].dropna().unique()}
    data = pd.read_csv(DATA_CSV)
    data["_url_norm"] = data["URL"].map(_normalize_url)
    matched = data[data["_url_norm"].isin(golden_urls)].drop(columns=["_url_norm"])
    matched.to_csv(DATA_GOLDEN_OUT, index=False, encoding="utf-8-sig")
    return len(golden_urls), len(matched)


def _integrity_report(claims: pd.DataFrame, mapping: pd.DataFrame, verdict: pd.DataFrame) -> str:
    claim_ids, mapping_ids, verdict_ids = set(claims["claim_id"]), set(mapping["claim_id"]), set(verdict["claim_id"])

    def fmt(ids: set) -> str:
        return ", ".join(sorted(ids)) if ids else "없음"

    return (
        "## 무결성 체크 (claim_id 기준)\n\n"
        f"- 추출 {len(claim_ids)}건 / 매핑 {len(mapping_ids)}건 / 판정(또는 유도) {len(verdict_ids)}건\n"
        f"- 추출엔 있는데 매핑에 없음: {fmt(claim_ids - mapping_ids)}\n"
        f"- 추출엔 있는데 판정에 없음: {fmt(claim_ids - verdict_ids)}\n"
    )


def main() -> None:
    claims_valid, claims_dropped, claims_had_b = merge_claims()
    valid_claim_ids = set(claims_valid["claim_id"])

    mapping_valid, mapping_dropped, mapping_had_b = merge_mapping(valid_claim_ids)
    verdict_valid, verdict_source = build_verdict(valid_claim_ids, mapping_valid)

    claims_valid.to_csv(CLAIMS_OUT, index=False, encoding="utf-8-sig")
    mapping_valid.to_csv(MAPPING_OUT, index=False, encoding="utf-8-sig")
    verdict_valid.to_csv(VERDICT_OUT, index=False, encoding="utf-8-sig")

    n_golden_urls, n_matched = build_data_set_golden(claims_valid)

    parts = ["# 골든셋 병합 이슈 리포트\n\n"]
    parts.append(f"- B조(신영) 파일 존재 여부: 추출={claims_had_b}, 매핑={mapping_had_b}\n")
    parts.append(f"- 판정(verdict) 소스: {verdict_source}\n\n")

    parts.append(f"## 추출 — claim_sentence 빈 행 {len(claims_dropped)}건 (제외됨)\n\n")
    if len(claims_dropped):
        parts.append(claims_dropped[["claim_id", "article_id", "article_title"]].to_markdown(index=False))
        parts.append("\n\n")

    parts.append(f"## 매핑 — claim_id가 추출 병합본에 없는 행 {len(mapping_dropped)}건 (제외됨)\n\n")
    if len(mapping_dropped):
        cols = [c for c in ["claim_id", "kosis_table_id", "match_status"] if c in mapping_dropped.columns]
        parts.append(mapping_dropped[cols].to_markdown(index=False))
        parts.append("\n\n")

    if verdict_source.startswith("판정 골든셋 없음"):
        parts.append("## 유도된 verification_result 분포\n\n")
        parts.append(verdict_valid["verification_result"].value_counts().to_markdown())
        parts.append("\n\n")

    parts.append(
        f"## data_set_golden.csv\n\n"
        f"- 골든셋 고유 기사 URL: {n_golden_urls}건\n"
        f"- data.csv에서 실제로 매칭된 기사: {n_matched}건\n\n"
    )

    parts.append(_integrity_report(claims_valid, mapping_valid, verdict_valid))
    ISSUES_OUT.write_text("".join(parts), encoding="utf-8")

    print(f"[merge] 추출 {len(claims_valid)}건 (빈 문장 {len(claims_dropped)}건 제외) -> {CLAIMS_OUT.name}")
    print(f"[merge] 매핑 {len(mapping_valid)}건 ({len(mapping_dropped)}건 제외) -> {MAPPING_OUT.name}")
    print(f"[merge] 판정 {len(verdict_valid)}건 [{verdict_source}] -> {VERDICT_OUT.name}")
    print(f"[merge] data_set_golden.csv: 골든셋 URL {n_golden_urls}건 중 {n_matched}건 매칭 -> {DATA_GOLDEN_OUT.name}")
    print(f"[merge] 이슈 리포트 -> {ISSUES_OUT.name}")


if __name__ == "__main__":
    main()
