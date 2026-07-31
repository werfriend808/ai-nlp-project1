"""
agent/mapping/validate_catalog.py — table_catalog.json 품질 검증기

카탈로그가 53개 -> 100개 이상으로 늘어나도 사람이 매번 눈으로 훑지 않고
구조적 문제를 잡아내기 위한 검증기. CATALOG_GUIDELINE.md의 규칙을 코드로 옮긴 것.

체크 두 종류:
  - ERROR   : 명백한 결함, 즉시 수정 필요 (tblId 중복, catalog<->params 무결성, 필수 필드 누락)
  - WARNING : 사람이 판단해야 하는 후보 (keyword 충돌, embedding_text 불일치, 신규 category)
              특히 keyword substring 충돌은 자동으로 틀렸다고 판정하지 않는다 — 실제로
              같은 표 계열(전입/전입률 등)이 의도적으로 겹치는 경우가 더 많기 때문에,
              related_tblId로 이미 문서화됐는지 여부만 같이 보여주고 최종 판단은 사람이 한다.

실행:
    python -m agent.mapping.validate_catalog
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CATALOG_PATH = Path(__file__).parent / "table_catalog.json"
PARAMS_PATH = Path(__file__).parent.parent / "kosis" / "table_params.json"
CATEGORIES_PATH = Path(__file__).parent / "categories.json"

REQUIRED_FIELDS = ["tblId", "title", "category", "keywords", "description", "embedding_text"]


def _load_catalog(path: Path = CATALOG_PATH) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["tables"]


def _load_params(path: Path = PARAMS_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} 가 없습니다.")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_categories(path: Path = CATEGORIES_PATH) -> list[str]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("categories", [])


# ---------------------------------------------------------------------------
# ERROR 체크
# ---------------------------------------------------------------------------


def check_duplicate_tblid(tables: list[dict]) -> list[str]:
    errors = []
    seen: dict[str, int] = {}
    for e in tables:
        tid = e.get("tblId")
        seen[tid] = seen.get(tid, 0) + 1
    for tid, count in seen.items():
        if count > 1:
            errors.append(f"tblId 중복: '{tid}' 가 catalog에 {count}번 등장")
    return errors


def check_catalog_params_integrity(tables: list[dict], params: dict) -> list[str]:
    errors = []
    cat_ids = {e["tblId"] for e in tables}
    par_ids = set(params.keys())
    for tid in sorted(cat_ids - par_ids):
        errors.append(f"catalog에는 있지만 params 없음: '{tid}'")
    for tid in sorted(par_ids - cat_ids):
        errors.append(f"params에는 있지만 catalog 없음: '{tid}'")
    return errors


def check_required_fields(tables: list[dict]) -> list[str]:
    errors = []
    for e in tables:
        tid = e.get("tblId", "(tblId 없음)")
        missing = [f for f in REQUIRED_FIELDS if f not in e or e[f] in (None, "", [])]
        if missing:
            errors.append(f"{tid}: 필수 필드 누락 {missing}")
    return errors


# ---------------------------------------------------------------------------
# WARNING 체크
# ---------------------------------------------------------------------------


def check_keyword_exact_collision(tables: list[dict]) -> list[str]:
    warnings = []
    kw_to_entries: dict[str, list[dict]] = {}
    for e in tables:
        for kw in e.get("keywords", []):
            kw_to_entries.setdefault(kw, []).append(e)

    for kw, entries in sorted(kw_to_entries.items()):
        if len(entries) <= 1:
            continue
        lines = [f"[WARNING] keyword exact collision: '{kw}'"]
        for e in entries:
            lines.append(f"  - {e['tblId']}  ({e.get('title', '')})")
        warnings.append("\n".join(lines))
    return warnings


def check_keyword_substring_collision(tables: list[dict]) -> list[str]:
    warnings = []
    pairs = [(kw, e) for e in tables for kw in e.get("keywords", [])]
    seen_pairs: set[tuple] = set()

    for kw1, e1 in pairs:
        for kw2, e2 in pairs:
            if e1["tblId"] == e2["tblId"] or kw1 == kw2:
                continue
            if kw1 not in kw2 or len(kw1) < 2:
                continue
            key = tuple(sorted([kw1, kw2])) + tuple(sorted([e1["tblId"], e2["tblId"]]))
            if key in seen_pairs:
                continue
            seen_pairs.add(key)

            documented = e2["tblId"] in e1.get("related_tblId", {}) or e1["tblId"] in e2.get(
                "related_tblId", {}
            )
            warnings.append(
                "[WARNING] keyword substring candidate\n"
                f"\n{e1['tblId']}\nkeyword: {kw1}\n\ncontains in:\n\n"
                f"{e2['tblId']}\nkeyword: {kw2}\n\n"
                f"related_tblId 문서화 여부: {'YES' if documented else 'NO'}"
            )
    return warnings


def check_embedding_consistency(tables: list[dict]) -> list[str]:
    warnings = []
    for e in tables:
        emb = e.get("embedding_text", "")
        title = e.get("title", "")
        missing = []
        if title and title not in emb:
            missing.append(f"title: '{title}'")
        for kw in e.get("keywords", []):
            if kw not in emb:
                missing.append(kw)
        if missing:
            lines = ["[WARNING] embedding_text mismatch", f"\ntblId: {e['tblId']}", "missing:"]
            lines += [f"- {m}" for m in missing]
            warnings.append("\n".join(lines))
    return warnings


def check_category_vocabulary(tables: list[dict]) -> list[str]:
    warnings = []
    known_categories = set(_load_categories())
    used_categories = sorted(set(e.get("category", "") for e in tables))

    if known_categories:
        unknown = [c for c in used_categories if c not in known_categories]
        if unknown:
            warnings.append(
                "[WARNING] categories.json에 없는 신규 category 발견\n"
                + "\n".join(f"- {c}" for c in unknown)
            )

    warnings.append(
        "[WARNING] 현재 사용 중인 전체 category 목록 (사람이 훑어보고 판단)\n"
        + "\n".join(f"- {c}" for c in used_categories)
    )
    return warnings


# ---------------------------------------------------------------------------
# 실행/출력
# ---------------------------------------------------------------------------


def run_validation() -> int:
    tables = _load_catalog()
    params = _load_params()

    errors: list[str] = []
    errors += check_duplicate_tblid(tables)
    errors += check_catalog_params_integrity(tables, params)
    errors += check_required_fields(tables)

    warnings: list[str] = []
    warnings += check_keyword_exact_collision(tables)
    warnings += check_keyword_substring_collision(tables)
    warnings += check_embedding_consistency(tables)
    warnings += check_category_vocabulary(tables)

    print("=" * 20)
    print("ERROR")
    print("=" * 20)
    if errors:
        for i, e in enumerate(errors, 1):
            print(f"[{i}] {e}")
            print()
    else:
        print("No ERROR")
    print()

    print("=" * 20)
    print("WARNING")
    print("=" * 20)
    if warnings:
        for i, w in enumerate(warnings, 1):
            print(f"[{i}]")
            print(w)
            print()
    else:
        print("No WARNING")
    print()

    print("Summary")
    print(f"ERROR count: {len(errors)}")
    print(f"WARNING count: {len(warnings)}")

    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(run_validation())