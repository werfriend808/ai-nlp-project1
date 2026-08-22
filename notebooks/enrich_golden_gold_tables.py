"""
notebooks/enrich_golden_gold_tables.py — 골든셋에 나오는 정답표(19개)만 먼저
분류축 항목명까지 VDB 임베딩 텍스트에 포함시켜 재색인하는 실험 스크립트.

배경: 3단계 매핑 검증(Recall@1 17.1%)에서 "자살률" claim이 정답표(DT_1B34E13,
"시군구/사망원인(50항목)/성/사망자수...")를 못 찾는 사례를 발견했다 — 이 표는 제목에
"자살"이라는 단어가 아예 없고, "자살"은 표 안의 사망원인 50개 분류 항목 중 하나일
뿐이다. VDB는 지금 표 "제목"만 임베딩해서, 분류 항목명으로 찾아야 하는 표는 구조적으로
못 찾는다.

28.7만 개 전체를 재색인하려면 표마다 KOSIS API를 따로 호출해야 해서 부담이 크다 —
그래서 우선 골든셋 70건에 나오는 정답표 19개만 detail_cache.get_table_detail()로
분류축 항목명을 가져와 원래 텍스트 뒤에 붙이고, 그 텍스트로 다시 임베딩해서
kosis_vdb_tables를 업데이트한다. 이후 verify_stage3_on_golden_merged.py를 다시
돌려서 Recall@K/MRR이 실제로 개선되는지 확인하는 게 목적이다.

⚠️ 이 스크립트는 대상 19개 tbl_id의 text/embedding 컬럼을 직접 UPDATE한다 — 되돌리고
싶으면 사전에 원본 text를 백업해둔다(스크립트가 자동으로 .bak.json에 저장).

사용법 (원격 GPU 서버에서, Supabase 연결 필요):
    python -m notebooks.enrich_golden_gold_tables
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import psycopg2
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv(usecwd=True))

from agent.kosis.detail_cache import get_table_detail, DetailCacheUnavailableError
from agent.kosis.enrich_objl import ObjlFetchError

GOLDEN_PATH = Path(__file__).parent / "골든셋_통합.xlsx"
BACKUP_PATH = Path(__file__).parent / "enrich_golden_gold_tables.bak.json"
TABLE_NAME = "kosis_vdb_tables"


def _unique_gold_ids() -> list[str]:
    df = pd.read_excel(GOLDEN_PATH, sheet_name="7단계_판정목록")
    ids = df["matched_table_id(3단계)"].astype(str).str.strip()
    ids = ids[ids != "없음"]
    flat: list[str] = []
    for v in ids:
        flat.extend(x.strip() for x in v.split(","))
    return sorted(set(flat))


def main() -> None:
    conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
    cur = conn.cursor()

    tbl_ids = _unique_gold_ids()
    print(f"대상 표 {len(tbl_ids)}개: {tbl_ids}")

    cur.execute(
        f"select tbl_id, org_id, text from {TABLE_NAME} where tbl_id = ANY(%s)", (tbl_ids,)
    )
    rows = {r[0]: {"org_id": r[1], "text": r[2]} for r in cur.fetchall()}
    missing = [t for t in tbl_ids if t not in rows]
    if missing:
        print(f"[경고] VDB에 없는 tbl_id (건너뜀): {missing}")

    backup = {}
    enriched: dict[str, str] = {}
    for tbl_id in tbl_ids:
        if tbl_id not in rows:
            continue
        org_id = rows[tbl_id]["org_id"]
        original_text = rows[tbl_id]["text"]
        backup[tbl_id] = original_text

        try:
            detail = get_table_detail(tbl_id, org_id)
        except DetailCacheUnavailableError as e:
            print(f"[{tbl_id}] detail_cache 연결 실패, 건너뜀: {e}")
            continue

        if detail["status"] != "ok" or not detail.get("code_maps"):
            print(f"[{tbl_id}] 상세정보 없음(status={detail['status']}), 원본 텍스트 유지")
            continue

        labels: list[str] = []
        for axis_name, label_to_code in detail["code_maps"].items():
            labels.extend(label_to_code.keys())
        # 너무 긴 축(예: 전국 시군구 250개)은 노이즈만 늘리므로 항목 수가 합리적인
        # 범위(<=80)일 때만 붙인다 — 오늘 발견한 "사망원인 50항목" 같은 케이스는 포함,
        # "전국 시군구별" 같은 지리 축은 원래도 다른 신호(region_rank)로 커버되니 제외.
        if not labels or len(labels) > 80:
            print(f"[{tbl_id}] 항목 {len(labels)}개(0 또는 80 초과) — 원본 텍스트 유지")
            continue

        unique_labels = list(dict.fromkeys(labels))  # 순서 보존 중복 제거
        enriched_text = original_text + " " + " ".join(unique_labels)
        enriched[tbl_id] = enriched_text
        print(f"[{tbl_id}] 항목 {len(unique_labels)}개 추가 -> 길이 {len(original_text)} -> {len(enriched_text)}")

    BACKUP_PATH.write_text(json.dumps(backup, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n원본 텍스트 백업 -> {BACKUP_PATH}")

    if not enriched:
        print("보강할 표가 없음 — 종료")
        return

    print(f"\n{len(enriched)}개 표 재임베딩 중...")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("Qwen/Qwen3-Embedding-4B", truncate_dim=1024)
    tbl_ids_list = list(enriched.keys())
    texts_list = [enriched[t] for t in tbl_ids_list]
    # notebooks/vdb_embedding_colab.ipynb와 동일 — 문서(passage) 쪽은 프리픽스 없이 그대로 인코딩.
    vectors = model.encode(texts_list, convert_to_numpy=True, normalize_embeddings=True)

    for tbl_id, text, vec in zip(tbl_ids_list, texts_list, vectors):
        cur.execute(
            f"update {TABLE_NAME} set text = %s, embedding = %s::vector where tbl_id = %s",
            (text, vec.tolist(), tbl_id),
        )
    conn.commit()
    print(f"완료: {len(enriched)}개 표 text/embedding 갱신함")


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    main()
