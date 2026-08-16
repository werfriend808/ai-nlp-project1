"""
agent/kosis/prepare_vdb_export.py — 크롤링한 KOSIS 표 28만여 개를 코랩 임베딩용으로 정리한다.

배경: crawl_table_catalog.py가 만든 data/../agent/kosis/crawl_output/tables.jsonl에는
표 ID/이름/기관코드 같은 메타데이터만 있고 임베딩(벡터)은 없다. 28만여 개를 임베딩하는
건 로컬(RAM 7.4GB)에서는 무리라서, 기존 리랭커/임베딩 작업과 같은 방식으로 코랩에
넘겨서 처리한다 — 이 스크립트는 그 "넘길 파일"을 준비하는 역할만 한다.

지금은 표 이름(TBL_NM)만 임베딩 텍스트로 쓴다 — 64개 카탈로그처럼 풍부한 설명
(embedding_text)을 붙이려면 표 하나당 kosis_meta를 또 호출해야 하는데, 28만 번
추가 호출은 부담이 커서 1차로는 표 이름만으로 시작한다. 나중에 매칭 품질이 부족하면
그때 설명 보강을 검토한다.

사용법 (프로젝트 루트에서):
    python -m agent.kosis.prepare_vdb_export
    -> data/vdb_pending.jsonl 생성 (구글 드라이브에 업로드해서 코랩 노트북에서 사용)
"""

from __future__ import annotations

import json
from pathlib import Path

CRAWL_OUTPUT_PATH = Path(__file__).parent / "crawl_output" / "tables.jsonl"
EXPORT_PATH = Path(__file__).parent.parent.parent / "data" / "vdb_pending.jsonl"


def main() -> None:
    if not CRAWL_OUTPUT_PATH.exists():
        raise SystemExit(f"{CRAWL_OUTPUT_PATH}가 없습니다. crawl_table_catalog.py를 먼저 실행하세요.")

    seen_tbl_ids: set[str] = set()
    written = 0
    skipped_no_name = 0

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CRAWL_OUTPUT_PATH, encoding="utf-8") as in_f, open(
        EXPORT_PATH, "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tbl_id = row.get("TBL_ID")
            tbl_nm = row.get("TBL_NM")
            if not tbl_id or tbl_id in seen_tbl_ids:
                continue
            seen_tbl_ids.add(tbl_id)
            if not tbl_nm:
                skipped_no_name += 1
                continue

            out_f.write(
                json.dumps(
                    {
                        "tbl_id": tbl_id,
                        "org_id": row.get("ORG_ID"),
                        "text": tbl_nm,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"[export] 고유 표 {len(seen_tbl_ids)}개 중 {written}개를 {EXPORT_PATH}에 저장했습니다.")
    if skipped_no_name:
        print(f"  (표 이름이 없어서 제외된 것: {skipped_no_name}개)")
    print("다음: 이 파일을 구글 드라이브에 올리고 코랩 노트북에서 임베딩을 생성하세요.")


if __name__ == "__main__":
    main()
