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

2026-08-18: 골든셋으로 VDB 실측 평가해보니 Recall@5가 0%가 나왔는데, 원인은 표 이름
품질이 아니라 **언어**였다 — 크롤링 원본(tables.jsonl)에는 같은 표(TBL_ID)가 VW_CD별로
여러 행 중복 등장하는데(예: MT_OTITLE=한글 원제, MT_ETITLE=영문 제목), 예전엔 "파일에
먼저 나온 행을 그냥 채택"하는 방식이었다. 하필 인구/실업률/CPI처럼 뉴스에 자주 인용되는
주요 표들이 영문(MT_ETITLE) 행을 먼저 만나서 그대로 채택됐고, 그 결과 한국어 뉴스
claim으로 쿼리하는데 VDB엔 영어 제목이 저장돼 있어 유사도가 크게 떨어졌다(골든셋 정답
15개 중 10개, 67%가 영문으로 저장돼 있었음 — 실측 확인). VW_CD 전수 조사 결과 MT_ETITLE만
영문이고 나머지(MT_OTITLE/MT_STOP_TITLE/MT_HANKUK_TITLE/MT_CHOSUN_TITLE/MT_TM1_TITLE/
MT_TM2_TITLE/MT_BUKHAN/MT_RTITLE/MT_GTITLE01/02)는 전부 한글이었다. 그래서 같은 TBL_ID가
중복되면 MT_OTITLE(한글 원제, 26만2천 건으로 가장 많은 기본 뷰)을 최우선으로 채택하고,
MT_ETITLE(영문)만 맨 뒤로 미룬다 — 그 사이 값들은 전부 한글이라 우선순위를 세밀하게
가릴 필요가 없다.

사용법 (프로젝트 루트에서):
    python -m agent.kosis.prepare_vdb_export
    -> data/vdb_pending.jsonl 생성 (구글 드라이브에 업로드해서 코랩 노트북에서 사용)
"""

from __future__ import annotations

import json
from pathlib import Path

CRAWL_OUTPUT_PATH = Path(__file__).parent / "crawl_output" / "tables.jsonl"
EXPORT_PATH = Path(__file__).parent.parent.parent / "data" / "vdb_pending.jsonl"

_PREFERRED_VW_CD = "MT_OTITLE"  # 한글 원제(기본 뷰) — 있으면 무조건 최우선
_LOWEST_PRIORITY_VW_CD = "MT_ETITLE"  # 영문 제목 — 다른 대안이 없을 때만 마지막 수단


def _vw_cd_priority(vw_cd: object) -> int:
    """작을수록 우선순위가 높다. MT_OTITLE(한글 원제) 최우선, MT_ETITLE(영문) 최후순위,
    그 외(전부 한글 계열로 실측 확인됨)는 중간."""
    if vw_cd == _PREFERRED_VW_CD:
        return 0
    if vw_cd == _LOWEST_PRIORITY_VW_CD:
        return 2
    return 1


def main() -> None:
    if not CRAWL_OUTPUT_PATH.exists():
        raise SystemExit(f"{CRAWL_OUTPUT_PATH}가 없습니다. crawl_table_catalog.py를 먼저 실행하세요.")

    best_rows: dict[str, dict] = {}
    best_priority: dict[str, int] = {}
    total_rows = 0

    with open(CRAWL_OUTPUT_PATH, encoding="utf-8") as in_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            total_rows += 1
            tbl_id = row.get("TBL_ID")
            if not tbl_id:
                continue
            priority = _vw_cd_priority(row.get("VW_CD"))
            if tbl_id not in best_rows or priority < best_priority[tbl_id]:
                best_rows[tbl_id] = row
                best_priority[tbl_id] = priority

    written = 0
    skipped_no_name = 0
    demoted_english = 0

    EXPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(EXPORT_PATH, "w", encoding="utf-8") as out_f:
        for tbl_id, row in best_rows.items():
            tbl_nm = row.get("TBL_NM")
            if not tbl_nm:
                skipped_no_name += 1
                continue
            if row.get("VW_CD") == _LOWEST_PRIORITY_VW_CD:
                # 이 tbl_id는 한글 버전이 크롤링 원본에 아예 없어서 영문으로 남은 경우 —
                # 조회는 되지만 한국어 claim과의 유사도가 떨어질 수 있음을 알 수 있게 기록.
                demoted_english += 1

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

    print(f"[export] 크롤링 원본 {total_rows}행(중복 포함) 중 고유 표 {len(best_rows)}개, "
          f"그중 {written}개를 {EXPORT_PATH}에 저장했습니다.")
    if skipped_no_name:
        print(f"  (표 이름이 없어서 제외된 것: {skipped_no_name}개)")
    if demoted_english:
        print(f"  (한글 버전이 원본에 없어서 영문 제목 그대로 남은 표: {demoted_english}개)")
    print("다음: 이 파일을 구글 드라이브에 올리고 코랩 노트북에서 임베딩을 생성하세요.")


if __name__ == "__main__":
    main()
