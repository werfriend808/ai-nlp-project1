"""data/hide_article_from_list.py — 데모 시연용: 특정 기사를 "최근 검증한 기사" 목록/초기
export에서만 빼고(verifications.db 자체는 안 건드림), 그 URL을 라이브로 다시 제출하면
agent/api/server.py의 프리셋 단축 경로가 즉시 응답하면서 export를 다시 갱신해 목록에
다시 나타나게 한다 — "처음 보는 기사를 라이브로 처리하는 것처럼" 시연하기 위함.

⚠️ 이 스크립트가 만드는 "가려진" 상태는 이 실행 직후부터, 다음 번 아무 기사든 검증(라이브
또는 프리셋) 요청이 한 번이라도 들어오는 순간까지만 유지된다 — agent/api/server.py가
검증 완료 후 매번 DB 전체를 다시 export하기 때문(정상 동작). 시연 직전에 실행할 것.

사용법 (프로젝트 루트에서):
    python -m data.hide_article_from_list "<article_title>"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from db.export_json import (  # noqa: E402
    DEFAULT_ARTICLES_OUT_PATH,
    DEFAULT_DATES_OUT_PATH,
    DEFAULT_ORG_IDS_OUT_PATH,
    DEFAULT_OUT_PATH,
    export_article_dates,
    export_article_texts,
    export_table_org_ids,
    export_to_json,
)

FRONTEND_DATA_DIR = ROOT / "frontend" / "public" / "data"


def main(hidden_title: str) -> None:
    # 1) 평소처럼 DB 전체를 export(articles.json/articleDates.json/tableOrgIds.json은
    #    그대로 둬도 무방 — "최근 검증한 기사" 목록은 verifications.json만 그룹핑해서
    #    만들어지므로, 이 파일에서만 빼면 목록에서 사라진다).
    export_to_json(DEFAULT_OUT_PATH)
    export_article_texts(DEFAULT_ARTICLES_OUT_PATH)
    export_article_dates(DEFAULT_DATES_OUT_PATH)
    export_table_org_ids(DEFAULT_ORG_IDS_OUT_PATH)

    # 2) verifications_export.json에서만 해당 기사 레코드를 걸러내고 다시 저장.
    rows = json.loads(DEFAULT_OUT_PATH.read_text(encoding="utf-8"))
    before = len(rows)
    filtered = [r for r in rows if r.get("article_title") != hidden_title]
    removed = before - len(filtered)
    if removed == 0:
        print(f"[경고] '{hidden_title}' 제목과 일치하는 레코드가 없습니다 — 철자 확인 필요.")
        return
    DEFAULT_OUT_PATH.write_text(json.dumps(filtered, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) frontend/public/data로 복사(agent/api/server.py의 _refresh_frontend_exports와
    #    동일한 매핑) — verifications.json만 걸러진 버전으로, 나머지는 원본 그대로.
    mapping = {
        DEFAULT_OUT_PATH: "verifications.json",
        DEFAULT_ARTICLES_OUT_PATH: "articles.json",
        DEFAULT_DATES_OUT_PATH: "articleDates.json",
        DEFAULT_ORG_IDS_OUT_PATH: "tableOrgIds.json",
    }
    for src, dst_name in mapping.items():
        (FRONTEND_DATA_DIR / dst_name).write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"'{hidden_title}' — {removed}건 숨김, 목록엔 {len(filtered)}건 남음.")
    print("DB(verifications.db)는 그대로라 이 기사 URL을 다시 제출하면 즉시 프리셋으로 응답하고,")
    print("응답 직후 서버가 export를 다시 갱신해서 목록에 다시 나타납니다.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('사용법: python -m data.hide_article_from_list "<article_title>"')
        sys.exit(1)
    main(sys.argv[1])
