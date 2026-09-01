"""data/fill_golden_unverifiable_reasons.py — "판단불가" 프리셋 중 근거 텍스트가 비어있는
건을 실제 LLM(HCX)을 불러서 "왜 KOSIS로 확인이 안 되는지" 한 줄 설명을 채운다.

배경: notebooks/골든셋_통합.xlsx의 reason(판정 근거) 컬럼이 매칭 실패(matched_table_id
"없음") claim 중 상당수는 비어 있다 — 사람이 "표를 못 찾았다"는 사실 자체는 확인했지만
그 이유까지 매번 적어두진 않았기 때문. 화면에 판정 근거가 텅 비어 보이는 문제라, claim
문장·기사가 인용한 기관(source_org)·통계 표현(statistic_expression) 등 이미 가진 맥락을
근거로 LLM에게 "해외 통계", "KOSIS 미등재 통계(민간기업/협회 자체조사, 관세청/국세청 등
행정통계, 전망치·추정치 등)", "일회성 보도자료 수치" 중 실제로 맞는 이유를 스스로 판단해
한 문장으로 설명하게 한다 — 카테고리를 강제하지 않고 claim 내용을 보고 LLM이 직접 고르게
한다(하드코딩 금지 원칙).

사용법 (프로젝트 루트에서, 실제 HCX 호출 발생):
    python -m data.fill_golden_unverifiable_reasons
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.preprocessing.hcx_client import call_hcx  # noqa: E402
from db.store import DB_PATH  # noqa: E402

MODEL = "HCX-003"
SYSTEM_PROMPT = "아래 지시사항을 정확히 따르고, 반드시 지정된 JSON 형식으로만 응답하세요."

PROMPT_TEMPLATE = """다음은 뉴스 기사에서 뽑은 수치 주장인데, 국가데이터처(KOSIS, 옛 통계청)
공식 통계에서 대조할 표를 찾지 못해 "판단불가"로 처리됐다. 왜 KOSIS로 확인이 안 되는지
그 이유를 한 문장으로 설명하라.

기사 제목: {article_title}
주장 문장: "{claim_sentence}"
기사가 인용한 출처: {source_org}
주장이 다루는 통계: {statistic_expression}
시점: {time_expression}

가능한 이유 예시(실제로 해당하는 걸 판단해서 골라 쓰되, 다른 이유가 더 맞으면 그걸 써도 됨):
- 해외 기관/해외 통계라서 KOSIS(국내 국가승인통계)에 없음
- 국세청·관세청·예금보험공사 등 개별 행정기관이 자체 발표한 통계로, KOSIS에 등재된
  국가승인통계가 아님
- 민간 기업·협회·연구소가 자체 조사한 수치라서 KOSIS 미등재
- 전망치·추정치(예: 성장률 전망)라서 실적 통계가 아니어서 KOSIS에 대응하는 확정 표가 없음
- 보도자료로만 발표되고 정식 국가승인통계로 등재되지 않은 일회성 수치

출력 형식 (JSON만 출력, 다른 텍스트 금지):
{{"reason": "한 문장 설명 (예: '국세청이 자체 발표한 행정통계로 KOSIS 국가승인통계에는
등재되지 않음' 같은 톤)"}}"""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"JSON 못 찾음: {text!r}")
    return json.loads(m.group(0))


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT result_id, article_title, claim_sentence, statistic_expression, source_org, "
        "time_expression FROM verifications WHERE verification_result='판단불가' "
        "AND (evidence IS NULL OR evidence='') AND (ambiguity_reason IS NULL OR ambiguity_reason='')"
    ).fetchall()
    print(f"근거 없는 판단불가 {len(rows)}건")

    n_ok = n_fail = 0
    for r in rows:
        prompt = PROMPT_TEMPLATE.format(
            article_title=r["article_title"],
            claim_sentence=r["claim_sentence"],
            source_org=r["source_org"] or "명시 안 됨",
            statistic_expression=r["statistic_expression"] or "명시 안 됨",
            time_expression=r["time_expression"] or "명시 안 됨",
        )
        try:
            reply = call_hcx(model=MODEL, system_prompt=SYSTEM_PROMPT, user_content=prompt, temperature=0.0)
            reason = str(_extract_json(reply)["reason"]).strip()
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {r['result_id']}: {e}")
            n_fail += 1
            continue

        conn.execute(
            "UPDATE verifications SET evidence=?, ambiguity_reason=? WHERE result_id=?",
            (reason, reason, r["result_id"]),
        )
        n_ok += 1
        print(f"[OK] {r['claim_sentence'][:40]} -> {reason}")

    conn.commit()
    conn.close()
    print(f"\n총 {n_ok}건 채움, {n_fail}건 실패")


if __name__ == "__main__":
    main()
