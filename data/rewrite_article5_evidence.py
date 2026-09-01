"""data/rewrite_article5_evidence.py — 시현 영상 대상 기사("그냥 쉬었다는 20대...")의 claim
24건 evidence를 전부 자연스러운 판정문으로 다시 쓴다.

이전 rewrite_golden_evidence_as_llm.py는 "직접기록"(verification_possible='불가능') 46건만
대상으로 했는데, 실제로는 "가능"(judge() 규칙 기반 판정) 쪽도 "기사 수치(X)와 통계
계산값(Y) 차이가 허용 오차 이내... (규칙 기반 판정, LLM 미호출)"이라는 같은 문장이 숫자만
바뀐 채 15번 넘게 반복되어 데모에서 보면 기계적으로 보인다는 지적(2026-09-01)이 나왔다.
이 스크립트는 이 기사 하나(24건)에 한해 규칙기반/직접기록 둘 다 자연스럽게 다시 쓴다 —
사실(숫자·표·시점·최종 판정)은 그대로 두고 문장만 다양하게.

사용법 (프로젝트 루트에서, 실제 HCX 호출 발생):
    python -m data.rewrite_article5_evidence
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
from agent.verdict.judge import _find_compound_numbers  # noqa: E402
from db.store import DB_PATH  # noqa: E402

MODEL = "HCX-003"
SYSTEM_PROMPT = "아래 지시사항을 정확히 따르고, 반드시 지정된 JSON 형식으로만 응답하세요."

ARTICLE_TITLE = "“그냥 쉬었다”는 20대, 5년 새 가장 많아"

STYLE_EXAMPLES = """예시1: "기사 주장과 통계 계산값 모두 작년(2023년)을 기준으로 GDP 대비 가계신용 비율의 하락을 말하고 있고, 그 비율도 0.8%로 같아 실질적으로 같은 내용을 전한다."
예시2: "기사(1.7%)와 통계(1.6%)는 같은 연간 기준으로 비교 가능한데도 값이 다르고, 0.1%p 차이도 반올림으로 보기 어려워 불일치로 판단한다."
예시3: "통계표의 지표('1인당 연간 양곡소비량')와 기사 주장의 지표('1인당 기타양곡 소비량')는 서로 다른 개념이지만, 주장에서의 소비량 규모(8.6kg)와 통계 계산값(8.6kg)이 정확히 일치한다.\""""

PROMPT_TEMPLATE = """아래는 통계 팩트체크 시스템이 "판정 근거"로 남긴 메모다. 규칙 기반 판정은
"기사 수치(X)와 통계 계산값(Y) 차이가 허용 오차 이내... (규칙 기반 판정, LLM 미호출)" 같은
정형 문장이 반복되고, 일부는 표 ID·탭으로 나열된 원자료가 그대로 섞여 있다. 실제 사람이
분석 후 설명하듯 자연스러운 한국어 1~2문장으로 다시 써라.

절대 규칙:
- 판정 결과({verdict})는 이미 확정된 사실이다 — 절대 바꾸지 말고 그 결론을 그대로 설명하라.
- 원본에 있는 숫자·비교 대상·시점은 빠짐없이 반영하되, 새로운 숫자나 근거를 지어내지 마라.
- "(규칙 기반 판정, LLM 미호출)"이나 "KOSIS: DT_xxx(표명)" 같은 내부 표기·표 ID 나열은
  자연스러운 문장으로 풀어써라. 탭으로 구분된 숫자 목록은 그대로 나열하지 말고 요약하라
  (예: 12개월 숫자를 다 나열하지 말고 "12개월 중 4개월은 오히려 전월보다 높아"처럼).
- "허용 오차", "규칙 기반" 같은 시스템 내부 용어를 그대로 노출하지 말 것.
- 매번 표현을 다르게 써서 24건이 전부 똑같은 패턴으로 안 보이게 하라.
- 1인칭·해요체 쓰지 말고 담백한 설명체로.

문체 예시 (내용은 무관, 톤만 참고):
{style_examples}

기사 주장: "{claim_sentence}"
판정 결과: {verdict}
원본 판정 근거 메모: {raw_reason}

출력 형식 (JSON만 출력, 다른 텍스트 금지):
{{"reason": "다시 쓴 판정 근거 1~2문장"}}"""

_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def _extract_numbers(text: str) -> set[float]:
    """단순 자릿수 숫자("29091000")와 한글 복합 표기("2909만 1000", "9만7000")를 둘 다
    뽑는다. 원본 근거는 "29091000.0"처럼 순수 숫자로 적혀 있는데 LLM이 다시 쓰면서
    기사 스타일대로 "2909만 1000명"으로 바꿔 쓰는 경우가 실제로 자주 나왔다(2026-09-01
    실측) — 순수 자릿수 정규식만 쓰면 "2909"/"1000"을 서로 다른 별개의 작은 숫자로
    잘못 인식해서, 사실은 정확히 같은 값(29,091,000)인데 "숫자가 사라졌다"고 오판해
    멀쩡한 rewrite까지 거부하는 버그가 있었다. judge.py의 한글 복합 숫자 파서
    (_find_compound_numbers)를 그대로 재사용해서 두 표기 방식 모두에서 실제 값을 뽑는다."""
    nums = set()
    for m in _NUMBER_RE.finditer(text):
        try:
            v = float(m.group(0).replace(",", ""))
        except ValueError:
            continue
        if v >= 4:
            nums.add(v)
    for v, _start, _end in _find_compound_numbers(text):
        if v >= 4:
            nums.add(v)
    return nums


def _numbers_preserved(original: str, rewritten: str) -> bool:
    orig_nums = _extract_numbers(original)
    new_nums = _extract_numbers(rewritten)
    if not orig_nums:
        return True
    missing = [n for n in orig_nums if not any(abs(n - m) / max(abs(n), 1) < 0.02 for m in new_nums)]
    return len(missing) <= max(1, len(orig_nums) // 4)


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"JSON 못 찾음: {text!r}")
    return json.loads(m.group(0))


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT result_id, claim_sentence, verification_result, evidence FROM verifications "
        "WHERE article_title = ?",
        (ARTICLE_TITLE,),
    ).fetchall()
    print(f"대상 {len(rows)}건")

    n_ok = n_rejected = n_fail = 0
    for r in rows:
        raw_reason = r["evidence"]
        if not raw_reason:
            print(f"[스킵·원본없음] {r['result_id']}")
            continue

        prompt = PROMPT_TEMPLATE.format(
            style_examples=STYLE_EXAMPLES,
            claim_sentence=r["claim_sentence"],
            verdict=r["verification_result"],
            raw_reason=raw_reason,
        )
        try:
            reply = call_hcx(model=MODEL, system_prompt=SYSTEM_PROMPT, user_content=prompt, temperature=0.3)
            new_reason = str(_extract_json(reply)["reason"]).strip()
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {r['result_id']}: {e}")
            n_fail += 1
            continue

        if not _numbers_preserved(raw_reason, new_reason):
            print(f"[거부·숫자불일치] {r['claim_sentence'][:35]}\n  원본: {raw_reason[:100]}\n  거부됨: {new_reason}\n")
            n_rejected += 1
            continue

        conn.execute("UPDATE verifications SET evidence=? WHERE result_id=?", (new_reason, r["result_id"]))
        n_ok += 1
        print(f"[{r['verification_result']}] {r['claim_sentence'][:40]}\n  원본: {raw_reason[:80]}\n  ->  {new_reason}\n")

    conn.commit()
    conn.close()
    print(f"\n총 {n_ok}건 다시 씀, {n_rejected}건 숫자 불일치로 거부(원본 유지), {n_fail}건 실패")


if __name__ == "__main__":
    main()
