"""data/rewrite_golden_evidence_as_llm.py — "직접기록"(judge() 대신 골든셋의 사람이 쓴
reason(판정 근거) 원문을 그대로 저장한) 일치/불일치 46건의 evidence 문구를, 실제 운영
judge()가 LLM으로 판정했을 때 쓰는 것과 같은 자연스러운 문장 톤으로 다시 쓴다.

주의 — 문체만 바꾸고 사실은 안 바꾼다: 원본 reason에 있는 수치·KOSIS 표 정보·비교
결과는 그대로 보존하고, verdict(일치/불일치)도 그대로 고정해서 프롬프트에 넘긴다.
LLM에게 시키는 건 "이미 확정된 사실을 자연스러운 설명문으로 다시 쓰는" 것뿐이지, 새로
판정하거나 근거를 지어내는 게 아니다 — 원본 데이터(정답_verdict)는 이미 golden 근거로
검증됐으므로 여기서 다시 판정을 흔들면 안 된다.

원본 reason은 "KOSIS: DT_xxx(표명) 2024.07~2025.06 12개월의 ... (탭으로 나열된 12개
숫자) ... 이므로 기사와 일치하지 X"처럼 스프레드시트 메모체라 데모에서 "LLM이 쓴 것처럼"
안 보였다는 지적(2026-08-31, 사용자)에 따라, 실제 운영 judge()의 진짜 LLM 출력 문체
(few-shot 예시로 아래 EXAMPLES에 실제 이 DB에 있는 출력을 그대로 사용)에 맞춰 다시 쓴다.

사용법 (프로젝트 루트에서, 실제 HCX 호출 발생):
    python -m data.rewrite_golden_evidence_as_llm
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

# 실제 이 프로젝트의 judge()가 낸 진짜 출력들 — 문체 참고용 few-shot (내용은 이 rewrite
# 대상과 무관, 순전히 "이런 톤으로 써라"는 예시).
STYLE_EXAMPLES = """예시1: "기사 수치(59.2)와 통계 계산값(59.2 kg) 차이가 허용 오차(반올림 단위 추정 ±0.5kg) 이내이고 시점·단위 불일치도 없음."
예시2: "기사 주장과 통계 계산값 모두 작년(2023년)을 기준으로 GDP 대비 가계신용 비율의 하락을 말하고 있고, 그 비율도 0.8%로 같아 실질적으로 같은 내용을 전한다."
예시3: "기사(1.7%)와 통계(1.6%)는 같은 연간 기준으로 비교 가능한데도 값이 다르고, 0.1%p 차이도 반올림으로 보기 어려워 불일치로 판단한다.\""""

PROMPT_TEMPLATE = """아래는 한 통계 팩트체크 시스템이 "판정 근거"로 남긴 메모인데, 스프레드시트에
사람이 적어둔 원본이라 문장이 딱딱하고 표 ID·탭으로 나열된 숫자가 그대로 섞여 있다. 이걸
실제 LLM이 판정하면서 쓰는 것과 같은 자연스러운 한국어 설명문 1~2문장으로 다시 써라.

절대 규칙:
- 판정 결과({verdict})는 이미 확정된 사실이다 — 절대 바꾸지 말고 그 결론을 그대로 설명하라.
- 원본에 있는 숫자·비교 대상·시점은 빠짐없이 반영하되, 새로운 숫자나 근거를 지어내지 마라.
- "KOSIS: DT_xxx(표명)" 같은 표 ID 나열이나 탭으로 구분된 숫자 목록은 자연스러운 문장으로
  풀어써라(예: 12개월 숫자를 그대로 나열하지 말고 "12개월 중 4개월은 오히려 전월보다 높아"
  처럼 요약).
- 1인칭·해요체 쓰지 말고, 아래 문체 예시처럼 담백한 설명체로.

문체 예시 (내용은 무관, 톤만 참고):
{style_examples}

기사 주장: "{claim_sentence}"
판정 결과: {verdict}
원본 판정 근거 메모: {raw_reason}

출력 형식 (JSON만 출력, 다른 텍스트 금지):
{{"reason": "다시 쓴 판정 근거 1~2문장"}}"""


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"JSON 못 찾음: {text!r}")
    return json.loads(m.group(0))


_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def _extract_numbers(text: str) -> set[float]:
    nums = set()
    for m in _NUMBER_RE.finditer(text):
        try:
            v = float(m.group(0).replace(",", ""))
        except ValueError:
            continue
        if v >= 4:  # "1~2월", "3배" 같은 사소한 한 자리 숫자는 근거의 핵심 수치가 아니라 제외
            nums.add(v)
    return nums


def _numbers_preserved(original: str, rewritten: str) -> bool:
    """원본에 있던 숫자가 다시 쓴 글에도 (근사치 허용) 다 남아있는지 확인 — 근거를 자연스러운
    문장으로 바꾸다가 LLM이 숫자를 지어내거나 빠뜨리는 사고를 막기 위한 최소 안전장치
    (2026-08-31, 실제로 "자살률" claim의 근거가 "암 발생률"로, 숫자까지 완전히 다르게
    지어내진 사고가 한 번 났음 — 그건 애초에 원본 근거가 비어 있어서 생긴 별개 원인이었지만
    (build_golden_demo_presets.py에서 고침), 여기서도 이중으로 막아둔다)."""
    orig_nums = _extract_numbers(original)
    new_nums = _extract_numbers(rewritten)
    if not orig_nums:
        return True
    missing = [n for n in orig_nums if not any(abs(n - m) / max(abs(n), 1) < 0.02 for m in new_nums)]
    return len(missing) <= max(1, len(orig_nums) // 4)  # 소소한 반올림 표기 차이 한두 개는 허용


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT result_id, claim_sentence, verification_result, evidence FROM verifications "
        "WHERE verification_result IN ('일치','불일치') AND verification_possible='불가능'"
    ).fetchall()
    print(f"다시 쓸 대상 {len(rows)}건")

    n_ok = n_fail = n_skipped_empty = n_rejected = 0
    for r in rows:
        raw_reason = r["evidence"]
        if not raw_reason:
            # 원본 근거 자체가 비어 있으면 rewrite할 게 없다 — LLM에게 아무 근거 없이
            # 써보라고 하면 지어낼 위험만 있다(2026-08-31 실측). build_golden_demo_presets.py를
            # 고쳐서 이제 이런 행은 없어야 하지만, 혹시 남아있으면 안전하게 건너뛴다.
            print(f"[스킵·원본없음] {r['result_id']}: {r['claim_sentence'][:30]}")
            n_skipped_empty += 1
            continue

        prompt = PROMPT_TEMPLATE.format(
            style_examples=STYLE_EXAMPLES,
            claim_sentence=r["claim_sentence"],
            verdict=r["verification_result"],
            raw_reason=raw_reason,
        )
        try:
            reply = call_hcx(model=MODEL, system_prompt=SYSTEM_PROMPT, user_content=prompt, temperature=0.0)
            new_reason = str(_extract_json(reply)["reason"]).strip()
        except Exception as e:  # noqa: BLE001
            print(f"[FAIL] {r['result_id']}: {e}")
            n_fail += 1
            continue

        if not _numbers_preserved(raw_reason, new_reason):
            # 원본에 있던 숫자가 다시 쓴 글에서 상당수 사라짐/바뀜 — 원본을 그대로 둔다
            # (틀릴 위험이 있는 rewrite를 적용하느니 스프레드시트 메모체가 낫다).
            print(f"[거부·숫자불일치] {r['claim_sentence'][:35]}\n  원본: {raw_reason[:80]}\n  거부됨: {new_reason}\n")
            n_rejected += 1
            continue

        conn.execute(
            "UPDATE verifications SET evidence=? WHERE result_id=?",
            (new_reason, r["result_id"]),
        )
        n_ok += 1
        print(f"[{r['verification_result']}] {r['claim_sentence'][:35]}\n  -> {new_reason}\n")

    conn.commit()
    conn.close()
    print(
        f"총 {n_ok}건 다시 씀, {n_rejected}건 숫자 불일치로 거부(원본 유지), "
        f"{n_skipped_empty}건 원본 없어 스킵, {n_fail}건 호출 실패"
    )


if __name__ == "__main__":
    main()
