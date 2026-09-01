#!/bin/bash
# 국가데이터처(101) excluded_too_large 복구 자동 재시작 래퍼.
# recover_excluded.py가 원인 불명(환경 재시작 등)으로 조용히 죽는 게 반복 확인돼서,
# 죽을 때마다 DB에서 남은 대상만 다시 뽑아 자동으로 재시작한다.
# 대상이 0건이 되면(전부 success/recover_failed로 처리됨) 종료한다.
set -u
cd /home/ubuntu/ai-nlp-project-reembedding
source .venv/bin/activate
set -a; source .env; set +a

LOGDIR=/tmp/claude-1000/-home-ubuntu-ai-nlp-project-reembedding/f7632bfc-b06f-42f8-82f9-061eccb16739/scratchpad
IDS_FILE=backup/recover_ids_101_news_relevant.json

while true; do
  python3 -c "
import os, json, psycopg2
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.environ['SUPABASE_DB_URL'])
cur = conn.cursor()
cur.execute(\"select table_id from kosis_vdb_tables_qwen where metadata_status='excluded_too_large' and org_id='101' order by table_id\")
ids = [r[0] for r in cur.fetchall()]
print(len(ids))
with open('$IDS_FILE', 'w', encoding='utf-8') as f:
    json.dump(ids, f, ensure_ascii=False)
" > /tmp/_remaining_count.txt
  REMAINING=$(cat /tmp/_remaining_count.txt)
  echo "[autorestart] 남은 대상: $REMAINING 건 ($(date -u '+%Y-%m-%d %H:%M:%S') UTC)"

  if [ "$REMAINING" -eq 0 ]; then
    echo "[autorestart] 대상 0건 — 전체 완료, 종료."
    break
  fi

  # 2026-08-30: GPU를 다른 프로세스(실험 등)가 꽉 채우고 있으면 모델 로딩이 바로
  # OOM으로 죽고 5초마다 재시도만 반복해 낭비가 컸다. 최소 여유 메모리(3GB)가 생길
  # 때까지 30초 간격으로 기다렸다가 시작한다(무한 대기 방지로 최대 60회=30분).
  for i in $(seq 1 60); do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$FREE_MB" ] || [ "$FREE_MB" -ge 3000 ]; then
      break
    fi
    echo "[autorestart] GPU 여유 부족(${FREE_MB}MiB free) — 30초 대기 후 재확인 ($i/60)"
    sleep 30
  done

  RUNLOG="$LOGDIR/recover_org101_$(date +%Y%m%d_%H%M%S).log"
  echo "[autorestart] 재시작: $RUNLOG"
  python3 -m agent.kosis.recover_excluded \
    --ids-file "$IDS_FILE" \
    --concurrency 8 \
    --api-keys "${KOSIS_API_KEY},${KOSIS_API_KEY_2}" \
    >> "$RUNLOG" 2>&1
  EXIT=$?
  echo "[autorestart] 하위 프로세스 종료(exit=$EXIT) — 5초 후 재확인 후 재시작" | tee -a "$RUNLOG"
  sleep 5
done
