#!/bin/bash
# 819건 Verified Mapping 실험 GPU-aware 자동재시작 래퍼.
# backup/recover_org101_autorestart.sh 패턴을 그대로 이식: 무한 루프 -> checkpoint.json의
# current_phase=="done"이면 종료 -> GPU 여유 메모리(<3000MiB)면 30초 간격 최대 60회 대기 ->
# run819.py 실행(run.log에 누적 append) -> exit 후 5초 대기 후 재시작(run819.py 자체가
# checkpoint.json에서 자동 재개하므로 래퍼는 "죽으면 다시 켜기"만 한다).
set -u
cd /home/ubuntu/ai-nlp-project-reembedding
source .venv/bin/activate
set -a; source .env; set +a

EXP_DIR=benchmark/verified_mapping_819
CHECKPOINT="$EXP_DIR/checkpoint.json"
RUNLOG="$EXP_DIR/run.log"

while true; do
  if [ -f "$CHECKPOINT" ]; then
    PHASE=$(python3 -c "import json; print(json.load(open('$CHECKPOINT')).get('current_phase',''))" 2>/dev/null)
    if [ "$PHASE" = "done" ]; then
      echo "[autorestart] current_phase=done — 819건 전체 완료, 종료. ($(date -u '+%Y-%m-%d %H:%M:%S') UTC)" | tee -a "$RUNLOG"
      break
    fi
    echo "[autorestart] 체크포인트 발견: phase=$PHASE ($(date -u '+%Y-%m-%d %H:%M:%S') UTC)" | tee -a "$RUNLOG"
  else
    echo "[autorestart] 체크포인트 없음 — 처음부터 시작 ($(date -u '+%Y-%m-%d %H:%M:%S') UTC)" | tee -a "$RUNLOG"
  fi

  # GPU를 다른 프로세스가 꽉 채우고 있으면 모델 로딩이 바로 OOM으로 죽고 재시도만 반복해
  # 낭비가 크다. 최소 여유 메모리(3GB)가 생길 때까지 30초 간격으로 기다렸다가 시작한다
  # (무한 대기 방지로 최대 60회=30분).
  for i in $(seq 1 60); do
    FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -z "$FREE_MB" ] || [ "$FREE_MB" -ge 3000 ]; then
      break
    fi
    echo "[autorestart] GPU 여유 부족(${FREE_MB}MiB free) — 30초 대기 후 재확인 ($i/60)" | tee -a "$RUNLOG"
    sleep 30
  done

  echo "[autorestart] run819.py 시작 ($(date -u '+%Y-%m-%d %H:%M:%S') UTC)" | tee -a "$RUNLOG"
  python3 "$EXP_DIR/run819.py" >> "$RUNLOG" 2>&1
  EXIT=$?
  echo "[autorestart] 하위 프로세스 종료(exit=$EXIT) — 5초 후 재확인 후 재시작 ($(date -u '+%Y-%m-%d %H:%M:%S') UTC)" | tee -a "$RUNLOG"
  sleep 5
done
