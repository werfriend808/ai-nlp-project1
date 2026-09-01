#!/usr/bin/env bash
# agent/kosis/run_rebuild_loop.sh -- 10,000건 단위로 (기존 검증된) reembed_worker.py를
# 반복 호출하고, 매 청크 뒤 quality_gate_check.py로 감사하고, backup_db.sh로 백업한다.
#
# 기존 enrichment/period fix 로직(reembed_worker.py)은 절대 수정하지 않는다 -- 이 스크립트는
# 그 위를 감싸는 오케스트레이터일 뿐이다. 품질 게이트가 이상치를 감지하면(exit code 1)
# 그 즉시 루프를 멈추고 사람이 확인할 때까지 다음 청크를 실행하지 않는다.
#
# 사용법:
#   bash agent/kosis/run_rebuild_loop.sh SERVER_A [concurrency]
#   bash agent/kosis/run_rebuild_loop.sh SERVER_B [concurrency]
set -uo pipefail

ROLE="${1:?사용법: run_rebuild_loop.sh SERVER_A|SERVER_B [concurrency] [api_key_env_names(콤마구분)] [worker_module]}"
CONCURRENCY="${2:-2}"
API_KEY_ENV_NAMES="${3:-KOSIS_API_KEY}"
WORKER_MODULE="${4:-agent.kosis.reembed_worker}"
CHUNK=10000
PROJECT_DIR="/home/ubuntu/ai-nlp-project-reembedding"

cd "$PROJECT_DIR"
source .venv/bin/activate
set -a
source .env
set +a

remaining() {
    python3 -c "
import os, psycopg2
def load_env():
    with open('.env') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip())
load_env()
conn = psycopg2.connect(os.environ['SUPABASE_DB_URL'])
cur = conn.cursor()
cur.execute(\"select count(*) from kosis_reembed_checkpoint_qwen where server_role=%s and status != 'success'\", ('$ROLE',))
print(cur.fetchone()[0])
conn.close()
"
}

N=0
while true; do
    REM=$(remaining)
    echo "[$(date -u +%H:%M:%S)] [$ROLE] 남은 대상: $REM"
    if [ "$REM" -eq 0 ]; then
        echo "[$ROLE] 전체 완료 -- 더 처리할 표 없음."
        break
    fi

    # 콤마로 구분된 env var 이름들(예: "KOSIS_API_KEY,KOSIS_API_KEY_2")을 각각의
    # 실제 값으로 풀어서 --api-keys에 다시 콤마로 합쳐 넘긴다 -- 키가 여러 개면
    # reembed_worker(_fast).py가 키마다 독립 rate limiter를 쓴다(기존 로직 그대로).
    IFS=',' read -ra KEY_NAMES <<< "$API_KEY_ENV_NAMES"
    API_KEY_VALUES=""
    for kn in "${KEY_NAMES[@]}"; do
        v="${!kn}"
        if [ -z "$API_KEY_VALUES" ]; then API_KEY_VALUES="$v"; else API_KEY_VALUES="$API_KEY_VALUES,$v"; fi
    done
    python -m "$WORKER_MODULE" "$ROLE" --limit "$CHUNK" --concurrency "$CONCURRENCY" \
        --api-keys "$API_KEY_VALUES"
    WORKER_EXIT=$?
    if [ $WORKER_EXIT -ne 0 ]; then
        echo "[$ROLE] *** $WORKER_MODULE 비정상 종료(exit=$WORKER_EXIT) -- 루프 중단, 확인 필요 ***"
        exit 1
    fi

    N=$((N + 1))
    echo "[$ROLE] 품질 게이트 실행 중..."
    python -m agent.kosis.quality_gate_check "$ROLE"
    GATE_EXIT=$?
    if [ $GATE_EXIT -ne 0 ]; then
        echo "[$ROLE] *** 품질 게이트 이상치 감지 -- 루프 중단, 사람 확인 필요 ***"
        exit 1
    fi

    # 2026-08-27: DB가 11GB까지 커지면서 매 청크(1만 건) 덤프가 4~5분씩 걸리고 파일도
    # 1.4GB씩 쌓인다(완료 시점엔 각 3GB 예상). 청크마다 워커가 그만큼 놀고 디스크도
    # 위험해서 3청크(3만 건)마다 1회로 줄인다. 재개는 체크포인트가 담당하므로 덤프
    # 주기를 늘려도 중단 복구 능력에는 영향이 없다(덤프는 DB 자체 소실 대비용).
    if [ $((N % 3)) -eq 0 ]; then
        echo "[$ROLE] 백업 실행 중..."
        bash agent/kosis/backup_db.sh "checkpoint_${ROLE}_chunk${N}"
    else
        echo "[$ROLE] 백업 건너뜀 (3청크마다 1회, 현재 청크 ${N})"
    fi
done
