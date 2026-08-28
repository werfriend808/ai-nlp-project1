#!/usr/bin/env bash
# agent/kosis/colab_ssh_tunnel.sh -- Colab에서 7-1의 PostgreSQL(5432)로 SSH 터널을 연다.
# ngrok(카드 등록 필요) 대신 이미 열려있는 SSH(22번)를 재사용 -- 새 포트를 열지 않는다.
# 사용된 키는 authorized_keys에 restrict,port-forwarding으로 제한돼 있어
# 셸 접속은 안 되고 포트포워딩만 가능하다(7-1에서 실측 검증 완료).
#
# 사용법 (Colab 셀에서):
#   !bash agent/kosis/colab_ssh_tunnel.sh
#
# 필요한 환경변수(Colab Secrets에서 os.environ으로 미리 채워둘 것):
#   COLAB_SSH_KEY_PATH  -- 개인키 파일 경로 (예: /content/.ssh/colab_a100_key, 0600 권한)
#   SSH_HOST            -- 7-1 공인 IP (기본값 아래 지정)
#   LOCAL_DB_PORT        -- 로컬에서 쓸 포트 (기본 15432)
set -euo pipefail

SSH_HOST="${SSH_HOST:-51.20.253.79}"
SSH_USER="${SSH_USER:-ubuntu}"
LOCAL_DB_PORT="${LOCAL_DB_PORT:-15432}"
KEY_PATH="${COLAB_SSH_KEY_PATH:-/content/.ssh/colab_a100_key}"

if [ ! -f "$KEY_PATH" ]; then
    echo "[오류] 개인키 파일이 없습니다: $KEY_PATH" >&2
    echo "Colab 셀에서 COLAB_SSH_PRIVATE_KEY 시크릿 내용을 먼저 이 경로에 써야 합니다." >&2
    exit 1
fi
chmod 600 "$KEY_PATH"

# 이미 떠 있는 동일 포트 터널이 있으면 정리(재실행 대비, 멱등)
pkill -f "ssh .*-L ${LOCAL_DB_PORT}:127.0.0.1:5432" 2>/dev/null || true
sleep 1

ssh -f -N \
    -o StrictHostKeyChecking=no \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    -o ExitOnForwardFailure=yes \
    -i "$KEY_PATH" \
    -L "${LOCAL_DB_PORT}:127.0.0.1:5432" \
    "${SSH_USER}@${SSH_HOST}"

sleep 2

# 터널 자체가 살아있는지(TCP 리스닝) 확인 -- ss/netstat 둘 다 없을 수 있어 python으로 확인
python3 - "$LOCAL_DB_PORT" <<'PYEOF'
import socket, sys
port = int(sys.argv[1])
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.settimeout(3)
try:
    s.connect(("127.0.0.1", port))
    print(f"[OK] 로컬 포트 {port} 터널 정상 리스닝 중")
except Exception as e:
    print(f"[오류] 로컬 포트 {port} 연결 실패: {e}")
    sys.exit(1)
finally:
    s.close()
PYEOF

echo "터널 준비 완료: 127.0.0.1:${LOCAL_DB_PORT} -> 7-1(${SSH_HOST}):5432"
