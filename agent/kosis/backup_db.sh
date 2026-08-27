#!/usr/bin/env bash
# agent/kosis/backup_db.sh -- 재구축 중 주기적 PostgreSQL 백업.
# EBS 루트 볼륨(프로젝트 디렉토리 내부, /dev/root) 아래에만 저장한다 -- 인스턴스
# 스토리지(/home/ubuntu/data)는 절대 사용하지 않는다.
#
# 사용법:
#   bash agent/kosis/backup_db.sh <label>
#   예: bash agent/kosis/backup_db.sh checkpoint_10000
set -euo pipefail

LABEL="${1:?사용법: backup_db.sh <label>}"
BACKUP_DIR="/home/ubuntu/ai-nlp-project-reembedding/backup"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${BACKUP_DIR}/${LABEL}_${TS}.dump"

mkdir -p "$BACKUP_DIR"

# 접속 정보는 .env의 SUPABASE_DB_URL에서 읽는다 — 스크립트에 비밀번호를 박아두면
# git에 그대로 올라간다(2026-08-27 커밋 직전에 발견해서 제거).
#   사용법: set -a && source .env && set +a && bash agent/kosis/backup_db.sh <label>
: "${SUPABASE_DB_URL:?SUPABASE_DB_URL이 없습니다. .env를 먼저 source 하세요}"

echo "[backup] pg_dump 시작 -> $OUT"
pg_dump "$SUPABASE_DB_URL" -F c -f "$OUT"

SIZE=$(du -h "$OUT" | cut -f1)
echo "[backup] 완료: $OUT ($SIZE)"

# 2026-08-27: DB가 커지면서 덤프 1개가 1.4GB(완료 시 3GB 예상)라, 전부 쌓아두면
# 디스크가 찬다. 같은 label 계열(=같은 서버 역할)의 최근 2개만 남기고 지운다.
# 덤프는 "DB 통째로 날아갔을 때" 복구용이라 최신 2개면 충분하다(재개는 체크포인트 담당).
KEEP=2
LABEL_PREFIX="$(echo "$LABEL" | sed 's/_chunk[0-9]*$//')"
OLD=$(ls -1t "${BACKUP_DIR}/${LABEL_PREFIX}"*.dump 2>/dev/null | tail -n +$((KEEP + 1)))
if [ -n "$OLD" ]; then
    echo "[backup] 오래된 덤프 정리(최근 ${KEEP}개만 유지):"
    echo "$OLD" | while read -r f; do
        echo "  삭제: $(basename "$f") ($(du -h "$f" | cut -f1))"
        rm -f "$f"
    done
fi

# EBS 여부 재확인(백업 자체가 인스턴스 스토리지로 새지 않았는지)
FS=$(df --output=source "$BACKUP_DIR" | tail -1)
echo "[backup] 저장된 파일시스템: $FS"
