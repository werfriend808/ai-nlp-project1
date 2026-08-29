"""환경 지문 — 두 사람이 같은 환경에서 작업하는지 한 줄로 대조한다.

    python -m agent.kosis.env_fingerprint

git 리비전·의존성·DB 상태·BM25 인덱스·데이터 파일을 찍고 마지막에 FINGERPRINT 한 줄을
낸다. 그 줄이 서로 같으면 같은 환경이다. 다르면 * 표시된 줄을 위에서부터 대조하면 된다.

비밀값은 출력하지 않는다 — API 키·DB 비밀번호는 "설정됨/없음"과 길이만 찍는다.

2026-08-29에 A 트랙(excluded 복구)과 C 트랙(리랭커)을 병렬로 진행하면서 추가했다.
두 트랙이 같은 골든셋으로 수치를 재는데 DB 상태나 인덱스가 갈리면 서로의 측정이
조용히 오염된다 — 특히 A 트랙이 표를 재임베딩하면 C 트랙의 기준선이 바뀐다.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_crit: list[str] = []      # 지문에 들어가는 항목(불일치하면 작업이 갈린다)


def _sh(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, cwd=ROOT, capture_output=True,
                              text=True, timeout=20).stdout.strip()
    except Exception:
        return "?"


def _line(label: str, value: str, critical: bool = False) -> None:
    print(f" {'*' if critical else ' '} {label:<26} {value}")
    if critical:
        _crit.append(f"{label}={value}")


def main() -> None:
    print("=" * 68)
    print(" 환경 지문   ( * = 서로 일치해야 하는 항목 )")
    print("=" * 68)

    print("\n[git]")
    _line("remote", _sh("git config --get remote.origin.url").replace("https://", ""))
    _line("branch", _sh("git rev-parse --abbrev-ref HEAD"), True)
    _line("HEAD", _sh("git rev-parse --short HEAD"), True)
    dirty = _sh("git status --porcelain")
    _line("작업트리", "깨끗함" if not dirty else f"수정/미추적 {len(dirty.splitlines())}건")
    sb = _sh("git status -sb | head -1")
    _line("원격 대비", sb.split("[")[-1].rstrip("]") if "[" in sb else "동기화됨")

    print("\n[python]")
    _line("실행 파이썬", sys.executable.replace(str(ROOT) + "/", ""))
    _line("버전", sys.version.split()[0], True)
    for pkg in ("torch", "sentence_transformers", "transformers",
                "psycopg2", "scipy", "sklearn", "numpy"):
        try:
            mod = __import__(pkg)
            _line(pkg, getattr(mod, "__version__", "?"), True)
        except Exception as e:
            _line(pkg, f"없음 ({type(e).__name__})", True)

    print("\n[.env — 존재 여부만, 값은 안 찍음]")
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        print("    python-dotenv 없음")
    for key in ("SUPABASE_DB_URL", "KOSIS_API_KEY", "HCX_API_KEY"):
        val = os.environ.get(key, "")
        _line(key, f"설정됨 (길이 {len(val)})" if val else "없음", True)

    print("\n[DB]")
    try:
        import psycopg2
        from urllib.parse import urlparse

        url = urlparse(os.environ["SUPABASE_DB_URL"])
        _line("호스트", f"{url.hostname}:{url.port or 5432}{url.path}", True)
        conn = psycopg2.connect(os.environ["SUPABASE_DB_URL"])
        with conn.cursor() as cur:
            cur.execute("""select metadata_status, count(*) from kosis_vdb_tables_qwen
                           group by 1 order by 1""")
            for status, n in cur.fetchall():
                _line(f"  {status}", f"{n:,}", True)
            cur.execute("select count(*) from kosis_vdb_tables_qwen")
            _line("표 합계", f"{cur.fetchone()[0]:,}", True)
            cur.execute("""select embedding_dimension, count(*) from kosis_vdb_tables_qwen
                           group by 1 order by 1""")
            _line("임베딩 차원", ", ".join(f"{d}({n:,})" for d, n in cur.fetchall()), True)
        conn.close()
    except Exception as e:
        _line("연결", f"실패 — {type(e).__name__}: {str(e)[:60]}", True)

    print("\n[BM25 인덱스]")
    try:
        import pickle

        from agent.kosis.bm25_search import INDEX_DIR, MATRIX_FILE, META_FILE

        if MATRIX_FILE.exists() and META_FILE.exists():
            with META_FILE.open("rb") as f:
                meta = pickle.load(f)
            _line("경로", str(INDEX_DIR).replace(str(ROOT) + "/", ""))
            _line("문서 수", f"{meta.get('n_docs', 0):,}", True)
            _line("생성 시각", str(meta.get("built_at", "?"))[:19])
            _line("크기", f"{MATRIX_FILE.stat().st_size / 1e6:.0f}MB")
        else:
            _line("상태", "없음 — `python -m agent.kosis.build_bm25_index` 필요", True)
    except Exception as e:
        _line("상태", f"확인 실패 ({type(e).__name__})", True)

    print("\n[GPU]")
    _line("장치", _sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader") or "없음", True)
    _line("사용 중", _sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader") or "-")

    print("\n[데이터 파일]")
    for rel in ("benchmark/search_experiment/eval_set.json",
                "benchmark/search_experiment/claim_slots.json",
                "notebooks/골든셋_통합.xlsx",
                "agent/mapping/table_catalog.json"):
        path = ROOT / rel
        if path.exists():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
            _line(Path(rel).name, f"{digest}  ({path.stat().st_size:,}B)", True)
        else:
            _line(Path(rel).name, "없음", True)

    fingerprint = hashlib.sha256("\n".join(_crit).encode()).hexdigest()[:16]
    print("\n" + "=" * 68)
    print(f" FINGERPRINT   {fingerprint}")
    print("=" * 68)
    print(" 이 값이 서로 같으면 같은 환경입니다.")
    print(" 다르면 위에서 * 표시된 줄을 하나씩 대조하세요.")


if __name__ == "__main__":
    main()
