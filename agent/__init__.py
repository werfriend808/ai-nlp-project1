import os

# embedding_search.py/reranker.py가 sentence-transformers를 로딩하기 전에 이 env var를
# 설정해두지만, 그보다 먼저 pandas(numpy MKL)가 자기 OpenMP 런타임을 로드해버리면 이미
# 늦어서 세그폴트가 재현된다 (2026-08-05 확인 — golden_set.py 등 pandas를 쓰는 모듈이
# embedding_search보다 먼저 import되는 모든 경우에 해당). agent 패키지를 import하는 순간
# (agent.* 하위 모듈이 뭐가 됐든) 가장 먼저 실행되도록 여기서 설정한다.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")