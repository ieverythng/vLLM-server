#!/usr/bin/env bash
set -euo pipefail

for p in \
  /home/juanbeck/vLLM-server/venv/bin/python \
  /home/juanbeck/vLLM-server/.venv/bin/python \
  /home/juanbeck/vLLM-server/.venv_wsl/bin/python
  do
  if [ -x "$p" ]; then
    echo "PY=$p"
    "$p" -c 'import sys, importlib.util as u; s=u.find_spec("vllm"); s2=u.find_spec("vllm._C"); print("python",sys.version.split()[0]); print("vllm",bool(s), s.origin if s else None); print("vllm._C",bool(s2), s2.origin if s2 else None)'
  fi
done
