#!/usr/bin/env bash
set -euo pipefail

# Safe launcher for 27B AWQ staging with hard timeout + RAM guardrail.
# Usage:
#   scripts/stage_27b_safe.sh [profile] [timeout_seconds] [min_free_mem_mb]
# Example:
#   scripts/stage_27b_safe.sh qwen36_27b_awq_int4_stable 420 2200

PROFILE="${1:-qwen36_27b_awq_int4_stable}"
TIMEOUT_S="${2:-420}"
MIN_FREE_MEM_MB="${3:-2200}"
WORKDIR="/mnt/c/Users/Admin/PROJECTS/vLLM-server"
PYTHON_BIN="${VLLM_PYTHON_BIN:-$WORKDIR/.venv311/Scripts/python.exe}"

log(){ printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }

cleanup_kill(){
  log "KILL-SWITCH: stopping vLLM manager + matching API server processes"
  (cd "$WORKDIR" && "$PYTHON_BIN" vllm_manager.py stop || true)
  pkill -f "vllm.entrypoints.openai.api_server" || true
  sleep 2
}

check_mem_guard(){
  local free_mb
  free_mb="$(awk '/MemAvailable:/ {print int($2/1024)}' /proc/meminfo)"
  log "MemAvailable=${free_mb}MB (threshold=${MIN_FREE_MEM_MB}MB)"
  if [ "$free_mb" -lt "$MIN_FREE_MEM_MB" ]; then
    log "KILL-SWITCH: low RAM detected"
    cleanup_kill
    return 1
  fi
  return 0
}

cd "$WORKDIR"

log "Preflight (python=${PYTHON_BIN})"
"$PYTHON_BIN" vllm_manager.py preflight

log "Setting active profile => ${PROFILE}"
PROFILE="$PROFILE" "$PYTHON_BIN" - <<'PY'
import yaml
from pathlib import Path
import os
p=Path('/mnt/c/Users/Admin/PROJECTS/vLLM-server/config.yaml')
profile=os.environ['PROFILE']
d=yaml.safe_load(p.read_text(encoding='utf-8'))
d.setdefault('runtime',{})['active_model_profile']=profile
p.write_text(yaml.safe_dump(d, sort_keys=False), encoding='utf-8')
print('active profile set to', profile)
PY

log "Dry-run command"
"$PYTHON_BIN" vllm_manager.py dry-run

log "Starting vLLM in background (--no-wait)"
"$PYTHON_BIN" vllm_manager.py start --no-wait

start_ts="$(date +%s)"
while true; do
  now="$(date +%s)"
  elapsed=$((now - start_ts))

  if ! check_mem_guard; then
    log "Aborted due to memory pressure"
    exit 2
  fi

  if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then
    log "vLLM ready in ${elapsed}s"
    break
  fi

  if [ "$elapsed" -ge "$TIMEOUT_S" ]; then
    log "KILL-SWITCH: timeout ${TIMEOUT_S}s reached"
    cleanup_kill
    exit 3
  fi

  sleep 5
done

log "Smoke test (short generation, low tokens)"
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "cyankiwi/Qwen3.6-27B-AWQ-INT4",
    "messages": [{"role": "user", "content": "Reply with: READY"}],
    "max_tokens": 16,
    "temperature": 0.0
  }' | "$PYTHON_BIN" -m json.tool | sed -n '1,80p'

log "Done. Use '$PYTHON_BIN vllm_manager.py stop' when finished."
