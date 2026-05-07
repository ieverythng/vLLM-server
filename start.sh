#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${BASE_DIR}/venv"
VENV_PY="${VENV_DIR}/bin/python"
VLLM_MANAGER="${BASE_DIR}/vllm_manager.py"
VLLM_PID_FILE="${BASE_DIR}/vllm.pid"
GATEWAY_PID_FILE="${BASE_DIR}/gateway.pid"
VLLM_URL="http://127.0.0.1:8000"
GATEWAY_HOST="0.0.0.0"
GATEWAY_PORT="${GATEWAY_PORT:-8001}"
VLLM_TIMEOUT="${VLLM_TIMEOUT:-300}"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "ERROR: virtualenv python not found at ${VENV_PY}" >&2
  exit 1
fi

source "${VENV_DIR}/bin/activate"

log() {
  printf '[start.sh] %s\n' "$*"
}

pid_running() {
  local pid_file="$1"
  if [[ -f "${pid_file}" ]]; then
    local pid
    pid="$(<"${pid_file}")"
    if [[ "${pid}" =~ ^[0-9]+$ ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  fi
  return 1
}

cleanup_pidfile() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] && rm -f "${pid_file}"
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local timeout_seconds="$3"
  local start_ts elapsed
  start_ts="$(date +%s)"

  log "Waiting for ${label} at ${url}"
  while true; do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      log "${label} is ready"
      return 0
    fi
    elapsed=$(( $(date +%s) - start_ts ))
    if (( elapsed >= timeout_seconds )); then
      log "ERROR: timeout waiting for ${label}"
      return 1
    fi
    sleep 2
  done
}

print_zerotier_info() {
  echo "ZeroTier:"
  if command -v zerotier-cli >/dev/null 2>&1; then
    zerotier-cli info 2>/dev/null || true
    zerotier-cli listnetworks 2>/dev/null || true
  fi

  local zt_lines
  zt_lines="$(ip -o -4 addr show up scope global 2>/dev/null | awk '$2 ~ /^zt/ {print $2" " $4}' || true)"
  if [[ -n "${zt_lines}" ]]; then
    while IFS= read -r line; do
      [[ -n "${line}" ]] && echo "  ${line}"
    done <<< "${zt_lines}"
  else
    echo "  (no active ZeroTier interface found)"
  fi
}

start_vllm() {
  if pid_running "${VLLM_PID_FILE}"; then
    log "vLLM already running (PID $(<"${VLLM_PID_FILE}"))"
    return 0
  fi

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    log "ERROR: nvidia-smi not found. CUDA/GPU drivers are required."
    return 1
  fi

  log "GPU inventory:"
  nvidia-smi || true

  log "Starting vLLM backend via vllm_manager.py"
  "${VENV_PY}" "${VLLM_MANAGER}" start --no-wait

  wait_for_http "${VLLM_URL}/v1/models" "vLLM /v1/models" "${VLLM_TIMEOUT}"
}

start_gateway() {
  if pid_running "${GATEWAY_PID_FILE}"; then
    log "Gateway already running (PID $(<"${GATEWAY_PID_FILE}"))"
    return 0
  fi

  log "Starting FastAPI gateway on ${GATEWAY_HOST}:${GATEWAY_PORT}"
  nohup "${VENV_PY}" -m uvicorn server:app \
    --host "${GATEWAY_HOST}" \
    --port "${GATEWAY_PORT}" \
    --log-level info \
    > "${BASE_DIR}/gateway.log" 2>&1 &
  echo $! > "${GATEWAY_PID_FILE}"

  wait_for_http "http://127.0.0.1:${GATEWAY_PORT}/health" "gateway /health" 60
}

stop_gateway() {
  if pid_running "${GATEWAY_PID_FILE}"; then
    local pid
    pid="$(<"${GATEWAY_PID_FILE}")"
    log "Stopping gateway (PID ${pid})"
    kill "${pid}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      if ! kill -0 "${pid}" 2>/dev/null; then
        cleanup_pidfile "${GATEWAY_PID_FILE}"
        return 0
      fi
      sleep 1
    done
    kill -9 "${pid}" 2>/dev/null || true
  fi
  cleanup_pidfile "${GATEWAY_PID_FILE}"
}

stop_vllm() {
  if pid_running "${VLLM_PID_FILE}"; then
    log "Stopping vLLM via vllm_manager.py"
    "${VENV_PY}" "${VLLM_MANAGER}" stop || true
  else
    cleanup_pidfile "${VLLM_PID_FILE}"
  fi
}

print_status() {
  echo "Status:"
  if pid_running "${VLLM_PID_FILE}"; then
    echo "  vLLM: running (PID $(<"${VLLM_PID_FILE}"))"
  else
    echo "  vLLM: stopped"
  fi

  if pid_running "${GATEWAY_PID_FILE}"; then
    echo "  gateway: running (PID $(<"${GATEWAY_PID_FILE}"))"
  else
    echo "  gateway: stopped"
  fi

  echo "  backend: ${VLLM_URL}"
  echo "  gateway: http://127.0.0.1:${GATEWAY_PORT}"
  print_zerotier_info
}

case "${1:-start}" in
  start)
    start_vllm
    start_gateway
    print_status
    ;;
  stop)
    stop_gateway
    stop_vllm
    print_status
    ;;
  restart)
    stop_gateway
    stop_vllm
    start_vllm
    start_gateway
    print_status
    ;;
  status)
    print_status
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status}" >&2
    exit 1
    ;;
esac
