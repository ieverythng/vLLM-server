#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: codex_wrapper.sh --harness codex_cli --repo-path PATH --prompt-file PATH [--mode MODE] \
  [--timeout-s SECONDS] [--allow-write 0|1] [--allow-shell 0|1] [--worktree PATH]
EOF
}

harness=""
repo_path=""
prompt_file=""
mode="suggest"
timeout_s="900"
allow_write="0"
allow_shell="0"
worktree=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --harness)
      harness="${2:-}"; shift 2 ;;
    --repo-path)
      repo_path="${2:-}"; shift 2 ;;
    --prompt-file)
      prompt_file="${2:-}"; shift 2 ;;
    --mode)
      mode="${2:-}"; shift 2 ;;
    --timeout-s)
      timeout_s="${2:-}"; shift 2 ;;
    --allow-write)
      allow_write="${2:-}"; shift 2 ;;
    --allow-shell)
      allow_shell="${2:-}"; shift 2 ;;
    --worktree)
      worktree="${2:-}"; shift 2 ;;
    -h|--help)
      usage
      exit 0 ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2 ;;
  esac
done

if [[ -z "${harness}" || -z "${repo_path}" || -z "${prompt_file}" ]]; then
  echo "ERROR: --harness, --repo-path, and --prompt-file are required" >&2
  usage >&2
  exit 2
fi

if [[ "${harness}" != "codex_cli" ]]; then
  echo "ERROR: unsupported harness '${harness}'" >&2
  exit 2
fi

if [[ ! -f "${prompt_file}" ]]; then
  echo "ERROR: prompt file not found: ${prompt_file}" >&2
  exit 2
fi

if [[ ! -d "${repo_path}" ]]; then
  echo "ERROR: repo path not found: ${repo_path}" >&2
  exit 2
fi

target_dir="${worktree:-${repo_path}}"
if [[ ! -d "${target_dir}" ]]; then
  echo "ERROR: worktree path not found: ${target_dir}" >&2
  exit 2
fi

if ! git -C "${target_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "ERROR: target path is not inside a git work tree: ${target_dir}" >&2
  exit 2
fi

case "${mode}" in
  suggest)
    sandbox_mode="read-only"
    approval_mode="never"
    bypass_flag=""
    ;;
  auto_edit)
    sandbox_mode="workspace-write"
    approval_mode="never"
    bypass_flag=""
    ;;
  full_auto)
    if [[ "${allow_write}" == "1" && "${allow_shell}" == "1" ]]; then
      sandbox_mode="danger-full-access"
      approval_mode="never"
      bypass_flag="--dangerously-bypass-approvals-and-sandbox"
    elif [[ "${allow_write}" == "1" ]]; then
      sandbox_mode="workspace-write"
      approval_mode="never"
      bypass_flag=""
    else
      sandbox_mode="read-only"
      approval_mode="never"
      bypass_flag=""
    fi
    ;;
  *)
    echo "ERROR: unsupported mode '${mode}'" >&2
    exit 2
    ;;
esac

if [[ "${allow_write}" != "0" && "${allow_write}" != "1" ]]; then
  echo "ERROR: --allow-write must be 0 or 1" >&2
  exit 2
fi
if [[ "${allow_shell}" != "0" && "${allow_shell}" != "1" ]]; then
  echo "ERROR: --allow-shell must be 0 or 1" >&2
  exit 2
fi

if ! command -v codex >/dev/null 2>&1; then
  echo "ERROR: codex CLI not found on PATH" >&2
  exit 127
fi

if ! command -v timeout >/dev/null 2>&1; then
  echo "ERROR: timeout command not available" >&2
  exit 127
fi

tmp_dir="$(mktemp -d "${TMPDIR:-/tmp}/codex-wrapper.XXXXXX")"
stdout_file="${tmp_dir}/codex.stdout"
stderr_file="${tmp_dir}/codex.stderr"
last_message_file="${tmp_dir}/last-message.txt"
pre_status_file="${tmp_dir}/pre-status.txt"
post_status_file="${tmp_dir}/post-status.txt"
diff_file="${tmp_dir}/diff.patch"
changed_file_list="${tmp_dir}/changed-files.txt"
meta_file="${tmp_dir}/meta.json"
trap 'rm -rf "${tmp_dir}"' EXIT

git -C "${target_dir}" status --short --untracked-files=all > "${pre_status_file}" || true
start_ns="$(python3 - <<'PY'
import time
print(time.monotonic_ns())
PY
)"

set +e
timeout --signal=TERM --kill-after=10s "${timeout_s}" \
  codex ${bypass_flag:+$bypass_flag} --cd "${target_dir}" \
    --sandbox "${sandbox_mode}" \
    --ask-for-approval "${approval_mode}" \
    exec --json --ignore-user-config --ignore-rules --ephemeral \
    --output-last-message "${last_message_file}" \
    - < "${prompt_file}" \
    > "${stdout_file}" 2> "${stderr_file}"
exit_code=$?
set -e

if [[ -f "${last_message_file}" ]]; then
  :
fi

git -C "${target_dir}" status --short --untracked-files=all > "${post_status_file}" || true

git -C "${target_dir}" diff --name-only --diff-filter=ACMRTUXB > "${changed_file_list}" || true
# Include untracked files from status if present.
python3 - "${post_status_file}" "${changed_file_list}" <<'PY'
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
existing = set(line.strip() for line in out_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()) if out_path.exists() else set()
for line in status_path.read_text(encoding="utf-8", errors="replace").splitlines():
    line = line.rstrip("\n")
    if line.startswith("?? "):
        existing.add(line[3:])
out_path.write_text("\n".join(sorted(existing)) + ("\n" if existing else ""), encoding="utf-8")
PY

git -C "${target_dir}" diff --patch --binary --no-color --no-ext-diff > "${diff_file}" || true

end_ns="$(python3 - <<'PY'
import time
print(time.monotonic_ns())
PY
)"
runtime_s="$(python3 - <<PY
start = int(${start_ns})
end = int(${end_ns})
print((end - start) / 1_000_000_000)
PY
)"

timed_out="false"
if [[ ${exit_code} -eq 124 ]]; then
  timed_out="true"
fi

python3 - "${stdout_file}" "${stderr_file}" "${pre_status_file}" "${post_status_file}" "${changed_file_list}" "${diff_file}" "${meta_file}" "${exit_code}" "${runtime_s}" "${timed_out}" "${mode}" "${repo_path}" "${target_dir}" <<'PY'
import json
import sys
from pathlib import Path

stdout_path = Path(sys.argv[1])
stderr_path = Path(sys.argv[2])
pre_status_path = Path(sys.argv[3])
post_status_path = Path(sys.argv[4])
changed_path = Path(sys.argv[5])
diff_path = Path(sys.argv[6])
meta_path = Path(sys.argv[7])
exit_code = int(sys.argv[8])
runtime_s = float(sys.argv[9])
timed_out = sys.argv[10].lower() == "true"
mode = sys.argv[11]
repo_path = sys.argv[12]
target_dir = sys.argv[13]

stdout = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.exists() else ""
pre_status = pre_status_path.read_text(encoding="utf-8", errors="replace") if pre_status_path.exists() else ""
post_status = post_status_path.read_text(encoding="utf-8", errors="replace") if post_status_path.exists() else ""
changed_files = [line.strip() for line in changed_path.read_text(encoding="utf-8", errors="replace").splitlines() if line.strip()] if changed_path.exists() else []
diff_text = diff_path.read_text(encoding="utf-8", errors="replace") if diff_path.exists() else ""

payload = {
    "exit_code": exit_code,
    "stdout": stdout,
    "stderr": stderr,
    "changed_files": changed_files,
    "diff_text": diff_text,
    "runtime_s": runtime_s,
    "timed_out": timed_out,
    "pre_git_status": pre_status,
    "post_git_status": post_status,
    "metadata": {
        "mode": mode,
        "repo_path": repo_path,
        "target_dir": target_dir,
        "wrapper": "codex_wrapper.sh",
        "command": "codex exec",
    },
}
meta_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY

exit ${exit_code}
