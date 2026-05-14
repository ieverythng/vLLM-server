#!/usr/bin/env python3
"""Start, stop, inspect, and dry-run the configured vLLM backend."""

from __future__ import annotations

import os
import platform
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

import yaml

BASE_DIR = Path(__file__).parent.resolve()


class VLLMConfigError(RuntimeError):
    """Raised when the selected vLLM profile cannot be used."""


class VLLMManager:
    """Manages the vLLM backend process."""

    PROFILE_OPTIONS = {
        "quantization": "--quantization",
        "gpu_memory_utilization": "--gpu-memory-utilization",
        "max_model_len": "--max-model-len",
        "dtype": "--dtype",
        "tensor_parallel_size": "--tensor-parallel-size",
        "cpu_offload_gb": "--cpu-offload-gb",
        "max_num_seqs": "--max-num-seqs",
        "reasoning_parser": "--reasoning-parser",
        "tool_call_parser": "--tool-call-parser",
    }
    PROFILE_FLAGS = {
        "enable_prefix_caching": "--enable-prefix-caching",
        "language_model_only": "--language-model-only",
        "enable_auto_tool_choice": "--enable-auto-tool-choice",
    }

    def __init__(self, config_path: str | None = None, models_path: str | None = None):
        self.config_path = Path(config_path) if config_path else BASE_DIR / "config.yaml"
        self.models_path = Path(models_path) if models_path else BASE_DIR / "models.yaml"
        self.config = self._load_yaml(self.config_path)
        self.models = self._load_yaml(self.models_path)
        self.process: Optional[subprocess.Popen[str]] = None
        self.pid_file = BASE_DIR / "vllm.pid"
        self._supported_flags_cache: Optional[set[str]] = None

    @staticmethod
    def _load_yaml(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise VLLMConfigError(f"missing config file: {path}")
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _append_option(cmd: list[str], flag: str, value: object) -> None:
        if value not in (None, "", False):
            cmd.extend([flag, str(value)])

    @staticmethod
    def _append_flag(cmd: list[str], flag: str, enabled: bool) -> None:
        if enabled:
            cmd.append(flag)

    @staticmethod
    def _python_executable() -> str:
        override = os.environ.get("VLLM_PYTHON", "").strip()
        if override and Path(override).exists():
            return override

        current = Path(sys.executable)
        if current.exists():
            return str(current)

        if os.name == "nt":
            candidates = [
                BASE_DIR / ".venv311" / "Scripts" / "python.exe",
                BASE_DIR / ".venv" / "Scripts" / "python.exe",
            ]
        else:
            candidates = [
                BASE_DIR / "venv" / "bin" / "python",
                BASE_DIR / ".venv311" / "bin" / "python",
                BASE_DIR / ".venv" / "bin" / "python",
            ]
        for candidate in candidates:
            if candidate.exists():
                return str(candidate)
        return str(current)

    def _runtime_probe(self) -> dict[str, Any]:
        python_exe = self._python_executable()
        probe = (
            "import platform,sys; "
            "import vllm; "
            "print(f'python={sys.version.split()[0]} platform={platform.system()} vllm={vllm.__version__}'); "
            "import vllm._C; "
            "print('vllm._C=ok')"
        )
        result = subprocess.run(
            [python_exe, "-c", probe],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
        stdout = (result.stdout or "").strip()
        stderr = (result.stderr or "").strip()
        ok = result.returncode == 0 and "vllm._C=ok" in stdout
        message = "runtime_ok" if ok else "runtime_probe_failed"
        if (not ok) and "No module named 'vllm._C'" in stderr:
            message = "missing_vllm_compiled_runtime"
        return {
            "ok": ok,
            "python": python_exe,
            "message": message,
            "stdout": stdout,
            "stderr": stderr,
            "platform": platform.system(),
        }

    def _supported_cli_flags(self) -> set[str]:
        if self._supported_flags_cache is not None:
            return self._supported_flags_cache

        python_exe = self._python_executable()
        result = subprocess.run(
            [python_exe, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            capture_output=True,
            text=True,
            cwd=str(BASE_DIR),
        )
        help_text = f"{result.stdout or ''}\n{result.stderr or ''}"
        self._supported_flags_cache = set(re.findall(r"--[a-z0-9][a-z0-9-]*", help_text))
        return self._supported_flags_cache

    def preflight(self) -> bool:
        probe = self._runtime_probe()
        payload = {
            "python": probe["python"],
            "platform": probe["platform"],
            "runtime_ok": probe["ok"],
            "message": probe["message"],
            "stdout": probe["stdout"],
            "stderr": probe["stderr"],
            "gpu": self._get_gpu_info(),
            "active_profile": self.active_profile_name(),
            "model": self._model_ref(self.active_profile()),
        }
        print(yaml.dump(payload, default_flow_style=False, sort_keys=False))
        if probe["ok"]:
            return True
        if probe["message"] == "missing_vllm_compiled_runtime":
            print(
                "Hint: compiled vLLM runtime is missing in this interpreter. "
                "Use a Linux/WSL2 CUDA environment or an interpreter that includes vllm._C."
            )
        return False

    def active_profile_name(self) -> str:
        runtime = self.config.get("runtime", {})
        name = runtime.get("active_model_profile")
        if not name:
            raise VLLMConfigError("runtime.active_model_profile is required in config.yaml")
        return str(name)

    def active_profile(self) -> dict[str, Any]:
        profiles = self.models.get("profiles", {})
        name = self.active_profile_name()
        profile = profiles.get(name)
        if not isinstance(profile, dict):
            raise VLLMConfigError(f"active profile not found in models.yaml: {name}")
        if profile.get("backend") != "vllm":
            raise VLLMConfigError(f"profile {name} is not a vLLM profile")
        if not profile.get("model_id"):
            raise VLLMConfigError(f"profile {name} must define model_id")
        return profile

    @staticmethod
    def _resolve_local_model_path(local_path: str | None) -> str | None:
        if not local_path:
            return None

        raw = str(local_path).strip()
        if not raw:
            return None

        candidate = Path(raw)
        if candidate.exists():
            return str(candidate)

        # In WSL/Linux, map Windows drive paths like D:/MODELS/... or D:\MODELS\...
        # to /mnt/d/MODELS/... so local model caching can still be reused.
        if os.name != "nt":
            m = re.match(r"^([A-Za-z]):[\\/](.*)$", raw)
            if m:
                drive = m.group(1).lower()
                suffix = m.group(2).replace("\\", "/")
                wsl_candidate = Path(f"/mnt/{drive}/{suffix}")
                if wsl_candidate.exists():
                    return str(wsl_candidate)
        return None

    @staticmethod
    def _model_ref(profile: dict[str, Any]) -> str:
        local_path = VLLMManager._resolve_local_model_path(profile.get("local_path"))
        if profile.get("prefer_local_path") and local_path:
            return local_path
        return str(profile["model_id"])

    def _build_command(self) -> list[str]:
        """Build the vLLM OpenAI server command from the active profile."""
        profile = self.active_profile()
        backend_cfg = self.config.get("backend", {})
        supported_flags = self._supported_cli_flags()
        cmd = [
            self._python_executable(),
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            self._model_ref(profile),
            "--host",
            str(backend_cfg.get("host", "127.0.0.1")),
            "--port",
            str(backend_cfg.get("port", 8000)),
        ]

        for key, flag in self.PROFILE_OPTIONS.items():
            value = profile.get(key)
            if key == "tensor_parallel_size" and int(value or 1) <= 1:
                continue
            if flag not in supported_flags:
                continue
            self._append_option(cmd, flag, value)
        for key, flag in self.PROFILE_FLAGS.items():
            if flag not in supported_flags:
                continue
            self._append_flag(cmd, flag, bool(profile.get(key)))

        api_key = os.environ.get("VLLM_API_KEY") or self.config.get("auth", {}).get("api_key", "")
        if "--api-key" in supported_flags:
            self._append_option(cmd, "--api-key", api_key)
        return cmd

    def dry_run(self) -> list[str]:
        cmd = self._build_command()
        print("Active profile:", self.active_profile_name())
        print("Command:")
        print(" ".join(cmd))
        return cmd

    def start(self, wait_ready: bool = True, timeout: int = 300) -> bool:
        """Start the vLLM server. Returns True if successful."""
        if self.is_running():
            print("vLLM server is already running.")
            return True
        probe = self._runtime_probe()
        if not probe["ok"]:
            print("ERROR: vLLM runtime preflight failed.")
            print(f"  python: {probe['python']}")
            if probe["stdout"]:
                print(f"  stdout: {probe['stdout']}")
            if probe["stderr"]:
                print(f"  stderr: {probe['stderr']}")
            if probe["message"] == "missing_vllm_compiled_runtime":
                print(
                    "  hint: missing compiled vLLM runtime (vllm._C). "
                    "This environment cannot launch vLLM until that dependency is available."
                )
            return False
        if not self._check_gpu():
            print("ERROR: GPU not available or insufficient memory.")
            return False

        cmd = self._build_command()
        print("Starting vLLM server...")
        print(f"Command: {' '.join(cmd)}")

        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # VLLM_PYTHON is a manager-only override and is not recognized by vLLM.
        env.pop("VLLM_PYTHON", None)

        log_file = open(BASE_DIR / "vllm.log", "a", encoding="utf-8")
        self.process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        self.pid_file.write_text(str(self.process.pid), encoding="utf-8")
        print(f"vLLM server started (PID: {self.process.pid})")

        if wait_ready:
            return self._wait_ready(timeout)
        return True

    def _wait_ready(self, timeout: int) -> bool:
        """Wait until vLLM is ready to serve requests."""
        import httpx

        backend_cfg = self.config.get("backend", {})
        host = backend_cfg.get("host", "127.0.0.1")
        port = backend_cfg.get("port", 8000)
        url = f"http://{host}:{port}/v1/models"

        print(f"Waiting for vLLM to be ready at {url}...")
        start = time.time()
        client = httpx.Client(timeout=5.0)
        headers = self._backend_auth_headers()

        while time.time() - start < timeout:
            try:
                resp = client.get(url, headers=headers)
                if resp.status_code == 200:
                    print(f"vLLM is ready! Models: {self._model_ids(resp.json())}")
                    return True
            except Exception:
                pass
            time.sleep(3)
            elapsed = int(time.time() - start)
            if elapsed and elapsed % 15 == 0:
                print(f"  Still waiting... ({elapsed}s)")

        print("ERROR: vLLM failed to start within timeout.")
        self.stop()
        return False

    def stop(self, graceful: bool = True) -> bool:
        """Stop the vLLM server."""
        if not self.is_running():
            print("vLLM server is not running.")
            self.pid_file.unlink(missing_ok=True)
            return False

        pid = self._read_pid()
        print(f"Stopping vLLM server (PID: {pid})...")

        if graceful and pid is not None:
            try:
                os.kill(pid, signal.SIGTERM)
                for _ in range(30):
                    if not self._process_alive(pid):
                        print("vLLM server stopped gracefully.")
                        self.pid_file.unlink(missing_ok=True)
                        return True
                    time.sleep(1)
                print("Graceful shutdown timed out, forcing...")
            except ProcessLookupError:
                pass

        if pid is not None:
            try:
                subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], capture_output=True, text=True)
            except Exception:
                try:
                    os.kill(pid, signal.SIGTERM)
                except OSError:
                    pass
        self.pid_file.unlink(missing_ok=True)
        print("vLLM server stopped.")
        return True

    def restart(self) -> bool:
        """Restart the vLLM server."""
        self.stop()
        time.sleep(2)
        return self.start()

    def is_running(self) -> bool:
        """Check if vLLM server is running."""
        if not self.pid_file.exists():
            return False
        pid = self._read_pid()
        return pid is not None and self._process_alive(pid)

    def get_status(self) -> dict[str, Any]:
        """Get server status info."""
        running = self.is_running()
        pid = self._read_pid() if running else None
        profile = self.active_profile()
        return {
            "running": running,
            "pid": pid,
            "python": self._python_executable(),
            "active_profile": self.active_profile_name(),
            "model": self._model_ref(profile),
            "model_source": profile.get("model_id"),
            "runtime": self._runtime_probe(),
            "gpu": self._get_gpu_info(),
        }

    def _read_pid(self) -> Optional[int]:
        if not self.pid_file.exists():
            return None
        text = self.pid_file.read_text(encoding="utf-8").strip()
        return int(text) if text.isdigit() else None

    @staticmethod
    def _process_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    @staticmethod
    def _model_ids(payload: object) -> list[str]:
        models = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(models, list):
            return []
        return [model["id"] for model in models if isinstance(model, dict) and "id" in model]

    def _backend_auth_headers(self) -> dict[str, str]:
        api_key = os.environ.get("VLLM_API_KEY") or self.config.get("auth", {}).get("api_key", "")
        return {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def _check_gpu(self) -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                return False
            lines = result.stdout.strip().split("\n")[1:]
            for line in lines:
                total, used = line.split(",")
                total_mb = int(total.strip().replace(" MiB", ""))
                used_mb = int(used.strip().replace(" MiB", ""))
                free_mb = total_mb - used_mb
                print(f"GPU: {total_mb} MiB total, {used_mb} MiB used, {free_mb} MiB free")
                if free_mb < 12288:
                    print(f"WARNING: Only {free_mb} MiB GPU memory free. May need to free resources.")
            return True
        except FileNotFoundError:
            print("ERROR: nvidia-smi not found. Is CUDA installed?")
            return False

    def _get_gpu_info(self) -> dict[str, Any]:
        """Get current GPU status."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.used,temperature.gpu",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                parts = [part.strip() for part in result.stdout.strip().split(",")]
                return {
                    "name": parts[0] if len(parts) > 0 else "unknown",
                    "memory_total_mb": int(parts[1].replace(" MiB", "")) if len(parts) > 1 else 0,
                    "memory_used_mb": int(parts[2].replace(" MiB", "")) if len(parts) > 2 else 0,
                    "temperature_c": int(parts[3].replace(" C", "").replace(" degC", "")) if len(parts) > 3 else 0,
                }
        except Exception:
            pass
        return {}


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Server Manager")
    parser.add_argument("action", choices=["dry-run", "preflight", "start", "stop", "restart", "status"])
    parser.add_argument("--config", help="Path to config.yaml")
    parser.add_argument("--models", help="Path to models.yaml")
    parser.add_argument("--no-wait", action="store_true", help="Do not wait for server readiness")
    args = parser.parse_args()

    try:
        manager = VLLMManager(args.config, args.models)
        if args.action == "dry-run":
            manager.dry_run()
            return 0
        if args.action == "start":
            return 0 if manager.start(wait_ready=not args.no_wait) else 1
        if args.action == "preflight":
            return 0 if manager.preflight() else 1
        if args.action == "stop":
            manager.stop()
            return 0
        if args.action == "restart":
            return 0 if manager.restart() else 1
        if args.action == "status":
            print(yaml.dump(manager.get_status(), default_flow_style=False, sort_keys=False))
            return 0
    except VLLMConfigError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
