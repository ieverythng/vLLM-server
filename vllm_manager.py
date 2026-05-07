#!/usr/bin/env python3
"""
vLLM Manager — Start, stop, and monitor the vLLM server process.
Handles GPU memory management and graceful shutdown.
"""

import subprocess
import sys
import os
import signal
import time
import yaml
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).parent.resolve()


class VLLMManager:
    """Manages the vLLM server lifecycle."""

    def __init__(self, config_path: str | None = None):
        self.config_path = Path(config_path) if config_path else BASE_DIR / "config.yaml"
        self.config = self._load_config()
        self.process: Optional[subprocess.Popen] = None
        self.pid_file = BASE_DIR / "vllm.pid"

    def _load_config(self) -> dict:
        with open(self.config_path) as f:
            return yaml.safe_load(f)

    def _build_command(self) -> list[str]:
        """Build the vLLM serve command from config."""
        vllm_cfg = self.config.get("vllm", {})
        server_cfg = self.config.get("server", {})

        cmd = [
            str((BASE_DIR / "venv" / "bin" / "python").resolve()),
            "-m", "vllm.entrypoints.openai.api_server",
            "--model", vllm_cfg.get("default_model", "Qwen/Qwen3.5-27B"),
            "--host", server_cfg.get("host", "0.0.0.0"),
            "--port", str(server_cfg.get("port", 8000)),
        ]

        # Optional params
        if vllm_cfg.get("quantization"):
            cmd.extend(["--quantization", vllm_cfg["quantization"]])
        if vllm_cfg.get("gpu_memory_utilization"):
            cmd.extend(["--gpu-memory-utilization", str(vllm_cfg["gpu_memory_utilization"])])
        if vllm_cfg.get("max_model_len"):
            cmd.extend(["--max-model-len", str(vllm_cfg["max_model_len"])])
        if vllm_cfg.get("dtype"):
            cmd.extend(["--dtype", vllm_cfg["dtype"]])
        if vllm_cfg.get("tensor_parallel_size", 1) > 1:
            cmd.extend(["--tensor-parallel-size", str(vllm_cfg["tensor_parallel_size"])])
        if vllm_cfg.get("enable_prefix_caching"):
            cmd.append("--enable-prefix-caching")

        # API key
        api_key = os.environ.get("VLLM_API_KEY", self.config.get("auth", {}).get("api_key", ""))
        if api_key:
            cmd.extend(["--api-key", f"bearer {api_key}"])

        return cmd

    def start(self, wait_ready: bool = True, timeout: int = 300) -> bool:
        """Start the vLLM server. Returns True if successful."""
        if self.is_running():
            print("vLLM server is already running.")
            return True

        # Check GPU availability
        gpu_ok = self._check_gpu()
        if not gpu_ok:
            print("ERROR: GPU not available or insufficient memory.")
            return False

        cmd = self._build_command()
        print(f"Starting vLLM server...")
        print(f"Command: {' '.join(cmd)}")

        env = os.environ.copy()
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

        self.process = subprocess.Popen(
            cmd,
            cwd=str(BASE_DIR),
            env=env,
            stdout=open(BASE_DIR / "vllm.log", "a"),
            stderr=subprocess.STDOUT,
        )

        # Write PID file
        self.pid_file.write_text(str(self.process.pid))
        print(f"vLLM server started (PID: {self.process.pid})")

        if wait_ready:
            return self._wait_ready(timeout)
        return True

    def _wait_ready(self, timeout: int) -> bool:
        """Wait until vLLM is ready to serve requests."""
        import httpx

        server_cfg = self.config.get("server", {})
        host = server_cfg.get("host", "0.0.0.0")
        port = server_cfg.get("port", 8000)
        url = f"http://127.0.0.1:{port}/v1/models"

        print(f"Waiting for vLLM to be ready at {url}...")
        start = time.time()
        client = httpx.Client(timeout=5.0)

        while time.time() - start < timeout:
            try:
                resp = client.get(url)
                if resp.status_code == 200:
                    models = resp.json()
                    print(f"vLLM is ready! Models: {[m['id'] for m in models]}")
                    return True
            except Exception:
                pass
            time.sleep(3)
            # Print progress every 15 seconds
            elapsed = int(time.time() - start)
            if elapsed % 15 == 0:
                print(f"  Still waiting... ({elapsed}s)")

        print("ERROR: vLLM failed to start within timeout.")
        self.stop()
        return False

    def stop(self, graceful: bool = True) -> bool:
        """Stop the vLLM server."""
        if not self.is_running():
            print("vLLM server is not running.")
            return False

        pid = self._read_pid()
        print(f"Stopping vLLM server (PID: {pid})...")

        if graceful:
            try:
                os.kill(pid, signal.SIGTERM)
                # Wait up to 30 seconds for graceful shutdown
                for _ in range(30):
                    if not self._process_alive(pid):
                        print("vLLM server stopped gracefully.")
                        self.pid_file.unlink(missing_ok=True)
                        return True
                    time.sleep(1)
                # Force kill if still alive
                print("Graceful shutdown timed out, forcing...")
            except ProcessLookupError:
                pass

        os.kill(pid, signal.SIGKILL)
        self.pid_file.unlink(missing_ok=True)
        print("vLLM server killed.")
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
        return self._process_alive(pid)

    def get_status(self) -> dict:
        """Get server status info."""
        running = self.is_running()
        pid = self._read_pid() if running else None

        # Check GPU memory
        gpu_info = self._get_gpu_info()

        return {
            "running": running,
            "pid": pid,
            "model": self.config.get("vllm", {}).get("default_model"),
            "gpu": gpu_info,
        }

    def _read_pid(self) -> Optional[int]:
        if self.pid_file.exists():
            return int(self.pid_file.read_text().strip())
        return None

    def _process_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def _check_gpu(self) -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode != 0:
                return False
            lines = result.stdout.strip().split("\n")[1:]  # Skip header
            for line in lines:
                total, used = line.split(",")
                total_mb = int(total.strip().replace(" MiB", ""))
                used_mb = int(used.strip().replace(" MiB", ""))
                free_mb = total_mb - used_mb
                print(f"GPU: {total_mb} MiB total, {used_mb} MiB used, {free_mb} MiB free")
                # Need at least 12 GB free for a 27B Q3 model
                if free_mb < 12288:
                    print(f"WARNING: Only {free_mb} MiB GPU memory free. May need to free up resources.")
            return True
        except FileNotFoundError:
            print("ERROR: nvidia-smi not found. Is CUDA installed?")
            return False

    def _get_gpu_info(self) -> dict:
        """Get current GPU status."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,temperature.gpu",
                 "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                return {
                    "name": parts[0].strip() if len(parts) > 0 else "unknown",
                    "memory_total_mb": int(parts[1].strip().replace(" MiB", "")) if len(parts) > 1 else 0,
                    "memory_used_mb": int(parts[2].strip().replace(" MiB", "")) if len(parts) > 2 else 0,
                    "temperature_c": int(parts[3].strip().replace(" °C", "")) if len(parts) > 3 else 0,
                }
        except Exception:
            pass
        return {}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="vLLM Server Manager")
    parser.add_argument("action", choices=["start", "stop", "restart", "status"])
    parser.add_argument("--config", help="Path to config file")
    parser.add_argument("--no-wait", action="store_true", help="Don't wait for server ready")
    args = parser.parse_args()

    manager = VLLMManager(args.config)

    if args.action == "start":
        success = manager.start(wait_ready=not args.no_wait)
        sys.exit(0 if success else 1)
    elif args.action == "stop":
        manager.stop()
    elif args.action == "restart":
        manager.restart()
    elif args.action == "status":
        status = manager.get_status()
        print(yaml.dump(status, default_flow_style=False))


if __name__ == "__main__":
    main()
