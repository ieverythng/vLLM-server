#!/usr/bin/env python3
"""Controlled adapter for supervised coding-harness execution.

This module gives Hermes/Watson a thin abstraction for invoking external dev
harnesses in a reproducible way. The first supported target is `codex_cli`.

Contract (per handoff):
    run_dev_harness(
        harness: str,
        repo_path: str,
        prompt: str,
        mode: str = "suggest",
        timeout_s: int = 900,
        allow_write: bool = False,
        allow_shell: bool = False,
        worktree: str | None = None,
    ) -> HarnessRunResult
"""

from __future__ import annotations

import dataclasses
import json
import os
import signal
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_WRAPPER = ROOT_DIR / "scripts" / "codex_wrapper.sh"
SUPPORTED_HARNESSES = {"codex_cli"}
SUPPORTED_MODES = {"suggest", "auto_edit", "full_auto"}


@dataclass
class HarnessRunResult:
    harness: str
    repo_path: str
    prompt: str
    mode: str
    timeout_s: int
    allow_write: bool
    allow_shell: bool
    worktree: Optional[str]
    command: List[str]
    exit_code: int
    stdout: str
    stderr: str
    changed_files: List[str] = field(default_factory=list)
    diff_text: str = ""
    runtime_s: float = 0.0
    timed_out: bool = False
    pre_git_status: str = ""
    post_git_status: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class DevHarnessError(RuntimeError):
    """Raised when a harness invocation cannot be prepared or parsed."""


class DevHarnessAdapter:
    """Thin orchestrator for supervised coding harnesses.

    The adapter is intentionally small: it validates inputs, launches the
    wrapper, captures output, and returns a structured result object.
    """

    def __init__(self, wrapper_path: Path | None = None) -> None:
        self.wrapper_path = Path(wrapper_path) if wrapper_path else DEFAULT_WRAPPER

    def run_dev_harness(
        self,
        harness: str,
        repo_path: str,
        prompt: str,
        mode: str = "suggest",
        timeout_s: int = 900,
        allow_write: bool = False,
        allow_shell: bool = False,
        worktree: str | None = None,
    ) -> HarnessRunResult:
        repo = Path(repo_path).expanduser().resolve()
        if not repo.exists():
            raise DevHarnessError(f"repo_path does not exist: {repo}")
        if not repo.is_dir():
            raise DevHarnessError(f"repo_path is not a directory: {repo}")

        target_dir = Path(worktree).expanduser().resolve() if worktree else repo
        if not target_dir.exists():
            raise DevHarnessError(f"worktree does not exist: {target_dir}")
        if not target_dir.is_dir():
            raise DevHarnessError(f"worktree is not a directory: {target_dir}")

        if harness not in SUPPORTED_HARNESSES:
            raise DevHarnessError(f"unsupported harness: {harness}")
        if mode not in SUPPORTED_MODES:
            raise DevHarnessError(f"unsupported mode: {mode}")
        if timeout_s <= 0:
            raise DevHarnessError("timeout_s must be positive")
        if not self.wrapper_path.exists():
            raise DevHarnessError(f"wrapper script missing: {self.wrapper_path}")

        command = [
            str(self.wrapper_path),
            "--harness",
            harness,
            "--repo-path",
            str(repo),
            "--mode",
            mode,
            "--timeout-s",
            str(timeout_s),
            "--allow-write",
            "1" if allow_write else "0",
            "--allow-shell",
            "1" if allow_shell else "0",
        ]
        if worktree:
            command.extend(["--worktree", str(target_dir)])

        env = self._build_sandbox_env()
        env["PYTHONUNBUFFERED"] = "1"

        start = time.monotonic()
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False) as prompt_file:
            prompt_file.write(prompt)
            prompt_file_path = Path(prompt_file.name)

        command.extend(["--prompt-file", str(prompt_file_path)])

        proc = subprocess.Popen(
            command,
            cwd=str(ROOT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(timeout=timeout_s + 120)
            exit_code = proc.returncode if proc.returncode is not None else 1
            wrapper_timeout = False
        except subprocess.TimeoutExpired:
            wrapper_timeout = True
            self._terminate_process_group(proc)
            stdout, stderr = proc.communicate()
            exit_code = proc.returncode if proc.returncode is not None else 124
        finally:
            try:
                prompt_file_path.unlink(missing_ok=True)
            except Exception:
                pass

        runtime_s = time.monotonic() - start
        parsed = self._parse_wrapper_result(stdout, stderr)

        timed_out = wrapper_timeout or bool(parsed.get("timed_out", False))
        result = HarnessRunResult(
            harness=harness,
            repo_path=str(repo),
            prompt=prompt,
            mode=mode,
            timeout_s=timeout_s,
            allow_write=allow_write,
            allow_shell=allow_shell,
            worktree=str(target_dir) if worktree else None,
            command=command,
            exit_code=int(parsed.get("exit_code", exit_code)),
            stdout=str(parsed.get("stdout", stdout)),
            stderr=str(parsed.get("stderr", stderr)),
            changed_files=list(parsed.get("changed_files", [])),
            diff_text=str(parsed.get("diff_text", "")),
            runtime_s=float(parsed.get("runtime_s", runtime_s)),
            timed_out=timed_out,
            pre_git_status=str(parsed.get("pre_git_status", "")),
            post_git_status=str(parsed.get("post_git_status", "")),
            metadata=dict(parsed.get("metadata", {})),
        )
        if wrapper_timeout and not result.metadata.get("wrapper_timeout"):
            result.metadata["wrapper_timeout"] = True
        return result

    def _build_sandbox_env(self) -> Dict[str, str]:
        env = os.environ.copy()
        # Default to no outbound network access for supervised runs.
        env["HTTP_PROXY"] = "http://127.0.0.1:9"
        env["HTTPS_PROXY"] = "http://127.0.0.1:9"
        env["ALL_PROXY"] = "http://127.0.0.1:9"
        env["NO_PROXY"] = "127.0.0.1,localhost,::1"
        env["http_proxy"] = env["HTTP_PROXY"]
        env["https_proxy"] = env["HTTPS_PROXY"]
        env["all_proxy"] = env["ALL_PROXY"]
        env["no_proxy"] = env["NO_PROXY"]
        return env

    def _terminate_process_group(self, proc: subprocess.Popen[str]) -> None:
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception:
            pass

        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return
            time.sleep(0.2)

        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        except Exception:
            pass

    def _parse_wrapper_result(self, stdout: str, stderr: str) -> Dict[str, Any]:
        text = stdout.strip()
        if not text:
            return {}

        # The wrapper emits a single JSON object, but if the harness streams
        # additional logs we try the last non-empty line first.
        candidates = [line.strip() for line in text.splitlines() if line.strip()]
        for candidate in reversed(candidates):
            try:
                payload = json.loads(candidate)
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload

        try:
            payload = json.loads(text)
            if isinstance(payload, dict):
                return payload
        except Exception:
            return {
                "stdout": stdout,
                "stderr": stderr,
                "metadata": {"parse_error": "wrapper output was not valid JSON"},
            }
        return {}


def run_dev_harness(
    harness: str,
    repo_path: str,
    prompt: str,
    mode: str = "suggest",
    timeout_s: int = 900,
    allow_write: bool = False,
    allow_shell: bool = False,
    worktree: str | None = None,
) -> HarnessRunResult:
    """Convenience wrapper matching the handoff contract."""

    return DevHarnessAdapter().run_dev_harness(
        harness=harness,
        repo_path=repo_path,
        prompt=prompt,
        mode=mode,
        timeout_s=timeout_s,
        allow_write=allow_write,
        allow_shell=allow_shell,
        worktree=worktree,
    )


__all__ = [
    "DevHarnessAdapter",
    "DevHarnessError",
    "HarnessRunResult",
    "run_dev_harness",
]
