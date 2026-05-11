import os

import pytest

from vllm_manager import VLLMManager


@pytest.mark.gpu
def test_active_profile_dry_run_is_valid():
    cmd = VLLMManager()._build_command()

    assert "vllm.entrypoints.openai.api_server" in cmd
    assert "--model" in cmd
    assert "--max-model-len" in cmd


@pytest.mark.gpu
def test_gpu_smoke_model_override_is_documented():
    model = os.environ.get("VLLM_SMOKE_MODEL", "cyankiwi/Qwen3.6-27B-AWQ-INT4")

    assert model
