from pathlib import Path

import yaml

from vllm_manager import VLLMManager


def write_yaml(path: Path, data: dict) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def test_dry_run_builds_active_profile_command(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    config_path = tmp_path / "config.yaml"
    models_path = tmp_path / "models.yaml"
    write_yaml(
        config_path,
        {
            "runtime": {"active_model_profile": "qwen36"},
            "backend": {"host": "127.0.0.1", "port": 8000},
            "auth": {"api_key": ""},
        },
    )
    write_yaml(
        models_path,
        {
            "profiles": {
                "qwen36": {
                    "backend": "vllm",
                    "model_id": "cyankiwi/Qwen3.6-27B-AWQ-INT4",
                    "local_path": str(model_dir),
                    "prefer_local_path": True,
                    "quantization": "compressed-tensors",
                    "dtype": "auto",
                    "max_model_len": 16384,
                    "gpu_memory_utilization": 0.9,
                    "cpu_offload_gb": 8,
                    "swap_space": 8,
                    "max_num_seqs": 2,
                    "tensor_parallel_size": 1,
                    "enable_prefix_caching": True,
                    "language_model_only": True,
                    "reasoning_parser": "qwen3",
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "qwen3_coder",
                }
            }
        },
    )

    cmd = VLLMManager(str(config_path), str(models_path))._build_command()

    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == str(model_dir)
    assert "--quantization" in cmd
    assert cmd[cmd.index("--quantization") + 1] == "compressed-tensors"
    assert "--max-model-len" in cmd
    assert cmd[cmd.index("--max-model-len") + 1] == "16384"
    assert "--cpu-offload-gb" in cmd
    assert "--swap-space" in cmd
    assert "--max-num-seqs" in cmd
    assert "--enable-prefix-caching" in cmd
    assert "--language-model-only" in cmd
    assert "--enable-auto-tool-choice" in cmd
    assert "--tool-call-parser" in cmd
    assert "--tensor-parallel-size" not in cmd


def test_model_ids_support_openai_models_payload():
    payload = {"object": "list", "data": [{"id": "a"}, {"id": "b"}]}

    assert VLLMManager._model_ids(payload) == ["a", "b"]


def test_status_reports_active_model(tmp_path):
    config_path = tmp_path / "config.yaml"
    models_path = tmp_path / "models.yaml"
    write_yaml(
        config_path,
        {
            "runtime": {"active_model_profile": "qwen36"},
            "backend": {"host": "127.0.0.1", "port": 8000},
            "auth": {"api_key": ""},
        },
    )
    write_yaml(
        models_path,
        {
            "profiles": {
                "qwen36": {
                    "backend": "vllm",
                    "model_id": "cyankiwi/Qwen3.6-27B-AWQ-INT4",
                }
            }
        },
    )

    status = VLLMManager(str(config_path), str(models_path)).get_status()

    assert status["active_profile"] == "qwen36"
    assert status["model"] == "cyankiwi/Qwen3.6-27B-AWQ-INT4"
