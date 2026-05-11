import pytest

pytest.importorskip("fastapi", reason="gateway runtime dependencies are not installed")
pytest.importorskip("httpx", reason="gateway runtime dependencies are not installed")

from server import ChatCompletionRequest, Message, build_chat_payload, model_ids, resolve_model_name


def test_model_ids_support_openai_shape():
    assert model_ids({"data": [{"id": "model-a"}, {"id": "model-b"}]}) == ["model-a", "model-b"]


def test_default_model_resolves_to_active_profile():
    resolved = resolve_model_name("default")

    assert resolved == "cyankiwi/Qwen3.6-27B-AWQ-INT4"


def test_build_chat_payload_applies_role_defaults_and_default_model():
    request = ChatCompletionRequest(
        model="default",
        user="iTrader",
        messages=[Message(role="user", content="Return JSON.")],
    )

    payload = build_chat_payload(request)

    assert payload["model"] == "cyankiwi/Qwen3.6-27B-AWQ-INT4"
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 2048
