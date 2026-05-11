import pytest

pytest.importorskip("httpx", reason="benchmark runtime dependencies are not installed")

from scripts.benchmark import normalize_base_url, percentile, try_parse_json


def test_normalize_base_url_strips_v1_suffix():
    assert normalize_base_url("http://localhost:8001/v1") == "http://localhost:8001"
    assert normalize_base_url("http://localhost:8001/") == "http://localhost:8001"


def test_percentile_interpolates():
    assert percentile([10, 20, 30], 0.5) == 20
    assert percentile([10, 20], 0.95) == 19.5


def test_try_parse_json_strips_markdown_fence():
    ok, payload = try_parse_json('```json\n{"symbol":"AAPL"}\n```')

    assert ok is True
    assert payload == {"symbol": "AAPL"}
