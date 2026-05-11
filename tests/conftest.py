import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--run-gpu",
        action="store_true",
        default=False,
        help="run tests that may validate or start a real GPU-backed vLLM service",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="need --run-gpu to run GPU/vLLM tests")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
