# Provide isolated local fixtures for the collaboration runtime tests.

import pytest
from aos.workspace import Workspace


# Keep local fixture requests independent of external proxy configuration.
@pytest.fixture(autouse=True)
def local_test_transport(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)


# Initialize a fresh artifact workspace for each task and workspace test.
@pytest.fixture
def ws(tmp_path):
    return Workspace.create(tmp_path / "ws", goal="test goal")
