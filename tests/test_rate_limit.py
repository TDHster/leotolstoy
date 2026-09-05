"""Тест rate limiting: при низком лимите лишние запросы → 429."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from conftest import FakeRetriever  # noqa: E402


@pytest.fixture
def low_limit_client(monkeypatch):
    monkeypatch.setenv("APP_USER", "tester")
    monkeypatch.setenv("APP_PASS", "secretpass123")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("AUTH_FAIL_DELAY", "0")
    monkeypatch.setenv("RATE_LIMIT_LOGIN", "3/minute")  # низкий лимит на вход
    monkeypatch.setenv("RATE_LIMIT_CHAT", "3/minute")

    for m in ("app", "embed", "retriever"):
        sys.modules.pop(m, None)
    import embed
    monkeypatch.setattr(embed, "embed_query", lambda t: np.zeros(3, dtype=np.float32))
    import app as app_module
    monkeypatch.setattr(app_module, "Retriever", FakeRetriever)
    monkeypatch.setattr(app_module, "embed_query", lambda t: np.zeros(3, dtype=np.float32))

    from fastapi.testclient import TestClient
    with TestClient(app_module.app) as c:
        yield c


def test_login_rate_limited(low_limit_client):
    auth = ("tester", "secretpass123")
    codes = [low_limit_client.get("/", auth=auth).status_code for _ in range(6)]
    assert 200 in codes            # часть прошла
    assert 429 in codes            # часть отбита лимитом
    # первые 3 — 200, дальше 429
    assert codes[:3] == [200, 200, 200]
    assert codes[3] == 429
