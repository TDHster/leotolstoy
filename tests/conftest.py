"""Общие фикстуры для API-тестов: мокаем эмбеддер, индекс и LLM,
чтобы тесты не ходили в сеть, не грузили модель и не тратили токены."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


class FakeRetriever:
    """Мини-заглушка индекса — без файлов и модели."""
    def __init__(self, *a, **k):
        pass

    def search(self, qvec, top_k=5, types=None):
        return [
            {"text": "Труд есть условие жизни.", "source": "Письмо №1, 1880",
             "meta": {"type": "письмо", "year": "1880"}, "score": 0.7},
        ]

    def __len__(self):
        return 1


@pytest.fixture
def client(monkeypatch):
    """FastAPI TestClient с замоканными эмбеддером/индексом/LLM и заданными кредами."""
    # креды и провайдер — до импорта app
    monkeypatch.setenv("APP_USER", "tester")
    monkeypatch.setenv("APP_PASS", "secretpass123")
    monkeypatch.setenv("LLM_API_KEY", "fake-key")
    monkeypatch.setenv("LLM_MODEL", "fake/model")
    monkeypatch.setenv("AUTH_FAIL_DELAY", "0")  # без задержки в тестах
    monkeypatch.setenv("RATE_LIMIT_CHAT", "1000/minute")  # не мешает обычным тестам
    monkeypatch.setenv("RATE_LIMIT_LOGIN", "1000/minute")

    # свежий импорт app с этими env
    for m in ("app", "embed", "retriever"):
        sys.modules.pop(m, None)
    import embed
    monkeypatch.setattr(embed, "embed_query", lambda text: np.zeros(3, dtype=np.float32))

    import app as app_module
    monkeypatch.setattr(app_module, "Retriever", FakeRetriever)
    monkeypatch.setattr(app_module, "embed_query", lambda text: np.zeros(3, dtype=np.float32))

    # мок LLM: httpx.AsyncClient.post → фейковый ответ OpenAI-формата
    class FakeResp:
        status_code = 200
        text = ""
        def json(self):
            return {"choices": [{"message": {"content": "Отвечаю в духе Толстого."}}]}

    class FakeAsyncClient:
        def __init__(self, *a, **k): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def post(self, *a, **k): return FakeResp()

    monkeypatch.setattr(app_module.httpx, "AsyncClient", FakeAsyncClient)

    from fastapi.testclient import TestClient
    with TestClient(app_module.app) as c:
        c._app_module = app_module  # доступ к модулю из тестов при нужде
        yield c
