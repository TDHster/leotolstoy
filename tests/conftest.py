"""Общие фикстуры для API-тестов: мокаем эмбеддер, индекс и LLM,
чтобы тесты не ходили в сеть, не грузили модель и не тратили токены."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))


class FakeEmbedder:
    """Заглушка эмбеддера — возвращает нулевые векторы."""
    @property
    def dimensions(self):
        return 3

    @property
    def model_name(self):
        return "fake-model"

    def embed_query(self, text):
        return np.zeros(3, dtype=np.float32)

    def embed_passages(self, texts, batch_size=32):
        return np.zeros((len(texts), 3), dtype=np.float32)


class FakeRetriever:
    """Мини-заглушка индекса — без файлов и модели."""
    def __init__(self, *a, **k):
        self.metadata = {
            "model": "fake",
            "dimensions": 3,
            "api": False,
            "display_name": "Fake Model"
        }

    def search(self, qvec, top_k=5, types=None, addressee=None,
               year_from=None, year_to=None, volume=None):
        return [
            {"text": "Труд есть условие жизни.", "source": "Письмо №1, 1880",
             "meta": {"type": "письмо", "year": "1880", "age": 52, "doc": "0"}, "score": 0.7},
        ]

    def browse(self, types=None, addressee=None, year_from=None, year_to=None,
               volume=None, limit=20, offset=0):
        records = [{
            "doc": "0", "type": "письмо", "date": "1880-01-01", "year": "1880",
            "volume": "63", "addressee": "Страхов Николай Николаевич",
            "number": "1", "entry": "1. Страхов Николай Николаевич, 1880",
            "preview": "Труд есть условие жизни.",
        }]
        return len(records), records[offset:offset + limit]

    def grep(self, pattern, whole_word=False, types=None, addressee=None,
             year_from=None, year_to=None, volume=None, limit=20, offset=0,
             full=False):
        if pattern.lower() not in "труд есть условие жизни.".lower():
            return {"count": 0, "chunks_matched": 0, "results": []}
        if whole_word and pattern.casefold() == "труд есть":
            return {"count": 0, "chunks_matched": 0, "results": []}
        return {
            "count": 1,
            "chunks_matched": 1,
            "results": [{
                # full=False — имитация обрезанного сниппета
                "text": "Труд есть условие жизни." if full else "Труд есть…",
                "source": "Письмо №1, 1880",
                "meta": {"type": "письмо", "year": "1880", "doc": "0"},
                "chunk_id": "v63_001#0",
                "matches": 1,
                "spans": [[0, 4]],
            }],
        }

    def facets(self):
        return {
            "records_total": 1,
            "types": {"письмо": 1},
            "years": {"min": "1880", "max": "1880"},
            "addressees": {"total": 1, "top": [{"name": "Страхов Николай Николаевич", "letters": 1}]},
            "volumes": {"63": 1},
        }

    def get_embedder(self):
        return FakeEmbedder()

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
    monkeypatch.setenv("MCP_API_KEY", "test-mcp-token")  # /mcp смонтирован

    # свежий импорт app с этими env
    for m in ("app", "embed", "retriever", "embedders"):
        sys.modules.pop(m, None)

    import app as app_module
    monkeypatch.setattr(app_module, "Retriever", FakeRetriever)

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

