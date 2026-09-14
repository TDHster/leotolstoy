"""Тесты OpenRouterEmbedder (server/embedders.py): пин провайдера и валидация.

httpx.Client подменяется: реальной сети нет, проверяем только payload и
обработку ответов.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

import embedders  # noqa: E402


@pytest.fixture
def api_mock(monkeypatch):
    """Подменяет embedders.httpx.Client; отвечает payload'ами по очереди."""
    calls = []

    def make(payloads):
        payloads = payloads if isinstance(payloads, list) else [payloads]
        it = iter(payloads)

        class FakeResponse:
            def __init__(self, payload):
                self._payload = payload
                self.status_code = 200
                self.headers = {}

            def json(self):
                return self._payload

        class FakeClient:
            def __init__(self, *args, **kwargs):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def post(self, url, json=None, headers=None):
                calls.append({"url": url, "json": json, "headers": headers})
                return FakeResponse(next(it))

        monkeypatch.setattr(embedders.httpx, "Client", FakeClient)
        return calls

    return make


def _ok_payload(dim=4, provider="Nebius"):
    return {"data": [{"embedding": [0.1] * dim}], "provider": provider}


def _embedder(**kwargs):
    return embedders.OpenRouterEmbedder(model="qwen/qwen3-embedding-8b", api_key="k", **kwargs)


def test_provider_pin_in_payload(api_mock):
    calls = api_mock(_ok_payload())
    _embedder(provider="nebius").embed_query("текст")
    assert calls[0]["json"]["provider"] == {"only": ["nebius"], "allow_fallbacks": False}


def test_no_pin_without_provider(api_mock):
    calls = api_mock(_ok_payload())
    _embedder(provider=None).embed_query("текст")
    assert "provider" not in calls[0]["json"]


def test_provider_list_split(api_mock):
    calls = api_mock(_ok_payload())
    _embedder(provider="nebius, deepinfra").embed_query("текст")
    assert calls[0]["json"]["provider"] == {
        "only": ["nebius", "deepinfra"],
        "allow_fallbacks": False,
    }


def test_embed_query_normalizes(api_mock):
    api_mock({"data": [{"embedding": [3.0, 4.0]}]})
    v = _embedder().embed_query("текст")
    np.testing.assert_allclose(v, [0.6, 0.8], rtol=1e-6)
    assert v.dtype == np.float32


def test_nan_response_raises(api_mock):
    api_mock({"data": [{"embedding": [np.nan, 0.1]}]})
    with pytest.raises(RuntimeError, match="NaN"):
        _embedder().embed_query("текст")


def test_inf_response_raises(api_mock):
    api_mock({"data": [{"embedding": [np.inf, 0.1]}]})
    with pytest.raises(RuntimeError, match="NaN"):
        _embedder().embed_query("текст")


def test_dims_change_raises(api_mock):
    """Провайдер сменил размерность между вызовами — громкая ошибка."""
    api_mock([_ok_payload(dim=4), _ok_payload(dim=3)])
    e = _embedder()
    e.embed_query("текст")  # размерность 4 запомнена
    with pytest.raises(RuntimeError, match="размерность"):
        e.embed_query("текст")


def test_dims_checked_against_preset(api_mock):
    """retriever заранее ставит _dimensions из metadata индекса — сверка работает."""
    api_mock(_ok_payload(dim=3))
    e = _embedder()
    e._dimensions = 4  # как в retriever.get_embedder()
    with pytest.raises(RuntimeError, match="размерность"):
        e.embed_query("текст")
