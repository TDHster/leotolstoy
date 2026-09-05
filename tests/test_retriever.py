"""Тесты косинусного поиска (server/retriever.py) на синтетическом индексе."""
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

from retriever import Retriever  # noqa: E402


def _make_index(tmp_path, vectors, chunks):
    np.save(tmp_path / "vectors.npy", np.asarray(vectors, dtype=np.float32))
    (tmp_path / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    return str(tmp_path)


@pytest.fixture
def tiny_index(tmp_path):
    # 3 ортогональных нормализованных вектора
    vecs = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    chunks = [
        {"text": "про труд", "source": "Письмо A", "meta": {"type": "письмо", "year": "1880"}},
        {"text": "про смерть", "source": "Дневник B", "meta": {"type": "дневник", "year": "1890"}},
        {"text": "про веру", "source": "Письмо C", "meta": {"type": "письмо", "year": "1900"}},
    ]
    return _make_index(tmp_path, vecs, chunks)


def test_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Retriever(index_dir=str(tmp_path))


def test_search_returns_closest_first(tiny_index):
    r = Retriever(index_dir=tiny_index)
    # запрос точно совпадает с первым вектором
    res = r.search(np.array([1, 0, 0], dtype=np.float32), top_k=3)
    assert res[0]["source"] == "Письмо A"
    assert res[0]["score"] == pytest.approx(1.0, abs=1e-5)
    # результаты отсортированы по убыванию score
    scores = [x["score"] for x in res]
    assert scores == sorted(scores, reverse=True)


def test_search_top_k_limit(tiny_index):
    r = Retriever(index_dir=tiny_index)
    res = r.search(np.array([1, 0, 0], dtype=np.float32), top_k=2)
    assert len(res) == 2


def test_search_type_filter(tiny_index):
    r = Retriever(index_dir=tiny_index)
    # ищем близко ко второму (дневник), но фильтруем только письма
    res = r.search(np.array([0, 1, 0], dtype=np.float32), top_k=3, types={"письмо"})
    assert all(x["meta"]["type"] == "письмо" for x in res)
    assert all(x["source"] != "Дневник B" for x in res)


def test_len(tiny_index):
    r = Retriever(index_dir=tiny_index)
    assert len(r) == 3


def test_desync_raises(tmp_path):
    # 2 вектора, но 1 чанк — рассинхрон
    np.save(tmp_path / "vectors.npy", np.zeros((2, 3), dtype=np.float32))
    (tmp_path / "chunks.json").write_text(
        json.dumps([{"text": "x", "source": "s", "meta": {}}]), encoding="utf-8"
    )
    with pytest.raises(AssertionError):
        Retriever(index_dir=str(tmp_path))
