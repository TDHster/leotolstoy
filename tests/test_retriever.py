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


def _make_int8_index(tmp_path, vectors, chunks):
    """int8-индекс с metadata (как у index.qwen3-embedding-8b на сервере)."""
    np.save(tmp_path / "vectors.npy", np.asarray(vectors, dtype=np.int8))
    (tmp_path / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "model": "qwen/qwen3-embedding-8b",
                "dimensions": int(np.asarray(vectors).shape[1]),
                "api": True,
                "quantized": True,
                "dtype": "int8",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return str(tmp_path)


def test_search_int8_no_overflow(tmp_path):
    """Регрессия: int8 @ int8 переполнялся, все косинусы были ≤ 0.008.

    Векторы с нормой ровно 127 (как после build_embeddings_api.py), запрос —
    первый вектор. До фикса numpy умножал int8-матрицы в int8 и результат
    зажимался в [-128, 127] → максимум возможного косинуса = 127/16129 ≈ 0.008.
    """
    rng = np.random.default_rng(42)
    dim, n = 128, 200
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_i8 = (vecs * 127).astype(np.int8)
    # перенормировка до нормы 127, как в build_embeddings_api.py
    norms = np.linalg.norm(vecs_i8.astype(np.float32), axis=1, keepdims=True)
    vecs_i8 = (vecs_i8.astype(np.float32) / norms * 127).astype(np.int8)

    chunks = [
        {"text": f"фрагмент {i}", "source": f"Источник {i}",
         "meta": {"type": "дневник", "year": "1900"}}
        for i in range(n)
    ]
    r = Retriever(index_dir=_make_int8_index(tmp_path, vecs_i8, chunks))
    assert r.is_quantized

    # запрос = первый вектор (search сам квантизует float32 → int8)
    res = r.search(vecs[0], top_k=3)
    assert res[0]["source"] == "Источник 0"
    assert res[0]["score"] > 0.2  # на баге максимум 0.0079
    scores = [x["score"] for x in res]
    assert scores == sorted(scores, reverse=True)
    assert scores[0] > scores[-1] + 0.1  # не плоский список


def _make_f16_index(tmp_path, vectors, chunks):
    """float16-индекс с metadata (как после перехода с int8 на float16)."""
    np.save(tmp_path / "vectors.npy", np.asarray(vectors, dtype=np.float16))
    (tmp_path / "chunks.json").write_text(
        json.dumps(chunks, ensure_ascii=False), encoding="utf-8"
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "model": "qwen/qwen3-embedding-8b",
                "dimensions": int(np.asarray(vectors).shape[1]),
                "api": True,
                "quantized": False,
                "dtype": "float16",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return str(tmp_path)


def test_search_float16(tmp_path):
    """f16-индекс: поиск работает, score ≈ косинус (без деления на 127²)."""
    vecs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    chunks = [
        {"text": "про труд", "source": "Письмо A", "meta": {"type": "письмо"}},
        {"text": "про смерть", "source": "Дневник B", "meta": {"type": "дневник"}},
        {"text": "про веру", "source": "Письмо C", "meta": {"type": "письмо"}},
    ]
    r = Retriever(index_dir=_make_f16_index(tmp_path, vecs, chunks))
    assert not r.is_quantized
    assert r.dtype == "float16"

    res = r.search(np.array([1, 0, 0], dtype=np.float32), top_k=3)
    assert res[0]["source"] == "Письмо A"
    assert res[0]["score"] == pytest.approx(1.0, abs=1e-5)
    scores = [x["score"] for x in res]
    assert scores == sorted(scores, reverse=True)


def test_f16_matches_f32_ranking(tmp_path):
    """f16 не должен искажать ранжирование по сравнению с float32-эталоном."""
    rng = np.random.default_rng(7)
    dim, n, q_count = 512, 300, 20
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

    # запросы = документ + шум (~cos 0.5), чтобы топ был осмысленным
    idx = rng.integers(0, n, size=q_count)
    queries = vecs[idx] + rng.standard_normal((q_count, dim)).astype(np.float32) * 0.02
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    chunks = [
        {"text": f"фрагмент {i}", "source": f"Источник {i}", "meta": {"type": "проза"}}
        for i in range(n)
    ]
    f16_dir, f32_dir = tmp_path / "f16", tmp_path / "f32"
    f16_dir.mkdir(), f32_dir.mkdir()
    r16 = Retriever(index_dir=_make_f16_index(f16_dir, vecs, chunks))
    r32 = Retriever(index_dir=_make_index(f32_dir, vecs, chunks))

    for q in queries:
        top16 = [x["source"] for x in r16.search(q, top_k=10)]
        top32 = [x["source"] for x in r32.search(q, top_k=10)]
        assert top16[0] == top32[0]
        assert len(set(top16) & set(top32)) >= 9  # не более 1 расхождения из 10


def test_dtype_mismatch_raises(tmp_path):
    """Файл одного dtype при metadata другого — жёсткая ошибка, а не тихий мусор."""
    vecs = np.array([[1, 0, 0]], dtype=np.float32)
    chunks = [{"text": "x", "source": "s", "meta": {}}]

    # f16-файл + metadata "int8"
    np.save(tmp_path / "vectors.npy", vecs.astype(np.float16))
    (tmp_path / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"model": "m", "dimensions": 3, "api": True,
                    "quantized": True, "dtype": "int8"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dtype"):
        Retriever(index_dir=str(tmp_path))

    # int8-файл + metadata "float16"
    np.save(tmp_path / "vectors.npy", vecs.astype(np.int8))
    (tmp_path / "metadata.json").write_text(
        json.dumps({"model": "m", "dimensions": 3, "api": True,
                    "quantized": False, "dtype": "float16"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="dtype"):
        Retriever(index_dir=str(tmp_path))


def test_float16_nan_rows_rejected(tmp_path):
    """NaN-вектор от провайдера не должен попадать в обслуживаемый индекс."""
    vecs = np.array([[1, 0, 0], [np.nan, 1, 0]], dtype=np.float32)
    chunks = [
        {"text": "a", "source": "s1", "meta": {}},
        {"text": "b", "source": "s2", "meta": {}},
    ]
    with pytest.raises(ValueError, match="повреждённых"):
        Retriever(index_dir=_make_f16_index(tmp_path, vecs, chunks))


def test_float16_zero_row_rejected(tmp_path):
    """Нулевая строка = повреждённый вектор (константа «притягивается» к любым запросам)."""
    vecs = np.array([[1, 0, 0], [0, 0, 0]], dtype=np.float32)
    chunks = [
        {"text": "a", "source": "s1", "meta": {}},
        {"text": "b", "source": "s2", "meta": {}},
    ]
    with pytest.raises(ValueError, match="повреждённых"):
        Retriever(index_dir=_make_f16_index(tmp_path, vecs, chunks))


def test_search_nan_query_returns_empty(tiny_index):
    """NaN-запрос (битый ответ эмбеддера) не должен ронять поиск."""
    r = Retriever(index_dir=tiny_index)
    assert r.search(np.array([np.nan, 0, 0], dtype=np.float32), top_k=3) == []
