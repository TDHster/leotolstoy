"""Тесты чекпоинтов сборки внешнего индекса (scripts/build_embeddings_api.py)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import build_embeddings_api as bapi  # noqa: E402


def test_checkpoint_roundtrip(tmp_path):
    vecs = np.arange(12, dtype=np.float32).reshape(3, 4)
    bapi.save_checkpoint(str(tmp_path), 3, 100, vecs)
    loaded = bapi.load_checkpoint(str(tmp_path), 100)
    assert loaded is not None
    np.testing.assert_array_equal(loaded, vecs)


def test_checkpoint_rejected_for_different_corpus(tmp_path):
    # total не совпадает — корпус поменялся, чекпоинт не валиден
    bapi.save_checkpoint(str(tmp_path), 3, 100, np.zeros((3, 4), dtype=np.float32))
    assert bapi.load_checkpoint(str(tmp_path), 99) is None


def test_checkpoint_clear(tmp_path):
    bapi.save_checkpoint(str(tmp_path), 2, 10, np.zeros((2, 3), dtype=np.float32))
    bapi.clear_checkpoint(str(tmp_path))
    assert bapi.load_checkpoint(str(tmp_path), 10) is None


def test_checkpoint_missing(tmp_path):
    assert bapi.load_checkpoint(str(tmp_path), 10) is None


def test_checkpoint_corrupted(tmp_path):
    (tmp_path / ".partial_vectors.npy").write_bytes(b"not a numpy file")
    (tmp_path / ".partial_meta.json").write_text('{"processed": 1, "total": 10}')
    assert bapi.load_checkpoint(str(tmp_path), 10) is None


def test_checkpoint_fingerprint_roundtrip(tmp_path):
    """model/provider записываются в чекпоинт и проверяются при resume."""
    vecs = np.zeros((3, 4), dtype=np.float32)
    bapi.save_checkpoint(
        str(tmp_path), 3, 100, vecs,
        model="qwen/qwen3-embedding-8b", provider="nebius", n_batches=3,
    )
    assert bapi.load_checkpoint(
        str(tmp_path), 100, model="qwen/qwen3-embedding-8b", provider="nebius"
    ) is not None


def test_checkpoint_rejected_for_different_provider(tmp_path):
    # Провайдер сменился — resume смешал бы векторы из разных пространств
    bapi.save_checkpoint(str(tmp_path), 3, 100, np.zeros((3, 4), dtype=np.float32),
                         model="m", provider="nebius")
    assert bapi.load_checkpoint(str(tmp_path), 100, model="m", provider="deepinfra") is None


def test_checkpoint_rejected_for_different_model(tmp_path):
    bapi.save_checkpoint(str(tmp_path), 3, 100, np.zeros((3, 4), dtype=np.float32),
                         model="m", provider="nebius")
    assert bapi.load_checkpoint(str(tmp_path), 100, model="other", provider="nebius") is None


def test_checkpoint_legacy_rejected_when_pinned(tmp_path):
    # Легаси-чекпоинт без fingerprint нельзя resume с пином: не знаем, кто его делал
    bapi.save_checkpoint(str(tmp_path), 3, 100, np.zeros((3, 4), dtype=np.float32))
    assert bapi.load_checkpoint(str(tmp_path), 100, model="m", provider="nebius") is None


def test_checkpoint_dims_mismatch_rejected(tmp_path):
    # Файл векторов подменён (другая размерность) — чекпоинт не валиден
    bapi.save_checkpoint(str(tmp_path), 3, 100, np.zeros((3, 4), dtype=np.float32))
    np.save(str(tmp_path / ".partial_vectors.npy"), np.zeros((3, 5), dtype=np.float32))
    assert bapi.load_checkpoint(str(tmp_path), 100) is None


def test_validate_batch_ok():
    bapi._validate_batch(np.ones((2, 8), dtype=np.float32))  # не бросает
    bapi._validate_batch(np.ones((2, 8), dtype=np.float32), expected_dims=8)


def test_validate_batch_rejects_nan():
    vecs = np.ones((2, 8), dtype=np.float32)
    vecs[0, 0] = np.nan
    with pytest.raises(RuntimeError, match="NaN"):
        bapi._validate_batch(vecs)


def test_validate_batch_rejects_inf():
    vecs = np.ones((2, 8), dtype=np.float32)
    vecs[1, 3] = np.inf
    with pytest.raises(RuntimeError, match="NaN"):
        bapi._validate_batch(vecs)


def test_validate_batch_rejects_zero_row():
    with pytest.raises(RuntimeError, match="нулевые"):
        bapi._validate_batch(np.zeros((2, 8), dtype=np.float32))


def test_validate_batch_rejects_dim_change():
    with pytest.raises(RuntimeError, match="Размерность"):
        bapi._validate_batch(np.ones((2, 8), dtype=np.float32), expected_dims=4)
