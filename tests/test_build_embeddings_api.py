"""Тесты чекпоинтов сборки внешнего индекса (scripts/build_embeddings_api.py)."""
import os
import sys

import numpy as np

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
