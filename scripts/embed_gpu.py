#!/usr/bin/env python3
"""
GPU-версия эмбеддингов через sentence-transformers + PyTorch с CUDA.
Используется только для быстрой сборки индекса на мощных машинах.
На сервере всё равно используется CPU-версия (embed.py).
"""
from __future__ import annotations

import os
import numpy as np

MODEL_NAME = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))

_USE_E5_PREFIX = "e5" in MODEL_NAME.lower()

_model = None


def get_model():
    """Загрузка модели через sentence-transformers с поддержкой CUDA."""
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError:
        print("ERROR: sentence-transformers не установлен.")
        print("Установите: uv pip install sentence-transformers torch")
        raise

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Загрузка модели {MODEL_NAME} на {device.upper()}...")

    _model = SentenceTransformer(MODEL_NAME, device=device)
    return _model


def _embed(texts: list[str], prefix: str, batch_size: int = 64) -> np.ndarray:
    """Эмбеддинг с GPU-ускорением."""
    model = get_model()

    # Добавляем префиксы если нужно
    if prefix:
        texts = [prefix + t for t in texts]

    # sentence-transformers сам батчит и нормализует
    vecs = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-нормализация
    )

    return vecs.astype(np.float32)


def embed_passages(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Эмбеддинг документов (для индекса)."""
    prefix = "passage: " if _USE_E5_PREFIX else ""
    return _embed(texts, prefix, batch_size)


def embed_query(text: str) -> np.ndarray:
    """Эмбеддинг одного запроса. Возвращает вектор (EMBED_DIM,)."""
    prefix = "query: " if _USE_E5_PREFIX else ""
    return _embed([text], prefix)[0]
