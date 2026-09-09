#!/usr/bin/env python3
"""
Общий модуль эмбеддингов — используется и при сборке индекса, и на сервере.

Модель: intfloat/multilingual-e5-small через ONNX (fastembed, без torch).
 - документам добавляется префикс "passage: "
 - запросам добавляется префикс "query: "
 - векторы L2-нормализованы -> косинус = скалярное произведение

Одна и та же модель по обе стороны => векторы документов (при сборке) совместимы
с эмбеддингом запросов (сервер).
"""
from __future__ import annotations

import os
import numpy as np

# По умолчанию — sentence-transformers/paraphrase-multilingual-mpnet-base-v2
# (1.0 ГБ, 768 dims). Более точная модель, чем MiniLM, влезает в 2 ГБ RAM.
# Старая модель (384 dims): sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
MODEL_NAME = os.environ.get(
    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
)
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))

# e5-модели асимметричны и требуют префиксов query:/passage:.
# MiniLM-paraphrase симметрична — префиксы ей не нужны (и даже вредят).
_USE_E5_PREFIX = "e5" in MODEL_NAME.lower()

_model = None


def get_model():
    global _model
    if _model is not None:
        return _model
    from fastembed import TextEmbedding
    # ограничиваем потоки, чтобы придержать RAM/CPU на маленьком сервере
    threads = int(os.environ.get("EMBED_THREADS", "2"))
    # кэш модели: в Docker указываем фиксированный путь, чтобы модель
    # запекалась в образ и не качалась при старте на сервере
    cache_dir = os.environ.get("EMBED_CACHE_DIR") or None
    kwargs = {"model_name": MODEL_NAME, "threads": threads}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    _model = TextEmbedding(**kwargs)
    return _model


def _embed(texts: list[str], prefix: str, batch_size: int = 32) -> np.ndarray:
    model = get_model()
    prefixed = [prefix + t for t in texts]
    vecs = np.array(list(model.embed(prefixed, batch_size=batch_size)), dtype=np.float32)
    # L2-нормализация
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vecs / norms


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Эмбеддинг документов (для индекса)."""
    prefix = "passage: " if _USE_E5_PREFIX else ""
    return _embed(texts, prefix, batch_size)


def embed_query(text: str) -> np.ndarray:
    """Эмбеддинг одного запроса. Возвращает вектор (EMBED_DIM,)."""
    prefix = "query: " if _USE_E5_PREFIX else ""
    return _embed([text], prefix)[0]
