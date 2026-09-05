#!/usr/bin/env python3
"""
Загрузка индекса (vectors.npy + chunks.json) и косинусный поиск.

Корпус небольшой (~5-6 тыс. чанков), поэтому brute-force по numpy:
одно матричное умножение — доли миллисекунды. Векторной БД не нужно.
"""
from __future__ import annotations

import json
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_DIR = os.environ.get("INDEX_DIR", os.path.join(ROOT, "index"))


class Retriever:
    def __init__(self, index_dir: str = INDEX_DIR):
        vec_path = os.path.join(index_dir, "vectors.npy")
        chunks_path = os.path.join(index_dir, "chunks.json")
        if not os.path.exists(vec_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"Индекс не найден в {index_dir}. "
                f"Соберите его на Mac (make embeddings) и скопируйте index/ на сервер."
            )
        # vectors уже L2-нормализованы при сборке
        self.vectors: np.ndarray = np.load(vec_path).astype(np.float32)
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)
        assert len(self.chunks) == self.vectors.shape[0], "рассинхрон векторов и чанков"

    def search(self, query_vec: np.ndarray, top_k: int = 5,
               types: set[str] | None = None) -> list[dict]:
        """query_vec — L2-нормализованный вектор запроса (EMBED_DIM,)."""
        # косинус = скалярное произведение (обе стороны нормализованы)
        scores = self.vectors @ query_vec  # (N,)

        if types:
            mask = np.array(
                [c["meta"].get("type") in types for c in self.chunks], dtype=bool
            )
            scores = np.where(mask, scores, -np.inf)

        k = min(top_k, len(self.chunks))
        # частичная сортировка топ-k
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]

        results = []
        for i in idx:
            if not np.isfinite(scores[i]):
                continue
            c = self.chunks[int(i)]
            results.append({
                "text": c["text"],
                "source": c["source"],
                "meta": c["meta"],
                "score": float(scores[i]),
            })
        return results

    def __len__(self) -> int:
        return len(self.chunks)
