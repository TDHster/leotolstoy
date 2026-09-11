#!/usr/bin/env python3
"""
Загрузка индекса (vectors.npy + chunks.json + metadata.json) и косинусный поиск.

Поддерживает множество индексов одновременно (разные модели эмбеддингов).
Корпус небольшой (~5-6 тыс. чанков), поэтому brute-force по numpy:
одно матричное умножение — доли миллисекунды. Векторной БД не нужно.
"""
from __future__ import annotations

import json
import logging
import os

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_DIR = os.environ.get("INDEX_DIR", os.path.join(ROOT, "index"))

log = logging.getLogger("tolstoy.retriever")


class Retriever:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        vec_path = os.path.join(index_dir, "vectors.npy")
        chunks_path = os.path.join(index_dir, "chunks.json")
        metadata_path = os.path.join(index_dir, "metadata.json")

        if not os.path.exists(vec_path) or not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"Индекс не найден в {index_dir}. "
                f"Соберите его локально (make embeddings) и скопируйте index/ на сервер."
            )

        # metadata.json (опционально для обратной совместимости)
        if os.path.exists(metadata_path):
            with open(metadata_path, encoding="utf-8") as f:
                self.metadata: dict = json.load(f)
        else:
            # fallback для старых индексов без metadata
            log.warning("metadata.json не найден в %s — используем дефолтные значения", index_dir)
            self.metadata = {
                "model": "local",
                "dimensions": 384,  # будет перезаписано ниже
                "api": False,
                "display_name": "Локальная модель",
                "quantized": False,
                "dtype": "float32",
            }

        # Загрузка векторов с учётом квантизации
        self.vectors: np.ndarray = np.load(vec_path)
        self.is_quantized = self.metadata.get("quantized", False)
        self.dtype = self.metadata.get("dtype", "float32")

        if self.is_quantized:
            log.info("Загружен квантизованный индекс (%s)", self.dtype)
            # Векторы уже int8, ничего не делаем
        else:
            # float32 векторы
            self.vectors = self.vectors.astype(np.float32)

        # Обновить dimensions в metadata если она была дефолтной
        if "dimensions" not in self.metadata or self.metadata["dimensions"] == 384:
            self.metadata["dimensions"] = self.vectors.shape[1]

        # chunks.json
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)
        assert len(self.chunks) == self.vectors.shape[0], "рассинхрон векторов и чанков"

        # Проверка совместимости
        expected_dim = self.metadata.get("dimensions", self.vectors.shape[1])
        if self.vectors.shape[1] != expected_dim:
            raise ValueError(
                f"Размерность vectors.npy ({self.vectors.shape[1]}) не совпадает "
                f"с metadata.json ({expected_dim}). Пересоберите индекс."
            )

        log.info(
            "Загружен индекс: %s (%d чанков, %d dims, модель: %s, dtype: %s, RAM: %.1f МБ)",
            index_dir,
            len(self.chunks),
            self.vectors.shape[1],
            self.metadata.get("model", "unknown"),
            self.dtype,
            self.vectors.nbytes / 1e6,
        )

    def search(self, query_vec: np.ndarray, top_k: int = 5,
               types: set[str] | None = None) -> list[dict]:
        """
        query_vec — L2-нормализованный вектор запроса (EMBED_DIM,), float32.

        Если индекс квантизован (int8), конвертируем query_vec в int8 для поиска.
        """
        # Квантизация query если индекс квантизован
        if self.is_quantized:
            query_vec_search = (query_vec * 127).astype(np.int8)
        else:
            query_vec_search = query_vec

        # косинус = скалярное произведение (обе стороны нормализованы)
        scores = self.vectors @ query_vec_search  # (N,)

        # Для int8: результат в диапазоне [-127*127, 127*127], нормализуем обратно
        if self.is_quantized:
            scores = scores.astype(np.float32) / (127 * 127)

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

    def get_embedder(self):
        """
        Создаёт эмбеддер на основе metadata.json.

        Returns:
            BaseEmbedder — LocalEmbedder или OpenRouterEmbedder
        """
        # ленивый импорт, чтобы не грузить embedders при импорте retriever в тестах
        from embedders import LocalEmbedder, OpenRouterEmbedder

        if self.metadata.get("api"):
            # Внешняя API-модель
            model = self.metadata["model"]
            api_key = os.environ.get("EXTERNAL_EMBEDDER_API_KEY")
            if not api_key:
                raise RuntimeError(
                    f"Индекс '{self.index_dir}' требует API-ключа. "
                    f"Задайте EXTERNAL_EMBEDDER_API_KEY в .env"
                )
            base_url = os.environ.get("EXTERNAL_EMBEDDER_ENDPOINT", "https://openrouter.ai/api/v1")
            embedder = OpenRouterEmbedder(model=model, api_key=api_key, base_url=base_url)
            # Хак: устанавливаем размерность из metadata, чтобы не делать лишний вызов API
            embedder._dimensions = self.metadata["dimensions"]
            return embedder
        else:
            # Локальная ONNX модель
            model_name = self.metadata.get("model")
            threads = int(os.environ.get("EMBED_THREADS", "2"))
            return LocalEmbedder(model_name=model_name, threads=threads)
