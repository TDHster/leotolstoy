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

_INT8_SEARCH_BLOCK = 2048  # строк индекса на один float32-блок при int8-поиске (~32 МБ)


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
                "model": os.environ.get(
                    "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
                ),
                "dimensions": 384,  # будет перезаписано ниже
                "api": False,
                "display_name": "Локальная модель",
                "quantized": False,
                "dtype": "float32",
            }

        # Загрузка векторов с учётом dtype из metadata
        self.vectors: np.ndarray = np.load(vec_path)
        self.dtype = self.metadata.get("dtype", "float32")

        # Сверка фактического dtype с metadata: подменённый или недособранный
        # файл не должен молча ломать поиск (int8-файл как float32 → скоры
        # сжимаются в ~127 раз и т.п.)
        if self.vectors.dtype != np.dtype(self.dtype):
            raise ValueError(
                f"vectors.npy имеет dtype {self.vectors.dtype}, а metadata.json — "
                f"{self.dtype}. Файл подменён или индекс недособран. Пересоберите индекс."
            )
        self.is_quantized = self.dtype == "int8"  # атрибут для совместимости (тесты/легаси)

        if self.is_quantized:
            log.info("Загружен квантизованный индекс (%s)", self.dtype)
            # Векторы уже int8, ничего не делаем
        else:
            # float32/float16 — оставляем как есть (float16 НЕ кастим целиком
            # в float32: это удвоило бы RAM на сервере)
            log.info("Загружен индекс (%s)", self.dtype)

        # Обновить dimensions в metadata если она была дефолтной
        if "dimensions" not in self.metadata or self.metadata["dimensions"] == 384:
            self.metadata["dimensions"] = self.vectors.shape[1]

        # chunks.json
        with open(chunks_path, encoding="utf-8") as f:
            self.chunks: list[dict] = json.load(f)

        assert len(self.chunks) == self.vectors.shape[0], f"рассинхрон: {len(self.chunks)} чанков vs {self.vectors.shape[0]} векторов"

        # Проверка совместимости
        expected_dim = self.metadata.get("dimensions", self.vectors.shape[1])
        if self.vectors.shape[1] != expected_dim:
            raise ValueError(
                f"Размерность vectors.npy ({self.vectors.shape[1]}) не совпадает "
                f"с metadata.json ({expected_dim}). Пересоберите индекс."
            )
        self._validate_vectors()

        log.info(
            "Загружен индекс: %s (%d чанков, %d dims, модель: %s, dtype: %s, RAM: %.1f МБ)",
            index_dir,
            len(self.chunks),
            self.vectors.shape[1],
            self.metadata.get("model", "unknown"),
            self.dtype,
            self.vectors.nbytes / 1e6,
        )

    def _blockwise_dot(self, q: np.ndarray) -> np.ndarray:
        """index @ q блоками — для int8/float16 индексов.

        numpy не умножает int8/float16 через быстрый BLAS, а полный cast индекса
        в float32 (87K×4096 ≈ 1.4 ГБ) не влезает в RAM — поэтому блоками.
        """
        scores = np.empty(self.vectors.shape[0], dtype=np.float32)
        for start in range(0, self.vectors.shape[0], _INT8_SEARCH_BLOCK):
            end = min(start + _INT8_SEARCH_BLOCK, self.vectors.shape[0])
            scores[start:end] = self.vectors[start:end].astype(np.float32) @ q
        return scores

    def _validate_vectors(self) -> None:
        """Проверка индекса на повреждённые векторы (NaN/inf/нулевые строки).

        int8-файлы пропускаем: NaN в int8 не хранится (при касте превращается
        в мусор, но сам NaN-сигнал теряется). Блочно — чтобы не аллоцировать
        N×dim float32 целиком.
        """
        if self.vectors.dtype == np.int8:
            return
        bad = 0
        for start in range(0, self.vectors.shape[0], _INT8_SEARCH_BLOCK):
            end = min(start + _INT8_SEARCH_BLOCK, self.vectors.shape[0])
            block = self.vectors[start:end].astype(np.float32)
            bad += int((~np.isfinite(block).all(axis=1) | ~(block != 0).any(axis=1)).sum())
        if bad:
            raise ValueError(
                f"Индекс содержит {bad} повреждённых векторов (NaN/inf/нулевые "
                f"строки). Пересоберите индекс."
            )

    def search(self, query_vec: np.ndarray, top_k: int = 5,
               types: set[str] | None = None) -> list[dict]:
        """
        query_vec — L2-нормализованный вектор запроса (EMBED_DIM,), float32.

        int8-индекс: запрос квантизуется той же схемой, что и сборка.
        float16/float32-индекс: ищется как есть (нормы ≈ 1 → dot ≈ cosine).
        """
        if not np.isfinite(query_vec).all():
            log.warning("Вектор запроса содержит NaN/inf — поиск пропущен")
            return []

        # Квантизация query если индекс int8 (легаси-путь)
        if self.dtype == "int8":
            query_vec_int8 = (query_vec * 127).astype(np.int8)
            # Перенормализация после квантизации (чтобы норма была ровно 127)
            norm_int8 = np.linalg.norm(query_vec_int8.astype(np.float32))
            if norm_int8 > 0:
                query_vec_search = (query_vec_int8.astype(np.float32) / norm_int8 * 127).astype(np.int8)
            else:
                query_vec_search = query_vec_int8
        else:
            query_vec_search = query_vec.astype(np.float32)

        # косинус = скалярное произведение (обе стороны нормализованы)
        if self.dtype == "int8":
            # int8 @ int8 в numpy переполняется (dtype результата не расширяется),
            # поэтому считаем в float32 блоками
            scores = self._blockwise_dot(query_vec_search.astype(np.float32))
            # int8-векторы перенормированы до нормы 127 → косинус = dot / 127²
            scores /= 127 * 127
        elif self.dtype == "float16":
            scores = self._blockwise_dot(query_vec_search)
        else:
            scores = self.vectors @ query_vec_search

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
            # Пин провайдера: дефолт nebius (неквантизованный апстрим Qwen3-8B).
            # Пустая строка = авто-роутинг (не рекомендуется — провайдеры могут
            # квантизовать модель по-разному, векторы несравнимы).
            provider = os.environ.get("EXTERNAL_EMBEDDER_PROVIDER", "nebius").strip() or None
            embedder = OpenRouterEmbedder(
                model=model, api_key=api_key, base_url=base_url, provider=provider
            )
            # Размерность из metadata вместо лишнего вызова API; _call_api сверяет
            # фактическую размерность ответа с ней (защита от подмены модели)
            embedder._dimensions = self.metadata["dimensions"]
            meta_provider = self.metadata.get("provider")
            if provider and meta_provider and str(meta_provider).lower() != provider.lower():
                log.warning(
                    "Рантайм-провайдер '%s' отличается от провайдера сборки индекса "
                    "'%s' — векторы запроса и индекса могут быть несовместимы",
                    provider, meta_provider,
                )
            return embedder
        else:
            # Локальная ONNX модель
            model_name = self.metadata.get("model")
            threads = int(os.environ.get("EMBED_THREADS", "2"))
            return LocalEmbedder(model_name=model_name, threads=threads)
