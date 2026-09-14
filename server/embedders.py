#!/usr/bin/env python3
"""
Модульная система эмбеддеров — абстракция над разными моделями.

Поддерживает:
- LocalEmbedder: ONNX модели через fastembed (текущая логика)
- OpenRouterEmbedder: внешние модели через OpenRouter API

Легко добавить новые эмбеддеры (OpenAI, Cohere, etc) — просто наследуем BaseEmbedder.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

import httpx
import numpy as np

log = logging.getLogger("tolstoy.embedders")


class BaseEmbedder(ABC):
    """Базовый класс для всех эмбеддеров."""

    @abstractmethod
    def embed_query(self, text: str) -> np.ndarray:
        """
        Эмбеддинг одного запроса пользователя.

        Returns:
            L2-нормализованный вектор размерности self.dimensions
        """
        pass

    @abstractmethod
    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """
        Эмбеддинг списка документов (для сборки индекса).

        Args:
            texts: список текстов для эмбеддинга
            batch_size: размер батча (для контроля памяти/rate limits)

        Returns:
            Массив (len(texts), self.dimensions), L2-нормализованный
        """
        pass

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Размерность векторов."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Имя модели для логов и UI."""
        pass


class LocalEmbedder(BaseEmbedder):
    """
    Локальная ONNX модель через fastembed (без torch, легковесная).

    Переиспользует логику из scripts/embed.py — одна модель для сборки и runtime.
    """

    def __init__(self, model_name: str | None = None, threads: int = 2):
        self._model_name = model_name or os.environ.get(
            "EMBED_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self._dimensions = int(os.environ.get("EMBED_DIM", "384"))
        self._threads = threads
        self._model = None
        # e5-модели требуют префиксов query:/passage:, MiniLM-paraphrase — нет
        self._use_prefix = "e5" in self._model_name.lower()

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self):
        """Ленивая загрузка модели (при первом вызове)."""
        if self._model is not None:
            return self._model
        from fastembed import TextEmbedding

        cache_dir = os.environ.get("EMBED_CACHE_DIR") or None
        kwargs = {"model_name": self._model_name, "threads": self._threads}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        log.info("Загрузка локальной модели: %s", self._model_name)
        self._model = TextEmbedding(**kwargs)
        return self._model

    def _embed(self, texts: list[str], prefix: str, batch_size: int = 32) -> np.ndarray:
        """Внутренний метод эмбеддинга с префиксом."""
        model = self._get_model()
        prefixed = [prefix + t for t in texts]

        # Батчинг для больших объёмов
        if len(texts) <= batch_size:
            vecs = np.array(list(model.embed(prefixed, batch_size=batch_size)), dtype=np.float32)
        else:
            all_vecs = []
            for i in range(0, len(texts), batch_size):
                batch = prefixed[i : i + batch_size]
                batch_vecs = np.array(list(model.embed(batch, batch_size=batch_size)), dtype=np.float32)
                all_vecs.append(batch_vecs)
            vecs = np.vstack(all_vecs)

        # L2-нормализация
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_query(self, text: str) -> np.ndarray:
        """Эмбеддинг одного запроса."""
        prefix = "query: " if self._use_prefix else ""
        return self._embed([text], prefix, batch_size=1)[0]

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Эмбеддинг документов (для индекса)."""
        prefix = "passage: " if self._use_prefix else ""
        return self._embed(texts, prefix, batch_size)


class OpenRouterEmbedder(BaseEmbedder):
    """
    Эмбеддер через OpenRouter API (или любой OpenAI-совместимый эндпоинт).

    Использует /embeddings эндпоинт. Батчит запросы, retry logic, таймауты.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout: float = 10.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self._dimensions: int | None = None  # определяется при первом вызове

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            raise RuntimeError("Размерность неизвестна — вызовите embed_query/embed_passages сначала")
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self.model

    def _call_api(self, texts: list[str]) -> np.ndarray:
        """
        Один вызов API (до 100 текстов в запросе, лимит OpenRouter).

        Returns:
            Массив (len(texts), dimensions)
        """
        url = f"{self.base_url}/embeddings"
        payload = {"model": self.model, "input": texts}
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        for attempt in range(self.max_retries):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    embeddings = [item["embedding"] for item in data["data"]]
                    vecs = np.array(embeddings, dtype=np.float32)
                    # Запоминаем размерность при первом успешном вызове
                    if self._dimensions is None:
                        self._dimensions = vecs.shape[1]
                    return vecs
                elif resp.status_code == 429:  # rate limit
                    wait = 5 * 2 ** attempt  # exponential backoff (5с, 10с, 20с, ...)
                    log.warning("Rate limit (429), retry %d/%d через %ds", attempt + 1, self.max_retries, wait)
                    time.sleep(wait)
                else:
                    error_text = resp.text[:500]
                    if resp.status_code == 403:
                        log.error(
                            "OpenRouter API заблокирован (403). Возможные причины: "
                            "IP-адрес сервера в блок-листе, недостаточно средств на счёте, "
                            "или провайдер блокирует VPS-хостинг. Ответ: %s",
                            error_text
                        )
                        raise RuntimeError(
                            f"Access denied by OpenRouter (403). "
                            f"Провайдер блокирует запросы с этого сервера. "
                            f"Детали: {resp.text[:200]}"
                        )
                    else:
                        log.error("OpenRouter API error %d: %s", resp.status_code, error_text)
                        raise RuntimeError(f"API returned {resp.status_code}: {resp.text[:300]}")

            except httpx.RequestError as e:
                if attempt < self.max_retries - 1:
                    wait = 5 * 2 ** attempt
                    log.warning("Сеть недоступна (%s), retry %d/%d через %ds", e, attempt + 1, self.max_retries, wait)
                    time.sleep(wait)
                else:
                    raise RuntimeError(f"Не удалось связаться с OpenRouter после {self.max_retries} попыток: {e}") from e

        raise RuntimeError(f"API вызов провалился после {self.max_retries} попыток")

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        """L2-нормализация."""
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vecs / norms

    def embed_query(self, text: str) -> np.ndarray:
        """Эмбеддинг одного запроса. Возвращает float32 вектор (для совместимости с int8 индексом)."""
        vecs = self._call_api([text])
        return self._normalize(vecs)[0]  # возвращаем float32, конвертация в int8 будет в retriever

    def embed_passages(self, texts: list[str], batch_size: int = 100) -> np.ndarray:
        """
        Эмбеддинг документов батчами (OpenRouter лимит: ~100 текстов/запрос).

        Args:
            batch_size: максимум текстов в одном API-вызове (default: 100)
        """
        all_vecs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_vecs = self._call_api(batch)
            all_vecs.append(batch_vecs)
            log.info("Обработано %d/%d текстов", min(i + batch_size, len(texts)), len(texts))

        vecs = np.vstack(all_vecs) if len(all_vecs) > 1 else all_vecs[0]
        return self._normalize(vecs)
