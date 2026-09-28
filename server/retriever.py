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
import re

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_DIR = os.environ.get("INDEX_DIR", os.path.join(ROOT, "index"))

log = logging.getLogger("tolstoy.retriever")

_INT8_SEARCH_BLOCK = 2048  # строк индекса на один float32-блок при int8-поиске (~32 МБ)


def _year_bounds(year) -> tuple[int, int] | None:
    """(min, max) годов из поля meta["year"]: "1900" или диапазон "1900–1910"."""
    nums = re.findall(r"\d{4}", str(year or ""))
    if not nums:
        return None
    return int(nums[0]), int(nums[-1])


def _meta_matches(meta: dict, types: set[str] | None, addressee: str | None,
                  year_from: int | None, year_to: int | None,
                  volume: str | None) -> bool:
    """Фильтр чанка по метаданным. Пустые параметры фильтра не ограничивают."""
    if types and meta.get("type") not in types:
        return False
    if addressee:
        # подстрока: агент может написать «Чертков» вместо полного имени
        who = str(meta.get("addressee") or meta.get("to") or "")
        if addressee.casefold() not in who.casefold():
            return False
    if year_from is not None or year_to is not None:
        bounds = _year_bounds(meta.get("year"))
        if bounds is None:
            return False
        if year_from is not None and bounds[1] < year_from:
            return False
        if year_to is not None and bounds[0] > year_to:
            return False
    if volume is not None:
        vol = str(meta.get("vol") or meta.get("volume") or "")
        if vol != str(volume):
            return False
    return True


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

        self._facets: dict | None = None  # кеш facets() — считается лениво, один раз

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
               types: set[str] | None = None, addressee: str | None = None,
               year_from: int | None = None, year_to: int | None = None,
               volume: str | None = None) -> list[dict]:
        """
        query_vec — L2-нормализованный вектор запроса (EMBED_DIM,), float32.

        int8-индекс: запрос квантизуется той же схемой, что и сборка.
        float16/float32-индекс: ищется как есть (нормы ≈ 1 → dot ≈ cosine).
        Помимо types доступны фильтры по метаданным (addressee/год/том) —
        для MCP-инструментов; существующие вызовы без них работают как раньше.
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

        if types or addressee or year_from is not None or year_to is not None or volume is not None:
            mask = np.array(
                [_meta_matches(c["meta"], types, addressee, year_from, year_to, volume)
                 for c in self.chunks],
                dtype=bool,
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

    def browse(self, types: set[str] | None = None, addressee: str | None = None,
               year_from: int | None = None, year_to: int | None = None,
               volume: str | None = None, limit: int = 20,
               offset: int = 0) -> tuple[int, list[dict]]:
        """Список ЗАПИСЕЙ (группировка чанков по meta["doc"]) под фильтры.

        Возвращает (total, records): записи с превью первого чанка, по дате,
        с пагинацией. Для MCP-навигации по переписке («все письма Черткову
        за 1890-е») — полный текст записи берётся отдельно по doc.
        """
        recs: dict[str, dict] = {}
        for c in self.chunks:
            meta = c["meta"]
            if not _meta_matches(meta, types, addressee, year_from, year_to, volume):
                continue
            doc = str(meta.get("doc", ""))
            if doc in recs:
                continue
            recs[doc] = {
                "doc": doc,
                "type": meta.get("type", ""),
                "date": str(meta.get("date") or ""),
                "year": meta.get("year", ""),
                "volume": meta.get("vol", ""),
                "addressee": str(meta.get("addressee") or meta.get("to") or ""),
                "number": str(meta.get("number") or meta.get("num") or ""),
                "entry": str(meta.get("entry") or meta.get("title") or ""),
                "preview": c["text"][:200],
            }
        ordered = sorted(
            recs.values(),
            key=lambda r: (str(r["date"] or r["year"] or ""), r["doc"]),
        )
        return len(ordered), ordered[offset:offset + limit]

    def grep(self, pattern: str, whole_word: bool = False,
             types: set[str] | None = None, addressee: str | None = None,
             year_from: int | None = None, year_to: int | None = None,
             volume: str | None = None, limit: int = 20) -> dict:
        """Точный поиск слова/фразы по текстам чанков (без эмбеддинга).

        Возвращает суммарное число вхождений по всем подходящим чанкам и до
        limit чанков с контекстом ±150 символов вокруг первого вхождения.
        Фильтры дают частотность: «сколько раз слово X встречается в письмах
        Черткову за 1890-е». Регистронезависимый; whole_word=True — границы
        слова (юникодные \\b, работает с кириллицей).
        """
        rx = re.compile(
            rf"\b{re.escape(pattern)}\b" if whole_word else re.escape(pattern),
            re.IGNORECASE,
        )
        results: list[dict] = []
        total_matches = 0
        chunks_matched = 0
        for c in self.chunks:
            if not _meta_matches(c["meta"], types, addressee, year_from, year_to, volume):
                continue
            m = rx.search(c["text"])
            if m is None:
                continue
            chunks_matched += 1
            total_matches += len(rx.findall(c["text"]))
            if len(results) < limit:
                text = c["text"]
                start = max(0, m.start() - 150)
                end = min(len(text), m.end() + 150)
                snippet = ("…" if start > 0 else "") + text[start:end] + \
                          ("…" if end < len(text) else "")
                results.append({
                    "text": snippet,
                    "source": c["source"],
                    "meta": c["meta"],
                    "matches": len(rx.findall(text)),
                })
        return {
            "count": total_matches,
            "chunks_matched": chunks_matched,
            "results": results,
        }

    def facets(self) -> dict:
        """Словарь значений для фильтров — по ЗАПИСЯМ, не по чанкам.

        Считается один раз (полный проход по чанкам) и кешируется: типы со
        счётчиками записей, диапазон лет, топ-100 адресатов писем, тома.
        Для MCP: агент узнаёт валидные значения фильтров до запросов.
        """
        if self._facets is not None:
            return self._facets
        from collections import Counter

        seen: set[str] = set()
        types_c, years_c, addrs_c, vols_c = Counter(), Counter(), Counter(), Counter()
        for c in self.chunks:
            meta = c["meta"]
            doc = str(meta.get("doc", ""))
            if doc in seen:
                continue
            seen.add(doc)
            types_c[meta.get("type", "")] += 1
            years_c[str(meta.get("year") or "")] += 1
            if meta.get("type") == "письмо":
                who = str(meta.get("addressee") or meta.get("to") or "").strip()
                if who:
                    addrs_c[who] += 1
            vol = str(meta.get("vol") or meta.get("volume") or "")
            if vol:
                vols_c[vol] += 1

        year_vals = [y for y in years_c if re.match(r"\d{4}", y)]
        self._facets = {
            "records_total": len(seen),
            "types": {t: n for t, n in types_c.most_common()},
            "years": {
                "min": min(year_vals) if year_vals else None,
                "max": max(year_vals) if year_vals else None,
            },
            "addressees": {
                "total": len(addrs_c),
                "top": [{"name": n, "letters": cnt} for n, cnt in addrs_c.most_common(100)],
            },
            "volumes": {v: vols_c[v] for v in sorted(
                vols_c, key=lambda v: int(v) if v.isdigit() else 0)},
        }
        return self._facets

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
