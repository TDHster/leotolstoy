#!/usr/bin/env python3
"""
Сборка индекса эмбеддингов через OpenRouter API (быстро, платно).

Использует внешнюю модель для векторизации всех документов.
Результат сохраняется в index.<model-slug>/ с metadata.json.

Пример:
    EXTERNAL_EMBEDDER_MODEL=qwen/qwen3-embedding-8b \
    EXTERNAL_EMBEDDER_API_KEY=sk-or-v1-... \
    uv run python scripts/build_embeddings_api.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv

# Загрузка .env перед чтением переменных окружения
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from corpus import load_chunks, source_label  # noqa: E402

# Ленивый импорт embedders — он тянет httpx, который не нужен для локальной сборки
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MIN_CHARS = 40  # чанки короче — мусор


def main() -> int:
    # Конфигурация из окружения
    model = os.environ.get("EXTERNAL_EMBEDDER_MODEL", "qwen/qwen3-embedding-8b")
    api_key = os.environ.get("EXTERNAL_EMBEDDER_API_KEY")
    base_url = os.environ.get("EXTERNAL_EMBEDDER_ENDPOINT", "https://openrouter.ai/api/v1")

    if not api_key:
        print("ERROR: EXTERNAL_EMBEDDER_API_KEY не задан в .env", file=sys.stderr)
        return 1

    # Ограничение типов (опционально)
    include = None
    env_types = os.environ.get("EMBED_TYPES", "").strip()
    if env_types:
        include = {t.strip() for t in env_types.split(",") if t.strip()}
        print(f"Ограничение по типу: {include}")

    # Загрузка чанков
    chunks = load_chunks(include_types=include)
    before = len(chunks)
    chunks = [c for c in chunks if len(c.text) >= MIN_CHARS]
    print(f"Чанков: {len(chunks)} (отброшено коротких: {before - len(chunks)})")
    if not chunks:
        print("Нечего эмбеддить. Сначала: python3 scripts/epub_to_txt.py", file=sys.stderr)
        return 1

    texts = [c.text for c in chunks]

    # Импорт эмбеддера (здесь, чтобы не грузить лишнее если нет чанков)
    from embedders import OpenRouterEmbedder  # noqa: E402

    print(f"\nЭмбеддинг {len(texts)} чанков через {model}...")
    print(f"API: {base_url}")
    sys.stdout.flush()

    embedder = OpenRouterEmbedder(model=model, api_key=api_key, base_url=base_url, timeout=30.0)

    t0 = time.time()

    # Обёртка для embed_passages с прогресс-баром
    batch_size = 100
    all_vecs = []

    try:
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_vecs = embedder._call_api(batch)
            all_vecs.append(batch_vecs)

            # Прогресс
            processed = min(i + batch_size, len(texts))
            percent = 100 * processed / len(texts)
            elapsed = time.time() - t0
            rate = processed / elapsed if elapsed > 0 else 0
            eta_sec = (len(texts) - processed) / rate if rate > 0 else 0
            eta_min = eta_sec / 60

            # Прогресс-бар
            bar_width = 40
            filled = int(bar_width * processed / len(texts))
            bar = '█' * filled + '░' * (bar_width - filled)

            print(f"\r[{bar}] {percent:.1f}% | {processed}/{len(texts)} | "
                  f"{rate:.1f} чанков/с | ETA: {eta_min:.1f} мин", end='', flush=True)

        print()  # новая строка после прогресс-бара
    except Exception as e:
        print(f"\n\nОшибка при вызове API: {e}", file=sys.stderr)
        return 1

    vectors = np.vstack(all_vecs)
    # Нормализация
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    # Запоминаем размерность
    embedder._dimensions = vectors.shape[1]

    dt = time.time() - t0
    print(f"\nГотово за {dt:.1f}с ({len(texts)/dt:.1f} чанков/с). Форма: {vectors.shape}")

    # Квантизация float32 → int8 (4× экономия памяти, минимальная потеря качества)
    # ВАЖНО: после квантизации нужно заново нормализовать, чтобы норма int8 векторов была ровно 127
    print(f"\nКвантизация векторов: float32 → int8 (экономия RAM: 4×)")
    vectors_int8 = (vectors * 127).astype(np.int8)

    # Перенормализация int8 векторов (чтобы норма была ровно 127, а не ~106)
    norms_int8 = np.linalg.norm(vectors_int8.astype(np.float32), axis=1, keepdims=True)
    norms_int8[norms_int8 == 0] = 1.0
    vectors_int8 = (vectors_int8.astype(np.float32) / norms_int8 * 127).astype(np.int8)

    # Проверка
    sample_norm = np.linalg.norm(vectors_int8[0].astype(np.float32))
    print(f"  Форма int8: {vectors_int8.shape}, размер: {vectors_int8.nbytes / 1e6:.1f} МБ")
    print(f"  Норма после квантизации: {sample_norm:.1f} (ожидается ~127)")

    # Оценка стоимости (приблизительная, ~250 токенов на чанк)
    est_tokens = len(texts) * 250
    est_cost_per_1m = 0.01  # для qwen3-embedding-8b
    if "qwen3-embedding-8b" in model:
        est_cost_per_1m = 0.01
    elif "qwen3-embedding-4b" in model:
        est_cost_per_1m = 0.02
    elif "nemotron" in model:
        est_cost_per_1m = 0.0
    est_cost = (est_tokens / 1_000_000) * est_cost_per_1m
    print(f"Оценка: ~{est_tokens:,} токенов, ~${est_cost:.2f} ({model})")

    # Определить каталог индекса
    # model_id для фронтенда: "qwen/qwen3-embedding-8b" → "qwen3-embedding-8b" (только последняя часть)
    model_id = model.split("/")[-1] if "/" in model else model
    index_dir = os.path.join(ROOT, f"index.{model_id}")
    os.makedirs(index_dir, exist_ok=True)

    # Сохранение vectors.npy (int8 для экономии памяти)
    vectors_path = os.path.join(index_dir, "vectors.npy")
    np.save(vectors_path, vectors_int8)

    # Также сохраним float32 версию для сравнения (опционально, можно удалить потом)
    vectors_f32_path = os.path.join(index_dir, "vectors_float32.npy")
    np.save(vectors_f32_path, vectors)
    print(f"\nСохранены две версии векторов:")
    print(f"  vectors.npy (int8):       {os.path.getsize(vectors_path) / 1e6:.1f} МБ ← используется")
    print(f"  vectors_float32.npy:      {os.path.getsize(vectors_f32_path) / 1e6:.1f} МБ (для сравнения)")
    print(f"  Экономия: {(1 - vectors_int8.nbytes / vectors.nbytes) * 100:.0f}%")

    # Сохранение chunks.json (тот же, что для локальной модели)
    # Можно использовать symlink для экономии памяти и диска
    chunks_path = os.path.join(index_dir, "chunks.json")
    local_chunks_path = os.path.join(ROOT, "index", "chunks.json")

    if os.path.exists(local_chunks_path) and not os.path.exists(chunks_path):
        # Создаём symlink вместо копирования (экономия ~227 МБ)
        try:
            os.symlink(local_chunks_path, chunks_path)
            print(f"\nСоздан symlink для chunks.json → экономия ~227 МБ")
        except OSError:
            # На Windows может не работать — копируем
            print(f"\nSymlink не поддерживается, копируем chunks.json...")
            with open(local_chunks_path, encoding="utf-8") as f:
                local_chunks = json.load(f)
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(local_chunks, f, ensure_ascii=False)
    elif not os.path.exists(chunks_path):
        # Локального индекса нет — сохраняем chunks заново
        meta_out = []
        for c in chunks:
            meta_out.append({
                "text": c.text,
                "source": source_label(c.meta),
                "meta": c.meta,
            })
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(meta_out, f, ensure_ascii=False)
    else:
        print(f"\nchunks.json уже существует, пропускаем")

    # Сохранение metadata.json (НОВОЕ)
    metadata = {
        "model": model,
        "dimensions": int(vectors.shape[1]),
        "api": True,
        "quantized": True,  # НОВОЕ: флаг квантизации
        "dtype": "int8",    # НОВОЕ: тип данных
        "display_name": _display_name(model),
        "latency_ms": 350,  # типичная latency для API
        "cost_per_1k": est_cost_per_1m * 1000 / 1_000_000,  # стоимость за 1000 запросов
        "description": f"Внешняя модель через OpenRouter API (int8 quantized)",
    }
    metadata_path = os.path.join(index_dir, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # Статистика
    vec_mb = os.path.getsize(vectors_path) / 1e6
    json_mb = os.path.getsize(chunks_path) / 1e6
    print(f"\nСохранено в {index_dir}/:")
    print(f"  vectors.npy     {vec_mb:.1f} МБ")
    print(f"  chunks.json     {json_mb:.1f} МБ")
    print(f"  metadata.json   {os.path.getsize(metadata_path) / 1e3:.1f} КБ")
    print(f"\nПеренос на сервер: ./deploy.sh (rsync автоматически синкает)")
    print(f"Локально: make run, затем открыть http://localhost:8000 и переключить модель")

    return 0


def _display_name(model: str) -> str:
    """Красивое имя модели для UI."""
    if "qwen3-embedding-8b" in model:
        return "Qwen3 Embedding 8B"
    elif "qwen3-embedding-4b" in model:
        return "Qwen3 Embedding 4B"
    elif "nemotron" in model:
        return "NVIDIA Nemotron 3B"
    elif "openai" in model and "text-embedding-3-large" in model:
        return "OpenAI Text Embedding 3 Large"
    elif "openai" in model and "text-embedding-3-small" in model:
        return "OpenAI Text Embedding 3 Small"
    else:
        return model.replace("/", " ").replace("-", " ").title()


if __name__ == "__main__":
    raise SystemExit(main())
