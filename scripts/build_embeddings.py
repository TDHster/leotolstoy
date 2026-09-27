#!/usr/bin/env python3
"""
Сборка индекса эмбеддингов (`make embeddings` — локально или на сервере).

1. читает jsonl корпуса Verbatis (data/verbatis/…) -> чанки (scripts/corpus.py)
2. эмбеддит их локальной ONNX-моделью (scripts/embed.py, MiniLM по умолчанию)
3. сохраняет index/vectors.npy + index/chunks.json + index/docs.json

Индексы не ходят в git и rsync (.rsyncignore: index*/) — сборка делается
там же, где индекс будет использоваться. На сервере эмбеддится только
запрос пользователя — той же моделью, что и при сборке.

Для тестовых прогонов: EMBED_INDEX_DIR (куда писать, дефолт index) и
EMBED_LIMIT (сколько чанков максимум, дефолт — все).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
from dotenv import load_dotenv

# Загрузка .env для чтения EMBED_TYPES
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
from corpus import MIN_CHARS, load_chunks, load_docs, source_label  # noqa: E402
from embed import embed_passages, EMBED_DIM, MODEL_NAME   # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_idx = os.environ.get("EMBED_INDEX_DIR", "index")
INDEX_DIR = _idx if os.path.isabs(_idx) else os.path.join(ROOT, _idx)


def main() -> int:
    include = None
    # можно ограничить типы: EMBED_TYPES="письмо" -> только письма
    env_types = os.environ.get("EMBED_TYPES", "").strip()
    if env_types:
        include = {t.strip() for t in env_types.split(",") if t.strip()}
        print(f"Ограничение по типу: {include}")

    print(f"Индекс: {INDEX_DIR}")

    chunks = load_chunks(include_types=include)
    total = len(chunks)
    chunks = [c for c in chunks if len(c.text) >= MIN_CHARS]
    print(f"Чанков: {len(chunks)} (отброшено коротких: {total - len(chunks)})")
    if not chunks:
        print("Нечего эмбеддить. Сначала: make verbatis-fetch", file=sys.stderr)
        return 1

    limit_s = os.environ.get("EMBED_LIMIT", "").strip()
    if limit_s.isdigit() and int(limit_s) > 0:
        chunks = chunks[:int(limit_s)]
        print(f"EMBED_LIMIT={limit_s}: эмбеддим первые {len(chunks)} чанков")

    texts = [c.text for c in chunks]

    print(f"Эмбеддинг {len(texts)} чанков моделью {MODEL_NAME}...")
    sys.stdout.flush()
    t0 = time.time()

    # Батчи по 64 для прогресса, внутри model.embed обрабатывает по 8
    batch_size = 64
    all_vectors = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_vecs = embed_passages(batch, batch_size=8)  # маленький batch_size для model.embed
        all_vectors.append(batch_vecs)

        processed = min(i + batch_size, len(texts))
        pct = 100 * processed / len(texts)
        elapsed = time.time() - t0
        rate = processed / elapsed if elapsed > 0 else 0
        eta = (len(texts) - processed) / rate if rate > 0 else 0

        # прогресс-бар в стиле build_embeddings_api.py
        bar_width = 40
        filled = int(bar_width * processed / len(texts))
        bar = '█' * filled + '░' * (bar_width - filled)
        eta_min = eta / 60
        print(f"\r[{bar}] {pct:.1f}% | {processed}/{len(texts)} | "
              f"{rate:.1f} чанков/с | ETA: {eta_min:.1f} мин", end='', flush=True)

    print()  # новая строка после прогресс-бара

    vectors = np.vstack(all_vectors)
    dt = time.time() - t0
    print(f"Готово за {dt:.1f}с ({len(texts)/dt:.1f} чанков/с). Форма: {vectors.shape}", flush=True)

    assert vectors.shape[1] == EMBED_DIM, f"неожиданная размерность {vectors.shape[1]}"

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(os.path.join(INDEX_DIR, "vectors.npy"), vectors)

    meta_out = []
    for c in chunks:
        meta_out.append({
            "text": c.text,
            "source": source_label(c.meta),
            "meta": c.meta,
        })
    with open(os.path.join(INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(meta_out, f, ensure_ascii=False)

    # Полные тексты записей для GET /doc (с подписями — в отличие от чанков).
    # Тот же прогон и тот же include_types, что у chunks.json — doc_id совпадают.
    docs = load_docs(include_types=include)
    with open(os.path.join(INDEX_DIR, "docs.json"), "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False)

    # метаданные индекса: без них /models не покажет модель в UI
    # (формат как в build_embeddings_api.py)
    index_meta = {
        "model": MODEL_NAME,
        "dimensions": EMBED_DIM,
        "api": False,
        "display_name": "Локальная (MiniLM)",
        "latency_ms": 50,
        "cost_per_1k": 0,
        "description": "Локальная ONNX-модель, эмбеддинг без внешнего API",
    }
    with open(os.path.join(INDEX_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(index_meta, f, ensure_ascii=False, indent=2)

    vec_mb = os.path.getsize(os.path.join(INDEX_DIR, "vectors.npy")) / 1e6
    json_mb = os.path.getsize(os.path.join(INDEX_DIR, "chunks.json")) / 1e6
    docs_mb = os.path.getsize(os.path.join(INDEX_DIR, "docs.json")) / 1e6
    print(f"\nСохранено:")
    print(f"  {INDEX_DIR}/vectors.npy   {vec_mb:.1f} МБ")
    print(f"  {INDEX_DIR}/chunks.json   {json_mb:.1f} МБ")
    print(f"  {INDEX_DIR}/docs.json     {docs_mb:.1f} МБ ({len(docs)} записей)")
    print(f"\nСервер: индексы rsync'ом не синкаются (.rsyncignore: index*/) —")
    print("соберите индекс на сервере: make embeddings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
