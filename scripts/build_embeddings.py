#!/usr/bin/env python3
"""
Сборка индекса эмбеддингов (запускать на Mac: `make embeddings`).

1. читает все data/*.txt -> чанки (scripts/corpus.py)
2. эмбеддит их e5-small через ONNX (scripts/embed.py)
3. сохраняет index/vectors.npy + index/chunks.json

Эти два файла копируются на сервер (`make deploy-index` или вручную scp).
На сервере эмбеддится только запрос пользователя — той же моделью.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from corpus import load_chunks, source_label  # noqa: E402
from embed import embed_passages, EMBED_DIM   # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_DIR = os.path.join(ROOT, "index")

MIN_CHARS = 40  # чанки короче — мусор (обрывки), выбрасываем


def main() -> int:
    include = None
    # можно ограничить типы: EMBED_TYPES="письмо" -> только письма
    env_types = os.environ.get("EMBED_TYPES", "").strip()
    if env_types:
        include = {t.strip() for t in env_types.split(",") if t.strip()}
        print(f"Ограничение по типу: {include}")

    chunks = load_chunks(include_types=include)
    before = len(chunks)
    chunks = [c for c in chunks if len(c.text) >= MIN_CHARS]
    print(f"Чанков: {len(chunks)} (отброшено коротких: {before - len(chunks)})")
    if not chunks:
        print("Нечего эмбеддить. Сначала: python3 scripts/epub_to_txt.py", file=sys.stderr)
        return 1

    texts = [c.text for c in chunks]

    print(f"Эмбеддинг {len(texts)} чанков моделью e5-small (ONNX)...")
    t0 = time.time()
    vectors = embed_passages(texts)
    dt = time.time() - t0
    print(f"Готово за {dt:.1f}с ({len(texts)/dt:.0f} чанков/с). Форма: {vectors.shape}")

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

    vec_mb = os.path.getsize(os.path.join(INDEX_DIR, "vectors.npy")) / 1e6
    json_mb = os.path.getsize(os.path.join(INDEX_DIR, "chunks.json")) / 1e6
    print(f"\nСохранено:")
    print(f"  index/vectors.npy   {vec_mb:.1f} МБ")
    print(f"  index/chunks.json   {json_mb:.1f} МБ")
    print(f"\nПеренос на сервер: scp -r index/ user@server:/path/to/app/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
