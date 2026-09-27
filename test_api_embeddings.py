#!/usr/bin/env python3
"""Тест OpenRouter API эмбеддингов на 5 чанках."""
import os
import sys
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, "scripts")
sys.path.insert(0, "server")

from corpus import load_chunks
from embedders import OpenRouterEmbedder

# Конфигурация
model = os.environ.get("EXTERNAL_EMBEDDER_MODEL", "qwen/qwen3-embedding-8b")
api_key = os.environ.get("EXTERNAL_EMBEDDER_API_KEY")
base_url = os.environ.get("EXTERNAL_EMBEDDER_ENDPOINT", "https://openrouter.ai/api/v1")

if not api_key:
    print("ERROR: EXTERNAL_EMBEDDER_API_KEY не задан")
    sys.exit(1)

print(f"Модель: {model}")
print(f"API: {base_url}")
print(f"Ключ: {api_key[:20]}...")

# Загрузить первые 5 чанков
chunks = load_chunks()[:5]
texts = [c.text for c in chunks]

print(f"\nЗагружено {len(texts)} тестовых чанков:")
for i, t in enumerate(texts, 1):
    print(f"  {i}. {t[:80]}...")

# Создать эмбеддер
print(f"\nСоздаём OpenRouterEmbedder...")
embedder = OpenRouterEmbedder(model=model, api_key=api_key, base_url=base_url, timeout=30.0)

# Тест 1: один запрос
print(f"\n=== Тест 1: embed_query (один текст) ===")
try:
    vec = embedder.embed_query(texts[0])
    print(f"✅ Успешно! Размерность: {vec.shape[0]}, норма: {vec.dot(vec):.3f}")
    print(f"   Первые 5 элементов: {vec[:5]}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    sys.exit(1)

# Тест 2: батч из 5 текстов
print(f"\n=== Тест 2: embed_passages (батч из 5) ===")
try:
    vecs = embedder.embed_passages(texts, batch_size=5)
    print(f"✅ Успешно! Форма: {vecs.shape}")
    print(f"   Размерность каждого вектора: {vecs.shape[1]}")
    print(f"   Нормы: {[f'{v.dot(v):.3f}' for v in vecs]}")

    # Проверка косинусного сходства
    sim = vecs[0].dot(vecs[1])
    print(f"   Косинус между 1-м и 2-м чанком: {sim:.3f}")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    sys.exit(1)

print(f"\n🎉 Все тесты прошли! API работает корректно.")
print(f"   Запуск билдера: uv run python scripts/build_embeddings_api.py (таргет make embeddings-qwen удалён — Qwen отключён)")
