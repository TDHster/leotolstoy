#!/usr/bin/env python3
"""
Предзагрузка модели эмбеддингов в кеш.

Использование:
    python scripts/download_model.py

Скачивает модель один раз в EMBED_CACHE_DIR (или ~/.cache/fastembed).
На сервере запускается перед первым docker compose up.
"""
import os
import sys
from pathlib import Path

# Добавляем scripts/ в путь для импорта embed
sys.path.insert(0, str(Path(__file__).parent))

from embed import get_model

def main():
    cache_dir = os.getenv("EMBED_CACHE_DIR", str(Path.home() / ".cache" / "fastembed"))
    print(f"Загрузка модели в {cache_dir}...")

    try:
        # Просто создаём модель — она закешируется автоматически
        model = get_model()
        print(f"✅ Модель загружена и закеширована")

        # Проверим что кеш не пустой
        cache_path = Path(cache_dir)
        if cache_path.exists():
            try:
                size = sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
                print(f"   Размер кеша: {size / 1024 / 1024:.1f} МБ")
            except Exception as e:
                print(f"   Не удалось проверить размер кеша: {e}")

        return 0
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}", file=sys.stderr)
        return 1

if __name__ == "__main__":
    sys.exit(main())
