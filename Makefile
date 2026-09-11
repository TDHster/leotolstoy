.PHONY: help install convert embeddings embeddings-gpu run clean deploy-index lock

help:
	@echo "Толстой-чат — команды (через uv):"
	@echo "  make install     — uv sync (создать окружение + поставить зависимости)"
	@echo "  make convert     — EPUB (все найденные тома) -> data/*.txt"
	@echo "  make embeddings  — собрать индекс из data/*.txt -> index/ (CPU, медленно)"
	@echo "  make embeddings-gpu — собрать индекс на GPU (требует CUDA, быстро)"
	@echo "  make run         — запустить сервер локально (http://localhost:8000)"
	@echo "  make deploy-index HOST=user@server DIR=/path — scp индекса на сервер"
	@echo "  make lock        — обновить uv.lock"
	@echo "  make clean       — удалить окружение и индекс"

install:
	uv sync

lock:
	uv lock

# EPUB -> txt (без сторонних зависимостей, но гоним через uv для единообразия)
convert:
	uv run python scripts/epub_to_txt.py

# Главная цель: обработать все data/*.txt и собрать эмбеддинги.
# Только письма:  make embeddings EMBED_TYPES=письмо
embeddings:
	uv run python scripts/build_embeddings.py

# GPU-версия (требует: uv pip install sentence-transformers torch)
embeddings-gpu:
	uv run python scripts/build_embeddings_gpu.py

run:
	uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1

deploy-index:
	@test -n "$(HOST)" || (echo "Укажите HOST=user@server"; exit 1)
	@test -n "$(DIR)" || (echo "Укажите DIR=/path/to/app"; exit 1)
	scp -r index/ "$(HOST):$(DIR)/"

clean:
	rm -rf .venv index
