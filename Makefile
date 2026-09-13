.PHONY: help install convert embeddings embeddings-gpu embeddings-qwen run clean deploy-index lock

help:
	@echo "Толстой-чат — команды (через uv):"
	@echo "  make install     — uv sync (создать окружение + поставить зависимости)"
	@echo "  make convert     — EPUB (все найденные тома) -> data/*.txt"
	@echo "  make embeddings  — собрать индекс из data/*.txt -> index/ (CPU, медленно)"
	@echo "  make embeddings-gpu — собрать индекс на GPU (требует CUDA, быстро)"
	@echo "  make embeddings-qwen — собрать индекс через OpenRouter API (Qwen3-8B, быстро, платно)"
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
	@if [ "$$(uname)" != "Darwin" ] && command -v docker compose &> /dev/null && [ -f docker-compose.yml ]; then \
		echo "🔄 Запуск Docker контейнеров (up -d)..."; \
		docker compose up -d; \
	fi

# GPU-версия (требует: uv pip install sentence-transformers torch)
embeddings-gpu:
	uv run python scripts/build_embeddings_gpu.py

# API-версия (требует EXTERNAL_EMBEDDER_API_KEY в .env)
embeddings-qwen:
	uv run python scripts/build_embeddings_api.py
	@if [ "$$(uname)" != "Darwin" ] && command -v docker compose &> /dev/null && [ -f docker-compose.yml ]; then \
		echo "🔄 Запуск Docker контейнеров (up -d)..."; \
		docker compose up -d; \
	fi

run:
	uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1

deploy-index:
	@test -n "$(HOST)" || (echo "Укажите HOST=user@server"; exit 1)
	@test -n "$(DIR)" || (echo "Укажите DIR=/path/to/app"; exit 1)
	scp -r index/ "$(HOST):$(DIR)/"

clean:
	rm -rf .venv index
