.PHONY: help install verbatis-fetch embeddings embeddings-gpu visualization run clean deploy-index lock

help:
	@echo "Толстой-чат — команды (через uv):"
	@echo "  make install     — uv sync (создать окружение + поставить зависимости)"
	@echo "  make verbatis-fetch — скачать/обновить корпус Verbatis (единственный источник текстов; приватный репо Nikulov-IB/Verbatis)"
	@echo "  make embeddings  — собрать индекс из data/verbatis/*.jsonl -> index/ (CPU, медленно)"
	@echo "  make embeddings-gpu — собрать индекс на GPU (требует CUDA, быстро)"
	@echo "  make visualization — собрать/пересобрать визуализацию из index/ (сам определит, сменился ли индекс; работает и на сервере)"
	@echo "  make run         — запустить сервер локально (http://localhost:8000)"
	@echo "  make deploy-index HOST=user@server DIR=/path — scp индекса на сервер"
	@echo "  make lock        — обновить uv.lock"
	@echo "  make clean       — удалить окружение и индекс"

install:
	uv sync

lock:
	uv lock

# Корпус Verbatis (приватный репо Nikulov-IB/Verbatis) — единственный источник
# текстов: 5 jsonl-корпусов ПСС-90 (письма, дневники, записные книжки, статьи,
# трактаты). Клонируется в data/verbatis (вне git: data/* игнорируется).
#
# Legacy (файлы остаются в репо, но в пайплайн не входят, таргетов нет):
#   scripts/epub_to_txt.py    — EPUB -> data/tom*.txt (проза, архив)
#   scripts/tei_to_txt.py     — TEI/XML -> data/tei_*.txt (заменено Verbatis'ом)
verbatis-fetch:
	@if [ -d data/verbatis/.git ]; then \
		echo "Обновляю data/verbatis (git pull)..."; \
		git -C data/verbatis pull; \
	else \
		echo "Клонирую Verbatis в data/verbatis..."; \
		git clone https://github.com/Nikulov-IB/Verbatis.git data/verbatis; \
	fi

# Главная цель: прочитать data/verbatis/*.jsonl и собрать эмбеддинги локальной моделью.
# Только письма:  make embeddings EMBED_TYPES=письмо
embeddings:
	uv run python scripts/build_embeddings.py
	@if [ "$$(uname)" != "Darwin" ] && command -v docker compose &> /dev/null && [ -f docker-compose.yml ]; then \
		echo "🔄 Перезапуск Docker-контейнеров (force-recreate — подхватить новый индекс)..."; \
		docker compose up -d --force-recreate; \
	fi

# GPU-версия (требует: uv pip install sentence-transformers torch)
embeddings-gpu:
	uv run python scripts/build_embeddings_gpu.py

# Визуализация эволюции тем: статические страницы из index/ (visualization/out/).
# Работает и локально (просмотр двойным кликом), и на сервере — там запускать
# после make embeddings, чтобы https://verbatis.ru/viz/ показала свежую карту.
# run.py сам сравнивает штамп index/ с кэшем: сменился индекс — пересчитает всё
# (UMAP + кластеры + LLM-названия ~15–30 мин на сервере), нет — отработает за секунды.
visualization:
	cd visualization && uv sync && uv run python run.py

run:
	uv run uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 1

deploy-index:
	@test -n "$(HOST)" || (echo "Укажите HOST=user@server"; exit 1)
	@test -n "$(DIR)" || (echo "Укажите DIR=/path/to/app"; exit 1)
	scp -r index/ "$(HOST):$(DIR)/"

clean:
	rm -rf .venv index
