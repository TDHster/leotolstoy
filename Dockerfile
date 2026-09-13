# syntax=docker/dockerfile:1

FROM python:3.12-slim

# uv для установки зависимостей (тот же lock, что и локально)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    EMBED_CACHE_DIR=/app/model_cache \
    EMBED_THREADS=2 \
    OMP_NUM_THREADS=2 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Зависимости (кэшируемый слой) — строго из lock
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2. Код
COPY scripts/ ./scripts/
COPY server/ ./server/

# index/, .env и model_cache монтируются как volume при запуске (см. compose)
EXPOSE 8000

# TLS терминирует Caddy, приложение работает на HTTP
CMD ["uv", "run", "--frozen", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
