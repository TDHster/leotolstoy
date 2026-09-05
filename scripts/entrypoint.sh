#!/bin/sh
# Entrypoint контейнера: включает TLS, если смонтированы сертификаты.
# TLS_CERT/TLS_KEY задаются в docker-compose.yml. Если файлов нет — HTTP
# (например при локальном запуске без сертификатов).
set -e

PORT="${PORT:-8000}"
ARGS="server.app:app --host 0.0.0.0 --port ${PORT} --workers 1"

if [ -n "${TLS_CERT:-}" ] && [ -f "${TLS_CERT}" ] && [ -f "${TLS_KEY:-}" ]; then
    echo "TLS включён: ${TLS_CERT}"
    ARGS="${ARGS} --ssl-certfile ${TLS_CERT} --ssl-keyfile ${TLS_KEY}"
else
    echo "TLS выключен — HTTP на порту ${PORT}"
fi

exec uv run --frozen uvicorn ${ARGS}
