#!/bin/bash
set -euo pipefail

# DEPLOY SCRIPT

cd "$(dirname "$0")"

# Load server configuration from .deploy
if [ ! -f .deploy ]; then
    echo "Missing .deploy config file" >&2
    exit 1
fi
source .deploy

# Validate required variables
: "${SSH_HOST:?not set in .deploy}"
: "${PROJECT_DIR:?not set in .deploy}"
: "${SSH_PORT:?not set in .deploy}"
: "${REMOTE_COMMAND:?not set in .deploy}"

# .env синкается основным rsync'ом (см. .rsyncignore — он там НЕ исключён).
# Проверяем наличие ДО sync: без него rsync --delete удалил бы секреты на сервере.
if [ ! -f .env ]; then
    echo "ERROR: .env отсутствует локально — деплой остановлен," \
         "чтобы rsync --delete не удалил секреты на сервере" >&2
    exit 1
fi

# Гейт на содержимое .env: секрет не должен быть пустым или дефолтным.
check_secret() {
    local key="$1" value
    value="$(sed -n "s/^[[:space:]]*${key}=//p" .env | tail -n 1)"
    value="${value%"${value##*[![:space:]]}"}"   # обрезаем хвостовые пробелы
    if [ -z "$value" ]; then
        echo "ERROR: $key не задан в .env (закомментирован или пуст) — деплой остановлен" >&2
        exit 1
    fi
    case "$value" in
        CHANGE_ME|смените-на-длинный-пароль|testpass123|admin)
            echo "ERROR: $key содержит дефолтное/тестовое значение '$value' — деплой остановлен" >&2
            exit 1
            ;;
    esac
    if [ "$key" = "APP_PASS" ] && [ "${#value}" -lt 8 ]; then
        echo "ERROR: APP_PASS короче 8 символов — деплой остановлен" >&2
        exit 1
    fi
}
check_secret APP_PASS
check_secret LLM_API_KEY

# Run tests if tests/ directory exists
if [ -d "tests" ]; then
    echo "=== Running tests ==="

    if command -v uv &> /dev/null && [ -f "pyproject.toml" ]; then
        RUN_PYTEST=(uv run pytest)
    elif command -v pytest &> /dev/null; then
        RUN_PYTEST=(pytest)
    else
        echo "pytest not found. Install pytest or use uv." >&2
        exit 1
    fi

    if ! "${RUN_PYTEST[@]}" tests/; then
        echo "Tests failed" >&2
        exit 1
    fi
    echo "✅ Tests passed."
fi

echo "=== Syncing local files with remote server ==="
echo "From: $(pwd)"
echo "To:   $SSH_HOST:$PROJECT_DIR"

# Sync via rsync using exclusion file
RSYNC_EXCLUDE=()
if [ -f .rsyncignore ]; then
    RSYNC_EXCLUDE=(--exclude-from=.rsyncignore)
fi

if rsync -avzP --delete -e "ssh -p $SSH_PORT" "${RSYNC_EXCLUDE[@]}" ./ "$SSH_HOST:$PROJECT_DIR/"; then
    echo "✅ Sync completed successfully."

    # --- Caddy: конфиг живёт в репо и возится на сервер (Caddy — системный
    # сервис на хосте, НЕ в docker: compose-рестарт его конфиг не применит).
    if [ -f Caddyfile ]; then
        if rsync -e "ssh -p $SSH_PORT" Caddyfile "$SSH_HOST:/etc/caddy/Caddyfile"; then
            echo "✅ Caddyfile обновлён, reload Caddy..."
            if ! ssh -p "$SSH_PORT" "$SSH_HOST" \
                "caddy reload --config /etc/caddy/Caddyfile 2>/dev/null || systemctl reload caddy"; then
                echo "❌ Не удалось перезагрузить Caddy (route /viz может не заработать)" >&2
            else
                echo "✅ Caddy перезагружен: https://verbatis.ru/viz/"
            fi
        else
            echo "❌ Ошибка копирования Caddyfile" >&2
            exit 1
        fi
    fi

    # --- Визуализация: bootstrap статики на сервере. Основной rsync out/ не
    # возит (см. .rsyncignore): после первого деплоя страница обновляется на
    # сервере командой `make visualization` (после make embeddings там же).
    # $PROJECT_DIR раскрывается локально (в ssh-команде кавычек вокруг него нет).
    if ssh -p "$SSH_PORT" "$SSH_HOST" "test -f $PROJECT_DIR/visualization/out/index.html"; then
        echo "ℹ️  Серверная визуализация уже есть — локальная out/ не перезаписывает её"
    elif [ -f visualization/out/index.html ]; then
        echo "📦 Первая выкладка визуализации (на сервере её ещё нет)..."
        rsync -a -e "ssh -p $SSH_PORT" visualization/out/ "$SSH_HOST:$PROJECT_DIR/visualization/out/"
        echo "✅ Статика визуализации выложена: https://verbatis.ru/viz/"
    else
        echo "⚠️  Локально нет visualization/out/index.html — страница не собрана." \
             "Соберите: make visualization, затем повторите деплой." >&2
    fi

    # --- uv на сервере: нужен для `make visualization` там (после make embeddings).
    if ! ssh -p "$SSH_PORT" "$SSH_HOST" "command -v uv >/dev/null"; then
        echo "⚠️  На сервере нет uv — `make visualization` там не заработает." \
             "Установите один раз: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
    fi

    echo "🚀 Running command on server: $REMOTE_COMMAND"
    echo "Note: Press Ctrl+C to exit logs (if running)"

    # -t only when running in a terminal (skip in cron/pipe/CI)
    SSH_FLAGS=(-p "$SSH_PORT")
    if [ -t 1 ]; then
        SSH_FLAGS=(-t "${SSH_FLAGS[@]}")
    fi

    if ssh "${SSH_FLAGS[@]}" "$SSH_HOST" "cd \"$PROJECT_DIR\" && $REMOTE_COMMAND"; then
        echo "✅ Command completed successfully at $(date)"
    else
        echo "❌ Error running command on remote server."
        exit 1
    fi
else
    echo "❌ Error syncing files."
    exit 1
fi
