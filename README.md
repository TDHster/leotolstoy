# Беседа с Л. Н. Толстым

Чат, отвечающий в стиле Льва Толстого. RAG поверх его писем, дневников,
записных книжек, статей и трактатов из 90-томного собрания сочинений
(тексты — из приватного репозитория Verbatis), генерация — через внешнюю LLM
(любой OpenAI-совместимый API).

## Как это устроено

```
Локальная машина (разово)             Сервер (~4 ГБ RAM)
├─ make verbatis-fetch  корпус Verbatis  └─ Caddy (reverse proxy, :443)
├─ make embeddings  jsonl → index/       └─ TLS терминация (Let's Encrypt)
└─ ./deploy.sh  rsync кода (индексы собир. на сервере) └─ Docker (FastAPI, :8000 HTTP)
                                                   ├─ Basic Auth (1 пользователь)
                                                   ├─ эмбеддинг запроса (MiniLM/ONNX)
                                                   ├─ косинус-поиск top-k (numpy)
                                                   └─ генерация (внешняя LLM)
```

- **Эмбеддинги — локально** (ONNX, без torch), одна модель при сборке и на сервере.
  Векторы документов считаются один раз; на сервере эмбеддится только запрос.
- **Поиск — numpy** (корпус ~84 тыс. фрагментов, brute-force косинус — доли мс).
- **Генерация — любой OpenAI-совместимый API** (эндпоинт и модель в `.env`).
- **Продакшн**: https://verbatis.ru (Caddy + Let's Encrypt автоматически).

## Быстрый старт (локально)

Зависимости управляются через [uv](https://docs.astral.sh/uv/).

```bash
make install       # uv sync — окружение + зависимости (из uv.lock)
make verbatis-fetch  # клонировать корпус Verbatis в data/verbatis
make embeddings    # data/verbatis/*.jsonl → index/vectors.npy + index/chunks.json
cp .env.example .env   # впишите APP_PASS и LLM_API_KEY
make run           # http://localhost:8000  (логин из .env)
```

Под капотом всё гоняется через `uv run`, отдельную активацию venv делать не нужно.
Прямые вызовы тоже работают: `uv run python scripts/build_embeddings.py`.

## Данные

Корпус — репозиторий [Verbatis](https://github.com/Nikulov-IB/Verbatis) (приватный):
сырые тексты ПСС-90, специально обработанные. `make verbatis-fetch` клонирует его
в `data/verbatis`; `scripts/corpus.py` читает 5 jsonl-корпусов
(`data/verbatis/data/raw/tolstoy/`):

| Файл | Записей | Тип |
|---|---:|---|
| `tolstoy_letters.jsonl` | 9087 | письмо |
| `tolstoy_diaries.jsonl` | 4584 | дневник |
| `tolstoy_notes_dated.jsonl` | 576 | записная книжка |
| `tolstoy_articles.jsonl` | 278 | статья |
| `tolstoy_treatises.jsonl` | 26 | трактат |

Вся метадата записи (адресат, дата, место, том, страницы, возраст Толстого,
`in_ss22` и т.д.) сохраняется в индекс — она видна в «Источниках» ответа
и позволяет фильтровать (`SRC_TYPES` в `.env`).

Legacy: `scripts/epub_to_txt.py` (EPUB → tom*.txt, проза) и `scripts/tei_to_txt.py`
(TEI/XML → tei_*.txt) остаются в репо как справочные — в пайплайн не входят.

## MCP для Cursor (инструменты корпуса)

MCP-сервер встроен в приложение: `https://verbatis.ru/mcp` (streamable HTTP).
Даёт агенту Cursor фактуру корпуса — **без вызовов внешней LLM** (генерацию
агент делает сам):

| Инструмент | Что делает |
|---|---|
| `tolstoy_search` | семантический поиск фрагментов с фильтрами (тип, адресат, годы, том) |
| `tolstoy_browse` | списки записей по метаданным (напр. «письма Черткову за 1890-е») |
| `tolstoy_facets` | словарь значений фильтров (типы, годы, адресаты, тома) |
| `tolstoy_doc` | полный текст записи по id |
| `tolstoy_grep` | точный поиск слова/фразы с контекстом и счётчиками |

**Настройка (на машине Cursor ничего ставить не нужно):**

1. На сервере в `.env` задать `MCP_API_KEY` (сгенерировать:
   `python3 -c "import secrets; print(secrets.token_urlsafe(32))"`).
2. В Cursor создать `~/.cursor/mcp.json` (глобально, вне репозитория):

```json
{
  "mcpServers": {
    "tolstoy": {
      "url": "https://verbatis.ru/mcp",
      "headers": { "Authorization": "Bearer ВАШ_MCP_API_KEY" }
    }
  }
}
```

3. Перезапустить Cursor — сервер появится в Settings → Tools & Integrations → MCP.

Доступ — только по токену: без него `/mcp` возвращает 401, а если токен не
задан на сервере — маршрута нет вовсе. Пример конфига — `.cursor/mcp.json.example`.

## Деплой на сервер (Docker + Caddy + Let's Encrypt)

- **Локально** — HTTP, без Docker (`make run`), для разработки.
- **Сервер** — Caddy (HTTPS, порт 443) → Docker (HTTP, localhost:8000).

На сервере нужен Docker + docker compose + Caddy. Индексы собираются там же,
где используются (`make embeddings`), в образ не пекутся и rsync'ом не
синкаются (`.rsyncignore`: `index*/`).

### Полная инструкция

См. [CADDY_SETUP.md](CADDY_SETUP.md) — пошаговая инструкция по переходу на домен
с автоматическими Let's Encrypt сертификатами.

### Краткая версия

```bash
# 1. Настроить DNS: verbatis.ru → IP сервера (A-запись)

# 2. Локально: деплой обновлённой конфигурации
./deploy.sh

# 3. На сервере: установить Caddy
ssh root@<SERVER_IP>
cd /opt/leotolstoy
chmod +x scripts/setup_caddy.sh
sudo ./scripts/setup_caddy.sh

# 4. Проверка
curl https://verbatis.ru/health
# Должно вернуть: {"status":"ok"}
```

Caddy автоматически получит и обновит Let's Encrypt сертификаты.

Заход: **https://verbatis.ru** — валидный зелёный замочек, без предупреждений браузера.

Управление:

```bash
docker compose ps           # статус
docker compose logs -f      # логи
docker compose restart      # перезапуск
docker compose down         # остановить
docker compose up -d --build   # пересобрать после изменений кода
```

Проверка: `curl -s localhost:8000/health`, затем в браузере `http://<IP-сервера>:8000`.
Первый старт ~10-15 с (качается модель эмбеддингов в volume `model_cache`,
дальше — мгновенно). Память под нагрузкой ~650 МБ.

После смены модели/провайдера правьте `.env` и `docker compose restart` —
пересборка образа не нужна (переменные читаются из `.env` при старте).

## ⚠️ Совместимость эмбеддингов

Индекс (`index/`) и сервер должны использовать **одну и ту же модель И версию
`fastembed`**. Разные версии могут менять пуллинг → векторы запроса не совпадут
с индексом, и поиск деградирует. Версия зафиксирована в `uv.lock` (и локально, и на
сервере ставится через `uv sync --frozen`). При обновлении fastembed (`uv lock`)
пересоберите индекс (`make embeddings`) и залейте заново.

## Настройка стиля и модели

Провайдер генерации — любой OpenAI-совместимый API, задаётся в `.env`:
- **`LLM_BASE_URL`** — базовый URL (без `/chat/completions`). Примеры: OpenRouter,
  OpenAI, DeepSeek напрямую, Together, локальный Ollama — см. `.env.example`.
- **`LLM_MODEL`** — имя модели.
- **`LLM_API_KEY`** — ключ.
- Старые `OPENROUTER_API_KEY` / `OPENROUTER_MODEL` тоже работают (fallback).

Прочее:
- **Системный промпт** (характер речи Толстого) — `SYSTEM_PROMPT` в `server/app.py`.
- **Только письма / дневники / статьи…** — `SRC_TYPES=письмо` / `письмо,дневник`
  и т.п. в `.env` (типы: письмо, дневник, записная книжка, статья, трактат).
- **Сколько фрагментов в контексте** — `TOP_K` в `.env`.

## Тесты

```bash
uv run pytest        # 118 тестов, ~1с, без сети и без загрузки модели
```

Покрытие: парсинг/чанкинг корпуса, косинусный поиск (на синтетическом индексе),
API через FastAPI TestClient с замоканными эмбеддером и LLM — auth, rate limit,
лимит длины сообщения, health, favicon; MCP — протокол streamable HTTP, токен,
5 инструментов.

**Деплой прогоняет тесты сам:** `deploy.sh` перед синхронизацией запускает
`pytest tests/` и **прерывает деплой (exit 1) при любом провале** — на сервер
ничего не уедет, пока тесты красные. pytest — dev-зависимость (`--no-dev` в
Docker), в prod-образ не попадает.

## Безопасность

- Basic Auth на всех роутах; пароль только в `.env` (не в коде, не в git —
  `*.env` в `.dockerignore` и `.gitignore`).
- **Сервер: HTTPS** (Let's Encrypt через Caddy) — трафик, включая пароль Basic Auth,
  шифруется валидным сертификатом. Автоматическое обновление сертификатов.
- **Локально: HTTP** без TLS — только для разработки (`make run`).
- Docker контейнер доступен только через localhost:8000, публично открыт только
  Caddy на портах 80 (для ACME challenge) и 443 (HTTPS).

### Защита от абьюза (rate limiting)

Каждый запрос к `/chat` = платный вызов LLM, поэтому есть лимиты (настраиваются
в `.env`, значения по умолчанию):

- `RATE_LIMIT_CHAT=20/minute;200/day` — потолок запросов к чату на IP. Защищает
  API-счёт: скрипт-абьюз упрётся в 429.
- `RATE_LIMIT_LOGIN=30/minute` + `AUTH_FAIL_DELAY=0.5` — тормозят перебор пароля.
- `MAX_MESSAGE_CHARS=2000` — длиннее → 422 (не даёт слать мегабайты в LLM).
- `LLM_TIMEOUT=120` — таймаут запроса к провайдеру.

Rate limit по IP: если сервер стоит за реверс-прокси, прокси должен передавать
реальный IP клиента (`X-Forwarded-For`), иначе все запросы будут считаться с
одного адреса. При прямом доступе (как в текущем compose) это работает из коробки.
