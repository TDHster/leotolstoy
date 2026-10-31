# CLAUDE.md

Гайд для быстрого понимания проекта (для Claude Code и разработчиков).

## Что это

**«Беседа с Л. Н. Толстым»** — чат, отвечающий в стиле Льва Толстого. RAG поверх
текстов ПСС-90 из приватного репозитория [Verbatis](https://github.com/Nikulov-IB/Verbatis):
**5 корпусов** (письма 9087, дневники 4584, записные книжки 576, статьи 278,
трактаты 26 записей — всего ~14.5 тыс. записей, ~84 тыс. чанков). Поиск релевантных
фрагментов и эмбеддинг запроса — локальной ONNX-моделью на сервере; генерация
ответа — через внешнюю LLM с системным промптом «говори как Толстой». В UI по
клику на фрагмент открывается полный текст источника.

## Архитектура

```
Локальная машина (разработка, подготовка)    Сервер (прод, 89.125.117.142)
├─ make verbatis-fetch  клон Verbatis         ├─ Caddy (reverse proxy, :443, TLS Let's Encrypt)
│  (data/verbatis/, приватный репо)          ├─ Docker (FastAPI, :8000 HTTP localhost)
├─ make run       локально по HTTP (dev)     │    ├─ Basic Auth (1 пользователь)
└─ ./deploy.sh    тесты + rsync + compose    │    ├─ эмбеддинг запроса (ONNX, локально)
                                             │    ├─ косинус-поиск top-k (numpy) по index/
                                             │    ├─ GET /doc/{id} — полный текст (index/docs.json)
                                             │    ├─ /mcp — MCP-сервер для Cursor
                                             │    │   (streamable HTTP, Bearer MCP_API_KEY,
                                             │    │    5 инструментов фактуры БЕЗ вызова LLM)
                                             │    └─ генерация (внешняя LLM)
                                             └─ make embeddings — сборка индекса НА СЕРВЕРЕ
                                                  (rsync индексы не возит: .rsyncignore index*/)
```

Ключевые решения:
- **Единственный источник текстов — Verbatis** (ПСС-90, «специально обработанные
  для нас»). `scripts/corpus.py` читает 5 jsonl из `data/verbatis/data/raw/tolstoy/`,
  вся метадата записи сохраняется в чанк (id, date, date_start/end, age, title,
  volume, pages, file, in_ss22, sent, у писем addressee/place/number) + составные
  ключи UI (type, vol, to, num, year, entry). Обновление корпуса: `make
  verbatis-fetch` → `./deploy.sh` → пересборка индекса на сервере.
- **Эмбеддинги — только локальная модель (ONNX)** для документов и запросов.
  Qwen отключён: код (`build_embeddings_api.py`, OpenRouterEmbedder) дремлет в
  репо, таргета `embeddings-qwen` нет, `index.qwen3-embedding-8b/` с сервера
  удалён. **Индекс собирается там же, где используется**: прод — на сервере
  (`make embeddings`, ~6 ч CPU на ~84 тыс. чанков), локально — только для dev.
  Сборка и рантайм — одной версией fastembed из `uv.lock` (см. Подводные камни).
- **Поиск — numpy** (brute-force косинус, ~84 тыс. фрагментов = миллисекунды). БД нет.
- **Чанкинг: 1 абзац = 1 чанк**, без перекрытий; подписи/даты в конце писем из
  выдачи вырезаются (см. «Индекс»).
- **Полный текст источника**: `index/docs.json` + `GET /doc/{id}` → модалка в UI
  с подсветкой кликнутого абзаца.
- **Генерация — любой OpenAI-совместимый API** (эндпоинт в `.env`).
- **MCP-сервер для Cursor** встроен в FastAPI: `/mcp` (streamable HTTP через
  FastMCP), 5 инструментов фактуры — поиск фрагментов, навигация по
  метаданным, полные тексты. **Внешнюю LLM из MCP НЕ вызывает** (генерацию
  делает агент сам). Доступ только по Bearer-токену `MCP_API_KEY` (пусто =
  /mcp выключен). На машине Cursor ничего ставить не нужно — URL + заголовок.
- **Локально = HTTP без Docker; сервер = Caddy (HTTPS, Let's Encrypt) → Docker (HTTP)**.
- **Домен**: https://verbatis.ru (доступ через валидный TLS-сертификат).

## Стек

- Python 3.12, зависимости через **uv** (`pyproject.toml` + `uv.lock`).
- FastAPI + uvicorn, httpx (вызов LLM), slowapi (rate limit).
- Эмбеддинги: **fastembed** (ONNX, без torch), локальная модель
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims, ~120 МБ).
- Docker + docker compose, тесты — pytest.

## Файлы

| Путь | Назначение |
|---|---|
| `scripts/epub_to_txt.py` | **legacy**: EPUB → `data/tom*.txt` (не в пайплайне; выход в архиве `data/90 томов Толстого/`) |
| `scripts/tei_to_txt.py` | **legacy**: TEI/XML писем и дневников → `data/tei_*.txt` (заменён Verbatis'ом; не в пайплайне) |
| `scripts/corpus.py` | чтение 5 jsonl Verbatis (`data/verbatis/…`), нарезка на чанки (1 абзац = 1 чанк), вся метадата записи в meta, фильтр подписей писем, `load_docs` (полные тексты для /doc) |
| `scripts/embed.py` | общий модуль эмбеддингов (сборка и сервер) |
| `scripts/build_embeddings.py` | сборка локального индекса → `index/vectors.npy` + `chunks.json` + `docs.json`; тестовые переключатели `EMBED_INDEX_DIR` / `EMBED_LIMIT` |
| `scripts/build_embeddings_gpu.py` | сборка локального индекса на GPU (CUDA; альтернатива CPU-билдеру) |
| `scripts/build_embeddings_api.py` | **дремлет** (Qwen отключён): сборка индекса внешней моделью через OpenRouter |
| `server/embedders.py` | LocalEmbedder (fastembed) + OpenRouterEmbedder (дремлет) — выбор по metadata индекса |
| `scripts/entrypoint.sh` | Docker entrypoint: включает TLS, если смонтированы сертификаты (устарело, TLS теперь на Caddy) |
| `scripts/gen_cert.sh` | генерация самоподписанного TLS-сертификата на IP (устарело, используется Let's Encrypt через Caddy) |
| `scripts/setup_caddy.sh` | установка и настройка Caddy на сервере Ubuntu |
| `Caddyfile` | конфигурация Caddy reverse proxy (копируется на сервер в `/etc/caddy/Caddyfile`) |
| `CADDY_SETUP.md` | инструкция по переходу на Caddy + домен verbatis.ru |
| `server/app.py` | FastAPI: auth, rate limit, RAG, вызов LLM, `GET /doc/{id}` (полный текст источника), **MCP-сервер на /mcp** (FastMCP, 5 инструментов, Bearer `MCP_API_KEY`) |
| `server/retriever.py` | загрузка индекса + косинусный поиск + метаданные (browse/grep/facets для MCP) |
| `server/static/index.html` | фронт (чат, показ фрагментов-источников, клик по фрагменту → модалка с полным текстом) |
| `tests/` | pytest: corpus, retriever, API (LLM/эмбеддер замоканы), MCP (`test_mcp.py`) |
| `.cursor/mcp.json.example` | конфиг MCP в Cursor (пример; реальный `.cursor/mcp.json` с токеном — вне git) |
| `deploy.sh` + `.deploy` | деплой: прогон тестов → rsync → compose на сервере |

## Индекс (не в git)

- `index/vectors.npy` — массив N×384 float32, L2-нормализован (MiniLM-L12-v2).
- **Источник — jsonl корпуса Verbatis** (`data/verbatis/data/raw/tolstoy/*.jsonl`;
  ~84 тыс. чанков: письмо ~37 тыс., дневник ~14.7 тыс., записная книжка ~3.8 тыс.,
  статья ~12 тыс., трактат ~16.8 тыс.). Метаданные: id, дата (date, date_start,
  date_end), возраст Толстого, титул, том ПСС, страницы, путь к XML, in_ss22,
  признак отправки письма, у писем — адресат/место/№ письма.
- **Чанкинг: 1 абзац = 1 чанк.** Абзацы длиннее ~675 симв. режутся по предложениям
  БЕЗ перекрытий; короткие (<40 симв.) склеиваются со следующим абзацем. Дублей
  на стыках чанков нет. Записи целиком короче 40 симв. дают 1 чанк, который
  выпадает из индекса (фильтр в билдере), но остаётся в docs.json.
- **Фильтр подписей/дат в письмах** (уровень чанков, данные не тронуты): хвостовые
  короткие строки-подписи («Л. Толстой.», «Граф Лев Толстой.», «Л. Т.», «Léon
  Tolstoï.», «Ваш Гр. Л. Толстой.», формулы «С совершенным почтением…») и даты
  («31 окт. 1909.», «Апреля 12-го дня 1847 года.») вырезаются из чанков писем
  (`corpus._filter_letter_signature`). Консервативно: только хвост, только ≤80
  симв., явные паттерны; телеграммы с подписью в той же строке («Ждешь меня…
  Толстой.») НЕ трогаются.
- **Qwen отключён**: `index.qwen3-embedding-8b/` с сервера удалён, таргета
  `embeddings-qwen` нет. Код внешнего эмбеддера дремлет (`build_embeddings_api.py`,
  `server/embedders.py` OpenRouterEmbedder, `EXTERNAL_EMBEDDER_*` в .env) — при
  желании вернуть: вернуть таргет и пересобрать индекс на сервере.
- `index/chunks.json` — тексты фрагментов + метаданные (~150 МБ на проде),
  `index/vectors.npy` (~129 МБ для 84 тыс. чанков float32).
- `index/docs.json` — полные тексты записей БЕЗ фильтра подписей для
  `GET /doc/{id}`: 14 541 запись, ~100 МБ. Создаётся ТОЛЬКО локальным билдером
  (`make embeddings`), в том же прогоне, что и chunks.json. `doc_id` — сквозной
  счётчик записей, совпадает с `meta["doc"]` в chunks.json (общий итератор
  `corpus._iter_indexed_records`).
- Связь по позиции: i-й вектор ↔ i-й элемент chunks.json.
- Индексы собираются там же, где используются: прод — на сервере (`make
  embeddings`), локально — только для dev. В `.rsyncignore` — `index*/`,
  deploy.sh индексы не копирует и не удаляет.
- Векторизация занимает ~6 ч на CPU сервера (~84 тыс. чанков, сентябрь 2026).

## MCP для Cursor (/mcp)

MCP-сервер (streamable HTTP, FastMCP 3.x) встроен в FastAPI — инструменты
работают in-process рядом с индексом. **Внешнюю LLM не вызывает**: агент Cursor
получает фактуру и генерирует сам. Инструменты:

| Инструмент | Что делает |
|---|---|
| `tolstoy_search` | семантический поиск фрагментов (локальный эмбеддинг + numpy) с фильтрами: types / addressee / year_from / year_to / volume, `min_score` (отсечка). Возвращает текст, source, score, полную meta |
| `tolstoy_browse` | список ЗАПИСЕЙ по метаданным, по дате, с превью и пагинацией |
| `tolstoy_facets` | словарь значений фильтров: типы, годы, топ-100 адресатов, тома (кешируется в Retriever) |
| `tolstoy_doc` | полный текст записи + метаданные по doc_id (index/docs.json) |
| `tolstoy_grep` | точный поиск слова/фразы по чанкам (регэксп, IGNORECASE, `whole_word` — `\b`), контекст ±150, счётчики вхождений, пагинация `offset` (limit ≤ 50) |

- Auth: Bearer `MCP_API_KEY` — проверяется ASGI-мидлварью `_MCPTokenAuth`
  (FastAPI `Depends` на mount не работают). Пусто = /mcp не смонтирован.
- Конфиг Cursor: `.cursor/mcp.json.example` (URL `https://verbatis.ru/mcp` +
  заголовок Authorization); реальный файл с токеном — в `~/.cursor/mcp.json`
  или проектном `.cursor/mcp.json` (в .gitignore).
- Фильтры метаданных реализованы в `server/retriever.py` (`_meta_matches`,
  `browse`, `grep`, `facets`) поверх уже загруженных в RAM чанков — индекс
  пересобирать не нужно.

## Команды

```bash
make install      # uv sync
make verbatis-fetch # скачать/обновить корпус Verbatis (git clone/pull data/verbatis)
make embeddings   # data/verbatis/*.jsonl → index/ (локальная модель; на проде — на сервере)
make embeddings-gpu # то же на GPU (требует CUDA)
make visualization # визуализация тем: index/ → visualization/out/ (локально ИЛИ на сервере после make embeddings)
make run          # локальный сервер (HTTP, http://localhost:8000)
uv run pytest     # тесты (~0.4с, без сети/модели)
./deploy.sh       # деплой (тесты → rsync → Caddy reload → docker compose на сервере; индексы НЕ возит)
```

**Визуализация (visualization/)**: статические страницы (карта смыслов + потоки
тем) собираются из index/ командой `make visualization` — run.py сравнивает
штамп index/ с кэшем и при смене индекса пересчитывает всё (UMAP + кластеры +
LLM-названия, ~15–30 мин на 2 ядрах сервера), иначе отрабатывает за секунды.
URL: https://verbatis.ru/viz/ — статику отдаёт Caddy (route в Caddyfile, мимо
FastAPI, публично — там только тексты Толстого). После обновления индекса на
сервере (`make embeddings`) запустить там же `make visualization` — страница
обновится (out/ на сервере авторитетна: deploy.sh её не перезаписывает,
bootstrap только при первом деплое; см. .rsyncignore). Для билда на сервере
нужен uv (установлен: /usr/local/bin/uv). Память сервера впритык (3.8 ГиБ):
билд — ручная команда, не часть deploy.sh.

Docker (на сервере): `docker compose up -d --build`, `logs -f`, `restart`, `down`.

**Полный цикл обновления корпуса/индекса на проде**:
локально `make verbatis-fetch` (новые тексты Verbatis) → `./deploy.sh` (jsonl
уедут на сервер) → на сервере `make embeddings` — Makefile сам перезапустит
контейнер (`docker compose up -d --force-recreate`) → `make visualization`.

## Конфиг (.env)

Провайдер LLM: `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` (примеры в `.env.example`;
старые `OPENROUTER_*` работают как fallback). Auth: `APP_USER`, `APP_PASS`.
**MCP**: `MCP_API_KEY` (Bearer-токен /mcp; пусто = MCP выключен).
Поиск: `TOP_K` (потолок фрагментов), отсечка слабых — `MIN_SCORE` (абс. косинус),
`REL_SCORE` (доля от лучшего), `MIN_K` (сколько оставить, если порог не прошёл
никто; тогда LLM получает пометку «прямой опоры нет»); 0 = выключено. Калибровка
на MiniLM: темы корпуса ~0.65–0.9, запросы «мимо» ~0.3–0.5 → 0.6 / 0.9. `SRC_TYPES` (типы: письмо/дневник/записная книжка/статья/трактат;
пусто = все; также ограничивает MCP-инструменты). Сборка: `EMBED_TYPES` (пусто = все типы Verbatis).
Лимиты: `RATE_LIMIT_CHAT`, `RATE_LIMIT_LOGIN`, `RATE_LIMIT_DOC` (GET /doc/{id}),
`MAX_MESSAGE_CHARS`, `AUTH_FAIL_DELAY`.
Логи: `LOG_LEVEL`, `LOG_QUERIES` (текст запросов — ПД, по умолчанию выкл).

## ⚠️ Подводные камни (важно)

- **fastmcp закреплён `>=3.2.4,<4`** — fastmcp 4.x требует starlette>=1.0,
  а fastapi 0.115.x требует starlette<0.47 (конфликт); fastmcp 3.x также
  требует uvicorn>=0.35 (пин в pyproject поднят до `<0.55`). При смене версии
  fastmcp проверять совместимость с fastapi/starlette.
- **Combined lifespan обязателен** (`_combined_lifespan` в app.py): Starlette
  не форвардит lifespan в mount-подприложения, а без `_mcp_app.lifespan` у
  FastMCP не инициализируется session manager — /mcp отдавал бы 500.
- **Auth на /mcp — только ASGI-мидлварью** (`_MCPTokenAuth`): FastAPI
  `Depends` на mount не работают. Токен сравнивается через `compare_digest`,
  ошибка — 401 JSON с задержкой `AUTH_FAIL_DELAY`.
- **MCP-сессии — в памяти процесса**: `--workers 1` обязателен (уже в
  Dockerfile и Makefile — не менять). После рестарта контейнера Cursor
  переподключается сам.
- **`/mcp` → 307 на `/mcp/`** — стандартное поведение Starlette для mount
  (redirect_slashes); MCP-клиенты следуют редиректу. Чтобы Location генерился
  со схемой https за Caddy, uvicorn запускается с `--proxy-headers` и
  `FORWARDED_ALLOW_IPS=*` (в Dockerfile): без второго контейнер видит
  источником адрес docker-моста, а не loopback, и заголовки отбрасывает.
- **НЕ добавлять `from __future__ import annotations` в `server/app.py`** — slowapi
  оборачивает роуты, при строковых аннотациях FastAPI теряет тело запроса (body
  становится query-параметром → все запросы 422). Держим реальные аннотации.
- **Совместимость эмбеддингов**: модель И версия `fastembed` при сборке индекса
  и в рантайме должны совпадать (иначе pooling разойдётся, поиск деградирует).
  Индекс собирается на том же хосте, где используется, — обе стороны берут
  `uv sync --frozen` из `uv.lock`, поэтому совместимы по построению; локальный
  Mac-индекс — только для dev-запуска тем же `uv.lock`. При обновлении fastembed
  или смене модели — пересобрать индекс (`make embeddings` на сервере). Новые
  версии fastembed предупреждают о смене pooling (CLS → mean) — это ещё один
  повод не менять версию без пересборки.
- **Корпус на сервере — только 5 jsonl**: `.rsyncignore` исключает из
  `data/verbatis/` всё, кроме `tolstoy_*.jsonl` (csv/txt/_for_embedding/.git не
  едут). Если файлов нет при старте билдера — он подскажет `make verbatis-fetch`.
- **`data/tom*.txt` и `data/tei_*.txt` уехали в архив** `data/90 томов Толстого/`
  (сентябрь 2026, переход на Verbatis). Корпус их не читает (glob только по
  `data/verbatis/…`). На сервере их нет: tom*.txt удалил rsync `--delete`, tei_*.txt
  снесены вручную (rsync исключённые не удаляет). Не запускать legacy-конвертеры —
  их выход ничего не изменит, но создаст мусорные файлы.
- **Поле `text` не должно попадать в meta чанка** — `_verbatis_meta` его
  вырезает; иначе полный текст записи дублировался бы в каждый чанк и раздул
  chunks.json на десятки МБ.
- **Предзагрузка модели на сервере**: при первом деплое `deploy.sh` автоматически
  скачивает модель эмбеддингов (~120 МБ) в Docker volume `model_cache` **до** запуска
  контейнера. Это нужно чтобы избежать OOM kill при одновременной загрузке модели
  и индекса (compose: mem_reservation 2000m, mem_limit 3000m). Volume переживает
  пересборки контейнера.
- **Индексы не едут через deploy.sh**: `index*/` в `.rsyncignore` — rsync их не
  копирует и не удаляет (`--delete` не трогает excluded). Индексы собираются
  и пересобираются на сервере (`make embeddings`).
- **Индекс читается в память при старте сервера** (lifespan загружает
  `chunks.json` + `vectors.npy` в RAM; uvicorn в проде без `--reload`). Поэтому
  после пересборки индекса контейнер нужно перезапустить. Хвост Makefile
  (`make embeddings`) делает это сам: `docker compose up -d --force-recreate`.
  Раньше там был `up -d` — для уже работающего контейнера это no-op («Container
  … Running», ничего не делает), и сервер молча продолжал отвечать по старому
  индексу из памяти.
- **Секреты**: `.env*`, `.deploy`, `certs/`, `favicon.ico` в `.gitignore`; в
  `.rsyncignore` их НЕТ (должны уехать на сервер). EPUB-издание 1978 г. под
  авторским правом — тоже не в git (`data/` игнорируется). Verbatis — приватный
  репозиторий: клон работает через git keychain-credentials на этой машине.
- **`docs.json` создаёт только локальный билдер** (`make embeddings`) — модалка
  полных текстов живёт на нём.
- **curl в Docker-образе нет** — healthcheck и внутренние проверки через Python.

## Open-source

MIT LICENSE (tdhster). Данные (EPUB, Verbatis), индекс, сертификаты, favicon,
секреты — вне репозитория. `data/README.md` объясняет структуру данных.
