# CLAUDE.md

Гайд для быстрого понимания проекта (для Claude Code и разработчиков).

## Что это

**«Беседа с Л. Н. Толстым»** — чат, отвечающий в стиле Льва Толстого. RAG поверх
всех 22 томов собрания сочинений в EPUB (художественные произведения, публицистика,
письма и дневники). Поиск релевантных фрагментов — локально; генерация ответа —
через внешнюю LLM с системным промптом «говори как Толстой».

## Архитектура

```
Локальная машина (разработка, подготовка)   Сервер (прод)
├─ make convert   EPUB → data/*.txt          └─ Caddy (reverse proxy, :443)
├─ make embeddings data/*.txt → index/            └─ TLS терминация (Let's Encrypt)
│     эмбеддинги локально (ONNX)                      └─ Docker (FastAPI, :8000 HTTP localhost)
├─ make run       локально по HTTP                        ├─ Basic Auth (1 пользователь)
└─ ./deploy.sh    rsync + тесты + compose                 ├─ эмбеддинг запроса (ONNX)
                                                           ├─ косинус-поиск top-k (numpy)
                                                           └─ генерация (внешняя LLM)
```

Ключевые решения:
- **Эмбеддинги — локально**, одна модель при сборке и на сервере. Векторы документов
  считаются один раз локально; на сервере эмбеддится только запрос.
- **Поиск — numpy** (brute-force косинус, ~19.6 тыс. фрагментов = ~1-3 мс). БД нет.
- **Генерация — любой OpenAI-совместимый API** (эндпоинт в `.env`).
- **Локально = HTTP без Docker; сервер = Caddy (HTTPS, Let's Encrypt) → Docker (HTTP)**.
- **Домен**: https://verbatis.ru (доступ через валидный TLS-сертификат).

## Стек

- Python 3.12, зависимости через **uv** (`pyproject.toml` + `uv.lock`).
- FastAPI + uvicorn, httpx (вызов LLM), slowapi (rate limit).
- Эмбеддинги: **fastembed** (ONNX, без torch), локальная модель
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims, ~120 МБ).
  Опционально вторая модель `qwen/qwen3-embedding-8b` (4096 dims) через OpenRouter
  API — переключается в UI, индексы живут параллельно (`index/` и `index.qwen3-embedding-8b/`).
- Docker + docker compose, тесты — pytest.

## Файлы

| Путь | Назначение |
|---|---|
| `scripts/epub_to_txt.py` | EPUB (все тома) → `data/*.txt`: тома 18-22 с детальной структурой (письма/дневники), остальные — весь текст целиком |
| `scripts/corpus.py` | чтение txt, нарезка на чанки + метаданные |
| `scripts/embed.py` | общий модуль эмбеддингов (сборка и сервер) |
| `scripts/build_embeddings.py` | сборка локального индекса → `index/vectors.npy` + `chunks.json` |
| `scripts/build_embeddings_api.py` | сборка индекса внешней моделью (Qwen через OpenRouter) → `index.<model>/` |
| `server/embedders.py` | LocalEmbedder (fastembed) + OpenRouterEmbedder (API) — выбор по metadata индекса |
| `scripts/entrypoint.sh` | Docker entrypoint: включает TLS, если смонтированы сертификаты (устарело, TLS теперь на Caddy) |
| `scripts/gen_cert.sh` | генерация самоподписанного TLS-сертификата на IP (устарело, используется Let's Encrypt через Caddy) |
| `scripts/setup_caddy.sh` | установка и настройка Caddy на сервере Ubuntu |
| `Caddyfile` | конфигурация Caddy reverse proxy (копируется на сервер в `/etc/caddy/Caddyfile`) |
| `CADDY_SETUP.md` | инструкция по переходу на Caddy + домен verbatis.ru |
| `server/app.py` | FastAPI: auth, rate limit, RAG, вызов LLM |
| `server/retriever.py` | загрузка индекса + косинусный поиск |
| `server/static/index.html` | фронт (чат, показ фрагментов-источников) |
| `tests/` | pytest: corpus, retriever, API (LLM/эмбеддер замоканы) |
| `deploy.sh` + `.deploy` | деплой: прогон тестов → rsync → compose на сервере |

## Индекс (не в git)

- `index/vectors.npy` — массив N×384 float32, L2-нормализован (MiniLM-L12-v2).
  Размер: ~30 МБ для ~19.6 тыс. фрагментов.
- `index.qwen3-embedding-8b/` — внешний индекс Qwen3-8B: `vectors.npy` N×4096 int8
  (~80 МБ), `chunks.json` — symlink на `index/chunks.json`, `metadata.json`
  помечает `api: true` / `quantized: true`.
- `index/chunks.json` — тексты фрагментов + метаданные (том, тип, адресат, год).
  Размер: ~43 МБ.
- Связь по позиции: i-й вектор ↔ i-й элемент chunks.json.
- Индексы собираются там же, где используются: локальный — `make embeddings`,
  внешний — `make embeddings-qwen` (на сервере). В `.rsyncignore` — `index*/`,
  deploy.sh индексы не копирует и не удаляет.
- Векторизация всех 22 томов занимает ~2-5 минут на обычном CPU.

### Альтернатива: векторизация через OpenRouter API (2026)

Вместо локального CPU можно использовать облачные embedding модели через OpenRouter.
Это значительно быстрее (1 час vs 10 часов на MacBook без GPU) и дёшево:

**Цены на embedding модели (2026):**
- **Qwen3 Embedding 8B**: $0.01 за 1M токенов (32K context, 8B params — лучшее качество)
  - Для ~19.6K чанков (~5M токенов): **$0.05** (~5₽)
- **Qwen3 Embedding 4B**: $0.02 за 1M токенов (32K context, 4B params)
  - Для ~19.6K чанков: **$0.10** (~10₽)
- **LiquidAI LFM2.5-Embedding-350M**: **бесплатно**, но только 512 tokens context

**Плюсы API:**
- Скорость: 1 час вместо 10 (на GPU, можно параллелить)
- Не греет локальную машину
- Дёшево для одноразовой векторизации

**Минусы API:**
- Зависимость от стороннего сервиса
- Latency на inference (хотя можно использовать API только для сборки индекса, 
  а runtime — локально с той же моделью)
- Нужна совместимость: если используете API для сборки, нужна та же модель 
  для эмбеддинга запросов на сервере

**Текущий подход**: обе модели живут параллельно — локальная MiniLM (бесплатная,
основная) и внешняя Qwen3-8B (переключается в UI). Индекс внешней модели
пересобирается на сервере: `make embeddings-qwen`.

## Команды

```bash
make install      # uv sync
make convert      # EPUB → data/*.txt
make embeddings   # data/*.txt → index/
make run          # локальный сервер (HTTP, http://localhost:8000)
uv run pytest     # тесты (~0.3с, без сети/модели)
./deploy.sh       # деплой (тесты → rsync → docker compose на сервере)
```

Docker (на сервере): `docker compose up -d --build`, `logs -f`, `restart`, `down`.

## Конфиг (.env)

Провайдер LLM: `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` (примеры в `.env.example`;
старые `OPENROUTER_*` работают как fallback). Auth: `APP_USER`, `APP_PASS`.
Лимиты: `RATE_LIMIT_CHAT`, `RATE_LIMIT_LOGIN`, `MAX_MESSAGE_CHARS`, `AUTH_FAIL_DELAY`.
Логи: `LOG_LEVEL`, `LOG_QUERIES` (текст запросов — ПД, по умолчанию выкл).

## ⚠️ Подводные камни (важно)

- **НЕ добавлять `from __future__ import annotations` в `server/app.py`** — slowapi
  оборачивает роуты, при строковых аннотациях FastAPI теряет тело запроса (body
  становится query-параметром → все запросы 422). Держим реальные аннотации.
- **Совместимость эмбеддингов**: модель И версия `fastembed` на сервере должны
  совпадать с теми, чем собран индекс локально (иначе pooling разойдётся, поиск
  деградирует). Зафиксировано в `uv.lock`, ставится через `uv sync --frozen`.
  При обновлении fastembed или смене модели — пересобрать индекс (`make embeddings`).
  **Текущая локальная модель: `paraphrase-multilingual-MiniLM-L12-v2`** (384 dims).
  Внешняя: `qwen/qwen3-embedding-8b` (4096 dims, int8-индекс) — runtime-запросы
  к ней идут через OpenRouter, на сервере нужен `EXTERNAL_EMBEDDER_API_KEY` в `.env`.
- **Предзагрузка модели на сервере**: при первом деплое `deploy.sh` автоматически
  скачивает модель эмбеддингов (~120 МБ) в Docker volume `model_cache` **до** запуска
  контейнера. Это нужно чтобы избежать OOM kill при одновременной загрузке модели
  и индекса (mem_limit: 2000m). Volume переживает пересборки контейнера.
- **Индексы не едут через deploy.sh**: `index*/` в `.rsyncignore` — rsync их не
  копирует и не удаляет (`--delete` не трогает excluded). Индексы собираются
  и пересобираются на сервере (`make embeddings` / `make embeddings-qwen`).
  Если каталог индекса отсутствует или повреждён при старте — сервер
  запускается без этой модели, остальные работают.
- **Секреты**: `.env*`, `.deploy`, `certs/`, `favicon.ico` в `.gitignore`; в
  `.rsyncignore` их НЕТ (должны уехать на сервер). EPUB-издание 1978 г. под
  авторским правом — тоже не в git (`data/` игнорируется).
- **Издание 1978 г. — современная орфография** (не дореформенная), нормализация
  не нужна.
- **curl в Docker-образе нет** — healthcheck и внутренние проверки через Python.

## Open-source

MIT LICENSE (tdhster). Данные (EPUB), индекс, сертификаты, favicon, секреты — вне
репозитория. `data/README.md` объясняет пользователю, куда класть свои EPUB.
