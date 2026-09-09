# CLAUDE.md

Гайд для быстрого понимания проекта (для Claude Code и разработчиков).

## Что это

**«Беседа с Л. Н. Толстым»** — чат, отвечающий в стиле Льва Толстого. RAG поверх
всех 22 томов собрания сочинений в EPUB (художественные произведения, публицистика,
письма и дневники). Поиск релевантных фрагментов — локально; генерация ответа —
через внешнюю LLM с системным промптом «говори как Толстой».

## Архитектура

```
Локальная машина (разработка, подготовка)   Сервер (прод, Docker)
├─ make convert   EPUB → data/*.txt          └─ Docker + HTTPS на порту 8443
├─ make embeddings data/*.txt → index/            └─ FastAPI (uvicorn, 1 worker)
│     эмбеддинги локально (ONNX)                      ├─ Basic Auth (1 пользователь)
├─ make run       локально по HTTP                    ├─ эмбеддинг запроса (ONNX, локально)
└─ ./deploy.sh    rsync + тесты + compose             ├─ косинус-поиск top-k (numpy)
                                                       └─ генерация (внешняя LLM)
```

Ключевые решения:
- **Эмбеддинги — локально**, одна модель при сборке и на сервере. Векторы документов
  считаются один раз локально; на сервере эмбеддится только запрос.
- **Поиск — numpy** (brute-force косинус, ~24.7 тыс. фрагментов = ~1-3 мс). БД нет.
- **Генерация — любой OpenAI-совместимый API** (эндпоинт в `.env`).
- **Локально = HTTP без Docker; сервер = Docker + HTTPS** (самоподписанный серт на IP, 8443).

## Стек

- Python 3.12, зависимости через **uv** (`pyproject.toml` + `uv.lock`).
- FastAPI + uvicorn, httpx (вызов LLM), slowapi (rate limit).
- Эмбеддинги: **fastembed** (ONNX, без torch), модель
  `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dims, ~0.22 ГБ).
- Docker + docker compose, тесты — pytest.

## Файлы

| Путь | Назначение |
|---|---|
| `scripts/epub_to_txt.py` | EPUB (все тома) → `data/*.txt`: тома 18-22 с детальной структурой (письма/дневники), остальные — весь текст целиком |
| `scripts/corpus.py` | чтение txt, нарезка на чанки + метаданные |
| `scripts/embed.py` | общий модуль эмбеддингов (сборка и сервер) |
| `scripts/build_embeddings.py` | сборка индекса → `index/vectors.npy` + `chunks.json` |
| `scripts/entrypoint.sh` | Docker entrypoint: включает TLS, если смонтированы сертификаты |
| `scripts/gen_cert.sh` | генерация самоподписанного TLS-сертификата на IP |
| `server/app.py` | FastAPI: auth, rate limit, RAG, вызов LLM |
| `server/retriever.py` | загрузка индекса + косинусный поиск |
| `server/static/index.html` | фронт (чат, показ фрагментов-источников) |
| `tests/` | pytest: corpus, retriever, API (LLM/эмбеддер замоканы) |
| `deploy.sh` + `.deploy` | деплой: прогон тестов → rsync → compose на сервере |

## Индекс (не в git)

- `index/vectors.npy` — массив N×384, только числа (векторы, L2-нормализованы).
  Размер: ~38 МБ для ~24.7 тыс. фрагментов.
- `index/chunks.json` — тексты фрагментов + метаданные (том, тип, адресат, год).
  Размер: ~50-60 МБ.
- Связь по позиции: i-й вектор ↔ i-й элемент chunks.json.
- Собирается локально (`make embeddings`), переносится на сервер (rsync/scp).
- Векторизация всех 22 томов занимает ~2-5 минут на обычном CPU.

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
  При обновлении fastembed — пересобрать индекс.
- **Секреты**: `.env*`, `.deploy`, `certs/`, `favicon.ico` в `.gitignore`; в
  `.rsyncignore` их НЕТ (должны уехать на сервер). EPUB-издание 1978 г. под
  авторским правом — тоже не в git (`data/` игнорируется).
- **Издание 1978 г. — современная орфография** (не дореформенная), нормализация
  не нужна.
- **curl в Docker-образе нет** — healthcheck и внутренние проверки через Python.

## Open-source

MIT LICENSE (tdhster). Данные (EPUB), индекс, сертификаты, favicon, секреты — вне
репозитория. `data/README.md` объясняет пользователю, куда класть свои EPUB.
