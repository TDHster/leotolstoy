# CLAUDE.md

Гайд для быстрого понимания проекта (для Claude Code и разработчиков).

## Что это

**«Беседа с Л. Н. Толстым»** — чат, отвечающий в стиле Льва Толстого. RAG поверх
писем и дневников из TEI-корпуса tolstoydigital (90-томное ПСС: ~9 тыс. писем и
~4.3 тыс. записей дневников; проза из EPUB — legacy, в индексе её нет). Поиск
релевантных фрагментов и эмбеддинг запроса — локальной ONNX-моделью на сервере;
генерация ответа — через внешнюю LLM с системным промптом «говори как Толстой».
В UI по клику на фрагмент открывается полный текст письма/записи дневника.

## Архитектура

```
Локальная машина (разработка, подготовка)    Сервер (прод, 89.125.117.142)
├─ make tei-fetch TEI-корпус (sparse clone)  ├─ Caddy (reverse proxy, :443, TLS Let's Encrypt)
├─ make convert   TEI/XML → data/*.txt       ├─ Docker (FastAPI, :8000 HTTP localhost)
├─ make run       локально по HTTP (dev)     │    ├─ Basic Auth (1 пользователь)
└─ ./deploy.sh    тесты + rsync + compose    │    ├─ эмбеддинг запроса (ONNX, локально)
                                             │    ├─ косинус-поиск top-k (numpy) по index/
                                             │    ├─ GET /doc/{id} — полный текст (index/docs.json)
                                             │    └─ генерация (внешняя LLM)
                                             └─ make embeddings — сборка индекса НА СЕРВЕРЕ
                                                  (rsync индексы не возит: .rsyncignore index*/)
```

Ключевые решения:
- **Эмбеддинги — локально (ONNX), одна модель** для документов и запросов.
  **Индекс собирается там же, где используется**: прод — на сервере
  (`make embeddings`, ~4 ч CPU на 54.4 тыс. чанков), локально — только для dev.
  Сборка и рантайм — одной версией fastembed из `uv.lock` (см. Подводные камни).
- **Поиск — numpy** (brute-force косинус, 54.4 тыс. фрагментов = миллисекунды). БД нет.
- **Чанкинг: 1 абзац = 1 чанк**, без перекрытий; подписи/даты в конце писем из
  выдачи вырезаются (см. «Индекс»).
- **Полный текст источника**: `index/docs.json` + `GET /doc/{id}` → модалка в UI
  с подсветкой кликнутого абзаца.
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
| `scripts/epub_to_txt.py` | **legacy**: EPUB → `data/tom*.txt` (до переезда на digital Tolstoy; проза tom01–45 пока оттуда) |
| `scripts/tei_to_txt.py` | TEI/XML писем и дневников (tolstoydigital/TEI, CC BY-SA) → `data/tei_letters.txt`, `data/tei_diaries.txt` |
| `scripts/corpus.py` | чтение `data/tom*.txt` + `data/tei_*.txt`, нарезка на чанки (1 абзац = 1 чанк) + метаданные, фильтр подписей писем, `load_docs` (полные тексты для /doc) |
| `scripts/embed.py` | общий модуль эмбеддингов (сборка и сервер) |
| `scripts/build_embeddings.py` | сборка локального индекса → `index/vectors.npy` + `chunks.json` + `docs.json` |
| `scripts/build_embeddings_gpu.py` | сборка локального индекса на GPU (CUDA; альтернатива CPU-билдеру) |
| `scripts/build_embeddings_api.py` | сборка индекса внешней моделью (Qwen через OpenRouter) → `index.<model>/` |
| `server/embedders.py` | LocalEmbedder (fastembed) + OpenRouterEmbedder (API) — выбор по metadata индекса |
| `scripts/entrypoint.sh` | Docker entrypoint: включает TLS, если смонтированы сертификаты (устарело, TLS теперь на Caddy) |
| `scripts/gen_cert.sh` | генерация самоподписанного TLS-сертификата на IP (устарело, используется Let's Encrypt через Caddy) |
| `scripts/setup_caddy.sh` | установка и настройка Caddy на сервере Ubuntu |
| `Caddyfile` | конфигурация Caddy reverse proxy (копируется на сервер в `/etc/caddy/Caddyfile`) |
| `CADDY_SETUP.md` | инструкция по переходу на Caddy + домен verbatis.ru |
| `server/app.py` | FastAPI: auth, rate limit, RAG, вызов LLM, `GET /doc/{id}` (полный текст источника) |
| `server/retriever.py` | загрузка индекса + косинусный поиск |
| `server/static/index.html` | фронт (чат, показ фрагментов-источников, клик по фрагменту → модалка с полным текстом) |
| `tests/` | pytest: corpus, retriever, API (LLM/эмбеддер замоканы) |
| `deploy.sh` + `.deploy` | деплой: прогон тестов → rsync → compose на сервере |

## Индекс (не в git)

- `index/vectors.npy` — массив N×384 float32, L2-нормализован (MiniLM-L12-v2).
- **Письма и дневники — из TEI-корпуса tolstoydigital** (`data/tei_letters.txt`,
  `data/tei_diaries.txt`; 54.4 тыс. чанков на проде при EMBED_TYPES=письмо,дневник).
  Метаданные: адресат, дата, место, № письма в томе, титул, библиографическая
  ссылка ПСС, страницы, год издания, признак дореформенной орфографии.
- **Чанкинг (сентябрь 2026): 1 абзац = 1 чанк.** Абзацы длиннее ~675 симв. режутся
  по предложениям БЕЗ перекрытий; короткие (<40 симв., иначе выпали бы при
  эмбеддинге) склеиваются со следующим абзацем (хвост — с предыдущим/отдельным
  чанком). Дублей на стыках чанков нет.
- **Фильтр подписей/дат в письмах** (уровень чанков, данные не тронуты): хвостовые
  короткие строки-подписи («Л. Толстой.», «Граф Лев Толстой.», «Л. Т.», «Léon
  Tolstoï.», «Ваш Гр. Л. Толстой.», формулы «С совершенным почтением…») и даты
  («31 окт. 1909.», «Апреля 12-го дня 1847 года.») вырезаются из чанков писем
  (`corpus._filter_letter_signature`). Консервативно: только хвост, только ≤80
  симв., явные паттерны; телеграммы с подписью в той же строке («Ждешь меня…
  Толстой.») НЕ трогаются.
- Проза (tom01–45) — из EPUB, legacy; в индексе её нет (фильтр EMBED_TYPES).
  Старые EPUB-тома писем/дневников (46–90) лежат в `data/unused_tom/` — вне
  корпуса (глоб читает только `data/*.txt`, не подкаталоги).
- `index.qwen3-embedding-8b/` — внешний индекс Qwen3-8B: `vectors.npy` N×4096 float32
  (~0.9 ГБ для 54.4 тыс. чанков; влезает в 4 ГБ RAM сервера, mem_limit 3000m).
  Раньше int8 — усечение обнуляло ~39% компонент и искажало топ-10.
  `EMBED_INDEX_DTYPE` позволяет собрать float16 при нехватке памяти. `chunks.json` —
  symlink на `index/chunks.json`, `metadata.json` помечает `api: true` / `dtype` / `provider`.
- `index/chunks.json` — тексты фрагментов + метаданные (том, тип, адресат, год,
  `doc` — id полного текста, `chunk` — № чанка в записи). На проде: ~72 МБ
  (vectors.npy ~84 МБ) для 54.4 тыс. чанков (письма+дневники).
- `index/docs.json` — полные тексты записей (письмо целиком / запись дневника,
  БЕЗ фильтра подписей) для `GET /doc/{id}`: 13 379 записей, ~38 МБ.
  Создаётся ТОЛЬКО локальным билдером (`make embeddings`), в том же прогоне, что
  и chunks.json. `doc_id` — сквозной счётчик записей, совпадает с `meta["doc"]`
  в chunks.json (общий итератор `corpus._iter_indexed_records`).
- Связь по позиции: i-й вектор ↔ i-й элемент chunks.json.
- Индексы собираются там же, где используются: локальный — `make embeddings`,
  внешний — `make embeddings-qwen` (на сервере). В `.rsyncignore` — `index*/`,
  deploy.sh индексы не копирует и не удаляет.
- Векторизация писем и дневников занимает ~4 часа на CPU сервера
  (3.9 чанков/с, 54.4 тыс. чанков, сентябрь 2026).

### Альтернатива: векторизация через OpenRouter API (2026)

Вместо локального CPU можно использовать облачные embedding модели через OpenRouter.
Это заметно быстрее ~4 часов локального CPU на сервере и дёшево:

**Цены на embedding модели (2026):**
- **Qwen3 Embedding 8B**: $0.01 за 1M токенов (32K context, 8B params — лучшее качество)
  - Для 54.4K чанков (~5M токенов): **~$0.05** (~5₽)
- **Qwen3 Embedding 4B**: $0.02 за 1M токенов (32K context, 4B params)
  - Для 54.4K чанков: **~$0.10** (~10₽)
- **LiquidAI LFM2.5-Embedding-350M**: **бесплатно**, но только 512 tokens context

**Плюсы API:**
- Скорость: заметно быстрее ~4 часов локального CPU (можно параллелить)
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
make tei-fetch    # скачать/обновить TEI-корпус (sparse: только texts/letters + texts/diaries, без аудио/картинок)
make convert      # TEI/XML → data/tei_letters.txt, data/tei_diaries.txt
make tei          # tei-fetch + convert
make embeddings   # data/*.txt → index/ (локальная модель; на проде — на сервере)
make embeddings-gpu # то же на GPU (требует CUDA)
make embeddings-qwen # сборка внешней моделью (Qwen через OpenRouter)
make visualization # визуализация тем: index/ → visualization/out/ (локально ИЛИ на сервере после make embeddings)
make run          # локальный сервер (HTTP, http://localhost:8000)
uv run pytest     # тесты (~0.3с, без сети/модели)
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

**Полный цикл обновления индекса на проде** (после смены данных/нарезки):
`./deploy.sh` → на сервере `make embeddings` — Makefile сам перезапустит
контейнер (`docker compose up -d --force-recreate`).

## Конфиг (.env)

Провайдер LLM: `LLM_API_KEY`, `LLM_MODEL`, `LLM_BASE_URL` (примеры в `.env.example`;
старые `OPENROUTER_*` работают как fallback). Auth: `APP_USER`, `APP_PASS`.
Поиск: `TOP_K`, `SRC_TYPES` (письмо/дневник).
Лимиты: `RATE_LIMIT_CHAT`, `RATE_LIMIT_LOGIN`, `RATE_LIMIT_DOC` (GET /doc/{id}),
`MAX_MESSAGE_CHARS`, `AUTH_FAIL_DELAY`.
Логи: `LOG_LEVEL`, `LOG_QUERIES` (текст запросов — ПД, по умолчанию выкл).

## ⚠️ Подводные камни (важно)

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
  **Текущая локальная модель: `paraphrase-multilingual-MiniLM-L12-v2`** (384 dims).
  Внешняя: `qwen/qwen3-embedding-8b` (4096 dims, float32-индекс) — runtime-запросы
  к ней идут через OpenRouter, на сервере нужен `EXTERNAL_EMBEDDER_API_KEY` в `.env`.
  **Провайдер-пин обязателен**: `EXTERNAL_EMBEDDER_PROVIDER` (дефолт `nebius`)
  должен быть одинаковым при сборке и в рантайме — OpenRouter роутит запросы к
  разным апстримам, один из которых (siliconflow) отдаёт модель в fp8-квантизации,
  и векторы индекса/запроса оказываются в разных пространствах. Чекпоинт сборки
  хранит fingerprint (model/provider/dims) и не резюмируется при несовпадении.
- **Предзагрузка модели на сервере**: при первом деплое `deploy.sh` автоматически
  скачивает модель эмбеддингов (~120 МБ) в Docker volume `model_cache` **до** запуска
  контейнера. Это нужно чтобы избежать OOM kill при одновременной загрузке модели
  и индекса (compose: mem_reservation 2000m, mem_limit 3000m). Volume переживает
  пересборки контейнера.
- **Индексы не едут через deploy.sh**: `index*/` в `.rsyncignore` — rsync их не
  копирует и не удаляет (`--delete` не трогает excluded). Индексы собираются
  и пересобираются на сервере (`make embeddings` / `make embeddings-qwen`).
  Если каталог индекса отсутствует или повреждён при старте — сервер
  запускается без этой модели, остальные работают.
- **Индекс читается в память при старте сервера** (lifespan загружает
  `chunks.json` + `vectors.npy` в RAM; uvicorn в проде без `--reload`). Поэтому
  после пересборки индекса контейнер нужно перезапустить. Хвост Makefile
  (`make embeddings` / `make embeddings-qwen`) делает это сам:
  `docker compose up -d --force-recreate`. Раньше там был `up -d` — для уже
  работающего контейнера это no-op («Container … Running», ничего не делает),
  и сервер молча продолжал отвечать по старому индексу из памяти.
- **Пересборка локального индекса рассинхронизирует Qwen**: `index.qwen3-embedding-8b/chunks.json`
  — symlink на `index/chunks.json`, а его `vectors.npy` — старые векторы. После
  `make embeddings` число чанков меняется → Retriever падает на assert «рассинхрон»,
  и выбор Qwen в UI отдаёт 503. Лечится пересборкой `make embeddings-qwen` на сервере.
  Qwen-билдер использует тот же `corpus.load_chunks` — наследует новую нарезку
  и фильтр подписей автоматически.
- **`docs.json` создаёт только локальный билдер**: если на сервере собрать только
  `make embeddings-qwen`, `GET /doc/{id}` будет отдавать 404 (с подсказкой) —
  для модалки полных текстов нужен прогон `make embeddings`.
- **Секреты**: `.env*`, `.deploy`, `certs/`, `favicon.ico` в `.gitignore`; в
  `.rsyncignore` их НЕТ (должны уехать на сервер). EPUB-издание 1978 г. под
  авторским правом — тоже не в git (`data/` игнорируется).
- **Орфография TEI**: в файлах параллельно `<orig>` (дореформенная) и `<reg>`
  (современная) — `tei_to_txt.py` берёт `<reg>`, нормализация в corpus.py
  осталась как страховка.
- **Не запускать legacy `scripts/epub_to_txt.py`**: он снова создаст
  `data/tom46–90.txt` (дубли писем/дневников) — в этом случае удалить их
  повторно или перенести в `data/unused_tom/`.
- **curl в Docker-образе нет** — healthcheck и внутренние проверки через Python.

## Open-source

MIT LICENSE (tdhster). Данные (EPUB), индекс, сертификаты, favicon, секреты — вне
репозитория. `data/README.md` объясняет пользователю, куда класть свои EPUB.
