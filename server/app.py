#!/usr/bin/env python3
"""
Чат «в стиле Льва Толстого» — RAG поверх его писем и дневников.

 - эмбеддинг запроса (e5/MiniLM, ONNX, локально)
 - косинусный поиск top-k по index/ (numpy)
 - генерация через любой OpenAI-совместимый API (эндпоинт задаётся в .env)
 - Basic Auth на всех роутах, один пользователь
 - rate limiting (slowapi) против абьюза и брутфорса
"""
# ВНИМАНИЕ: без `from __future__ import annotations` намеренно — slowapi
# оборачивает роуты, и при строковых аннотациях FastAPI не распознаёт тело
# запроса (принимает его за query-параметр). Держим реальные аннотации.
import asyncio
import glob
import json
import logging
import os
import secrets
import sys
import time
from contextlib import asynccontextmanager

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))  # server/ (retriever)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))  # embed, corpus
from retriever import Retriever  # noqa: E402

# --- логирование -------------------------------------------------------------
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("tolstoy")

# --- конфиг из окружения -----------------------------------------------------
APP_USER = os.environ.get("APP_USER", "admin")
APP_PASS = os.environ.get("APP_PASS", "")
# Провайдер генерации — любой OpenAI-совместимый эндпоинт.
# Новые имена LLM_*, со старыми OPENROUTER_* как fallback (обратная совместимость).
LLM_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
LLM_MODEL = (os.environ.get("LLM_MODEL")
             or os.environ.get("OPENROUTER_MODEL", "deepseek/deepseek-chat"))
# База API без хвоста /chat/completions — его добавим сами.
LLM_BASE_URL = (os.environ.get("LLM_BASE_URL")
                or "https://openrouter.ai/api/v1").rstrip("/")
LLM_URL = LLM_BASE_URL + "/chat/completions"
LLM_TIMEOUT = float(os.environ.get("LLM_TIMEOUT", "120"))
TOP_K = int(os.environ.get("TOP_K", "5"))
# ограничение типов источников: "письмо" / "дневник" / "письмо,дневник"
SRC_TYPES = os.environ.get("SRC_TYPES", "").strip()
SRC_TYPES_SET = {t.strip() for t in SRC_TYPES.split(",") if t.strip()} or None
# лимиты
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "2000"))
RATE_LIMIT_CHAT = os.environ.get("RATE_LIMIT_CHAT", "20/minute;200/day")
RATE_LIMIT_LOGIN = os.environ.get("RATE_LIMIT_LOGIN", "30/minute")
RATE_LIMIT_DOC = os.environ.get("RATE_LIMIT_DOC", "30/minute")
AUTH_FAIL_DELAY = float(os.environ.get("AUTH_FAIL_DELAY", "0.5"))
# Логировать текст запросов/ответов. По умолчанию ВЫКЛ — это персональные данные
# пользователей (их вопросы). Включайте осознанно на своём сервере: LOG_QUERIES=1.
LOG_QUERIES = os.environ.get("LOG_QUERIES", "").strip().lower() in ("1", "true", "yes")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

SYSTEM_PROMPT = """\
Ты отвечаешь от лица Льва Николаевича Толстого. Твоя речь — это речь Толстого \
из его писем и дневников: живая, искренняя, с длинными обдуманными периодами, \
с нравственным вниманием к человеку, без нарочитой архаики и без карикатуры.

Правила:
1. Пиши на современном русском языке, но в интонации и складе речи Толстого: \
рассудительно, прямо, с внутренней честностью, порой с самокритикой.
2. Опирайся на приведённые ниже подлинные фрагменты писем и дневников Толстого \
как на образец мысли и голоса. Можешь развивать их темы, но не выдумывай \
биографических фактов, которых там нет.
3. Не притворяйся, что помнишь конкретные события, если их нет во фрагментах. \
Лучше рассуждай о предмете так, как рассуждал бы Толстой.
4. Не начинай ответ со слов «Как Лев Толстой» — просто говори от первого лица.
5. Отвечай по существу вопроса собеседника, не уходя в пустое морализаторство."""


# --- lifespan (загрузка индекса и прогрев эмбеддера при старте) --------------
_retrievers: dict[str, Retriever] = {}  # кеш: model_id -> Retriever
MAX_LOADED_INDICES = 1  # максимум индексов в памяти одновременно (для экономии RAM)


def _get_index_dir(model_id: str) -> str:
    """Маппинг model_id → каталог индекса."""
    root = os.path.join(os.path.dirname(__file__), "..")

    # INDEX_DIR в окружении переопределяет ТОЛЬКО локальный индекс —
    # иначе оба переключателя в UI вели бы в один и тот же каталог.
    if model_id == "local":
        index_dir_env = os.environ.get("INDEX_DIR")
        if index_dir_env:
            return os.path.join(root, index_dir_env)
        return os.path.join(root, "index")

    # Для внешних моделей: index.<model_id>/
    return os.path.join(root, f"index.{model_id}")


def _get_retriever(model_id: str) -> Retriever:
    """Получить retriever для модели (с кешем в памяти и автоматической выгрузкой)."""
    if model_id not in _retrievers:
        index_dir = _get_index_dir(model_id)
        if not os.path.exists(index_dir):
            raise HTTPException(404, f"Индекс для модели '{model_id}' не найден. Доступные модели: GET /models")

        # Если достигли лимита загруженных индексов → выгружаем старый
        if len(_retrievers) >= MAX_LOADED_INDICES:
            old_id = list(_retrievers.keys())[0]
            log.info("Выгружаем индекс '%s' для освобождения памяти (limit: %d)", old_id, MAX_LOADED_INDICES)
            del _retrievers[old_id]
            # Принудительная сборка мусора для немедленного освобождения памяти
            import gc
            gc.collect()

        # Загружаем новый индекс. Фоллбек: если индекс повреждён/неконсистентен
        # (например, пересобирается прямо сейчас) — не падаем, а отдаём 503
        # с понятным объяснением; остальные модели продолжают работать.
        log.info("Загружаем индекс '%s' из %s", model_id, index_dir)
        try:
            _retrievers[model_id] = Retriever(index_dir=index_dir)
        except Exception as e:  # noqa: BLE001
            log.error("Не удалось загрузить индекс '%s': %s", model_id, e)
            raise HTTPException(
                503,
                f"Индекс модели '{model_id}' не удалось загрузить: {e}. "
                f"Возможно, он пересобирается — выберите другую модель.",
            ) from e

    return _retrievers[model_id]


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not APP_PASS:
        log.warning("APP_PASS не задан — авторизация будет отклонять всех. Задайте APP_PASS в .env")
    if not LLM_KEY:
        log.warning("LLM_API_KEY (или OPENROUTER_API_KEY) не задан — генерация работать не будет.")

    # Загрузка локального индекса при старте (для обратной совместимости).
    # Фоллбек: если индекс отсутствует или повреждён (например, пересобирается),
    # сервер стартует без него — остальные модели остаются доступны.
    try:
        _retrievers["local"] = Retriever(index_dir=_get_index_dir("local"))
        log.info("Индекс 'local' загружен: %d фрагментов", len(_retrievers["local"]))
    except Exception as e:  # noqa: BLE001
        log.warning("Локальный индекс не загружен (сервер работает без него): %s", e)

    # Прогрев локальной модели (если она есть)
    if "local" in _retrievers:
        try:
            embedder = _retrievers["local"].get_embedder()
            await run_in_threadpool(embedder.embed_query, "прогрев")
            log.info("Локальная модель прогрета")
        except Exception as e:  # noqa: BLE001
            log.warning("Не удалось прогреть локальную модель: %s", e)

    log.info("Сервер готов. Модель генерации: %s @ %s", LLM_MODEL, LLM_BASE_URL)
    yield


# --- приложение --------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Толстой-чат", lifespan=lifespan)
app.state.limiter = limiter
security = HTTPBasic()


@app.exception_handler(RateLimitExceeded)
async def _rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return Response(
        content="Слишком много запросов. Подождите немного и попробуйте снова.",
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        media_type="text/plain; charset=utf-8",
    )


async def auth(cred: HTTPBasicCredentials = Depends(security)) -> str:
    ok_user = secrets.compare_digest(cred.username, APP_USER)
    ok_pass = bool(APP_PASS) and secrets.compare_digest(cred.password, APP_PASS)
    if not (ok_user and ok_pass):
        # задержка удорожает перебор пароля
        if AUTH_FAIL_DELAY > 0:
            await asyncio.sleep(AUTH_FAIL_DELAY)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный логин или пароль",
            headers={"WWW-Authenticate": "Basic"},
        )
    return cred.username


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    model: str = Field(default="local")  # "local" или id другой модели (например "qwen3-embedding-8b")


class ChatOut(BaseModel):
    reply: str
    sources: list[dict]


def build_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[Фрагмент {i}. {r['source']}]\n{r['text']}")
    return "\n\n".join(blocks)


def _display_name(model_id: str, meta: dict) -> str:
    """Имя модели для UI.

    Локальная модель берётся из конфига (EMBED_MODEL в .env) — честное имя
    модели без префикса sentence-transformers/, а не захардкоженная подпись
    из metadata. Для внешних моделей — display_name из metadata индекса.
    """
    if model_id == "local":
        model = os.environ.get("EMBED_MODEL") or meta.get("model") or ""
        if model:
            return model.split("/")[-1]
        return "Локальная модель"
    return meta.get("display_name", model_id)


@app.get("/models")
async def list_models(_user: str = Depends(auth)):
    """Список доступных моделей эмбеддингов с их характеристиками."""
    available = []

    # Сканируем директорию проекта на наличие индексов
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    for entry in glob.glob(os.path.join(root_dir, "index*")):
        if not os.path.isdir(entry):
            continue

        metadata_path = os.path.join(entry, "metadata.json")
        if not os.path.exists(metadata_path):
            continue

        try:
            with open(metadata_path, encoding="utf-8") as f:
                meta = json.load(f)

            # Определяем model_id из имени директории
            dir_name = os.path.basename(entry)
            if dir_name == "index":
                model_id = "local"
            else:
                # index.qwen3-embedding-8b -> qwen3-embedding-8b
                model_id = dir_name.replace("index.", "")

            # Формируем описание для UI
            display_name = _display_name(model_id, meta)
            dims = meta.get("dimensions", 0)
            is_api = meta.get("api", False)
            latency_ms = meta.get("latency_ms", 50 if not is_api else 350)
            cost_per_1k = meta.get("cost_per_1k", 0)

            # Форматируем стоимость
            if cost_per_1k == 0:
                cost_str = "бесплатно"
            else:
                cost_str = f"${cost_per_1k * 1000:.2f} за 1000 запросов"

            available.append({
                "id": model_id,
                "name": display_name,
                "dimensions": dims,
                "latency": f"~{latency_ms}ms",
                "cost": cost_str,
                "is_api": is_api,
            })
        except Exception as e:  # noqa: BLE001
            log.warning("Не удалось прочитать metadata для %s: %s", entry, e)
            continue

    # Сортируем: сначала local, потом остальные по имени
    available.sort(key=lambda m: (m["id"] != "local", m["id"]))

    return {"models": available}


@app.get("/config")
async def get_config(_user: str = Depends(auth)):
    """Конфигурация сервера (LLM модель и т.д.)."""
    return {
        "llm_model": LLM_MODEL,
        "llm_provider": LLM_BASE_URL,
    }


@app.post("/chat", response_model=ChatOut)
@limiter.limit(RATE_LIMIT_CHAT)
async def chat(request: Request, body: ChatIn, _user: str = Depends(auth)):
    msg = body.message.strip()
    if not msg:
        raise HTTPException(400, "Пустое сообщение")
    if not LLM_KEY:
        raise HTTPException(503, "LLM_API_KEY не настроен на сервере")

    t0 = time.monotonic()
    if LOG_QUERIES:
        # логируем запрос сразу — попадёт в лог даже если LLM потом упадёт
        client_ip = request.client.host if request.client else "?"
        log.info("запрос от %s (%s), модель '%s': %r", _user, client_ip, body.model, msg)

    # Получить retriever для выбранной модели
    try:
        retriever = _get_retriever(body.model)
    except HTTPException:
        raise
    except Exception as e:
        log.error("Не удалось загрузить индекс для модели '%s': %s", body.model, e)
        raise HTTPException(500, f"Ошибка загрузки индекса: {e}") from e

    # Получить эмбеддер для этого индекса (API-модели требуют ключ в .env)
    try:
        embedder = retriever.get_embedder()
    except RuntimeError as e:
        log.error("Не удалось создать эмбеддер для модели '%s': %s", body.model, e)
        raise HTTPException(503, str(e)) from e

    # 1. поиск релевантных фрагментов (эмбеддинг CPU-bound → в threadpool,
    #    чтобы не блокировать event loop)
    t_embed = time.monotonic()
    try:
        qvec = await run_in_threadpool(embedder.embed_query, msg)
        # защита от рассинхрона размерностей (напр. провайдер сменил модель)
        vectors = getattr(retriever, "vectors", None)
        if vectors is not None and (qvec.ndim != 1 or qvec.shape[0] != vectors.shape[1]):
            raise HTTPException(
                503,
                f"Модель '{body.model}' вернула вектор размерности {qvec.shape}, "
                f"а индекс требует {vectors.shape[1]}. Выберите другую модель.",
            )
    except RuntimeError as e:
        error_msg = str(e)
        # Распознаём типичные ошибки API и даём понятное объяснение
        if "403" in error_msg or "Access denied" in error_msg:
            log.error("Embedding API заблокирован (403): %s", error_msg)
            raise HTTPException(
                503,
                f"Модель '{body.model}' временно недоступна: API-провайдер заблокировал запросы с этого сервера. "
                f"Попробуйте локальную модель или свяжитесь с администратором."
            ) from e
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            log.error("Embedding API rate limit: %s", error_msg)
            raise HTTPException(429, f"Превышен лимит запросов для модели '{body.model}'. Попробуйте позже.") from e
        else:
            log.error("Ошибка эмбеддинга для модели '%s': %s", body.model, error_msg)
            raise HTTPException(503, f"Не удалось обработать запрос с моделью '{body.model}': {error_msg[:200]}") from e
    embed_ms = int((time.monotonic() - t_embed) * 1000)

    results = retriever.search(qvec, top_k=TOP_K, types=SRC_TYPES_SET)
    context = build_context(results)
    top_score = results[0]["score"] if results else 0.0

    # 2. промпт для LLM
    user_content = (
        f"Вопрос собеседника: {msg}\n\n"
        f"Подлинные фрагменты писем и дневников Толстого для опоры:\n\n{context}"
    )
    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "temperature": 0.8,
    }
    headers = {
        "Authorization": f"Bearer {LLM_KEY}",
        "Content-Type": "application/json",
        # безвредно для не-OpenRouter провайдеров (лишние заголовки игнорируются)
        "X-Title": "Tolstoy Chat",
    }

    # 3. вызов LLM (любой OpenAI-совместимый эндпоинт)
    t_llm = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=LLM_TIMEOUT) as client:
            resp = await client.post(LLM_URL, json=payload, headers=headers)
    except httpx.RequestError as e:
        log.error("Сбой связи с LLM (%s): %s", LLM_URL, e)
        raise HTTPException(502, f"Не удалось связаться с LLM-провайдером: {e}") from e
    llm_ms = int((time.monotonic() - t_llm) * 1000)

    if resp.status_code != 200:
        log.error("LLM вернул %s: %s", resp.status_code, resp.text[:500])
        raise HTTPException(502, f"LLM-провайдер вернул {resp.status_code}: {resp.text[:300]}")

    data = resp.json()
    try:
        reply = data["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError) as e:
        log.error("Неожиданный ответ LLM: %s", str(data)[:500])
        raise HTTPException(502, f"Неожиданный ответ LLM-провайдера: {str(data)[:300]}") from e

    total_ms = int((time.monotonic() - t0) * 1000)
    log.info(
        "chat: model=%s len=%d top_score=%.3f embed=%dms llm=%dms total=%dms",
        body.model,
        len(msg),
        top_score,
        embed_ms,
        llm_ms,
        total_ms,
    )
    if LOG_QUERIES:
        log.info("ответ: %r", reply)

    sources = [
        {
            "source": r["source"],
            "score": round(r["score"], 3),
            "text": r["text"],  # оригинальный фрагмент, ушедший в модель
            "type": r["meta"].get("type", ""),
            "year": r["meta"].get("year", ""),
            "age": r["meta"].get("age", ""),  # возраст Толстого (полные годы на дату записи)
            "doc": r["meta"].get("doc", ""),  # id полного текста в index/docs.json ("" для старых индексов)
        }
        for r in results
    ]
    return ChatOut(reply=reply, sources=sources)


# --- полный текст источника (клик по фрагменту в UI) -------------------------
_docs_cache: list[dict] | None = None  # index/docs.json, лениво, один раз на процесс


def _load_docs() -> list[dict]:
    global _docs_cache
    if _docs_cache is None:
        path = os.path.join(_get_index_dir("local"), "docs.json")
        if not os.path.exists(path):
            raise HTTPException(404, "Полные тексты недоступны: index/docs.json не найден. "
                                     "Пересоберите индекс (make embeddings).")
        with open(path, encoding="utf-8") as f:
            _docs_cache = json.load(f)
    return _docs_cache


@app.get("/doc/{doc_id}")
@limiter.limit(RATE_LIMIT_DOC)
async def get_doc(request: Request, doc_id: int, _user: str = Depends(auth)):
    try:
        docs = await run_in_threadpool(_load_docs)  # json.load ~23 МБ — не блокируем event loop
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        log.error("Не удалось прочитать index/docs.json: %s", e)
        raise HTTPException(404, "Полные тексты недоступны (повреждён index/docs.json).") from e
    if doc_id < 0 or doc_id >= len(docs):
        raise HTTPException(404, f"Документ {doc_id} не найден")
    return docs[doc_id]


@app.get("/")
@limiter.limit(RATE_LIMIT_LOGIN)
async def index(request: Request, _user: str = Depends(auth)):
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/favicon.ico")
def favicon():
    # без авторизации: браузер запрашивает иконку автоматически (в т.ч. на
    # странице логина), под Basic Auth она бы не загрузилась.
    # Файл не в репозитории (лицензия картинки) — если его нет, отдаём 204,
    # чтобы не сыпать 404 в лог. Положите свой server/static/favicon.ico.
    path = os.path.join(STATIC_DIR, "favicon.ico")
    if os.path.exists(path):
        return FileResponse(path, media_type="image/x-icon")
    return Response(status_code=204)


@app.get("/health")
def health():
    # Список загруженных моделей
    models_info = {model_id: len(r) for model_id, r in _retrievers.items()}
    return {
        "status": "ok",
        "models": models_info,
        "llm_model": LLM_MODEL,
        "llm_endpoint": LLM_BASE_URL,
    }


# статика (css/js если понадобится) — тоже под авторизацией не делаем,
# т.к. это просто ассеты; сам index.html отдаётся через "/" с auth.
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
