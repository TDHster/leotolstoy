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
from embed import embed_query  # noqa: E402
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
_retriever: Retriever | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever
    if not APP_PASS:
        log.warning("APP_PASS не задан — авторизация будет отклонять всех. Задайте APP_PASS в .env")
    if not LLM_KEY:
        log.warning("LLM_API_KEY (или OPENROUTER_API_KEY) не задан — генерация работать не будет.")
    _retriever = Retriever()
    try:
        await run_in_threadpool(embed_query, "прогрев")  # прогрев модели
    except Exception as e:  # noqa: BLE001
        log.warning("Не удалось прогреть эмбеддер: %s", e)
    log.info("Индекс загружен: %d фрагментов. Модель: %s @ %s",
             len(_retriever), LLM_MODEL, LLM_BASE_URL)
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


class ChatOut(BaseModel):
    reply: str
    sources: list[dict]


def build_context(results: list[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        blocks.append(f"[Фрагмент {i}. {r['source']}]\n{r['text']}")
    return "\n\n".join(blocks)


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
        log.info("запрос от %s (%s): %r", _user, client_ip, msg)

    # 1. поиск релевантных фрагментов (эмбеддинг CPU-bound → в threadpool,
    #    чтобы не блокировать event loop)
    qvec = await run_in_threadpool(embed_query, msg)
    results = _retriever.search(qvec, top_k=TOP_K, types=SRC_TYPES_SET)
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
    log.info("chat: len=%d top_score=%.3f llm=%dms total=%dms",
             len(msg), top_score, llm_ms, total_ms)
    if LOG_QUERIES:
        log.info("ответ: %r", reply)

    sources = [
        {
            "source": r["source"],
            "score": round(r["score"], 3),
            "text": r["text"],           # оригинальный фрагмент, ушедший в модель
            "type": r["meta"].get("type", ""),
            "year": r["meta"].get("year", ""),
        }
        for r in results
    ]
    return ChatOut(reply=reply, sources=sources)


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
    n = len(_retriever) if _retriever else 0
    return {"status": "ok", "chunks": n, "model": LLM_MODEL, "endpoint": LLM_BASE_URL}


# статика (css/js если понадобится) — тоже под авторизацией не делаем,
# т.к. это просто ассеты; сам index.html отдаётся через "/" с auth.
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
