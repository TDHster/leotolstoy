"""Названия тем: c-TF-IDF (pymorphy3) + LLM из .env проекта → JSON-кэш.

Корректировка ТЗ: LLM — существующий конфиг проекта (LLM_API_KEY / LLM_BASE_URL /
LLM_MODEL из корневого ../.env, OpenAI-совместимый /chat/completions через
httpx), а не пакет anthropic с несуществующей моделью claude-sonnet-4-6.

Лемматизация: только ~2000 самых частых словоформ корпуса (полная лемматизация
всех ~2.5 млн слов заняла бы десятки минут, а в топ-10 c-TF-IDF попадают именно
частые слова; редкие слова — имена собственные, которым лемма и не нужна).
"""
from __future__ import annotations

import json
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import httpx
import numpy as np
from dotenv import dotenv_values
from pymorphy3 import MorphAnalyzer
from tqdm import tqdm

from . import load as load_mod

CTFIDF_CACHE = "ctfidf.json"
LABELS_GLOBAL_CACHE = "labels_global.json"
LABELS_EPOCH_CACHE = "labels_epoch.json"

# Русские стоп-слова: служебная лексика + сверхчастотные глаголы
STOPWORDS = frozenset(
    """а без более бы был была были было быть в вам вас весь во вот все всего всех вы где да даже
    для до его ее её ему если есть еще ещё же за здесь и из или им именно их к как какая какой когда
    кое конечно который кто куда ли либо лишь между меня мне много может можно мой моя мы на над надо
    наконец нас наш не него нее её ней нем нему нет ни нибудь ниже никак никогда ним ними них ничего
    но ну об один около он она они оно опять от очень перед по под после потом потому почти при про
    просто раз разве с сам свое своей свои свой себе себя сейчас сказал сказала сказать так также такой
    там те тебе тем теперь то тогда того тоже той только том тот тут ты у уж уже хотя чего чей чем
    через что чтобы чтоб чуть эта эти этим этих это этого этой этом этот эту я""".split()
)

_WORD_RE = re.compile(r"[а-яёa-z]{2,}")
_JSON_RE = re.compile(r"\{.*\}", re.S)

# Параметры LLM (ТЗ 5.4): температура, лимит токенов, параллелизм, ретраи
TEMPERATURE = 0.3
MAX_TOKENS = 300
MAX_WORKERS = 5
RETRIES = 3
FRAGMENTS = 8  # ближайших к центроиду фрагментов в промпте
FRAGMENT_CHARS = 400
TOP_TERMS = 10


# ---------------------------------------------------------------- c-TF-IDF

def _tokens(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall(text.lower()) if w not in STOPWORDS]


def run_ctfidf(data: dict, force: bool = False) -> dict:
    """Топ-10 лемм по c-TF-IDF для глобальных кластеров и узлов эпох."""
    cache_path = load_mod.CACHE_DIR / CTFIDF_CACHE
    if not force and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    chunks = data["chunks"]
    with open(load_mod.CACHE_DIR / "hdbscan.json", encoding="utf-8") as f:
        hdb = json.load(f)
    with open(load_mod.CACHE_DIR / "epochs.json", encoding="utf-8") as f:
        epochs = json.load(f)

    # Один проход по корпусу: токены каждого чанка + частоты форм
    preproc = []
    form_counts = Counter()
    for c in tqdm(chunks, desc="Токенизация корпуса"):
        toks = _tokens(c["text"])
        preproc.append(" ".join(toks))
        form_counts.update(toks)

    # Леммы только для частых форм
    morph = MorphAnalyzer()
    lemma_map: dict[str, str] = {}
    for form, _ in form_counts.most_common(2000):
        try:
            lemma = morph.parse(form)[0].normal_form
        except Exception:
            continue
        if lemma != form:
            lemma_map[form] = lemma

    def doc_tokens(member_indices) -> Counter:
        counts = Counter()
        for i in member_indices:
            for w in preproc[i].split():
                counts[lemma_map.get(w, w)] += 1
        return counts

    labels_arr = np.array(hdb["labels"])
    global_docs = {
        str(cid): doc_tokens(np.flatnonzero(labels_arr == cid))
        for cid in sorted(set(labels_arr))
        if cid != -1
    }
    epoch_docs = {
        node["id"]: doc_tokens(node["members"])
        for e in epochs
        for node in e["nodes"]
    }
    result = {
        "global": _ctfidf_top(global_docs),
        "epoch": _ctfidf_top(epoch_docs),
    }
    load_mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return result


def _ctfidf_top(docs: dict[str, Counter], top: int = TOP_TERMS) -> dict[str, list[str]]:
    """c-TF-IDF: tf_норм × log(1 + N/(1+df)); на выходе топ-термы на документ."""
    n = len(docs)
    df = Counter()
    for counts in docs.values():
        df.update(counts.keys())
    out = {}
    for doc_id, counts in docs.items():
        total = sum(counts.values()) or 1
        scored = {
            term: (count / total) * np.log(1 + n / (1 + df[term]))
            for term, count in counts.items()
        }
        out[doc_id] = [term for term, _ in Counter(scored).most_common(top)]
    return out


# ---------------------------------------------------------------- LLM

def _env_llm() -> tuple[str, str, str]:
    """Конфиг провайдера из корневого .env (как у сервера проекта)."""
    env = dotenv_values(load_mod.INDEX_DIR.parent / ".env")
    key = env.get("LLM_API_KEY") or env.get("OPENROUTER_API_KEY") or ""
    base = (env.get("LLM_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    model = env.get("LLM_MODEL") or env.get("OPENROUTER_MODEL") or "deepseek-chat"
    return key, base, model


def _parse_json(content: str | None) -> dict | None:
    """JSON из ответа LLM с очисткой от ```-обёрток и лишнего текста (ТЗ 5.4)."""
    if not content:
        return None
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    m = _JSON_RE.search(content)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    if isinstance(obj, dict) and obj.get("title"):
        return {
            "title": str(obj["title"]).strip()[:80],
            "description": str(obj.get("description", "")).strip()[:300],
        }
    return None


def _call_llm(
    client: httpx.Client,
    base: str,
    model: str,
    key: str,
    messages: list[dict],
    calls: list,
) -> dict | None:
    """Один запрос с ретраями и бэкоффом (ТЗ 5.4). Возвращает разобранный JSON или None."""
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    for attempt in range(RETRIES):
        calls.append(1)  # счётчик HTTP-запросов для отчёта
        try:
            r = client.post(
                f"{base}/chat/completions", json=payload, headers=headers, timeout=60.0
            )
            r.raise_for_status()
            parsed = _parse_json(r.json()["choices"][0]["message"]["content"])
            if parsed:
                return parsed
        except Exception:
            if attempt < RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


def _prompt(
    snippets: list[str],
    terms: list[str],
    year_range: str,
    issued_titles: list[str] | None,
) -> list[dict]:
    parts = [
        f"Фрагменты писем и дневников Л. Н. Толстого ({year_range} годы):",
        *[f"{i}. «{s}»" for i, s in enumerate(snippets, 1)],
        "",
        "Ключевые слова: " + ", ".join(terms) + ".",
    ]
    if issued_titles:
        parts.append("Уже названы темы: " + ", ".join(issued_titles) + ". Не повторяйся.")
    parts.append(
        'Дай название темы из 2–4 слов и описание в одно предложение. '
        'Ответь строго JSON без пояснений: {"title": "…", "description": "…"}'
    )
    return [
        {
            "role": "system",
            "content": (
                "Ты — литературовед, изучаешь письма и дневники Л. Н. Толстого. "
                "Даёшь темам короткие ёмкие названия."
            ),
        },
        {"role": "user", "content": "\n".join(parts)},
    ]


def _fallback(terms: list[str]) -> dict:
    """Фолбэк без падения пайплайна (ТЗ 5.4): топ-3 термина как название."""
    return {
        "title": " · ".join(terms[:3]),
        "description": "Тема выделена по ключевым словам (LLM недоступен).",
        "fallback": True,
    }


def _snippets(data: dict, centroid: list, members: list[int]) -> list[str]:
    """8 ближайших к центроиду фрагментов, обрезанных до 400 символов (ТЗ 5.4)."""
    vectors = data["vectors"]
    sims = vectors[members] @ np.array(centroid)  # косинус: векторы нормализованы
    top = np.asarray(members)[np.argsort(-sims)[:FRAGMENTS]]
    out = []
    for i in top:
        text = " ".join(data["chunks"][i]["text"].split())
        if len(text) > FRAGMENT_CHARS:
            text = text[:FRAGMENT_CHARS] + "…"
        out.append(text)
    return out


def _year_range(data: dict, members: list[int]) -> str:
    ys = data["years"][np.asarray(members, dtype=int)]
    ys = ys[ys >= 0]
    if len(ys) == 0:
        return "год неизвестен"
    return str(int(ys.min())) if ys.min() == ys.max() else f"{int(ys.min())}–{int(ys.max())}"


def run_labels(data: dict, ctfidf: dict, force: bool = False) -> tuple[dict, dict, dict]:
    """Названия тем от LLM. Кэш labels_global.json / labels_epoch.json."""
    cache_global = load_mod.CACHE_DIR / LABELS_GLOBAL_CACHE
    cache_epoch = load_mod.CACHE_DIR / LABELS_EPOCH_CACHE
    if not force and cache_global.exists() and cache_epoch.exists():
        with open(cache_global, encoding="utf-8") as f:
            labels_global = json.load(f)
        with open(cache_epoch, encoding="utf-8") as f:
            labels_epoch = json.load(f)
        return labels_global, labels_epoch, {"calls": 0, "fallbacks": 0, "skipped": "из кэша"}

    with open(load_mod.CACHE_DIR / "hdbscan.json", encoding="utf-8") as f:
        hdb = json.load(f)
    with open(load_mod.CACHE_DIR / "epochs.json", encoding="utf-8") as f:
        epochs = json.load(f)

    key, base, model = _env_llm()
    if not key:
        print("  LLM_API_KEY не задан в ../.env — все названия из фолбэка (топ-3 термина)")

    labels_arr = np.array(hdb["labels"])
    calls: list = []

    def label_task(
        doc_id: str,
        members: list[int],
        centroid: list,
        terms: list[str],
        issued_titles: list[str] | None,
    ) -> dict:
        messages = _prompt(
            _snippets(data, centroid, members), terms, _year_range(data, members),
            issued_titles,
        )
        if key:
            with httpx.Client() as client:
                parsed = _call_llm(client, base, model, key, messages, calls)
            if parsed:
                parsed["fallback"] = False
                return parsed
        return _fallback(terms)

    # Глобальные кластеры — последовательно, каждый видит уже выданные названия
    # (критерий ТЗ: названия тем различимы; параллельность тут не нужна — вызовы быстрые)
    tasks_global = [
        (cid, np.flatnonzero(labels_arr == int(cid)).tolist(), centroid,
         ctfidf["global"][cid], None)
        for cid, centroid in hdb["centroids"].items()
    ]
    labels_global = {}
    issued_global: list[str] = []
    for cid, members, centroid, terms, _ in tqdm(
        tasks_global, desc="Названия глобальных тем"
    ):
        label = label_task(cid, members, centroid, terms, issued_global)
        if label["title"] in issued_global:
            # страховка для фолбэка: тот же топ-терминов у соседних кластеров
            label["title"] = f"{label['title']} ({cid})"
        labels_global[cid] = label
        issued_global.append(label["title"])

    # Узлы эпох: внутри эпохи последовательно, чтобы передавать уже выданные
    # названия той же эпохи (ТЗ 5.4: «не повторяться»); эпохи — параллельно.
    labels_epoch = {}

    def label_epoch(epoch: dict) -> None:
        issued: list[str] = []
        for node in epoch["nodes"]:
            label = label_task(
                node["id"], node["members"], node["centroid"],
                ctfidf["epoch"][node["id"]], issued,
            )
            labels_epoch[node["id"]] = label
            issued.append(label["title"])

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        list(tqdm(
            pool.map(label_epoch, epochs),
            total=len(epochs),
            desc="Названия тем по эпохам",
        ))

    load_mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_global, "w", encoding="utf-8") as f:
        json.dump(labels_global, f, ensure_ascii=False, indent=1)
    with open(cache_epoch, "w", encoding="utf-8") as f:
        json.dump(labels_epoch, f, ensure_ascii=False, indent=1)

    stats = {
        "calls": len(calls),
        "fallbacks": sum(1 for l in labels_global.values() if l.get("fallback"))
        + sum(1 for l in labels_epoch.values() if l.get("fallback")),
    }
    return labels_global, labels_epoch, stats
