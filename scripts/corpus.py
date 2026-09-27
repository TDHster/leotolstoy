#!/usr/bin/env python3
"""
Чтение корпуса Verbatis (data/verbatis/data/raw/tolstoy/*.jsonl) и нарезка на чанки.

Единственный источник текстов — репозиторий Verbatis (ПСС-90, «специально
обработанные для нас» тексты): 5 корпусов в jsonl. Читаем только основные
*.jsonl (без *_for_embedding.jsonl). Вся метадата записи сохраняется в чанк
как есть + составные ключи для UI (type, vol, to, num, year, entry).

Legacy: scripts/epub_to_txt.py и scripts/tei_to_txt.py больше не участвуют
в пайплайне (остаются в репо как справочные).
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass, field

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))


def _verbatis_dir() -> str:
    """Каталог jsonl-корпусов. Вычисляется на каждый вызов: тесты
    подменяют DATA_DIR через monkeypatch, и путь должен следовать за ним."""
    return os.path.join(DATA_DIR, "verbatis", "data", "raw", "tolstoy")

# Целевой размер чанка в символах. MiniLM-L12-v2 эмбеддит только первые ~128 токенов
# любого текста (~3.3 симв/токен на кириллице, измерено токенизатором fastembed —
# т.е. окно ≈ 420-460 символов), поэтому длинные абзацы режем по предложениям
# так, чтобы вектор покрывал весь текст чанка, а не только его «голову».
CHUNK_CHARS = 450
MIN_CHARS = 40  # короче — склеиваем с соседним абзацем (иначе выпал бы при эмбеддинге)
SIG_MAX_CHARS = 80  # подписи/даты длиннее не трогаем (консервативно)

# Справочный аппарат изданий (указатели имён, оглавления) не несёт смысловой
# нагрузки для RAG — такие записи пропускаем при сборке индекса.
JUNK_TITLE_RE = re.compile(r"^(УКАЗАТЕЛЬ|СОДЕРЖАНИЕ|ОГЛАВЛЕНИЕ)\b", re.IGNORECASE)

# Маппинг файлов корпуса Verbatis → тип документа в нашем индексе.
TYPE_BY_STEM = {
    "tolstoy_letters":     "письмо",
    "tolstoy_diaries":     "дневник",
    "tolstoy_notes_dated": "записная книжка",
    "tolstoy_articles":    "статья",
    "tolstoy_treatises":   "трактат",
}


def _is_junk_record(meta: dict) -> bool:
    """True, если запись — справочный мусор (указатель/оглавление), а не текст."""
    title = (meta.get("entry") or meta.get("title") or meta.get("to") or "").strip()
    return bool(JUNK_TITLE_RE.match(title))


@dataclass
class Chunk:
    text: str                       # текст для эмбеддинга и показа
    meta: dict = field(default_factory=dict)  # метадата записи Verbatis + doc/chunk


def _year_of(rec: dict) -> str | None:
    """Год из date_start/date_end (fallback date); диапазон разных годов → «1900–1910»."""
    m1 = re.match(r"^\d{4}", str(rec.get("date_start") or ""))
    m2 = re.match(r"^\d{4}", str(rec.get("date_end") or ""))
    if m1 and m2 and m1.group(0) != m2.group(0):
        return f"{m1.group(0)}–{m2.group(0)}"
    for key in ("date_start", "date_end", "date"):
        m = re.match(r"^\d{4}", str(rec.get(key) or ""))
        if m:
            return m.group(0)
    return None


def _verbatis_meta(rec: dict, rec_type: str) -> dict:
    """Метаданные записи: ВСЕ поля Verbatis как есть + составные ключи для UI
    (type, vol, to, num, year, entry). «Максимум метаданных» — ничего не теряем.
    Поле text исключается: текст живёт в абзацах, дублировать его в каждый чанк
    нельзя (раздул бы chunks.json на десятки МБ)."""
    meta = dict(rec)  # id, date, date_start, date_end, age, title, volume, pages,
                      # file, in_ss22, sent + у писем addressee/place/number
    meta.pop("text", None)
    meta["type"] = rec_type
    if rec.get("volume") is not None:
        meta["vol"] = str(rec["volume"])
    if rec.get("addressee") is not None:
        meta["to"] = str(rec["addressee"])
    if rec.get("number") is not None:
        meta["num"] = str(rec["number"])
    year = _year_of(rec)
    if year:
        meta["year"] = year
    title = str(rec.get("title") or "").strip()
    if rec_type == "письмо":
        # entry как раньше собирал tei_to_txt: «N. Адресат, год, место, с. страницы»
        if rec.get("number") is not None and rec.get("addressee"):
            parts = [str(rec["addressee"])]
            if year:
                parts.append(year)
            if rec.get("place"):
                parts.append(str(rec["place"]))
            if rec.get("pages") is not None:
                parts.append(f"с. {rec['pages']}")
            entry = f"{rec['number']}. {', '.join(parts)}"
        else:
            entry = title or None
    elif rec_type == "дневник":
        entry = title if title and title.rstrip(".,;: ") != "Дневник" else None
    else:  # записная книжка / статья / трактат
        entry = title or None
    if entry:
        meta["entry"] = entry
    return meta


def _iter_jsonl_records(path: str, rec_type: str):
    """(meta, [абзацы]) по строкам jsonl. Битые строки и пустые тексты пропускаются."""
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                print(f"[corpus] битая строка jsonl {path}:{lineno} — пропущена",
                      file=sys.stderr)
                continue
            text = str(rec.get("text") or "").strip()
            if not text:
                continue
            paras = [p.strip() for p in text.split("\n\n") if p.strip()]
            if paras:
                yield _verbatis_meta(rec, rec_type), paras


# --- фильтр подписей/дат в конце писем --------------------------------------
# Подписи («Л. Толстой.», «Граф Лев Толстой.») и даты («31 окт. 1909.») — это
# короткие хвостовые абзацы письма. В выдаче они не нужны (шум для поиска и для
# LLM), поэтому вырезаем их на уровне чанков — сами данные не трогаем.
# Фильтр консервативный: только хвост записи, только короткие строки,
# только подходящие под явные паттерны подписи/даты.

# Фамилия в любой падежной/дореформенной форме + пунктуация в конце строки
_NAME_END_RE = re.compile(
    r"(Толст(ой|аго|ого|ому|ым|ая|ых)?|Tolst(oy|oï))\s*[.,:;»!?…]*\s*$", re.IGNORECASE)
# «Л. Толстой. 1844 Мая 30.» — фамилия + дата-хвост в одной строке
_NAME_WITH_DATE_RE = re.compile(r"(толст|tolst)", re.IGNORECASE)
_DATE_TAIL_RE = re.compile(
    r"\d{4}\s+(Января|Февраля|Марта|Апреля|Мая|Июня|Июля|Августа|"
    r"Сентября|Октября|Ноября|Декабря)\s+\d{1,2}\.?\s*$", re.IGNORECASE)
# Подписные формулы без фамилии: «Ваш», «Твой Л.», «Любящий вас», «Брат ваш»…
_OPENING_RE = re.compile(
    r"^(Ваш|Твой|Любящ\w*|Уважающ\w*|Брат\s+(ваш|твой)"
    r"|С\s+(совершенным|искренним|истинным)\s+(уважением|почтением)"
    r"|Искренне\s+(уважающ\w*|ваш)"
    r"|Остаюсь\s+(ваш|с\s+))\b",
    re.IGNORECASE)
_INITIALS_RE = re.compile(r"^Л\.\s*[ТT]\.?$", re.IGNORECASE)
_DATE_RE = re.compile(
    r"^\d{4}\s*(г\.?)?\s*\.?\s*$"                                   # 1907. / 1907 г.
    r"|^\d{4}\s*(г\.?)?\s*[.,]?\s*\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|"
    r"июля|августа|сентября|октября|ноября|декабря)\s*\.?\s*$"       # 1903. 20 марта.
    r"|^\d{4}\s*(г\.?)?\s*[.,]?\s*(Января|Февраля|Марта|Апреля|Мая|Июня|Июля|Августа|"
    r"Сентября|Октября|Ноября|Декабря)\s+\d{1,2}\.?\s*$"             # 1894 г. Сентября 24.
    r"|^\d{1,2}(-го)?\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|"
    r"октября|ноября|декабря|янв\.|февр?\.|мар\.|апр\.|мая|июн\.|июл\.|авг\.|сент\.|"
    r"окт\.|нояб\.|дек\.)\s*\.?\s*\d{4}\s*(г(ода)?\.?)?\.?\s*$"      # 1 апреля 1903. / 31 окт. 1909.
    r"|^(Января|Февраля|Марта|Апреля|Мая|Июня|Июля|Августа|Сентября|Октября|"
    r"Ноября|Декабря)\s+\d{1,2}(-го)?\s+дня\s+\d{4}\s+года\.?$"      # Апреля 12-го дня 1847 года.
    r"|^\d{4}\s+(Января|Февраля|Марта|Апреля|Мая|Июня|Июля|Августа|Сентября|"
    r"Октября|Ноября|Декабря)\s+\d{1,2}\.?\s*$",                     # 1844 Мая 30.
    re.IGNORECASE)
# «На конверте: …» — почтовые пометки, не подпись, не трогаем
_ENVELOPE_RE = re.compile(r"^(На конверте|На обороте|На письме|В конверте)\b", re.IGNORECASE)


def _name_match(para: str) -> tuple[int, int] | None:
    """Позиция подписи-фамилии в конце строки, или None."""
    m = _NAME_END_RE.search(para)
    if m:
        return m.start(), m.end()
    # «Л. Толстой. 1844 Мая 30.» — фамилия + дата-хвост
    m = _NAME_WITH_DATE_RE.search(para)
    if m and _DATE_TAIL_RE.search(para):
        return m.start(), m.end()
    return None


def _pre_name_ok(pre: str) -> bool:
    """Часть строки ДО фамилии — только титулы/инициалы («Граф Лев », «Ваш
    Гр. Л. », «Ея Сиятельству Графине », «Léon »). Иначе это текст с подписью
    в той же строке (телеграмма «Ждешь меня. Что больной. Толстой.») — не трогаем."""
    if not pre:
        return True
    words = pre.split()
    if not words:
        return True
    word_re = re.compile(r"^[А-ЯЁA-Z][а-яёa-zéè]*\.?$")
    return all(word_re.fullmatch(w) for w in words)


def _is_signature_or_date(para: str) -> bool:
    """True, если короткая строка похожа на подпись или дату (хвост письма)."""
    if len(para) > SIG_MAX_CHARS or _ENVELOPE_RE.match(para):
        return False
    if _OPENING_RE.match(para) or _INITIALS_RE.match(para) or _DATE_RE.match(para):
        return True
    m = _name_match(para)
    if m is None:
        return False
    pre = para[:m[0]].strip()
    return _pre_name_ok(pre)


def _filter_letter_signature(meta: dict, paras: list[str]) -> list[str]:
    """Для писем: убираем ХВОСТОВЫЕ подписи/даты. Останавливаемся на первом
    не-подходящем абзаце. Может вернуть пустой список (письмо из одной подписи)."""
    if meta.get("type") != "письмо":
        return paras
    out = list(paras)
    while out and _is_signature_or_date(out[-1]):
        out.pop()
    return out


def _chunk_paragraphs(paras: list[str]) -> list[str]:
    """1 абзац = 1 чанк. Абзацы короче MIN_CHARS приклеиваются к следующему
    (хвост из коротких — к предыдущему), чтобы ничего не выпало при эмбеддинге.
    Абзацы длиннее CHUNK_CHARS*1.5 режутся по предложениям БЕЗ перекрытия."""
    chunks: list[str] = []
    short_buf: list[str] = []
    for para in paras:
        if len(para) < MIN_CHARS:
            short_buf.append(para)          # копим короткие — серия коротких
            continue                        # абзацев вся уйдёт в следующий
        if short_buf:                       # короткие + следующий абзац → один чанк
            para = "\n\n".join(short_buf + [para])
            short_buf = []
        if len(para) > CHUNK_CHARS * 1.5:
            chunks.extend(_split_long(para))
        else:
            chunks.append(para)
    if short_buf:                           # хвост из коротких
        tail = "\n\n".join(short_buf)
        if chunks and len(chunks[-1]) + len(tail) + 2 <= CHUNK_CHARS * 1.5:
            chunks[-1] = chunks[-1] + "\n\n" + tail
        elif len(tail) >= MIN_CHARS:
            chunks.append(tail)             # самодостаточный хвост — отдельным чанком
        elif chunks:
            # короткий хвост к последнему чанку: лучше лёгкое превышение окна,
            # чем потеря текста при эмбеддинге
            chunks[-1] = chunks[-1] + "\n\n" + tail
        else:
            chunks.append(tail)             # вырожденный случай: вся запись < MIN_CHARS
    return chunks


def _split_long(para: str) -> list[str]:
    """Делит длинный абзац по предложениям на куски ~CHUNK_CHARS без перекрытий."""
    sents = re.split(r"(?<=[.!?…])\s+", para)
    out, buf, blen = [], [], 0
    for s in sents:
        if blen + len(s) > CHUNK_CHARS and buf:
            out.append(" ".join(buf))
            buf, blen = [], 0
        # гигантское предложение длиннее чанка — идёт отдельным куском как есть
        buf.append(s)
        blen += len(s)
    if buf:
        out.append(" ".join(buf))
    return out


def _iter_indexed_records(include_types: set[str] | None = None):
    """Единый порядок перечисления записей Verbatis: файлы sorted(glob), записи
    по порядку строк, одинаковые фильтры include_types/junk. doc_id — сквозной
    счётчик записей. Его используют И load_chunks, И load_docs — нумерация
    совпадает по построению."""
    doc_id = 0
    paths = sorted(
        p for p in glob.glob(os.path.join(_verbatis_dir(), "*.jsonl"))
        if not os.path.basename(p).endswith("_for_embedding.jsonl")
    )
    if not paths:
        print("[corpus] Verbatis не найден — сначала: make verbatis-fetch",
              file=sys.stderr)
    for path in paths:
        stem = os.path.basename(path)[: -len(".jsonl")]
        rec_type = TYPE_BY_STEM.get(stem)
        if rec_type is None:
            print(f"[corpus] пропущен неизвестный файл: {path}", file=sys.stderr)
            continue
        for meta, paras in _iter_jsonl_records(path, rec_type):
            if include_types and meta.get("type") not in include_types:
                continue
            if _is_junk_record(meta):
                continue
            yield doc_id, meta, paras
            doc_id += 1


def load_chunks(include_types: set[str] | None = None) -> list[Chunk]:
    """
    Читает jsonl корпуса Verbatis (data/verbatis/…), режет на чанки (1 абзац = 1 чанк).
    include_types — например {"письмо"} чтобы взять только письма. None = всё.
    """
    chunks: list[Chunk] = []
    for doc_id, meta, paras in _iter_indexed_records(include_types):
        if meta.get("type") == "письмо":
            # подписи/даты в конце письма не идут в чанки (но остаются в load_docs)
            paras = _filter_letter_signature(meta, paras)
        for i, ctext in enumerate(_chunk_paragraphs(paras)):
            m = dict(meta)
            m["chunk"] = str(i)
            m["doc"] = str(doc_id)  # строка: "0" — truthy в JS на фронте
            chunks.append(Chunk(text=ctext, meta=m))
    return chunks


def load_docs(include_types: set[str] | None = None) -> list[dict]:
    """Полные тексты записей (БЕЗ фильтра подписей) для эндпоинта /doc.
    doc_id совпадает с meta["doc"] в чанках из load_chunks — общий итератор."""
    docs = []
    for doc_id, meta, paras in _iter_indexed_records(include_types):
        docs.append({
            "doc": str(doc_id),
            "label": source_label(meta),
            "text": "\n\n".join(paras),
            "meta": meta,
        })
    return docs


def source_label(meta: dict) -> str:
    """Человекочитаемая ссылка на источник для показа в ответе."""
    t = meta.get("type", "")
    vol = meta.get("vol", "")
    year = meta.get("year", "")
    entry = meta.get("entry", "")  # полное название из TOC / составной заголовок

    if t == "письмо":
        # Если есть полное название из TOC, используем его
        if entry:
            return f"Письмо: {entry}, том {vol}"
        # Иначе формируем из полей
        who = meta.get("to", "")
        num = meta.get("num", "")
        parts = []
        if num:
            parts.append(f"№{num}")
        if who:
            parts.append(f"к {who}")
        if year:
            parts.append(year)
        return "Письмо " + ", ".join(parts) if parts else f"Том {vol}"
    elif t == "дневник":
        if entry and entry != year:  # если есть название раздела, не только год
            return f"Дневник: {entry}, {year}".strip().rstrip(",")
        return f"Дневник, {year}".strip().rstrip(",")
    elif t in ("статья", "трактат", "записная книжка"):
        base = {"статья": "Статья", "трактат": "Трактат",
                "записная книжка": "Записная книжка"}[t]
        title = meta.get("title") or entry
        parts = [p for p in (title, year) if p]
        return f"{base}: {', '.join(parts)}" if parts else base
    else:
        title = meta.get("title") or ""
        return title or (f"Том {vol}" if vol else "")


if __name__ == "__main__":
    cs = load_chunks()
    print(f"Всего чанков: {len(cs)}")
    from collections import Counter
    by_type = Counter(c.meta.get("type") for c in cs)
    print("По типу:", dict(by_type))
    lens = [len(c.text) for c in cs]
    print(f"Длина чанка: min={min(lens)} avg={sum(lens)//len(lens)} max={max(lens)}")
    docs = load_docs()
    print(f"Документов (для /doc): {len(docs)}")
    print(f"Чанков без meta['doc']: {sum(1 for c in cs if 'doc' not in c.meta)}")
    print("\nПример:")
    print(source_label(cs[0].meta))
    print(cs[0].text[:300])
