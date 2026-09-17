#!/usr/bin/env python3
"""
Чтение txt файлов (результат epub_to_txt.py) и нарезка на чанки.

Формат входа:
    ### META | vol=5 | title=Война и мир | type=проза
    <абзацы через пустую строку>

Метаданные (vol, title, type) копируются в каждый чанк.
"""
from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

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


def _is_junk_record(meta: dict) -> bool:
    """True, если запись — справочный мусор (указатель/оглавление), а не текст."""
    title = (meta.get("entry") or meta.get("title") or meta.get("to") or "").strip()
    return bool(JUNK_TITLE_RE.match(title))


@dataclass
class Chunk:
    text: str                       # текст для эмбеддинга и показа
    meta: dict = field(default_factory=dict)  # vol, title, type


def _parse_meta(line: str) -> dict:
    """Парсит строку ### META | vol=5 | title=... | type=..."""
    line = line[len("### META"):].strip()
    if line.startswith("|"):
        line = line[1:]
    meta = {}
    for part in line.split("|"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        k, v = part.split("=", 1)
        meta[k.strip()] = v.strip()
    return meta


def _normalize_old_russian(text: str) -> str:
    """Преобразует дореформенные буквы в современные."""
    # Старорусские буквы → современные
    replacements = {
        'ѣ': 'е',  # ять
        'Ѣ': 'Е',
        'і': 'и',  # i десятеричное
        'І': 'И',
        'ѳ': 'ф',  # фита
        'Ѳ': 'Ф',
        'ѵ': 'и',  # ижица
        'Ѵ': 'И',
        'ѐ': 'е',  # ё с грависом
        'Ѐ': 'Е',
        'ъ': '',   # твёрдый знак в конце слов (граница слова)
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    # Убираем конечный ъ (граница слова): слово + ъ + пробел/конец → слово + пробел/конец
    text = re.sub(r'ъ\b', '', text)
    return text


def _strip_editorial_notes(text: str) -> str:
    """
    Удаляет редакторские комментарии из начала и конца текста.

    В томах полного собрания сочинений могут быть:
    - Издательская информация ("Электронное издание", "Настоящее издание")
    - Редакторские введения (историко-литературная справка)
    - Текстологические примечания ("К окончательному тексту...")
    - Варианты редакций ("Вместо: ... — в ж. ред.: ...")
    - Ссылки на страницы ("Стр. 5, строка 10")
    - Библиографические ссылки

    Возвращает только основной текст до первого маркера комментариев.
    """
    lines = text.split('\n')
    end_boundary = len(lines)  # конец основного текста

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Издательские и редакторские маркеры — сразу обрезаем
        if any(marker in line for marker in [
            'Электронное издание',
            'Российской государственной библиотекой',
            'Настоящее издание представляет',
            'доступно на портале',
            'Государственное издательство',
            'Подготовлено на основе',
            '90-томного собрания',
            'Если Вы нашли ошибку',
            'Непосредственно вслед за напечатанием',
            'В начале варианта обозначается страница',
            'исправленное по новым корректурам',
            'Первый слой рукописи',
            'РЕДАКТОР',
            'Рукописей «Войны и мира» сохранилось',
            'В том 14 входят',
            'Из писем к Бартеневу',
            'Наборная рукопись отправлялась',
            'Перепечатка разрешается безвозмездно',
            'Reproduction libre',
            'печатать роман по авторским корректурам',
            'Существующие печатные тексты',
            'черновым автографам',
            'При отборе вариантов редакторы',
            'Располагая варианты по рукописям',
            'Работа над статьей',
            'Запись в Дневнике',
            'черновым его рукописям',
            'В настоящем издании',
            'впервые была напечатана',
            'Рукописи сказки не сохранились',
            'продиктовал в фонограф'
        ]):
            end_boundary = i
            break

        # Маркер "конец." — самый надежный для художественных произведений
        if stripped == 'конец.':
            end_boundary = i
            break

        # Начало текстологических комментариев (длинный абзац со специфическими фразами)
        if len(line) > 300 and any(marker in line for marker in [
            'К окончательному тексту',
            'варианты текста',
            'в основу которого',
            'Рукопись, автограф',
            'исправленный Толстым'
        ]):
            end_boundary = i
            break

        # Ссылки на страницы и строки (ТОЛЬКО в начале строки, не в скобках внутри текста)
        # Паттерн: "Стр. 5, строк" или "*Стр. 10" (с отступом или звёздочкой)
        if re.match(r'^\s*\*?\s*Стр\.\s+\d+,?\s+строк', line):
            end_boundary = i
            break

        # Варианты редакций
        if re.search(r'Вместо:.*—\s*в\s+ж\.\s*ред\.:', line):
            end_boundary = i
            break

        # Ссылки на части и главы (формат: "Ч. V, гл. I.")
        if re.match(r'^Ч\.\s+[IVXLCDM]+,\s+гл\.\s+[IVXLCDM]+\.$', stripped):
            end_boundary = i
            break

        # Датировка + топоним (авторская подпись): "24 сентября." → пустая строка → "Ясная Поляна."
        if re.match(r'^\d{1,2}\s+(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\.$', stripped):
            # Проверяем следующие 2-3 строки на наличие топонима
            next_lines = lines[i+1:i+4] if i+4 < len(lines) else lines[i+1:]
            if any(re.match(r'^(Ясная Поляна|Москва|Петербург|Женева|Лондон)\.?$', nl.strip()) for nl in next_lines):
                end_boundary = i
                break

        # Библиографические ссылки с упоминанием изданий
        if re.search(r'«(Русский вестник|Современник|Красная новь)».*\d{4}', line):
            end_boundary = i
            break

        # Начало библиографических ссылок (одинокая цифра, потом Ibid/Письмо/Цитируется)
        if i > 0 and re.match(r'^\d+$', stripped):
            # Проверяем следующие 2 строки
            next_lines = lines[i+1:i+3] if i+3 < len(lines) else lines[i+1:]
            if any(re.search(r'Ibid\.|Письмо не издано|Цитируется по', nl) for nl in next_lines):
                end_boundary = i
                break

    # Возвращаем только основной текст
    return '\n'.join(lines[:end_boundary])


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


def _iter_records(path: str):
    """Отдаёт (meta, [абзацы]) для каждой записи в файле."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    # Нормализация дореформенной орфографии
    content = _normalize_old_russian(content)

    # делим по строкам-заголовкам META
    parts = re.split(r"(?m)^### META.*$", content)
    headers = re.findall(r"(?m)^### META.*$", content)

    # parts[0] — то, что до первого META (обычно пусто)
    for header, body in zip(headers, parts[1:]):
        # Удаляем редакторские комментарии из каждой записи
        body = _strip_editorial_notes(body)

        meta = _parse_meta(header)
        paras = [p.strip() for p in body.split("\n\n") if p.strip()]
        if paras:
            yield meta, paras


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
    """Единый порядок перечисления записей: файлы sorted(glob), записи по порядку,
    одинаковые фильтры include_types/junk. doc_id — сквозной счётчик записей.
    Его используют И load_chunks, И load_docs — нумерация совпадает по построению."""
    doc_id = 0
    # tom*.txt — художественные произведения (EPUB, legacy); tei_*.txt — письма
    # и дневники из TEI-корпуса tolstoydigital (см. scripts/tei_to_txt.py).
    files = sorted(
        glob.glob(os.path.join(DATA_DIR, "tom*.txt"))
        + glob.glob(os.path.join(DATA_DIR, "tei_*.txt"))
    )
    for path in files:
        for meta, paras in _iter_records(path):
            if include_types and meta.get("type") not in include_types:
                continue
            if _is_junk_record(meta):
                continue
            yield doc_id, meta, paras
            doc_id += 1


def load_chunks(include_types: set[str] | None = None) -> list[Chunk]:
    """
    Читает все tom*.txt и tei_*.txt в data/, режет на чанки (1 абзац = 1 чанк).
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
    entry = meta.get("entry", "")  # полное название из TOC

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
    elif t == "проза":
        title = meta.get("title", "")
        if title and "том" not in title.lower():
            return f"{title}, том {vol}"
        return f"Том {vol} (художественные произведения)"
    else:
        return f"Том {vol}"


if __name__ == "__main__":
    cs = load_chunks()
    print(f"Всего чанков: {len(cs)}")
    from collections import Counter
    by_type = Counter(c.meta.get("type") for c in cs)
    print("По типу:", dict(by_type))
    lens = [len(c.text) for c in cs]
    print(f"Длина чанка: min={min(lens)} avg={sum(lens)//len(lens)} max={max(lens)}")
    print(f"Документов (для /doc): {len(load_docs())}")
    print(f"Чанков без meta['doc']: {sum(1 for c in cs if 'doc' not in c.meta)}")
    print("\nПример:")
    print(source_label(cs[0].meta))
    print(cs[0].text[:300])
