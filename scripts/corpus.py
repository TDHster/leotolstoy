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
# т.е. окно ≈ 420-460 символов), поэтому чанк выбираем размером с окно: вектор
# покрывает весь текст чанка, а не только его «голову».
CHUNK_CHARS = 450
CHUNK_OVERLAP_PARAS = 1  # сколько абзацев перекрытия между соседними чанками

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
    """Склеиваем абзацы в чанки ~CHUNK_CHARS с небольшим перекрытием."""
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for para in paras:
        # если один абзац сам по себе огромный — режем его по предложениям
        if len(para) > CHUNK_CHARS * 1.5:
            if buf:
                chunks.append("\n\n".join(buf))
                buf, buf_len = [], 0
            chunks.extend(_split_long(para))
            continue
        if buf_len + len(para) > CHUNK_CHARS and buf:
            chunks.append("\n\n".join(buf))
            # перекрытие: оставляем последние N абзацев
            buf = buf[-CHUNK_OVERLAP_PARAS:] if CHUNK_OVERLAP_PARAS else []
            buf_len = sum(len(p) for p in buf)
        # перекрытие + новый абзац не влезают в чанк — сбрасываем перекрытие,
        # чтобы чанк не разрастался вдвое за счёт дубля
        if buf_len + len(para) > CHUNK_CHARS and buf:
            buf, buf_len = [], 0
        buf.append(para)
        buf_len += len(para)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks


def _split_long(para: str) -> list[str]:
    sents = re.split(r"(?<=[.!?…])\s+", para)
    out, buf, blen = [], [], 0
    for s in sents:
        if blen + len(s) > CHUNK_CHARS and buf:
            out.append(" ".join(buf))
            # перекрытие: последнее предложение переходит в следующий кусок,
            # чтобы граница не рвала контекст
            buf, blen = buf[-1:], len(buf[-1])
        # перекрытие не влезает (гигантское предложение) — без него
        if blen + len(s) > CHUNK_CHARS and buf:
            buf, blen = [], 0
        buf.append(s)
        blen += len(s)
    if buf:
        out.append(" ".join(buf))
    return out


def load_chunks(include_types: set[str] | None = None) -> list[Chunk]:
    """
    Читает все tom*.txt в data/, режет на чанки.
    include_types — например {"проза"} чтобы взять только художественные произведения.
    None = всё.
    """
    chunks: list[Chunk] = []
    files = sorted(glob.glob(os.path.join(DATA_DIR, "tom*.txt")))
    for path in files:
        for meta, paras in _iter_records(path):
            if include_types and meta.get("type") not in include_types:
                continue
            if _is_junk_record(meta):
                continue
            for i, ctext in enumerate(_chunk_paragraphs(paras)):
                m = dict(meta)
                m["chunk"] = str(i)
                chunks.append(Chunk(text=ctext, meta=m))
    return chunks


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
    print("\nПример:")
    print(source_label(cs[0].meta))
    print(cs[0].text[:300])
