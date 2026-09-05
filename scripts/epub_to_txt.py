#!/usr/bin/env python3
"""
Конвертер EPUB -> структурированный txt.

Обрабатывает тома собрания сочинений Толстого (письма и дневники).
На каждую запись (письмо или дневниковый год) пишет блок:

    ### META | vol=18 | type=письмо | year=1849 | num=4 | to=С. Н. Толстому | date=... | note=...
    <текст записи, абзацы через пустую строку>

Затем этот txt читает scripts/build_embeddings.py.

Зависимостей нет — только стандартная библиотека.
"""
from __future__ import annotations

import html
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
DATA_DIR = os.path.abspath(DATA_DIR)

# Какие тома чем являются. Всё остальное игнорируем.
VOLUME_TYPE = {
    18: "письмо",
    19: "письмо",
    20: "письмо",
    21: "дневник",
    22: "дневник",
}

# ---------------------------------------------------------------------------
# Разбор HTML секции
# ---------------------------------------------------------------------------

_ANCHOR_RE = re.compile(r"<a\b[^>]*>.*?</a>", re.S | re.I)  # сноски-ссылки целиком
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t ]+")


def _clean_text(fragment: str) -> str:
    """HTML-фрагмент -> чистый текст: убираем сноски, теги, entities, лишние пробелы."""
    fragment = _ANCHOR_RE.sub("", fragment)      # выкидываем сноски целиком
    fragment = _TAG_RE.sub("", fragment)          # снимаем оставшиеся теги (<em> и т.п.)
    fragment = html.unescape(fragment)            # &lt; &#160; -> символы
    fragment = fragment.replace(" ", " ")
    fragment = _WS_RE.sub(" ", fragment)
    return fragment.strip()


def _extract_paragraphs(section_html: str) -> tuple[list[str], str]:
    """Возвращает (абзацы_тела, дата_из_первого_em_в_p.drop)."""
    date = ""
    drop_m = re.search(r'<p[^>]*class="[^"]*drop[^"]*"[^>]*>(.*?)</p>', section_html, re.S | re.I)
    if drop_m:
        em_m = re.search(r"<em>(.*?)</em>", drop_m.group(1), re.S | re.I)
        if em_m:
            date = _clean_text(em_m.group(1))

    paragraphs = []
    for m in re.finditer(r"<p\b[^>]*>(.*?)</p>", section_html, re.S | re.I):
        text = _clean_text(m.group(1))
        if text:
            paragraphs.append(text)
    return paragraphs, date


# ---------------------------------------------------------------------------
# Разбор оглавления (toc.ncx) и метаданных записей
# ---------------------------------------------------------------------------

_NAVPOINT_RE = re.compile(
    r"<navLabel>\s*<text>(.*?)</text>\s*</navLabel>\s*<content\s+src=\"([^\"]+)\"",
    re.S | re.I,
)
_YEAR_RE = re.compile(r"^\d{4}$")
# "* 4. С. Н. Толстому  <перевод с французского>"
_LETTER_RE = re.compile(r"^\*?\s*(\d+)\.\s*(.*?)\s*(?:<(.*?)>)?\s*$")


@dataclass
class Record:
    vol: int
    rtype: str
    year: str
    num: str = ""
    to: str = ""
    date: str = ""
    note: str = ""
    paragraphs: list[str] = field(default_factory=list)


def _find_section_file(zf: zipfile.ZipFile, src: str, names: dict[str, str]) -> str | None:
    """Ищем файл секции регистронезависимо (в разных томах text/ vs Text/)."""
    src = src.split("#")[0]
    base = os.path.basename(src).lower()
    # прямые кандидаты
    for cand in (f"oebps/text/{base}", f"oebps/Text/{base}".lower()):
        if cand in names:
            return names[cand]
    # запасной путь: по имени файла
    for low, real in names.items():
        if low.endswith("/" + base):
            return names[low]
    return None


def parse_volume(epub_path: str, vol: int) -> list[Record]:
    rtype = VOLUME_TYPE[vol]
    records: list[Record] = []

    with zipfile.ZipFile(epub_path) as zf:
        names = {n.lower(): n for n in zf.namelist()}
        toc_name = next((names[n] for n in names if n.endswith("toc.ncx")), None)
        if not toc_name:
            print(f"  [!] Том {vol}: нет toc.ncx, пропускаю", file=sys.stderr)
            return records
        toc = zf.read(toc_name).decode("utf-8-sig", errors="replace")

        current_year = ""
        for label_raw, src in _NAVPOINT_RE.findall(toc):
            label = html.unescape(label_raw).replace(" ", " ").strip()

            if _YEAR_RE.match(label):
                current_year = label
                if rtype == "письмо":
                    # для писем год-заголовок содержания не несёт текста — пропускаем
                    continue
                # для дневников год = отдельный файл с текстом за год -> читаем ниже
                num, to, note = "", "", ""
            else:
                m = _LETTER_RE.match(label)
                if rtype == "письмо":
                    if not m:
                        continue  # не письмо (обложка, лицензия и т.п.)
                    num, to, note = m.group(1), m.group(2).strip(), (m.group(3) or "").strip()
                else:
                    # дневник: именованный раздел ("Дневник помещика")
                    num, to, note = "", label, ""

            section_file = _find_section_file(zf, src, names)
            if not section_file:
                continue
            section_html = zf.read(section_file).decode("utf-8-sig", errors="replace")
            paragraphs, date = _extract_paragraphs(section_html)
            if not paragraphs:
                continue

            records.append(Record(
                vol=vol, rtype=rtype, year=current_year,
                num=num, to=to, date=date, note=note, paragraphs=paragraphs,
            ))
    return records


# ---------------------------------------------------------------------------
# Запись txt
# ---------------------------------------------------------------------------

def _meta_line(r: Record) -> str:
    def esc(s: str) -> str:
        return s.replace("|", "/").replace("\n", " ").strip()
    parts = [
        f"vol={r.vol}", f"type={r.rtype}", f"year={esc(r.year)}",
    ]
    if r.num:
        parts.append(f"num={esc(r.num)}")
    if r.to:
        parts.append(f"to={esc(r.to)}")
    if r.date:
        parts.append(f"date={esc(r.date)}")
    if r.note:
        parts.append(f"note={esc(r.note)}")
    return "### META | " + " | ".join(parts)


def write_txt(records: list[Record], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(_meta_line(r) + "\n")
            f.write("\n\n".join(r.paragraphs))
            f.write("\n\n")


def find_epubs() -> list[tuple[int, str]]:
    found = []
    for root, _dirs, files in os.walk(DATA_DIR):
        for fn in files:
            if not fn.lower().endswith(".epub"):
                continue
            m = re.search(r"Том\s*(\d+)", fn)
            if not m:
                continue
            vol = int(m.group(1))
            if vol in VOLUME_TYPE:
                found.append((vol, os.path.join(root, fn)))
    found.sort()
    return found


def main() -> int:
    epubs = find_epubs()
    if not epubs:
        print(f"EPUB с письмами/дневниками (тома {sorted(VOLUME_TYPE)}) не найдены в {DATA_DIR}",
              file=sys.stderr)
        return 1

    total = 0
    for vol, path in epubs:
        rtype = VOLUME_TYPE[vol]
        records = parse_volume(path, vol)
        out_name = f"tom{vol}_{'pisma' if rtype == 'письмо' else 'dnevniki'}.txt"
        out_path = os.path.join(DATA_DIR, out_name)
        write_txt(records, out_path)
        total += len(records)
        print(f"Том {vol} ({rtype}): {len(records):>5} записей -> data/{out_name}")

    print(f"\nИтого: {total} записей.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
