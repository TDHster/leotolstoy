#!/usr/bin/env python3
"""
Конвертер EPUB -> txt для 90-томного собрания.

Извлекает текст + базовые метаданные (номер тома, название, тип).
Формат выхода:

    ### META | vol=5 | title=Война и мир. Том 1 | type=проза
    <текст, абзацы через пустую строку>
"""
from __future__ import annotations

import html
import os
import re
import sys
import zipfile
from dataclasses import dataclass, field

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "90 томов Толстого")
DATA_DIR = os.path.abspath(DATA_DIR)


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


def _extract_epub_metadata(zf: zipfile.ZipFile) -> tuple[str, str]:
    """
    Извлекает название тома и тип из EPUB метаданных.
    Возвращает (title, type).
    """
    names = {n.lower(): n for n in zf.namelist()}

    # Ищем content.opf или package.opf
    opf_name = None
    for low, real in names.items():
        if low.endswith('.opf') and ('content' in low or 'package' in low):
            opf_name = real
            break

    if not opf_name:
        return "", ""

    try:
        opf_content = zf.read(opf_name).decode('utf-8', errors='replace')

        # Извлекаем <dc:title>
        title_m = re.search(r'<dc:title[^>]*>(.*?)</dc:title>', opf_content, re.S | re.I)
        title = _clean_text(title_m.group(1)) if title_m else ""

        # Определяем тип по ключевым словам в названии
        title_lower = title.lower()
        if 'письм' in title_lower or 'letter' in title_lower:
            rtype = "письма"
        elif 'дневник' in title_lower or 'diary' in title_lower:
            rtype = "дневники"
        else:
            rtype = "проза"

        return title, rtype
    except:
        return "", ""


@dataclass
class Record:
    """Запись с базовыми метаданными."""
    vol: int
    title: str = ""  # название тома из EPUB
    rtype: str = ""  # тип: проза, письма, дневники
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
    rtype = STRUCTURED_VOLUMES[vol]
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


def parse_prose_volume(epub_path: str, vol: int) -> list[Record]:
    """Парсинг любого тома: весь текст + базовые метаданные."""
    records: list[Record] = []

    with zipfile.ZipFile(epub_path) as zf:
        # Извлекаем метаданные
        title, rtype = _extract_epub_metadata(zf)

        names = {n.lower(): n for n in zf.namelist()}

        # Собираем все параграфы из всех HTML-файлов
        all_paragraphs = []

        # Ищем все HTML/XHTML файлы в text/ или Text/
        for low_name, real_name in names.items():
            if '/text/' in low_name and (low_name.endswith('.html') or low_name.endswith('.xhtml')):
                # Пропускаем служебные файлы (обложка, лицензия и т.п.)
                basename = os.path.basename(low_name)
                if basename in ('cover.xhtml', 'cover.html', 'title.xhtml', 'title.html',
                               'license.xhtml', 'license.html', 'titlepage.xhtml', 'titlepage.html',
                               'annotation.xhtml', 'annotation.html'):
                    continue

                html = zf.read(real_name).decode('utf-8-sig', errors='replace')
                paragraphs, _ = _extract_paragraphs(html)
                all_paragraphs.extend(paragraphs)

        if all_paragraphs:
            records.append(Record(
                vol=vol,
                title=title,
                rtype=rtype,
                paragraphs=all_paragraphs
            ))

    return records


# ---------------------------------------------------------------------------
# Запись txt
# ---------------------------------------------------------------------------

def write_txt(records: list[Record], out_path: str) -> None:
    """Записываем текст с метаданными."""
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            # Формируем строку META
            parts = [f"vol={r.vol}"]
            if r.title:
                # Экранируем | в названии
                title_safe = r.title.replace("|", "/")
                parts.append(f"title={title_safe}")
            if r.rtype:
                parts.append(f"type={r.rtype}")

            f.write("### META | " + " | ".join(parts) + "\n")
            f.write("\n\n".join(r.paragraphs))
            f.write("\n\n")


def find_epubs() -> list[tuple[int, str]]:
    """Ищет все EPUB-файлы с номером тома в имени."""
    found = []
    for root, _dirs, files in os.walk(DATA_DIR):
        for fn in files:
            if not fn.lower().endswith(".epub"):
                continue
            # Ищем паттерн: "Том" или "том" + цифры (включая Толстой_Том_01.epub)
            m = re.search(r"[Тт]ом[_\s]*(\d+)", fn)
            if not m:
                continue
            vol = int(m.group(1))
            found.append((vol, os.path.join(root, fn)))
    found.sort()
    return found


def main() -> int:
    epubs = find_epubs()
    if not epubs:
        print(f"EPUB-файлы с номером тома не найдены в {DATA_DIR}",
              file=sys.stderr)
        return 1

    # Выходные txt сохраняем в data/, а не в data/90 томов Толстого/
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_dir = os.path.abspath(output_dir)

    total = 0
    for vol, path in epubs:
        # Парсим все тома одинаково — извлекаем текст + метаданные
        records = parse_prose_volume(path, vol)

        # Упрощённые имена файлов: tom01.txt, tom02.txt, ...
        out_name = f"tom{vol:02d}.txt"
        out_path = os.path.join(output_dir, out_name)
        write_txt(records, out_path)
        total += len(records)

        # Показываем что извлекли
        info = []
        if records and records[0].title:
            info.append(f'"{records[0].title}"')
        if records and records[0].rtype:
            info.append(f"[{records[0].rtype}]")
        info_str = " ".join(info) if info else ""

        print(f"Том {vol:2d}: {len(records):>5} записей -> {out_name}  {info_str}")

    print(f"\nИтого: {total} записей из {len(epubs)} томов.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
