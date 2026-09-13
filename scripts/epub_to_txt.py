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

# Тома с детальной структурой (письма/дневники)
# Письма: тома 59-90 (32 тома)
# Дневники: тома 48, 49, 52, 53, 54, 58
STRUCTURED_VOLUMES = {
    # Дневники
    48: "дневник",
    49: "дневник",
    52: "дневник",
    53: "дневник",
    54: "дневник",
    58: "дневник",
    # Письма
    59: "письмо",
    60: "письмо",
    61: "письмо",
    62: "письмо",
    63: "письмо",
    64: "письмо",
    65: "письмо",
    66: "письмо",
    67: "письмо",
    68: "письмо",
    69: "письмо",
    70: "письмо",
    71: "письмо",
    72: "письмо",
    73: "письмо",
    74: "письмо",
    75: "письмо",
    76: "письмо",
    77: "письмо",
    78: "письмо",
    79: "письмо",
    80: "письмо",
    81: "письмо",
    82: "письмо",
    83: "письмо",
    84: "письмо",
    85: "письмо",
    86: "письмо",
    87: "письмо",
    88: "письмо",
    89: "письмо",
    90: "письмо",
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
    # Дополнительные поля для писем/дневников
    year: str = ""   # год (для писем/дневников)
    num: str = ""    # номер письма
    to: str = ""     # адресат письма или название раздела дневника
    date: str = ""   # дата из <em> в p.drop
    note: str = ""   # примечание (например, "перевод с французского")
    entry_title: str = ""  # полное название записи из TOC (для поиска и отображения)


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


def _is_editorial_content(text: str) -> bool:
    """
    Определяет, является ли параграф редакторским комментарием.

    Признаки редакторских комментариев:
    - Начинается с цифры и буквы (сноски: "1Письмо это не сохранилось")
    - Начинается с "Впервые опубликовано", "Отрывок напечатан", "Датируется"
    - Упоминает издания, публикации, редакторов
    - Биографические справки (упоминает годы жизни в скобках)
    """
    text = text.strip()

    # Пустой или очень короткий — пропускаем
    if len(text) < 10:
        return True

    # Начинается с цифры и буквы — сноска (1Письмо, 2В 1856, 3В. И.)
    if re.match(r'^\d+[А-Яа-яA-Za-z]', text):
        return True

    # Редакторские маркеры
    editorial_markers = [
        'Впервые опубликовано',
        'Отрывок напечатан',
        'Датируется',
        'Автограф',
        'Печатается по',
        'Год определяется',
        'На четвертой странице:',
        'Адресовано:',
        'Письмо это не сохранилось',
        'См. т.',
        'см. т.',
    ]

    for marker in editorial_markers:
        if text.startswith(marker):
            return True

    # Биографические справки (годы жизни в скобках)
    if re.search(r'\(\d{4}—\d{4}\)', text):
        return True

    return False


def parse_volume(epub_path: str, vol: int) -> list[Record]:
    """
    Парсинг структурированного тома (письма/дневники) с поддержкой якорей.

    Для писем: несколько сотен записей в одном HTML-файле, разделены якорями.
    Используем BeautifulSoup для извлечения содержимого между якорями.

    Для дневников: каждая запись в отдельном файле, читаем целиком.
    """
    from bs4 import BeautifulSoup

    rtype = STRUCTURED_VOLUMES[vol]
    records: list[Record] = []

    with zipfile.ZipFile(epub_path) as zf:
        names = {n.lower(): n for n in zf.namelist()}
        toc_name = next((names[n] for n in names if n.endswith("toc.ncx")), None)
        if not toc_name:
            print(f"  [!] Том {vol}: нет toc.ncx, пропускаю", file=sys.stderr)
            return records
        toc = zf.read(toc_name).decode("utf-8-sig", errors="replace")

        # Группируем записи по файлам: {src_file: [(label, src_full, year), ...]}
        entries_by_file = {}
        current_year = ""

        for label_raw, src in _NAVPOINT_RE.findall(toc):
            label = html.unescape(label_raw).replace(" ", " ").strip()

            # Обработка года
            if _YEAR_RE.match(label):
                current_year = label
                if rtype == "письмо":
                    continue  # для писем год-заголовок не несёт текста

            # Парсим письмо/дневник
            m = _LETTER_RE.match(label)
            if rtype == "письмо":
                if not m:
                    continue  # не письмо (обложка и т.п.)
                num, to, note = m.group(1), m.group(2).strip(), (m.group(3) or "").strip()
            else:
                # дневник
                num, to, note = "", label, ""

            # Группируем по файлам
            src_file = src.split('#')[0]
            if src_file not in entries_by_file:
                entries_by_file[src_file] = []

            entries_by_file[src_file].append({
                'label': label,
                'src': src,
                'year': current_year,
                'num': num,
                'to': to,
                'note': note,
            })

        # Обрабатываем каждый файл
        for src_file, entries in entries_by_file.items():
            section_file = _find_section_file(zf, src_file, names)
            if not section_file:
                continue

            html_content = zf.read(section_file).decode("utf-8-sig", errors="replace")

            # Если один entry без якоря — читаем весь файл (дневники)
            if len(entries) == 1 and '#' not in entries[0]['src']:
                entry = entries[0]
                paragraphs, date = _extract_paragraphs(html_content)
                if paragraphs:
                    records.append(Record(
                        vol=vol, rtype=rtype, year=entry['year'],
                        num=entry['num'], to=entry['to'], date=date, note=entry['note'],
                        entry_title=entry['label'], paragraphs=paragraphs,
                    ))
            else:
                # Парсим через BeautifulSoup для извлечения по якорям
                soup = BeautifulSoup(html_content, 'html.parser')

                for entry in entries:
                    anchor = entry['src'].split('#')[1] if '#' in entry['src'] else None
                    if not anchor:
                        continue

                    # Ищем элемент с этим id
                    elem = soup.find(id=anchor)
                    if not elem:
                        continue

                    # Извлекаем параграфы до следующего <h3>
                    paragraphs = []
                    date = ""

                    for sibling in elem.find_all_next():
                        # Остановка на следующем заголовке
                        if sibling.name == 'h3' and sibling != elem:
                            break

                        # Извлекаем параграфы
                        if sibling.name == 'p':
                            text = _clean_text(str(sibling))
                            if not text:
                                continue

                            # Остановка на редакторских комментариях
                            if _is_editorial_content(text):
                                break

                            paragraphs.append(text)

                            # Извлекаем дату из первого параграфа
                            if not date and len(paragraphs) == 1:
                                date_match = re.search(r'\d{4}\s*г\..*?\.', paragraphs[0])
                                if date_match:
                                    date = date_match.group(0)

                    if paragraphs:
                        records.append(Record(
                            vol=vol, rtype=rtype, year=entry['year'],
                            num=entry['num'], to=entry['to'], date=date, note=entry['note'],
                            entry_title=entry['label'], paragraphs=paragraphs,
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
            # Дополнительные поля для писем/дневников
            if r.year:
                parts.append(f"year={r.year}")
            if r.num:
                parts.append(f"num={r.num}")
            if r.to:
                # Экранируем | в адресате
                to_safe = r.to.replace("|", "/")
                parts.append(f"to={to_safe}")
            if r.date:
                parts.append(f"date={r.date}")
            if r.note:
                note_safe = r.note.replace("|", "/")
                parts.append(f"note={note_safe}")
            if r.entry_title:
                # Полное название записи из TOC (для поиска и отображения)
                entry_safe = r.entry_title.replace("|", "/")
                parts.append(f"entry={entry_safe}")

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
        # Выбор парсера в зависимости от типа тома
        if vol in STRUCTURED_VOLUMES:
            records = parse_volume(path, vol)
        else:
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
