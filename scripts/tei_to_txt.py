#!/usr/bin/env python3
"""
Конвертация TEI/XML (репозиторий tolstoydigital/TEI, CC BY-SA) в data/*.txt
в формате записей, который читает scripts/corpus.py:

    ### META | vol=59 | type=письмо | num=5 | to=... | date=... | ...
    <абзацы через пустую строку>

Письма: data/tolstoy-tei/texts/letters/vNN_NNN_*.xml -> data/tei_letters.txt
Дневники: data/tolstoy-tei/texts/diaries/*.xml     -> data/tei_diaries.txt

Извлекается вся метаданная из шапки TEI (адресат, дата, место, том, №, титул,
библиографическая ссылка ПСС, страницы, год издания, признак дореформенной
орфографии). Тело: абзацы <p>; <choice> берём в современной орфографии (<reg>),
редакторские сноски <note>/<noteGrp> и их маркеры <ref> выбрасываем.

`make convert` — основной способ запуска; EPUB-конвертер (epub_to_txt.py)
остался legacy после перехода проекта на digital Tolstoy.
"""
from __future__ import annotations

import glob
import os
import re
import sys
import xml.etree.ElementTree as ET

NS = "http://www.tei-c.org/ns/1.0"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(ROOT, "data")
TEI_DIR = os.path.join(DATA_DIR, "tolstoy-tei", "texts")

LETTER_FILE_RE = re.compile(r"^v(\d+)_(\d+)_")
DIARY_FILE_RE = re.compile(r"^v(\d+)_(\d+)_(\d+)_(\d{4})_(\d{2})_(\d{2})")

# какие подкаталоги TEI-корпуса конвертируем и куда пишем результат
KINDS = {
    "letters": ("tei_letters.txt", "parse_letter"),
    "diaries": ("tei_diaries.txt", "parse_diary"),
}


def q(tag: str) -> str:
    """Имя тега с namespace TEI."""
    return f"{{{NS}}}{tag}"


def _clean(s: str) -> str:
    """Схлопываем пробелы/переводы строк — метаданные пишем в одну строку."""
    return re.sub(r"\s+", " ", s).strip()


def _fix_punct(s: str) -> str:
    """Убираем пробел перед знаками препинания (артефакт набора: «слово ,») ."""
    return re.sub(r"\s+([.,;:!?…])", r"\1", s)


def _text(elem: ET.Element) -> str:
    """
    Текст элемента для показа/эмбеддинга: рекурсивно, без сносок и разметки.
    <choice>: берём <reg> (современная орфография), <orig>/<sic> пропускаем;
    <note>, <noteGrp>, <ref> (маркеры сносок), <pb>, <opener> — пропускаем.
    """
    parts: list[str] = []

    def walk(el: ET.Element) -> None:
        tag = el.tag.rsplit("}", 1)[-1]
        if tag in ("note", "noteGrp", "orig", "sic", "ref", "pb", "opener"):
            return
        if el.text:
            parts.append(el.text)
        for child in el:
            walk(child)
            if child.tail:
                parts.append(child.tail)

    walk(elem)
    return _fix_punct(_clean("".join(parts)))


def _paragraphs(body: ET.Element) -> list[str]:
    """Все <p> тела документа, кроме тех, что внутри noteGrp/note."""
    paras: list[str] = []

    def walk(el: ET.Element, in_note: bool) -> None:
        for child in el:
            tag = child.tag.rsplit("}", 1)[-1]
            if tag in ("note", "noteGrp"):
                walk(child, True)
                continue
            if tag == "p" and not in_note:
                text = _text(child)
                if text:
                    paras.append(text)
            walk(child, in_note)

    walk(body, False)
    return paras


def _header_title(root: ET.Element, kind: str) -> str | None:
    """title[@type=...] из teiHeader."""
    for t in root.iter(q("title")):
        if t.get("type") == kind:
            text = _clean("".join(t.itertext()))
            return text or None
    return None


def _header_bibl(root: ET.Element) -> dict:
    """Библиография из sourceDesc/biblStruct: том, страницы, год издания, «публикуется впервые»."""
    out: dict = {}
    for bs in root.iter(q("biblScope")):
        unit = bs.get("unit", "")
        text = _clean(bs.text or "")
        if unit in ("vol", "volume") and text and "vol" not in out:
            out["vol"] = text
        elif unit == "page" and text and "pages" not in out:
            out["pages"] = text
    for d in root.iter(q("date")):
        # первый date в документе — год издания тома (imprint), шапка идёт раньше тела
        when = d.get("when")
        if when and re.match(r"^\d{4}", when):
            out["ed_year"] = when[:4]
            break
    sd = root.find(f".//{q('sourceDesc')}")
    if sd is not None:
        for b in sd.findall(q("bibl")):
            text = _clean("".join(b.itertext()))
            if text:
                out["first_pub"] = text
                break
    return out


def _has_old_orth(root: ET.Element) -> bool:
    """Дореформенная орфография: langUsage упоминает ru-petr1708."""
    return any(
        lg.get("ident") == "ru-petr1708" for lg in root.iter(q("language"))
    )


def _corresp_action(root: ET.Element, action_type: str) -> ET.Element | None:
    for ca in root.iter(q("correspAction")):
        if ca.get("type") == action_type:
            return ca
    return None


def _year_of(date: str | None) -> str | None:
    """Год из ISO-даты; диапазон "1900-01-01—1910-12-31" -> "1900–1910"."""
    if not date:
        return None
    m = re.match(r"^(\d{4})", date)
    if not m:
        return None
    if "—" in date:
        tail = date.split("—", 1)[1][:4]
        return f"{m.group(1)}–{tail}" if tail.isdigit() else m.group(1)
    return m.group(1)


def parse_letter(path: str) -> tuple[dict, list[str]] | None:
    """
    Письмо: возвращает (meta, [абзацы]); None — если тела нет/пусто.
    Raises ET.ParseError на битом XML (обрабатывает вызывающий).
    """
    root = ET.parse(path).getroot()
    base = os.path.basename(path)

    vol, num = None, None
    m = LETTER_FILE_RE.match(base)
    if m:
        vol, num = m.group(1), m.group(2)

    title = _header_title(root, "main")
    bibl = _header_title(root, "bibl")
    bib = _header_bibl(root)
    vol = vol or bib.get("vol")

    # № письма в томе (xenoData), если не взяли из имени файла
    if not num:
        for n in root.iter(q("number")):
            if n.get("type") == "in a volume":
                num = _clean(n.text or "") or None
                break
    if num and num.isdigit():
        num = str(int(num))  # убираем ведущие нули: 001 -> 1

    ca_send = _corresp_action(root, "sending")
    ca_recv = _corresp_action(root, "receiving")

    to, place, date = None, None, None
    if ca_recv is not None:
        pn = ca_recv.find(q("persName"))
        if pn is not None:
            to = _clean("".join(pn.itertext())) or None
    if ca_send is not None:
        pl = ca_send.find(q("placeName"))
        if pl is not None:
            place = _clean("".join(pl.itertext())) or None
        d = ca_send.find(q("date"))
        if d is not None:
            date = d.get("when")
            if not date:
                nb, na = d.get("notBefore"), d.get("notAfter")
                if nb:
                    date = f"{nb}—{na}" if na else nb

    year = _year_of(date)

    body = root.find(f".//{q('text')}/{q('body')}")
    if body is None:
        return None
    paras = _paragraphs(body)
    if not paras:
        return None

    # строка-подпись для UI (как entry у писем из EPUB)
    if to:
        parts = [to]
        if year:
            parts.append(year)
        if place:
            parts.append(place)
        if bib.get("pages"):
            parts.append(f"с. {bib['pages']}")
        entry = ", ".join(parts)
        if num:
            entry = f"{num}. {entry}"
    else:
        entry = title

    meta: dict = {}
    for k, v in (
        ("vol", vol), ("type", "письмо"), ("num", num), ("to", to),
        ("date", date), ("place", place), ("year", year), ("entry", entry),
        ("title", title), ("bibl", bibl), ("pages", bib.get("pages")),
        ("ed_year", bib.get("ed_year")), ("first_pub", bib.get("first_pub")),
    ):
        if v:
            meta[k] = v
    if _has_old_orth(root):
        meta["old_orth"] = "да"
    return meta, paras


def parse_diary(path: str) -> tuple[dict, list[str]] | None:
    """Дневниковая запись: (meta, [абзацы]); None — если тела нет/пусто."""
    root = ET.parse(path).getroot()
    base = os.path.basename(path)

    vol = None
    pages = None
    date = None
    m = DIARY_FILE_RE.match(base)
    if m:
        vol = m.group(1)
        pages = f"{int(m.group(2))}–{int(m.group(3))}"
        date = f"{m.group(4)}-{m.group(5)}-{m.group(6)}"

    # дата из шапки (creation) надёжнее, чем из имени файла
    for d in root.iter(q("date")):
        when = d.get("when") or d.get("from")
        if when and re.match(r"^\d{4}-\d{2}-\d{2}", when):
            date = when
            break

    title = _header_title(root, "main")
    bibl = _header_title(root, "bibl")
    bib = _header_bibl(root)
    vol = vol or bib.get("vol")
    pages = pages or bib.get("pages")
    year = _year_of(date)

    body = root.find(f".//{q('text')}/{q('body')}")
    if body is None:
        return None
    paras = _paragraphs(body)
    if not paras:
        return None

    # «Дневник» как название раздела не несёт информации — label соберётся из года
    if title and title.rstrip(".,;: ").strip() != "Дневник":
        entry = title.rstrip(".,;: ")
    else:
        entry = None

    meta: dict = {}
    for k, v in (
        ("vol", vol), ("type", "дневник"), ("date", date), ("year", year),
        ("entry", entry), ("title", title), ("bibl", bibl),
        ("pages", pages), ("ed_year", bib.get("ed_year")),
    ):
        if v:
            meta[k] = v
    if _has_old_orth(root):
        meta["old_orth"] = "да"
    return meta, paras


def format_record(meta: dict, paras: list[str]) -> str:
    """Запись в формате corpus.py: ### META | k=v ... + абзацы через пустую строку."""
    parts = [f"{k}={str(v).replace('|', '/')}" for k, v in meta.items()]
    return "### META | " + " | ".join(parts) + "\n" + "\n\n".join(paras) + "\n\n"


def convert(kind: str, out_name: str) -> int:
    """Конвертирует один подкаталог корпуса; возвращает 1 при ошибке каталога."""
    src_dir = os.path.join(TEI_DIR, kind)
    if not os.path.isdir(src_dir):
        print(f"[{kind}] каталог не найден: {src_dir}", file=sys.stderr)
        print(f"  Сначала: make tei-fetch", file=sys.stderr)
        return 1

    parse = globals()[KINDS[kind][1]]
    files = sorted(glob.glob(os.path.join(src_dir, "*.xml")))
    out_path = os.path.join(DATA_DIR, out_name)

    written = skipped = errors = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for path in files:
            try:
                rec = parse(path)
            except ET.ParseError:
                errors += 1
                continue
            if rec is None:
                skipped += 1
                continue
            meta, paras = rec
            f.write(format_record(meta, paras))
            written += 1

    size_mb = os.path.getsize(out_path) / 1e6
    print(f"[{kind}] файлов: {len(files)}, записей: {written}, "
          f"пустых: {skipped}, битых: {errors} -> {out_path} ({size_mb:.1f} МБ)")
    return 0


def main() -> int:
    ok = 0
    for kind, (out_name, _) in KINDS.items():
        ok += convert(kind, out_name)
    return 1 if ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
