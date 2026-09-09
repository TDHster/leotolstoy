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

# Целевой размер чанка в символах. mpnet-base-v2 берёт до ~512 токенов;
# ~1500 символов кириллицы (≈400 токенов) даёт больше контекста с запасом.
CHUNK_CHARS = 1500
CHUNK_OVERLAP_PARAS = 1  # сколько абзацев перекрытия между соседними чанками


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


def _iter_records(path: str):
    """Отдаёт (meta, [абзацы]) для каждой записи в файле."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    # делим по строкам-заголовкам META
    parts = re.split(r"(?m)^### META.*$", content)
    headers = re.findall(r"(?m)^### META.*$", content)
    # parts[0] — то, что до первого META (обычно пусто)
    for header, body in zip(headers, parts[1:]):
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
            for i, ctext in enumerate(_chunk_paragraphs(paras)):
                m = dict(meta)
                m["chunk"] = str(i)
                chunks.append(Chunk(text=ctext, meta=m))
    return chunks


def source_label(meta: dict) -> str:
    """Человекочитаемая ссылка на источник для показа в ответе."""
    vol = meta.get("vol", "")
    title = meta.get("title", "")
    t = meta.get("type", "")

    parts = []
    if vol:
        parts.append(f"Том {vol}")
    if title:
        parts.append(title)
    elif t:
        parts.append(t)

    return " — ".join(parts) if parts else "Неизвестный источник"


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
