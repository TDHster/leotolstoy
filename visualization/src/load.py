"""Загрузка и валидация индекса (vectors.npy + chunks.json) — только чтение.

Входные данные НЕ меняются (правило ТЗ: «не менять index/»). Векторы уже
L2-нормализованы сборщиком индекса — здесь это проверяется ассертом, а не
вычисляется (корректировка ТЗ: шаг «L2-нормализовать» — no-op).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np

VIZ_DIR = Path(__file__).resolve().parent.parent
INDEX_DIR = VIZ_DIR.parent / "index"
CACHE_DIR = VIZ_DIR / "cache"
OUT_DIR = VIZ_DIR / "out"

# Год смерти Толстого: всё позже в метаданных — битые даты TEI (нашлись «2012»).
MAX_YEAR = 1910

_YEAR_RE = re.compile(r"\d{4}")


def normalize_year(raw) -> int | None:
    """Год из meta['year']: первый 4-значный год диапазона; >MAX_YEAR/пусто → None.

    В TEI годы бывают диапазонами («1848–1848», «1888–1900») — для оси времени
    берём начало диапазона.
    """
    if raw is None:
        return None
    m = _YEAR_RE.search(str(raw))
    if not m:
        return None
    year = int(m.group())
    return year if year <= MAX_YEAR else None


def load_data(index_dir: Path = INDEX_DIR) -> dict:
    """Загрузить векторы и чанки, проверить согласованность (ТЗ, раздел 2)."""
    index_dir = Path(index_dir)
    vectors = np.load(index_dir / "vectors.npy")  # чистый float32 — allow_pickle не нужен
    with open(index_dir / "chunks.json", encoding="utf-8") as f:
        chunks = json.load(f)

    assert len(vectors) == len(chunks), (
        f"Рассинхрон: векторов {len(vectors)}, чанков {len(chunks)}"
    )
    assert vectors.ndim == 2 and vectors.dtype == np.float32, (vectors.shape, vectors.dtype)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-3), (
        f"Векторы не L2-нормализованы: нормы {norms.min():.4f}..{norms.max():.4f}"
    )

    # Год как int32; -1 = «без даты» (нет года или аномалия вроде «2012»).
    years = np.array(
        [normalize_year(c["meta"].get("year")) or -1 for c in chunks], dtype=np.int32
    )
    return {"vectors": vectors, "chunks": chunks, "years": years}


def summarize(data: dict) -> str:
    """Сводка данных (ТЗ: формы, типы, диапазон годов, десятилетия, пропуски)."""
    vectors, chunks, years = data["vectors"], data["chunks"], data["years"]
    types = Counter(c["meta"].get("type") for c in chunks)
    dated = years >= 0
    lines = [
        "=== Сводка данных (index/) ===",
        "Чанков: {:,} — {}".format(
            len(chunks),
            ", ".join(f"{t}: {n:,}" for t, n in sorted(types.items())),
        ),
        f"Векторы: {vectors.shape}, {vectors.dtype}, L2-норма 1.0",
        f"С годом: {int(dated.sum()):,} · без даты/аномалии: {int((~dated).sum()):,}",
    ]
    decades = Counter()
    for y in years[dated]:
        decades[y // 10 * 10] += 1
    lines.append(
        "Десятилетия: " + ", ".join(f"{d}: {n:,}" for d, n in sorted(decades.items()))
    )
    # примеры сырых годов, которые не нормализовались в год <= MAX_YEAR
    raw_odd = sorted(
        {
            str(c["meta"].get("year"))
            for c in chunks
            if c["meta"].get("year") and normalize_year(c["meta"].get("year")) is None
        }
    )
    if raw_odd:
        lines.append(f"Аномальные годы (примеры): {', '.join(raw_odd[:8])}")
    return "\n".join(lines)
