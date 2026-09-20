"""UMAP-проекции: 2D (карта) и 10D (кластеризация), кэш в cache/*.npy.

Параметры — по ТЗ 5.1. Векторы уже L2-нормализованы (проверено в load),
metric="cosine" на них эквивалентна угловой метрике.
"""
from __future__ import annotations

import numpy as np
import umap

from . import load as load_mod

UMAP_COMMON = dict(n_neighbors=30, metric="cosine", random_state=42, verbose=False)


def _project(vectors: np.ndarray, n_components: int, min_dist: float) -> np.ndarray:
    reducer = umap.UMAP(n_components=n_components, min_dist=min_dist, **UMAP_COMMON)
    return reducer.fit_transform(vectors).astype(np.float32)


def run(data: dict, force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """2D- и 10D-проекции; пересчитывает только отсутствующие файлы."""
    cache_2d = load_mod.CACHE_DIR / "umap2d.npy"
    cache_10d = load_mod.CACHE_DIR / "umap10d.npy"
    if not force and cache_2d.exists() and cache_10d.exists():
        return np.load(cache_2d), np.load(cache_10d)

    vectors = data["vectors"]
    print(f"  UMAP 2D ({len(vectors):,} точек)…")
    umap2d = _project(vectors, n_components=2, min_dist=0.05)
    print("  UMAP 10D…")
    umap10d = _project(vectors, n_components=10, min_dist=0.0)

    load_mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_2d, umap2d)
    np.save(cache_10d, umap10d)
    _sketch(umap2d, data["years"])
    return umap2d, umap10d


def _sketch(umap2d: np.ndarray, years: np.ndarray) -> None:
    """Быстрый PNG-скетч проекции (ТЗ 8: промежуточная проверка глазами)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    load_mod.OUT_DIR.mkdir(parents=True, exist_ok=True)
    dated = years >= 0
    decade = years[dated] // 10 * 10
    fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
    sc = ax.scatter(
        umap2d[dated, 0], umap2d[dated, 1],
        c=decade, cmap="turbo", norm=Normalize(decade.min(), decade.max()),
        s=0.8, alpha=0.5, rasterized=True,
    )
    ax.scatter(umap2d[~dated, 0], umap2d[~dated, 1], c="#555555", s=0.8, alpha=0.5)
    plt.colorbar(sc, label="десятилетие")
    ax.set_facecolor("#111111")
    fig.patch.set_facecolor("#111111")
    ax.set_title("UMAP 2D, цвет — десятилетие", color="#dddddd")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(load_mod.OUT_DIR / "sketch_umap2d.png", facecolor="#111111")
    plt.close(fig)
    print("  Скетч проекции: out/sketch_umap2d.png")
