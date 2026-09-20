"""Карта смыслов: DataMapPlot → out/map.html (тёмная тема, поиск, фильтр по годам).

Написано под фактический API datamapplot 0.7.x (deck.gl-рендер): текст метки —
это сама метка (массив строк на точку), отдельного label_text нет; поиск
включается enable_search=True; title/sub_title/гистограмма — параметры
render_html, пробрасываются из create_interactive_plot через **kwargs.
"""
from __future__ import annotations

import json
import re

import datamapplot
import numpy as np

from . import load as load_mod
from . import palette

HOVER_CHARS = 250
NOISE_LABEL = "Прочее"


def _hover_texts(chunks: list, years: np.ndarray, label_of: np.ndarray,
                 descriptions: dict[str, str]) -> list[str]:
    out = []
    for c, y, lab in zip(chunks, years, label_of):
        meta = c["meta"]
        parts = [f"{int(y) if y >= 0 else '—'}", meta.get("type", "")]
        to = meta.get("to")
        if to:
            parts.append(to)
        place = meta.get("place")
        if place and place != to:
            parts.append(place)
        text = " ".join(c["text"].split())
        if len(text) > HOVER_CHARS:
            text = text[:HOVER_CHARS] + "…"
        head = " · ".join(p for p in parts if p)
        desc = descriptions.get(lab, "")
        out.append(f"{head}\n{text}\nТема: {desc}" if desc else f"{head}\n{text}")
    return out


def run(data: dict, umap2d: np.ndarray, force: bool = False, standalone: bool = False) -> dict:
    out_path = load_mod.OUT_DIR / "map.html"
    if not force and out_path.exists():
        return {"file": str(out_path)}

    with open(load_mod.CACHE_DIR / "hdbscan.json", encoding="utf-8") as f:
        hdb = json.load(f)
    with open(load_mod.CACHE_DIR / "labels_global.json", encoding="utf-8") as f:
        labels_global = json.load(f)

    chunks, years = data["chunks"], data["years"]
    dated = years[years >= 0]
    min_year, max_year = int(dated.min()), int(dated.max())

    # Метка точки = название темы (текст метки — сама метка в datamapplot 0.7.x)
    titles = {cid: lab["title"] for cid, lab in labels_global.items()}
    descriptions = {lab["title"]: lab["description"] for lab in labels_global.values()}
    cluster_ids = np.array(hdb["labels"], dtype=object)
    label_texts = np.array(
        [titles[str(c)] if c != -1 else NOISE_LABEL for c in cluster_ids], dtype=object
    )
    label_color_map = {
        titles[cid]: palette.color_for(int(cid)) for cid in titles
    }

    # Гистограмма по годам. Без histogram_range: с явным range числовой путь
    # generate_bins_from_numeric_data в datamapplot 0.7.3 падает (edges на 1
    # короче индекса DataFrame) — баг их кода. Чанки без даты (-1) прижимаем
    # к min_year: они попадают в первый бин вместе с самым ранним периодом.
    hist_years = np.where(years < 0, min_year, years).astype(np.int64)

    # datamapplot 0.7.x возвращает InteractiveFigure (HTML в str()/методе save)
    plot = datamapplot.create_interactive_plot(
        umap2d.astype(np.float32),
        label_texts,  # *label_layers: массивы строк-меток на точку
        hover_text=_hover_texts(chunks, years, label_texts, descriptions),
        label_color_map=label_color_map,
        label_wrap_width=18,
        noise_label=NOISE_LABEL,
        noise_color=palette.NOISE,
        darkmode=True,
        # параметры render_html (пробрасываются через **kwargs):
        title="Лев Толстой: карта смыслов",
        sub_title=f"{len(chunks):,} фрагментов писем и дневников · {min_year}–{max_year}",
        enable_search=True,          # поиск по hover_text
        search_field="hover_text",
        histogram_data=hist_years,
        histogram_n_bins=max_year - min_year + 1,  # 1 бин = 1 год
    )
    html = str(plot)
    load_mod.OUT_DIR.mkdir(parents=True, exist_ok=True)
    if hasattr(plot, "save"):
        plot.save(str(out_path))
    else:
        out_path.write_text(html, encoding="utf-8")

    # JS-библиотеки datamapplot 0.7.x грузятся с CDN (unpkg) — для file:// нужен
    # интернет; данные встроены в файл (inline_data=True по умолчанию).
    external = sorted(set(re.findall(r'src="(https?://[^"]+)"', html)))
    if standalone:
        print("  --standalone: данные встроены; внешние скрипты: "
              + (", ".join(external) if external else "нет"))
    return {"file": str(out_path), "size_kb": len(html) // 1024}
