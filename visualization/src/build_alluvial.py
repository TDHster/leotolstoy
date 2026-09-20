"""Алювиальная диаграмма: Plotly Sankey → out/alluvial.html (тёмная тема).

Колонки узлов строго по эпохам слева направо (x-координаты), подписи эпох — над
колонками, цвет узла — по ближайшему глобальному кластеру (ТЗ 5.6), поэтому
родственные темы в разных эпохах одного цвета; карта красится той же палитрой.
"""
from __future__ import annotations

import json
import textwrap

import numpy as np
import plotly.graph_objects as go

from . import load as load_mod
from . import palette

BG = "#111111"
INK = "#e5e5e5"


def _wrap(text: str, width: int = 16) -> str:
    """Перенос длинных подписей узлов (ТЗ 5.6)."""
    parts = textwrap.wrap(text, width=width, break_long_words=True, break_on_hyphens=False)
    return "<br>".join(parts)


def run(data: dict, force: bool = False) -> dict:
    out_path = load_mod.OUT_DIR / "alluvial.html"
    if not force and out_path.exists():
        return {"file": str(out_path)}

    with open(load_mod.CACHE_DIR / "epochs.json", encoding="utf-8") as f:
        epochs = json.load(f)
    with open(load_mod.CACHE_DIR / "links.json", encoding="utf-8") as f:
        links = json.load(f)
    with open(load_mod.CACHE_DIR / "labels_epoch.json", encoding="utf-8") as f:
        labels_epoch = json.load(f)
    with open(load_mod.CACHE_DIR / "hdbscan.json", encoding="utf-8") as f:
        hdb = json.load(f)

    hdb_centroids = {int(cid): np.array(v) for cid, v in hdb["centroids"].items()}

    def node_color(node: dict) -> str:
        c = np.array(node["centroid"])
        best = max(hdb_centroids, key=lambda cid: float(c @ hdb_centroids[cid]))
        return palette.color_for(best)

    nodes = [n for e in epochs for n in e["nodes"]]
    id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
    n_epochs = len(epochs)
    x_step = 1.0 / (n_epochs - 1) if n_epochs > 1 else 0.5
    # y равномерно внутри колонки — стабильная раскладка без сюрпризов plotly
    y_by_node = {}
    for e in epochs:
        k = len(e["nodes"])
        for j, n in enumerate(e["nodes"]):
            y_by_node[n["id"]] = (j + 1) / (k + 1)

    fig = go.Figure(
        go.Sankey(
            arrangement="fixed",
            node=dict(
                label=[_wrap(labels_epoch[n["id"]]["title"]) for n in nodes],
                x=[n["epoch"] * x_step for n in nodes],
                y=[y_by_node[n["id"]] for n in nodes],
                color=[node_color(n) for n in nodes],
                pad=30,
                thickness=16,
                line=dict(color=BG, width=0),
                customdata=[
                    [labels_epoch[n["id"]]["description"], n["size"],
                     f"{n['years'][0]}–{n['years'][1]}"]
                    for n in nodes
                ],
                hovertemplate=(
                    "%{label}<br>%{customdata[0]}<br>"
                    "Фрагментов: %{customdata[1]:,}<br>Годы: %{customdata[2]}"
                    "<extra></extra>"
                ),
            ),
            link=dict(
                source=[id_to_idx[l["source"]] for l in links],
                target=[id_to_idx[l["target"]] for l in links],
                value=[l["value"] for l in links],
                color=[palette.rgba(node_color(nodes[id_to_idx[l["source"]]]), 0.3) for l in links],
            ),
        )
    )

    # Подписи эпох над колонками + заголовок (в полосе отступа сверху)
    max_year = int(data["years"][data["years"] >= 0].max())
    for e in epochs:
        end = min(int(e["end"]), max_year)  # хвост «1910–1919» → «1910»
        fig.add_annotation(
            x=e["epoch"] * x_step, y=0.955, xref="paper", yref="paper",
            text=f"{e['start']}–{end}", showarrow=False,
            font=dict(size=13, color="#aaaaaa"),
        )
    fig.add_annotation(
        x=0.5, y=1.0, xref="paper", yref="paper",
        text="Потоки тем: перетекание тем из десятилетия в десятилетие",
        showarrow=False, font=dict(size=18, color=INK),
    )

    fig.update_layout(
        paper_bgcolor=BG,
        font=dict(family='-apple-system, "Segoe UI", Arial, sans-serif', size=12, color="#c9c9c9"),
        height=860,
        margin=dict(t=100, b=10, l=10, r=10),
        hoverlabel=dict(bgcolor="#222222", font_color=INK, bordercolor="#444444"),
    )
    load_mod.OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs=True)  # встроенный JS — работает с file://
    return {"file": str(out_path), "size_kb": out_path.stat().st_size // 1024}
