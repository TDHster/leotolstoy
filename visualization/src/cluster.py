"""Кластеризация: HDBSCAN по 10D-проекции (карта) и KMeans по эпохам (алювиальная).

Кэш: cache/hdbscan.json, cache/epochs.json, cache/links.json.
Отчёт с примерами текста на кластер (ТЗ 8) — cache/cluster_report.txt.
"""
from __future__ import annotations

import json
import textwrap

import hdbscan
import numpy as np
from sklearn.cluster import KMeans

from . import load as load_mod

HDBSCAN_CACHE = "hdbscan.json"
EPOCHS_CACHE = "epochs.json"
LINKS_CACHE = "links.json"
REPORT_CACHE = "cluster_report.txt"


def run(
    data: dict,
    umap10d: np.ndarray,
    force: bool = False,
    k_per_epoch: int = 6,
    link_threshold: float = 0.75,
    min_epoch_size: int = 300,
) -> tuple[dict, list, list]:
    hdb = _hdbscan(data, umap10d, force)
    epochs, links = _epochs(data, force, k_per_epoch, link_threshold, min_epoch_size)
    return hdb, epochs, links


# Число кластеров KMeans, когда HDBSCAN вырождается (см. _hdbscan).
GLOBAL_KMEANS_K = 24


def _hdbscan(data: dict, umap10d: np.ndarray, force: bool) -> dict:
    cache_path = load_mod.CACHE_DIR / HDBSCAN_CACHE
    if not force and cache_path.exists():
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    vectors = data["vectors"]
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=60, min_samples=10, metric="euclidean"
    )
    labels = clusterer.fit_predict(umap10d)
    noise_share = float((labels == -1).mean())
    used = 60
    method = f"hdbscan{used}"
    if noise_share > 0.6:
        # ТЗ 5.2: шум > 60% → попробовать min_cluster_size=30 и сообщить
        print(f"  Шум {noise_share:.0%} > 60% — пересчёт с min_cluster_size=30")
        labels = hdbscan.HDBSCAN(
            min_cluster_size=30, min_samples=10, metric="euclidean"
        ).fit_predict(umap10d)
        noise_share = float((labels == -1).mean())
        used = 30
        method = f"hdbscan{used}"
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # ТЗ 5.2, цель: 10–30 осмысленных кластеров. Корпус писем/дневников
    # однороден — HDBSCAN даёт либо один блоб (mcs=60), либо мелкую крошку
    # с шумом ~67% (mcs=30). Тогда падаем на KMeans по исходным 384D-векторам:
    # детерминированно, покрывает все точки, «Прочее» не нужно.
    if not (10 <= n_clusters <= 30 and noise_share <= 0.6):
        print(
            f"  HDBSCAN вырожден ({n_clusters} кластеров, шум {noise_share:.0%}) — "
            f"фолбэк на KMeans k={GLOBAL_KMEANS_K} по 384D-векторам"
        )
        labels = KMeans(
            n_clusters=GLOBAL_KMEANS_K, n_init=10, random_state=42
        ).fit_predict(vectors)
        noise_share = 0.0
        n_clusters = GLOBAL_KMEANS_K
        method = f"kmeans{GLOBAL_KMEANS_K}"

    # Центроиды глобальных кластеров (для цвета тем и примеров в label.py)
    centroids = {}
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        c = vectors[labels == cid].mean(axis=0)
        centroids[str(cid)] = (c / np.linalg.norm(c)).tolist()

    result = {
        "method": method,
        "labels": [int(x) for x in labels],
        "noise_share": noise_share,
        "min_cluster_size": used,
        "n_clusters": n_clusters,
        "centroids": centroids,
    }
    load_mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    _cluster_examples(result, data["chunks"])
    return result


def _epochs(
    data: dict,
    force: bool,
    k_per_epoch: int,
    link_threshold: float,
    min_epoch_size: int,
) -> tuple[list, list]:
    cache_epochs = load_mod.CACHE_DIR / EPOCHS_CACHE
    cache_links = load_mod.CACHE_DIR / LINKS_CACHE
    if not force and cache_epochs.exists() and cache_links.exists():
        with open(cache_epochs, encoding="utf-8") as f:
            epochs = json.load(f)
        with open(cache_links, encoding="utf-8") as f:
            links = json.load(f)
        return epochs, links

    vectors, years = data["vectors"], data["years"]
    dated = years >= 0
    min_year, max_year = int(years[dated].min()), int(years[dated].max())

    # Десятилетия; эпохи мельче min_epoch_size сливаются с предыдущей (ТЗ 5.3)
    raw = []  # [start, end, member indices]
    for d in range(min_year // 10 * 10, max_year // 10 * 10 + 10, 10):
        idx = np.flatnonzero((years >= d) & (years < d + 10))
        if not raw:
            raw.append([d, d + 9, idx])
        elif len(idx) < min_epoch_size:
            raw[-1][1] = d + 9
            raw[-1][2] = np.concatenate([raw[-1][2], idx])
        else:
            raw.append([d, d + 9, idx])

    # KMeans внутри эпохи на исходных 384D-векторах (ТЗ 5.3)
    epochs = []
    for e, (start, end, members) in enumerate(raw):
        k = min(k_per_epoch, len(members))
        km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(vectors[members])
        nodes = []
        for j in range(k):
            mem_j = members[km.labels_ == j].astype(int)
            c = vectors[mem_j].mean(axis=0)
            c = c / np.linalg.norm(c)
            nodes.append(
                {
                    "id": f"e{e}_k{j}",
                    "epoch": e,
                    "size": int(len(mem_j)),
                    "members": mem_j.tolist(),
                    "centroid": c.tolist(),
                    "years": [int(years[mem_j].min()), int(years[mem_j].max())],
                }
            )
        epochs.append({"epoch": e, "start": int(start), "end": int(end), "nodes": nodes})

    # Связи между соседними эпохами (ТЗ 5.3): косинус центроидов ≥ порога, ≤3 на узел
    links = []
    for t in range(len(epochs) - 1):
        src_nodes, dst_nodes = epochs[t]["nodes"], epochs[t + 1]["nodes"]
        S = np.array([n["centroid"] for n in src_nodes])  # уже нормализованы
        D = np.array([n["centroid"] for n in dst_nodes])
        sim = S @ D.T
        for i, node in enumerate(src_nodes):
            for j in np.argsort(-sim[i])[:3]:
                if sim[i, j] < link_threshold:
                    continue
                links.append(
                    {
                        "source": node["id"],
                        "target": dst_nodes[j]["id"],
                        "similarity": float(sim[i, j]),
                        "value": float(sim[i, j] * node["size"]),  # вес = близость × размер
                    }
                )

    load_mod.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_epochs, "w", encoding="utf-8") as f:
        json.dump(epochs, f, ensure_ascii=False)
    with open(cache_links, "w", encoding="utf-8") as f:
        json.dump(links, f, ensure_ascii=False)
    return epochs, links


def _cluster_examples(result: dict, chunks: list) -> None:
    """ТЗ 8: по 3 примера текста на кластер, чтобы проверить осмысленность глазами."""
    labels = np.array(result["labels"])
    lines = [f"=== Глобальные кластеры ({result['method']}): "
             f"{result['n_clusters']} (шум {result['noise_share']:.0%}) ==="]
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        members = np.flatnonzero(labels == cid)
        lines.append(f"\nКластер {cid} ({len(members):,} чанков):")
        for i in members[:3]:
            one_line = " ".join(chunks[i]["text"].split())
            lines.append("  • " + textwrap.shorten(one_line, width=150, placeholder="…"))
    report = "\n".join(lines)
    print(report)
    (load_mod.CACHE_DIR / REPORT_CACHE).write_text(report, encoding="utf-8")
