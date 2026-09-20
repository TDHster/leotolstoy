#!/usr/bin/env python3
"""Сборка визуализации: точка входа, шаги с кэшем (повторный запуск — секунды).

Каждый тяжёлый шаг (UMAP, кластеризация, c-TF-IDF, LLM-названия) кэшируется в
cache/; --force STEP сбрасывает кэш конкретного шага (ТЗ 4, 7).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from src import build_alluvial, build_map, build_site, cluster, label, load, project

# шаг → какие файлы кэша и out/ он порождает (для --force)
FORCE_TARGETS = {
    "project": (["umap2d.npy", "umap10d.npy"], ["sketch_umap2d.png"]),
    "cluster": (["hdbscan.json", "epochs.json", "links.json", "cluster_report.txt"], []),
    "ctfidf": (["ctfidf.json"], []),
    "label": (["labels_global.json", "labels_epoch.json"], []),
    "map": ([], ["map.html"]),
    "alluvial": ([], ["alluvial.html"]),
    "site": ([], ["index.html"]),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--force", default="",
        help="пересчитать шаги заново: project,cluster,ctfidf,label,map,alluvial,site "
        "(через запятую) или all",
    )
    ap.add_argument("--k-per-epoch", type=int, default=6, help="кластеров KMeans на эпоху (ТЗ 5.3)")
    ap.add_argument("--link-threshold", type=float, default=0.75,
                    help="минимум косинусной близости центроидов для связи (ТЗ 5.3)")
    ap.add_argument("--min-epoch-size", type=int, default=300,
                    help="эпохи меньше этого числа чанков сливаются с предыдущей (ТЗ 5.3)")
    ap.add_argument("--split", action="store_true",
                    help="только out/map.html и out/alluvial.html, без index.html-обёртки (ТЗ 6)")
    ap.add_argument("--standalone", action="store_true",
                    help="встроить данные карты в HTML (offline_mode datamapplot)")
    args = ap.parse_args()

    force_steps = {s for s in args.force.split(",") if s} if args.force else set()
    if "all" in force_steps:
        force_steps = set(FORCE_TARGETS)
    for step in force_steps:
        if step not in FORCE_TARGETS:
            ap.error(f"неизвестный шаг: {step} (допустимо: {', '.join(FORCE_TARGETS)}, all)")
    for step in force_steps:
        for name in FORCE_TARGETS[step][0]:
            (load.CACHE_DIR / name).unlink(missing_ok=True)
        for name in FORCE_TARGETS[step][1]:
            (load.OUT_DIR / name).unlink(missing_ok=True)

    timings: dict[str, float] = {}

    def timed(name: str) -> None:
        timings[name] = time.time()

    def done(name: str) -> None:
        timings[name] = time.time() - timings[name]

    # 1. Данные (всегда; быстро, только чтение)
    timed("load")
    data = load.load_data()
    summary = load.summarize(data)
    print(summary)
    load.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (load.CACHE_DIR / "data_report.txt").write_text(summary, encoding="utf-8")
    done("load")

    # 2. Проекции UMAP 2D + 10D
    timed("project")
    umap2d, umap10d = project.run(data, force="project" in force_steps)
    done("project")

    # 3. Кластеризация: HDBSCAN + эпохи + связи
    timed("cluster")
    hdb, epochs, links = cluster.run(
        data, umap10d,
        force="cluster" in force_steps,
        k_per_epoch=args.k_per_epoch,
        link_threshold=args.link_threshold,
        min_epoch_size=args.min_epoch_size,
    )
    done("cluster")

    # 4. c-TF-IDF (топ-термы для кластеров и узлов)
    timed("ctfidf")
    ctfidf = label.run_ctfidf(data, force="ctfidf" in force_steps)
    done("ctfidf")

    # 5. Названия тем от LLM
    timed("label")
    labels_global, labels_epoch, llm_stats = label.run_labels(
        data, ctfidf, force="label" in force_steps
    )
    done("label")

    # 6. Карта смыслов
    timed("map")
    map_result = build_map.run(
        data, umap2d, force="map" in force_steps, standalone=args.standalone
    )
    done("map")

    # 7. Алювиальная диаграмма
    timed("alluvial")
    alluvial_result = build_alluvial.run(data, force="alluvial" in force_steps)
    done("alluvial")

    # 8. Обёртка с вкладками (если не --split)
    if not args.split:
        timed("site")
        site_result = build_site.run(data, force="site" in force_steps)
        done("site")

    # Отчёт (ТЗ, критерий 6)
    print("\n=== Отчёт ===")
    print(
        f"Глобальные кластеры ({hdb['method']}): {hdb['n_clusters']} · "
        f"шум {hdb['noise_share']:.1%} · эпох: {len(epochs)} · "
        f"узлов: {sum(len(e['nodes']) for e in epochs)} · связей: {len(links)}"
    )
    print(
        f"LLM-вызовов: {llm_stats['calls']} · фолбэков: {llm_stats['fallbacks']}"
    )
    for name, dt in timings.items():
        print(f"  {name}: {dt:.1f} с")
    if not args.split:
        print(f"\nГотово: {site_result['file']} — откройте двойным кликом")
    else:
        print(f"\nГотово (--split): {map_result['file']}, {alluvial_result['file']}")


if __name__ == "__main__":
    main()
