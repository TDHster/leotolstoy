"""Сборка out/index.html: вкладки «Карта смыслов» / «Потоки тем» с iframe.

Статическая страница: открывается двойным кликом (file://), без сервера.
Каждая вкладка — свой визуал в iframe + два предложения «как читать» (ТЗ 6).
"""
from __future__ import annotations

from . import load as load_mod

_TEMPLATE = """<!doctype html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Лев Толстой: эволюция тем</title>
<style>
  :root { color-scheme: dark; }
  body { margin: 0; background: #111111; color: #dddddd;
         font: 15px/1.5 -apple-system, "Segoe UI", Arial, sans-serif; }
  header { padding: 18px 24px 0; }
  h1 { margin: 0; font-size: 20px; font-weight: 600; color: #eeeeee; }
  p.lead { margin: 4px 0 10px; color: #999999; font-size: 13px; }
  .tabs { display: flex; gap: 6px; padding: 0 24px; }
  .tab { background: #1c1c1c; color: #bbbbbb; border: 1px solid #2e2e2e;
         border-bottom: none; border-radius: 8px 8px 0 0; padding: 8px 18px;
         cursor: pointer; font-size: 14px; }
  .tab.active { background: #262626; color: #ffffff; border-color: #3a3a3a; }
  .pane { display: none; }
  .pane.active { display: block; }
  iframe { width: 100%; height: calc(100vh - 160px); border: 0; display: block;
           background: #111111; }
  .hint { padding: 10px 24px 4px; color: #8a8a8a; font-size: 13px; max-width: 920px; }
</style>
</head>
<body>
<header>
  <h1>Лев Толстой: эволюция тем</h1>
  <p class="lead">Письма и дневники (Полное собрание сочинений, тт. 46–90) ·
  __N_CHUNKS__ фрагментов · __YEAR_SPAN__ · карта и потоки построены по эмбеддингам локальной модели</p>
</header>
<div class="tabs">
  <button class="tab active" id="tab-map" onclick="show('map')">Карта смыслов</button>
  <button class="tab" id="tab-alluvial" onclick="show('alluvial')">Потоки тем</button>
</div>
<div class="pane active" id="pane-map">
  <div class="hint">Каждая точка — фрагмент письма или дневника; близкие по смыслу лежат рядом.
  Выделите период на гистограмме внизу — карта покажет, какие области были активны в эти годы;
  наведите курсор на точку, чтобы прочесть фрагмент.</div>
  <iframe src="map.html" title="Карта смыслов"></iframe>
</div>
<div class="pane" id="pane-alluvial">
  <div class="hint">Колонки — десятилетия, внутри каждой — темы этой эпохи; ленты — как темы
  перетекают из десятилетия в десятилетие. Толщина ленты — вес связи; наведите курсор на узел,
  чтобы увидеть описание темы и число фрагментов.</div>
  <iframe src="alluvial.html" title="Потоки тем"></iframe>
</div>
<script>
function show(name) {
  document.getElementById('pane-map').classList.toggle('active', name === 'map');
  document.getElementById('pane-alluvial').classList.toggle('active', name === 'alluvial');
  document.getElementById('tab-map').classList.toggle('active', name === 'map');
  document.getElementById('tab-alluvial').classList.toggle('active', name === 'alluvial');
}
</script>
</body>
</html>
"""


def run(data: dict, force: bool = False) -> dict:
    out_path = load_mod.OUT_DIR / "index.html"
    if not force and out_path.exists():
        return {"file": str(out_path)}
    years = data["years"]
    dated = years[years >= 0]
    html = (
        _TEMPLATE
        .replace("__N_CHUNKS__", f"{len(data['chunks']):,}")
        .replace("__YEAR_SPAN__", f"{dated.min()}–{dated.max()}")
    )
    load_mod.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return {"file": str(out_path), "size_kb": len(html) // 1024}
