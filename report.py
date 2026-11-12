#!/usr/bin/env python
"""Descarga los datos más recientes, entrena y escribe METRICS.md (visible en GitHub).

Entrena dos veces, sin y con datos de jugadores, para medir cuánto aportan las alineaciones.
"""
from pathlib import Path

from fmp.pipeline import load_data, load_players, metrics_markdown, player_impact, train

ROOT = Path(__file__).parent

if __name__ == "__main__":
    data = load_data()
    players = load_players(data, verbose=True)
    tr = train(data, players=players)
    impact = player_impact(tr, train(data)) if tr["book"] else None
    if impact is not None:
        impact.to_json(ROOT / "player_impact.json", orient="records", indent=1, force_ascii=False)
    out = ROOT / "METRICS.md"
    out.write_text(metrics_markdown(tr, impact=impact), encoding="utf-8")
    print(f"Escrito {out}")
