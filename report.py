#!/usr/bin/env python
"""Descarga los datos más recientes, entrena y escribe METRICS.md (visible en GitHub)."""
from pathlib import Path

from fmp.pipeline import load_data, metrics_markdown, train

if __name__ == "__main__":
    tr = train(load_data())
    out = Path(__file__).with_name("METRICS.md")
    out.write_text(metrics_markdown(tr), encoding="utf-8")
    print(f"Escrito {out}")
