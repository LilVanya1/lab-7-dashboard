# -*- coding: utf-8 -*-
"""Экспорт графиков для отчёта laba7 (1.png … 4.png) без GUI."""
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import dashboard as d

OUT = ROOT / "report" / "img"


class _Var:
    def __init__(self, v):
        self._v = v

    def get(self):
        return self._v


class _Canvas:
    def draw_idle(self):
        pass


def _setup(**kw):
    d.fig = plt.Figure(figsize=(10, 6), dpi=120)
    d.canvas = _Canvas()
    d.df_raw = d.load_dataset()
    d.athlete_var = _Var(kw.get("athlete", "Все"))
    d.zone_var = _Var(kw.get("zone", "Все"))
    d.pace_limit_var = _Var(kw.get("pace", 12.0))
    d.roll_window_var = _Var(kw.get("roll", 10))
    d.current_chart = kw.get("chart", "line")
    d.agg_mode = kw.get("agg", "mean")
    plt.rcParams["font.family"] = ["Segoe UI", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    import seaborn as sns
    sns.set_theme(style="whitegrid")
    d.preprocess_data()


def _save(name: str):
    d.fig.savefig(OUT / name, dpi=150, bbox_inches="tight", facecolor="white")
    print("saved", name)


def main():
    _setup(chart="line")
    d.draw_line()
    _save("1.png")

    _setup(chart="bar", agg="mean")
    d.draw_bar()
    _save("2.png")

    _setup(chart="scatter")
    d.draw_scatter()
    _save("3.png")

    _setup(chart="heatmap")
    d.draw_heatmap()
    _save("4.png")

    _setup(chart="line", athlete="46", zone="3")
    d.draw_line()
    _save("5.png")


if __name__ == "__main__":
    main()
