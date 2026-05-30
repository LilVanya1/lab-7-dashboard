"""
Лабораторная работа 7: Создание интерактивного дашборда.
Вариант 10 — Спортивная аналитика.
Стек: pandas + seaborn + matplotlib + tkinter (процедурный стиль, без классов).
"""

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# ─── Глобальные переменные ──────────────────────────────────────────────────
df_raw   = None
df_work  = None
fig      = plt.Figure(figsize=(10, 6), dpi=100)
canvas   = None
root     = None

current_chart = "line"
agg_mode      = "mean"

athlete_var      = None
zone_var         = None
pace_limit_var   = None
roll_window_var  = None
status_var       = None

AGG_LABELS = {"mean": "среднее", "sum": "сумма", "median": "медиана"}


# ─── Русский тулбар (переопределяем подсказки и диалог настройки) ────────────
class RussianToolbar(NavigationToolbar2Tk):
    toolitems = (
        ("Начало",    "Начальный вид",             "home",         "home"),
        ("Назад",     "Предыдущий вид",            "back",         "back"),
        ("Вперёд",    "Следующий вид",             "forward",      "forward"),
        (None, None, None, None),
        ("Движение",  "Перемещение осей",          "move",         "pan"),
        ("Масштаб",   "Увеличить область",         "zoom_to_rect", "zoom"),
        ("Поля",      "Настроить поля графика",    "subplots",     "configure_subplots"),
        (None, None, None, None),
        ("Сохранить", "Сохранить рисунок",         "filesave",     "save_figure"),
    )

    def configure_subplots(self):
        """Русский диалог настройки полей (замена английского SubplotTool)."""
        win = tk.Toplevel(self.canvas.get_tk_widget())
        win.title("Настройка полей графика")
        win.resizable(False, False)
        win.configure(bg="#f3f5f7")

        sp = self.canvas.figure.subplotpars
        params = [
            ("Лево",          "left",   0.0, 1.0, sp.left),
            ("Низ",           "bottom", 0.0, 1.0, sp.bottom),
            ("Право",         "right",  0.0, 1.0, sp.right),
            ("Верх",          "top",    0.0, 1.0, sp.top),
            ("Зазор по ширине","wspace",0.0, 1.0, sp.wspace),
            ("Зазор по высоте","hspace",0.0, 1.0, sp.hspace),
        ]

        vars_ = {}
        for row, (label, key, lo, hi, init) in enumerate(params):
            tk.Label(win, text=label, width=18, anchor="e", bg="#f3f5f7").grid(
                row=row, column=0, padx=8, pady=4)
            var = tk.DoubleVar(value=round(init, 3))
            vars_[key] = var

            def _on_change(val, k=key, v=var):
                self.canvas.figure.subplots_adjust(**{k: float(v.get())})
                self.canvas.draw_idle()

            tk.Scale(
                win, variable=var, from_=lo, to=hi, resolution=0.001,
                orient=tk.HORIZONTAL, length=280, command=_on_change,
                bg="#f3f5f7",
            ).grid(row=row, column=1, padx=8, pady=4)
            lbl = tk.Label(win, textvariable=var, width=7, bg="#f3f5f7")
            lbl.grid(row=row, column=2, padx=4)

        def _reset():
            defaults = dict(left=0.125, bottom=0.11, right=0.9,
                            top=0.88, wspace=0.2, hspace=0.2)
            for k, var in vars_.items():
                var.set(defaults[k])
            self.canvas.figure.subplots_adjust(**defaults)
            self.canvas.draw_idle()

        tk.Button(win, text="Сбросить", command=_reset, width=14).grid(
            row=len(params), column=1, pady=10)


# ─── Загрузка данных ─────────────────────────────────────────────────────────
def load_dataset() -> pd.DataFrame:
    local  = Path("data.csv")
    parent = Path(__file__).resolve().parent.parent / "data.csv"
    source = local if local.exists() else parent
    if not source.exists():
        raise FileNotFoundError(
            "Файл data.csv не найден (ни в laba7/, ни в корне проекта)")
    return pd.read_csv(source)


# ─── Вспомогательная функция: IQR-обрезка по группам ────────────────────────
def grouped_iqr_clip(series: pd.Series, group_key: pd.Series) -> pd.Series:
    tmp = pd.DataFrame({"x": series, "g": group_key})
    q1  = tmp.groupby("g")["x"].transform(lambda s: s.quantile(0.25))
    q3  = tmp.groupby("g")["x"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    return tmp["x"].clip(lower=q1 - 1.5 * iqr, upper=q3 + 1.5 * iqr)


# ─── Этап 2: Предобработка и Feature Engineering ─────────────────────────────
def preprocess_data() -> pd.DataFrame:
    global df_work

    work = df_raw.copy()  # Изолируем изменения

    # 1. Фильтрация по условию варианта (очистка физически невозможных значений)
    work = work[
        np.isfinite(work["dist"]) &
        np.isfinite(work["pace"]) &
        np.isfinite(work["cal"])
    ].copy()
    work = work[(work["dist"] >= 0) & (work["pace"] >= 0)].copy()
    work["cal"] = np.where(work["cal"] < 0, 0.0, work["cal"])

    # Динамические фильтры из UI (df.query-логика через .loc)
    selected_athlete = athlete_var.get()
    selected_zone    = zone_var.get()
    max_pace         = float(pace_limit_var.get())

    if selected_athlete != "Все":
        work = work.loc[work["athlete_id"] == int(selected_athlete)].copy()
    if selected_zone != "Все":
        work = work.loc[work["zone"] == int(selected_zone)].copy()
    work = work.loc[work["pace"] <= max_pace].copy()

    # 2. Безопасное вычисление производных признаков
    work["speed_kmh"]  = work["dist"] / np.maximum(work["pace"] / 60.0, 1e-8)
    work["cal_per_km"] = work["cal"]  / np.maximum(work["dist"], 1e-8)

    # 3. Обрезка выбросов — IQR по группам (athlete_id)
    work["pace"]       = grouped_iqr_clip(work["pace"],       work["athlete_id"])
    work["cal_per_km"] = grouped_iqr_clip(work["cal_per_km"], work["athlete_id"])

    # 4. Оптимизация категориальных полей + pd.cut() биннинг дистанции
    work["athlete_id"] = work["athlete_id"].astype("category")
    work["zone"]       = work["zone"].astype("category")
    work["dist_bin"]   = pd.cut(
        work["dist"],
        bins=[0, 2, 5, 10, 20, np.inf],
        labels=["<2 км", "2–5 км", "5–10 км", "10–20 км", ">20 км"],
        right=False,
    )

    # Временная ось + скользящее среднее (df.rolling)
    work["dt"] = pd.to_datetime(work["ts"], unit="s", errors="coerce")
    work = work.dropna(subset=["dt"]).sort_values("dt")
    k = max(2, int(roll_window_var.get()))
    work["speed_roll"] = work["speed_kmh"].rolling(k, min_periods=1).mean()

    df_work = work
    return work


# ─── Этап 4: Функции отрисовки графиков ──────────────────────────────────────
def clear_figure() -> None:
    fig.clf()


def draw_line() -> None:
    """Линейный: скользящая средняя скорости (df.resample по часам)."""
    clear_figure()
    ax = fig.add_subplot(111)
    data = (
        df_work.set_index("dt")
               .resample("h")["speed_roll"]
               .mean()
               .dropna()
    )
    sns.lineplot(x=data.index, y=data.values, ax=ax, color="#2E86DE")
    ax.set_title("Скользящая средняя скорости (почасово)")
    ax.set_xlabel("Время")
    ax.set_ylabel("Скорость, км/ч")
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    canvas.draw_idle()


def draw_bar() -> None:
    """Столбчатый: калории по зонам пульса (агрегация — переключатель)."""
    clear_figure()
    ax = fig.add_subplot(111)
    grouped = df_work.groupby("zone", observed=True)["cal"]
    if agg_mode == "sum":
        s = grouped.sum()
    elif agg_mode == "median":
        s = grouped.median()
    else:
        s = grouped.mean()
    sns.barplot(x=s.index.astype(str), y=s.values, ax=ax,
                hue=s.index.astype(str), palette="viridis",
                legend=False, errorbar=None)
    ax.set_title(f"Калории по зонам пульса ({AGG_LABELS[agg_mode]})")
    ax.set_xlabel("Зона пульса")
    ax.set_ylabel("Калории")
    fig.tight_layout()
    canvas.draw_idle()


def draw_scatter() -> None:
    """Точечный: темп vs калории/км, цвет — зона пульса."""
    clear_figure()
    ax = fig.add_subplot(111)
    sample = df_work.sample(min(2500, len(df_work)), random_state=42)
    sns.scatterplot(
        data=sample,
        x="pace", y="cal_per_km", hue="zone",
        alpha=0.55, s=25, ax=ax,
    )
    ax.set_title("Темп и калории на километр")
    ax.set_xlabel("Темп, мин/км")
    ax.set_ylabel("Калории/км")
    fig.tight_layout()
    canvas.draw_idle()


def draw_heatmap() -> None:
    """Тепловая карта: средняя скорость (pivot_table: зона × бин дистанции)."""
    clear_figure()
    ax = fig.add_subplot(111)
    pivot = pd.pivot_table(
        df_work,
        index="zone",
        columns="dist_bin",
        values="speed_kmh",
        aggfunc="mean",
        observed=True,
    )
    if pivot.empty:
        ax.text(0.5, 0.5, "Нет данных для тепловой карты",
                ha="center", va="center")
        ax.axis("off")
    else:
        sns.heatmap(pivot, cmap="coolwarm", ax=ax,
                    cbar_kws={"label": "км/ч"}, annot=False)
    ax.set_title("Средняя скорость: зона × дистанция")
    ax.set_xlabel("Диапазон дистанции")
    ax.set_ylabel("Пульсовая зона")
    fig.tight_layout()
    canvas.draw_idle()


# ─── Маршрутизация и обновление ──────────────────────────────────────────────
def redraw_current_chart() -> None:
    if df_work is None or df_work.empty:
        clear_figure()
        ax = fig.add_subplot(111)
        ax.text(0.5, 0.5, "Нет данных после фильтрации",
                ha="center", va="center", fontsize=14)
        ax.axis("off")
        fig.tight_layout()
        canvas.draw_idle()
        return

    dispatch = {
        "line":    draw_line,
        "bar":     draw_bar,
        "scatter": draw_scatter,
        "heatmap": draw_heatmap,
    }
    dispatch.get(current_chart, draw_line)()


# ─── Этап 5: Панель управления и интерактивность ─────────────────────────────
def refresh_data() -> None:
    data = preprocess_data()
    n = len(data)
    u = data["athlete_id"].nunique() if n else 0
    status_var.set(f"Строк: {n}  |  Уникальных спортсменов: {u}  |  Тип графика: {current_chart}")
    redraw_current_chart()


def set_chart(name: str) -> None:
    global current_chart
    current_chart = name
    redraw_current_chart()


def set_agg(mode: str) -> None:
    global agg_mode
    agg_mode = mode
    if current_chart == "bar":
        redraw_current_chart()


def export_plot() -> None:
    filepath = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[("PNG изображение", "*.png"), ("PDF документ", "*.pdf")],
        title="Сохранить график",
    )
    if filepath:
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        messagebox.showinfo("Экспорт", f"График сохранён:\n{filepath}")


# ─── Этап 3: Встраивание Figure в Tkinter — построение GUI ───────────────────
def build_ui() -> None:
    global root, canvas, athlete_var, zone_var, \
           pace_limit_var, roll_window_var, status_var

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"]       = ["Segoe UI", "Arial", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    root = tk.Tk()
    root.title("Лабораторная 7 — Интерактивный дашборд (Вариант 10)")
    root.geometry("1280x820")
    root.configure(bg="#f3f5f7")

    # ── Переменные UI ──
    athlete_var     = tk.StringVar(value="Все")
    zone_var        = tk.StringVar(value="Все")
    pace_limit_var  = tk.DoubleVar(value=12.0)
    roll_window_var = tk.IntVar(value=10)
    status_var      = tk.StringVar(value="Готово")

    athlete_values = ["Все"] + sorted(
        df_raw["athlete_id"].dropna().astype(int).astype(str).unique().tolist())
    zone_values    = ["Все"] + sorted(
        df_raw["zone"].dropna().astype(int).astype(str).unique().tolist())

    # ── Верхняя панель фильтров ──
    top = tk.Frame(root, bg="#f3f5f7")
    top.pack(side=tk.TOP, fill=tk.X, padx=12, pady=8)

    tk.Label(top, text="Спортсмен:", bg="#f3f5f7").pack(side=tk.LEFT)
    ttk.Combobox(top, values=athlete_values, textvariable=athlete_var,
                 width=8, state="readonly").pack(side=tk.LEFT, padx=6)

    tk.Label(top, text="Зона:", bg="#f3f5f7").pack(side=tk.LEFT)
    ttk.Combobox(top, values=zone_values, textvariable=zone_var,
                 width=6, state="readonly").pack(side=tk.LEFT, padx=6)

    tk.Label(top, text="Макс. темп:", bg="#f3f5f7").pack(side=tk.LEFT, padx=(12, 2))
    tk.Scale(top, variable=pace_limit_var, from_=3.0, to=15.0,
             resolution=0.1, orient=tk.HORIZONTAL, length=190,
             bg="#f3f5f7").pack(side=tk.LEFT)

    tk.Label(top, text="Окно сглаживания k:", bg="#f3f5f7").pack(side=tk.LEFT, padx=(12, 2))
    tk.Scale(top, variable=roll_window_var, from_=2, to=60,
             resolution=1, orient=tk.HORIZONTAL, length=160,
             bg="#f3f5f7").pack(side=tk.LEFT)

    tk.Button(top, text="  Экспорт  ", command=export_plot).pack(side=tk.RIGHT, padx=4)
    tk.Button(top, text="  Обновить  ", command=refresh_data,
              bg="#2E86DE", fg="white", relief=tk.FLAT).pack(side=tk.RIGHT, padx=4)

    # ── Панель переключения типа графика и агрегации ──
    mid = tk.Frame(root, bg="#e8edf2", pady=5)
    mid.pack(side=tk.TOP, fill=tk.X, padx=0)

    tk.Label(mid, text="График:", bg="#e8edf2", font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(12, 4))
    for label, key in [("Линейный", "line"), ("Столбчатый", "bar"),
                        ("Точечный", "scatter"), ("Тепловая карта", "heatmap")]:
        tk.Button(mid, text=label, width=14,
                  command=lambda k=key: set_chart(k)).pack(side=tk.LEFT, padx=3)

    tk.Label(mid, text="  Агрегация:", bg="#e8edf2",
             font=("Segoe UI", 9, "bold")).pack(side=tk.LEFT, padx=(16, 4))
    for label, val in [("Среднее", "mean"), ("Сумма", "sum"), ("Медиана", "median")]:
        tk.Radiobutton(mid, text=label, value=val,
                       command=lambda v=val: set_agg(v),
                       bg="#e8edf2").pack(side=tk.LEFT)

    # ── Область графика ──
    plot_frame = tk.Frame(root, bg="white", relief=tk.SUNKEN, bd=1)
    plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=12, pady=6)

    canvas_widget = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas_widget.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    global canvas
    canvas = canvas_widget

    toolbar = RussianToolbar(canvas, plot_frame)
    toolbar.update()
    toolbar.pack(side=tk.TOP, fill=tk.X)

    # ── Строка статуса ──
    tk.Label(root, textvariable=status_var, anchor="w",
             bg="#d0d7df", font=("Segoe UI", 9)).pack(
        side=tk.BOTTOM, fill=tk.X, ipady=3)


# ─── Точка входа ─────────────────────────────────────────────────────────────
def main() -> None:
    global df_raw
    df_raw = load_dataset()
    build_ui()
    refresh_data()
    root.mainloop()


if __name__ == "__main__":
    main()
