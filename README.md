# Лабораторная 7 — интерактивный дашборд

**Студент:** Изместьев М.Н., гр. ИВТб-1302-06-00  
**Вариант:** 10 — спортивная аналитика (тот же `data.csv`, что ЛР 6)  
**GitHub (upload):** `lab-7-dashboard`

## Статус

| Компонент | Статус |
|-----------|--------|
| `dashboard.py` (4 графика + фильтры) | ✅ |
| `export_screenshots.py` | ✅ |
| Отчёт `report/main.tex` | ✅ |
| Скрины `report/img/1.png`…`5.png` | ✅ |
| `data.csv` (корень `proga/` или `LABA_6/`) | ✅ |
| Ссылка на GitHub в отчёте | ❌ |

## Стек

pandas + seaborn + matplotlib + tkinter (процедурный стиль).

## Структура

```
laba7/
├── dashboard.py
├── export_screenshots.py
├── drawio/              # вспомогательные схемы (не в отчёт)
├── REPORT_7.md          # черновик текста отчёта
└── report/
    ├── main.tex
    └── img/1.png … 5.png
```

## Запуск

```powershell
cd c:\Users\stud222640\Documents\proga
.\.venv\Scripts\python.exe laba7\dashboard.py
```

**В GUI:** переключение графиков, фильтры (спортсмен / зона / темп / окно k), «Обновить», «Экспорт» PNG/PDF.

## Скрины для отчёта

Автогенерация:
```powershell
python laba7\export_screenshots.py
```

| # | Содержание |
|---|------------|
| 1 | Line — скользящая средняя скорости |
| 2 | Bar — калории по зонам (среднее) |
| 3 | Scatter — темп vs cal/km |
| 4 | Heatmap — зона × дистанция |
| 5 | Line — фильтр: спортсмен 46, зона 3 |

## Предобработка (`preprocess_data`)

- фильтрация NaN/Inf и отрицательных значений;
- `speed_kmh`, `cal_per_km`;
- IQR по группам `athlete_id`;
- `pd.cut` дистанции, `rolling(k)`, `resample('h')`.

## Сдача

PDF из `report/main.tex` + ссылка на репозиторий.
