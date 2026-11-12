"""Simula un parque de baterias planificado por LP sobre varios dias y compara antes/despues.

    uv run python scripts/06_simula_lp.py

Periodo, GW, tramos y bordes en config/base.yaml (seccion simulacion_lp); bateria en la seccion
bateria. Hacen falta las curvas de todos los dias (scripts/02_descarga_curvas.py). Los supuestos
estan en pce68.dispatch.lp. Guarda en reports/simulacion_lp/ (AAAAMMDD = primer y ultimo dia):

    cuartos_*.csv   precio sin y con parque, potencia y energia del parque en cada periodo
    dias_*.csv      resumen por dia (columna analizado: los dias de los extremos no cuentan)
    tramos_*.csv    margen que vio cada tramo e ingreso a precios finales
    resumen_*.csv   totales de los dias analizados
    perfil_*.png    precio y potencia del parque en todo el periodo
    carga_*.png     energia almacenada e ingreso diario por MW
    bateria_*.png   estado de carga y potencia de carga/descarga del parque en todo el periodo
    simulacion_lp_*.xlsx  las mismas tablas en pestañas, con graficos nativos de Excel

Son SALIDAS: se regeneran enteras cada vez (para trabajar en Excel, haz una copia; cierra el
Excel antes de relanzar). Textos de figuras y Excel en ingles, como el entregable.
Con --desde-csv no se simula: Excel y figuras se rehacen con los CSV ya guardados.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from openpyxl import Workbook
from openpyxl.chart import BarChart, LineChart, Reference

from pce68.data.curvas import cargar, precios
from pce68.dispatch.battery import Battery
from pce68.dispatch.lp import despacho_lp
from pce68.excel import F_NOTA, F_TIT, F_TXT
from pce68.excel import tabla as tabla_excel
from pce68.features.metrics import daily_spread

ROOT = Path(__file__).resolve().parents[1]
TZ = "Europe/Madrid"
TINTA, TINTA_2, MUDO, REJILLA = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
AZUL, NARANJA = "#1c5cab", "#eb6834"


def estilo(ax) -> None:
    ax.grid(True, color=REJILLA, linewidth=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.tick_params(colors=TINTA_2, labelsize=9)


def resumen_dias(t: pd.DataFrame, mw: float, bat: Battery, s0_total: float, analizados: set) -> pd.DataFrame:
    filas, carga_previa = [], s0_total
    for d, g in t.groupby("dia", sort=True):
        h = g["resolucion_min"].to_numpy(float) / 60
        delta = g["mw"].to_numpy(float)
        descargada = float((np.clip(-delta, 0, None) * h).sum())
        filas.append({
            "dia": d,
            "analizado": d in analizados,
            "ingreso_eur": float((-delta * g["precio_parque"] * h).sum()),
            "energia_cargada_mwh": float((np.clip(delta, 0, None) * h).sum()),
            "energia_descargada_mwh": descargada,
            "ciclos_equivalentes": descargada / (mw * bat.duracion_horas),
            "carga_inicio_mwh": carga_previa,
            "carga_fin_mwh": float(g["carga_mwh"].iloc[-1]),
            "precio_medio_base": float(g["precio_base"].mean()),
            "precio_medio_parque": float(g["precio_parque"].mean()),
            "precio_min_base": float(g["precio_base"].min()),
            "precio_min_parque": float(g["precio_parque"].min()),
            "precio_max_base": float(g["precio_base"].max()),
            "precio_max_parque": float(g["precio_parque"].max()),
        })
        carga_previa = float(g["carga_mwh"].iloc[-1])
    r = pd.DataFrame(filas)
    r["ingreso_eur_mw"] = r["ingreso_eur"] / mw
    for col, nombre in (("precio_base", "spread_2h_base"), ("precio_parque", "spread_2h_parque")):
        s = daily_spread(t.rename(columns={col: "precio_es"}), bat.duracion_horas)
        r[nombre] = s.iloc[:, 1].to_numpy()
    return r


def figura_perfil(t: pd.DataFrame, gw: float, descartados: list, titulo: str, salida: Path) -> None:
    x = t["local"].dt.tz_localize(None)
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, height_ratios=[3, 2])
    ax.plot(x, t["precio_base"], color=MUDO, lw=1.0, ls="--", label="no batteries")
    ax.plot(x, t["precio_parque"], color=AZUL, lw=1.2, label=f"{gw:g} GW of batteries")
    ax.set_ylabel("€/MWh", color=TINTA_2)
    ax.set_title(f"Spanish day-ahead price, {titulo}: before and after the battery fleet "
                 "(LP over the whole period)", loc="left", color=TINTA, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=TINTA_2, loc="upper left")
    ax2.fill_between(x, t["mw"] / 1000, 0, step="post", color=AZUL, alpha=0.7, lw=0)
    ax2.axhline(0, color="#c3c2b7", lw=1)
    ax2.set_ylabel("GW on the grid", color=TINTA_2)
    ax2.set_title("Fleet: charging (+, extra demand) and discharging (−, extra supply). "
                  "Grey bands: boundary days, excluded from the results", loc="left", color=TINTA_2,
                  fontsize=10)
    for a in (ax, ax2):
        for d in descartados:
            inicio = pd.Timestamp(d)
            a.axvspan(inicio, inicio + pd.Timedelta(days=1), color=MUDO, alpha=0.12, lw=0)
        estilo(a)
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def figura_carga(t: pd.DataFrame, dias: pd.DataFrame, capacidad_mwh: float, salida: Path) -> None:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(13, 6.5), height_ratios=[3, 2])
    ax.plot(t["local"].dt.tz_localize(None), 100 * t["carga_mwh"] / capacidad_mwh, color=AZUL, lw=1.2)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of fleet energy capacity", color=TINTA_2)
    ax.set_title("Energy stored in the fleet (end of each quarter-hour)", loc="left", color=TINTA,
                 fontsize=12)
    a = dias[dias["analizado"]]
    ax2.bar([f"{d:%d/%m}" for d in a["dia"]], a["ingreso_eur_mw"], color=NARANJA)
    ax2.set_ylabel("€/MW installed", color=TINTA_2)
    ax2.set_title("Daily arbitrage revenue per installed MW (analysed days)", loc="left", color=TINTA_2,
                  fontsize=10)
    for x in (ax, ax2):
        estilo(x)
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


ETIQUETAS_RESUMEN = {
    "dias_analizados": ("Analysed days", "days"),
    "gw": ("Battery fleet", "GW"),
    "ingreso_eur": ("Fleet arbitrage revenue", "€"),
    "ingreso_eur_mw": ("Revenue per installed MW", "€/MW"),
    "ingreso_eur_mw_dia": ("Revenue per installed MW and day", "€/MW/day"),
    "aceptante_1mw_eur_mw_dia": ("Single 1 MW battery that does not move prices (ceiling)", "€/MW/day"),
    "perdida_pct": ("Revenue loss per MW of the fleet vs the single 1 MW battery", "%"),
    "ciclos_por_dia": ("Equivalent cycles per day", "cycles/day"),
    "carga_inicio_pct": ("Stored energy at the start of the analysed days", "% of capacity"),
    "carga_fin_pct": ("Stored energy at the end of the analysed days", "% of capacity"),
    "variacion_carga_mwh": ("Change in stored energy over the analysed days", "MWh"),
    "variacion_carga_valorada_eur": ("Same change, valued at the mean price with the fleet", "€"),
    "precio_medio_base": ("Mean price, no batteries", "€/MWh"),
    "precio_medio_parque": ("Mean price, with the fleet", "€/MWh"),
    "spread_2h_base": ("Mean daily 2 h spread, no batteries", "€/MWh"),
    "spread_2h_parque": ("Mean daily 2 h spread, with the fleet", "€/MWh"),
    "presupuesto_ciclos": ("Cycle budget over the whole period (boundary days included)", "cycles"),
    "ciclos_tramo1": ("Cycles used by tranche 1 over the whole period", "cycles"),
}
COLUMNAS_DIAS = {
    "dia": "Day",
    "analizado": "Analysed",
    "ingreso_eur_mw": "Revenue per installed MW (€/MW)",
    "ingreso_eur": "Fleet revenue (€)",
    "ciclos_equivalentes": "Equivalent cycles",
    "energia_cargada_mwh": "Energy charged (MWh)",
    "energia_descargada_mwh": "Energy discharged (MWh)",
    "carga_inicio_mwh": "Stored energy at 00:00 (MWh)",
    "carga_fin_mwh": "Stored energy at 24:00 (MWh)",
    "precio_medio_base": "Mean price, no batteries (€/MWh)",
    "precio_medio_parque": "Mean price, with fleet (€/MWh)",
    "precio_min_base": "Min price, no batteries",
    "precio_min_parque": "Min price, with fleet",
    "precio_max_base": "Max price, no batteries",
    "precio_max_parque": "Max price, with fleet",
    "spread_2h_base": "2 h spread, no batteries (€/MWh)",
    "spread_2h_parque": "2 h spread, with fleet (€/MWh)",
}
COLUMNAS_TRAMOS = {
    "tramo": "Tranche",
    "mw": "MW",
    "activo": "Active",
    "margen_visto_eur_mw": "Margin seen when placed (€/MW, whole period)",
    "ingreso_final_eur_mw": "Revenue at final prices (€/MW, whole period)",
    "ingreso_final_analizados_eur_mw": "Revenue at final prices (€/MW, analysed days)",
    "ciclos_equivalentes": "Equivalent cycles (whole period)",
}


def figura_bateria(t: pd.DataFrame, capacidad_mwh: float, descartados: list, titulo: str,
                   salida: Path) -> None:
    x = t["local"].dt.tz_localize(None)
    gw = t["mw"] / 1000
    soc = 100 * t["carga_mwh"] / capacidad_mwh
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    ax.fill_between(x, soc, 0, step="post", color=AZUL, alpha=0.2, lw=0)
    ax.step(x, soc, where="post", color=AZUL, lw=1.3)
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of energy capacity", color=TINTA_2)
    ax.set_title(f"Battery fleet state of charge, {titulo} (end of each quarter-hour)", loc="left",
                 color=TINTA, fontsize=12)
    ax2.fill_between(x, gw.clip(lower=0), 0, step="post", color=AZUL, lw=0, label="charging (extra demand)")
    ax2.fill_between(x, gw.clip(upper=0), 0, step="post", color=NARANJA, lw=0,
                     label="discharging (extra supply)")
    ax2.axhline(0, color="#c3c2b7", lw=1)
    ax2.set_ylabel("GW on the grid", color=TINTA_2)
    ax2.set_title("Charging and discharging power of the fleet. Grey bands: boundary days, excluded from the "
                  "results", loc="left", color=TINTA_2, fontsize=10)
    ax2.legend(frameon=False, fontsize=9, labelcolor=TINTA_2, loc="lower left", ncols=2)
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    for a in (ax, ax2):
        for d in descartados:
            inicio = pd.Timestamp(d)
            a.axvspan(inicio, inicio + pd.Timedelta(days=1), color=MUDO, alpha=0.12, lw=0)
        estilo(a)
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


# Condiciones de mercado de ESIOS (scripts/07_descarga_fundamentales.py): columna del parquet -> titulo
# en el Excel. MW medios de cada cuarto de hora; peninsulares salvo el ciclo combinado (nacional).
FUNDAMENTALES = {
    "demanda": "Demand (MW)",
    "eolica": "Wind (MW)",
    "solar_fv": "Solar PV (MW)",
    "solar_termica": "Solar thermal (MW)",
    "nuclear": "Nuclear (MW)",
    "ciclo_combinado": "Combined cycle, national (MW)",
    "carbon": "Coal (MW)",
    "cogeneracion": "Cogeneration and other (MW)",
    "hidraulica_gen": "Hydro generation (MW)",
    "bombeo_turbinacion": "Pumped storage generating (MW)",
    "bombeo_consumo": "Pumped storage pumping (MW, negative)",
    "hidraulica": "Hydro net of pumping (MW)",
    "intercambios": "Net imports (MW, + = import)",
    "demanda_residual": "Residual demand = demand - wind - solar PV (MW)",
}
FUNDAMENTALES_DIA = ["demanda", "eolica", "solar_fv", "demanda_residual", "bombeo_consumo", "intercambios"]


def lee_fundamentales(t: pd.DataFrame) -> pd.DataFrame | None:
    """Condiciones de mercado alineadas con los periodos de `t` (columnas con nombre del parquet)."""
    ruta = ROOT / "data" / "processed" / "fundamentales.parquet"
    if not ruta.exists():
        print("Sin condiciones de mercado: ejecuta scripts/07_descarga_fundamentales.py")
        return None
    f = pd.read_parquet(ruta)
    f["timestamp_utc"] = pd.to_datetime(f["timestamp_utc"], utc=True)
    cols = [c for c in FUNDAMENTALES if c in f]
    m = t[["timestamp_utc"]].merge(f[["timestamp_utc", *cols]], on="timestamp_utc", how="left")
    faltan = int(m[cols].isna().any(axis=1).sum())
    if faltan:
        print(f"Aviso: {faltan} de {len(m)} cuartos de hora sin condiciones de mercado completas")
    return m[cols].reset_index(drop=True)


def _serie(ws, col: int, fila0: int, fila1: int, titulo: str):
    from openpyxl.chart import Series

    return Series(Reference(ws, min_col=col, min_row=fila0, max_row=fila1), title=titulo)


def exporta_excel(t, dias, tramos, resumen, fund, params: dict, capacidad: float, salida: Path) -> None:
    """Libro con los resultados en pestañas y graficos nativos de Excel (textos en ingles)."""
    eur = "#,##0.00"
    wb = Workbook()

    ws = wb.active
    ws.title = "Read me"
    ws.sheet_view.showGridLines = False
    ws["A1"] = f"Battery fleet planned by linear programming, {params['period']}"
    ws["A1"].font = F_TIT
    textos = [
        "Generated by scripts/06_simula_lp.py. This file is an OUTPUT: it is fully regenerated",
        "every time the script runs. Save a copy under another name to work on it by hand.",
        "",
        "Sheets:",
        "  Summary        totals over the analysed days",
        "  Battery        chart of the fleet's charging/discharging power and state of charge",
        "  Days           one row per day (boundary days are flagged and excluded from the totals)",
        "  Tranches       margin each tranche saw when it was placed, and its revenue at final prices",
        "  Quarter-hours  price without and with the fleet, fleet power, stored energy and market conditions",
        "",
        "Market conditions: ESIOS (Red Eléctrica) 'T.Real' indicators, mean MW of each quarter-hour,",
        "  peninsular system except combined cycle (national). Pumped-storage pumping is negative;",
        "  net imports are positive when Spain imports (sign deduced from the demand balance).",
        "  The Days sheet repeats the daily means of the main ones.",
        "",
        "Parameters:",
        *[f"  {k}: {v}" for k, v in params.items()],
        "",
        "Main assumptions (details in src/pce68/dispatch/lp.py):",
        "  perfect foresight over the whole period (overestimates revenue and price impact);",
        "  stored energy fixed at the boundaries; cycle budget of about one cycle per day;",
        "  tranches placed one after another, never re-optimised, each one a price-taker;",
        "  fixed interconnection flows; pumped hydro and the other bids do not react to the batteries.",
        "  'Single 1 MW battery': a battery too small to move prices, i.e. the revenue ceiling.",
    ]
    for i, txt in enumerate(textos):
        ws.cell(row=3 + i, column=1, value=txt).font = F_TXT if txt else F_NOTA
    ws.column_dimensions["A"].width = 100

    ws = wb.create_sheet("Summary")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Totals over the analysed days"
    ws["A1"].font = F_TIT
    v = dict(resumen.iloc[0])
    v["perdida_pct"] = 100 * (1 - v["ingreso_eur_mw_dia"] / v["aceptante_1mw_eur_mw_dia"])
    df = pd.DataFrame([{"Indicator": e, "Value": v[k], "Unit": u} for k, (e, u) in ETIQUETAS_RESUMEN.items()])
    tabla_excel(ws, 3, df, {"Value": eur})
    ws.column_dimensions["A"].width = 62

    ws = wb.create_sheet("Days")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Results per day"
    ws["A2"] = "Boundary days carry the 50 % stored-energy conditions: not analysed."
    ws["A1"].font, ws["A2"].font = F_TIT, F_NOTA
    d = dias[list(COLUMNAS_DIAS)].rename(columns=COLUMNAS_DIAS)
    d["Day"] = pd.to_datetime(d["Day"]).dt.strftime("%d/%m/%Y")
    d["Analysed"] = np.where(dias["analizado"], "yes", "no (boundary)")
    if fund is not None:
        cols = [c for c in FUNDAMENTALES_DIA if c in fund]
        medias = fund[cols].groupby(t["dia"].to_numpy()).mean().reindex(dias["dia"])
        for c in cols:
            d[f"Mean {FUNDAMENTALES[c][0].lower()}{FUNDAMENTALES[c][1:]}"] = medias[c].to_numpy()
    fin = tabla_excel(ws, 4, d, {c: eur for c in d.columns if c not in ("Day", "Analysed")})
    ws.freeze_panes = "C5"
    filas = 5 + np.flatnonzero(dias["analizado"].to_numpy())
    i0, i1 = int(filas[0]), int(filas[-1])
    col = {c: j for j, c in enumerate(d.columns, start=1)}
    ch = BarChart()
    ch.title, ch.height, ch.width, ch.legend = "Daily revenue per installed MW (analysed days)", 9, 18, None
    ch.y_axis.title = "€/MW"
    ch.series.append(_serie(ws, col["Revenue per installed MW (€/MW)"], i0, i1, "Revenue"))
    ch.set_categories(Reference(ws, min_col=1, min_row=i0, max_row=i1))
    ch.x_axis.delete = ch.y_axis.delete = False
    ws.add_chart(ch, f"A{fin + 3}")
    ch = LineChart()
    ch.title, ch.height, ch.width = "Daily 2 h spread (€/MWh)", 9, 18
    for c in ("2 h spread, no batteries (€/MWh)", "2 h spread, with fleet (€/MWh)"):
        ch.series.append(_serie(ws, col[c], i0, i1, c.split(" (")[0]))
    ch.set_categories(Reference(ws, min_col=1, min_row=i0, max_row=i1))
    ch.x_axis.delete = ch.y_axis.delete = False
    ws.add_chart(ch, f"H{fin + 3}")

    ws = wb.create_sheet("Tranches")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Tranches of the fleet"
    ws["A2"] = "Placed one after another; each sees the prices left by the previous ones, never re-optimised."
    ws["A1"].font, ws["A2"].font = F_TIT, F_NOTA
    tr = tramos[list(COLUMNAS_TRAMOS)].rename(columns=COLUMNAS_TRAMOS)
    tr["Active"] = np.where(tramos["activo"], "yes", "no")
    fin = tabla_excel(ws, 4, tr, {c: eur for c in tr.columns if c not in ("Tranche", "Active")})
    ch = BarChart()
    ch.title, ch.height, ch.width = "Margin seen when placed vs revenue at final prices (€/MW)", 9, 22
    for c in ("Margin seen when placed (€/MW, whole period)", "Revenue at final prices (€/MW, whole period)"):
        ch.series.append(_serie(ws, list(tr.columns).index(c) + 1, 5, fin, c.split(" (")[0]))
    ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=fin))
    ch.x_axis.title = "tranche"
    ch.x_axis.delete = ch.y_axis.delete = False
    ws.add_chart(ch, "I4")

    ws = wb.create_sheet("Quarter-hours")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Quarter-hour results"
    ws["A2"] = "Fleet power: positive = charging (extra demand), negative = discharging."
    ws["A1"].font, ws["A2"].font = F_TIT, F_NOTA
    analizados = set(dias.loc[dias["analizado"], "dia"])
    q = pd.DataFrame({
        "Local time": t["local"].dt.strftime("%Y-%m-%d %H:%M"),
        "Analysed": np.where(t["dia"].isin(analizados), "yes", "no"),
        "Price, no batteries (€/MWh)": t["precio_base"],
        "Price, with fleet (€/MWh)": t["precio_parque"],
        "Fleet power (MW)": t["mw"],
        "Stored energy (MWh)": t["carga_mwh"],
        "Stored energy (% of capacity)": 100 * t["carga_mwh"] / capacidad,
    })
    if fund is not None:
        q = pd.concat([q.reset_index(drop=True), fund.rename(columns=FUNDAMENTALES)], axis=1)
    fin = tabla_excel(ws, 4, q, {c: eur for c in q.columns if c not in ("Local time", "Analysed")})
    ws.freeze_panes = "B5"
    ch = LineChart()
    ch.title, ch.height, ch.width = "Day-ahead price without and with the fleet (€/MWh)", 12, 32
    ch.add_data(Reference(ws, min_col=3, max_col=4, min_row=4, max_row=fin), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=fin))
    ch.x_axis.delete = ch.y_axis.delete = False
    ws.add_chart(ch, "W4")

    # Estado de carga y potencia del parque, con los datos de Quarter-hours
    qh, col = ws, {c: j for j, c in enumerate(q.columns, start=1)}
    ws = wb.create_sheet("Battery")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Battery fleet: charging, discharging and state of charge"
    ws["A2"] = ("Bars: fleet power on the grid (MW, positive = charging, negative = discharging). "
                "Line: stored energy (% of capacity, right axis). Data in the Quarter-hours sheet.")
    ws["A1"].font, ws["A2"].font = F_TIT, F_NOTA
    barras = BarChart()
    barras.type, barras.gapWidth = "col", 0
    barras.title, barras.height, barras.width = "Fleet power and state of charge", 15, 42
    barras.add_data(Reference(qh, min_col=col["Fleet power (MW)"], min_row=4, max_row=fin),
                    titles_from_data=True)
    barras.series[0].invertIfNegative = False
    barras.series[0].graphicalProperties.solidFill = AZUL.lstrip("#")
    barras.series[0].graphicalProperties.line.solidFill = AZUL.lstrip("#")
    barras.set_categories(Reference(qh, min_col=1, min_row=5, max_row=fin))
    barras.y_axis.title = "MW on the grid"
    linea = LineChart()
    linea.add_data(Reference(qh, min_col=col["Stored energy (% of capacity)"], min_row=4, max_row=fin),
                   titles_from_data=True)
    linea.series[0].graphicalProperties.line.solidFill = NARANJA.lstrip("#")
    linea.y_axis.axId = 200
    linea.y_axis.title = "% of energy capacity"
    linea.y_axis.scaling.min, linea.y_axis.scaling.max = 0, 100
    linea.y_axis.crosses = "max"
    barras.x_axis.delete = barras.y_axis.delete = linea.y_axis.delete = False
    barras += linea
    ws.add_chart(barras, "A4")
    wb.move_sheet("Battery", offset=-(len(wb.sheetnames) - 3))  # justo despues de Summary

    wb.save(salida)


def simula(sim: dict, bat: Battery, todos: list, analizados: list, ciclos: float, mw: float):
    """Corre el LP y devuelve las tablas por periodo, por dia, por tramo y el resumen."""
    curvas = cargar(ROOT / "data" / "processed" / "curvas", todos)
    t0 = time.perf_counter()
    r = despacho_lp(curvas, mw, bat, float(sim["paso_mw"]), ciclos,
                    float(sim["carga_bordes"]), float(sim["penalizacion_eur_mwh"]))
    base = precios(curvas)["precio_eur_mwh"].to_numpy(float)
    print(f"Simulacion: {len(todos)} dias, {len(r)} periodos, {len(r.attrs['tramos'])} tramos, "
          f"{time.perf_counter() - t0:.0f} s")

    t = r[["timestamp_utc", "resolucion_min"]].copy()
    t["local"] = t["timestamp_utc"].dt.tz_convert(TZ)
    t["dia"] = t["local"].dt.date
    t["precio_base"], t["precio_parque"] = base, r["precio_eur_mwh"].to_numpy(float)
    t["mw"], t["carga_mwh"] = r["delta_mw"].to_numpy(float), r["carga_mwh"].to_numpy(float)
    h = t["resolucion_min"].to_numpy(float) / 60
    capacidad = mw * bat.duracion_horas
    s0_total = capacidad * float(sim["carga_bordes"])

    dias = resumen_dias(t, mw, bat, s0_total, set(analizados))
    ventana = np.isin(t["dia"], analizados)
    a = dias[dias["analizado"]]

    planes = r.attrs["planes"]
    tramos = r.attrs["tramos"].copy()
    p_final = t["precio_parque"].to_numpy(float)
    tramos["ingreso_final_eur_mw"] = [-(pl * p_final * h).sum() for pl in planes]
    tramos["ingreso_final_analizados_eur_mw"] = [-(pl * p_final * h)[ventana].sum() for pl in planes]
    tramos["ciclos_equivalentes"] = [(np.clip(-pl, 0, None) * h).sum() / bat.duracion_horas for pl in planes]

    carga_ini, carga_fin = float(a["carga_inicio_mwh"].iloc[0]), float(a["carga_fin_mwh"].iloc[-1])
    precio_medio_ventana = float(t.loc[ventana, "precio_parque"].mean())
    resumen = pd.DataFrame([{
        "dias_analizados": len(a),
        "gw": float(sim["gw"]),
        "ingreso_eur": a["ingreso_eur"].sum(),
        "ingreso_eur_mw": a["ingreso_eur"].sum() / mw,
        "ingreso_eur_mw_dia": a["ingreso_eur"].sum() / mw / len(a),
        "aceptante_1mw_eur_mw_dia": float(-(planes[0] * base * h)[ventana].sum()) / len(a),
        "ciclos_por_dia": a["ciclos_equivalentes"].mean(),
        "carga_inicio_pct": 100 * carga_ini / capacidad,
        "carga_fin_pct": 100 * carga_fin / capacidad,
        "variacion_carga_mwh": carga_fin - carga_ini,
        "variacion_carga_valorada_eur": (carga_fin - carga_ini) * precio_medio_ventana,
        "precio_medio_base": float(t.loc[ventana, "precio_base"].mean()),
        "precio_medio_parque": precio_medio_ventana,
        "spread_2h_base": a["spread_2h_base"].mean(),
        "spread_2h_parque": a["spread_2h_parque"].mean(),
        "presupuesto_ciclos": ciclos,
        "ciclos_tramo1": float(tramos["ciclos_equivalentes"].iloc[0]),
    }])

    return t, dias, tramos, resumen


def lee_csv(carpeta: Path, suf: str):
    """Las tablas de una simulacion ya guardada (para rehacer Excel y figuras sin simular)."""
    t = pd.read_csv(carpeta / f"cuartos_{suf}.csv").drop(columns=["timestamp_local"])
    t["timestamp_utc"] = pd.to_datetime(t["timestamp_utc"], utc=True)
    t["local"] = t["timestamp_utc"].dt.tz_convert(TZ)
    t["dia"] = t["local"].dt.date
    dias = pd.read_csv(carpeta / f"dias_{suf}.csv")
    dias["dia"] = pd.to_datetime(dias["dia"]).dt.date
    return t, dias, pd.read_csv(carpeta / f"tramos_{suf}.csv"), pd.read_csv(carpeta / f"resumen_{suf}.csv")


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    sim = cfg["simulacion_lp"]
    p = argparse.ArgumentParser()
    p.add_argument("--desde-csv", action="store_true",
                   help="no simular: rehacer Excel y figuras con los CSV de reports/simulacion_lp")
    args = p.parse_args()
    bat = Battery(**cfg["bateria"])
    desde, hasta = date.fromisoformat(str(sim["desde"])), date.fromisoformat(str(sim["hasta"]))
    todos = [desde + timedelta(days=i) for i in range((hasta - desde).days + 1)]
    k = int(sim["dias_descartados"])
    analizados, descartados = todos[k:len(todos) - k], todos[:k] + todos[len(todos) - k:]
    ciclos = bat.ciclos_maximos_anuales * len(todos) / 365
    mw = float(sim["gw"]) * 1000
    capacidad = mw * bat.duracion_horas

    carpeta = ROOT / "reports" / "simulacion_lp"
    carpeta.mkdir(parents=True, exist_ok=True)
    suf = f"{desde:%Y%m%d}_{hasta:%Y%m%d}"
    if args.desde_csv:
        t, dias, tramos, resumen = lee_csv(carpeta, suf)
    else:
        t, dias, tramos, resumen = simula(sim, bat, todos, analizados, ciclos, mw)
        t.assign(timestamp_local=t["local"].dt.strftime("%Y-%m-%d %H:%M")).drop(columns=["local"]).to_csv(
            carpeta / f"cuartos_{suf}.csv", index=False, float_format="%.3f")
        dias.to_csv(carpeta / f"dias_{suf}.csv", index=False, float_format="%.2f")
        tramos.to_csv(carpeta / f"tramos_{suf}.csv", index=False, float_format="%.2f")
        resumen.to_csv(carpeta / f"resumen_{suf}.csv", index=False, float_format="%.2f")
    a = dias[dias["analizado"]]

    titulo = f"{desde:%d/%m/%Y}–{hasta:%d/%m/%Y}"
    figura_perfil(t, float(sim["gw"]), descartados, titulo, carpeta / f"perfil_{suf}.png")
    figura_carga(t, dias, capacidad, carpeta / f"carga_{suf}.png")
    figura_bateria(t, capacidad, descartados, titulo, carpeta / f"bateria_{suf}.png")
    params = {
        "period": titulo,
        "analysed days": f"{analizados[0]:%d/%m/%Y}–{analizados[-1]:%d/%m/%Y} ({len(analizados)} days)",
        "battery fleet": f"{float(sim['gw']):g} GW in tranches of {float(sim['paso_mw']):g} MW",
        "battery": f"{bat.duracion_horas:g} h, round-trip efficiency {bat.eficiencia_ida_vuelta:g}",
        "cycle budget": f"{ciclos:g} cycles over the period ({bat.ciclos_maximos_anuales} per year)",
        "stored energy at both ends": f"{100 * float(sim['carga_bordes']):g} %",
    }
    fund = lee_fundamentales(t)
    exporta_excel(t, dias, tramos, resumen, fund, params, capacidad, carpeta / f"simulacion_lp_{suf}.xlsx")

    pd.set_option("display.width", 200)
    print("\nResumen por dia (analizados):")
    cols = ["dia", "ingreso_eur_mw", "ciclos_equivalentes", "precio_medio_base", "precio_medio_parque",
            "spread_2h_base", "spread_2h_parque"]
    print(a[cols].round(2).to_string(index=False))
    print("\nTotales de los dias analizados:")
    print(resumen.T.round(2).to_string(header=False))
    print("\nTramos:")
    print(tramos.round(2).to_string(index=False))
    print(f"\nGuardado en {carpeta.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
