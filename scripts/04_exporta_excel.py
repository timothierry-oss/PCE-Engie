"""Exporta el analisis exploratorio a Excel.

    uv run python scripts/04_exporta_excel.py            # con los datos de 01_descarga_omie.py
    uv run python scripts/04_exporta_excel.py --demo     # precios sinteticos, para probar

El fichero que genera es una SALIDA, no un almacen: se regenera entero cada
vez y nunca se edita a mano. Los datos crudos viven en data/processed en
parquet, que es de donde leen el analisis y los modelos.

Hojas (los textos del Excel estan en ingles, como el entregable):
    Read me         que es esto y cuando se genero
    By regime       por que no se puede entrenar sobre toda la serie
    Yearly          precio, dispersion, horas a cero y spread por anno
    Monthly         la misma serie por meses, con grafico
    Hourly profile  precio medio por hora y anno: la curva del pato
"""

from __future__ import annotations

import argparse
import logging
from datetime import date
from pathlib import Path

import numpy as np
import openpyxl.utils
import pandas as pd
import yaml
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference

from pce68.dispatch.battery import Battery, simulate_price_taker
from pce68.excel import F_NOTA, F_TIT, F_TXT, tabla
from pce68.features.metrics import daily_spread
from pce68.features.regimenes import anotar, resumen

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
# Nombres de los regimenes en el Excel (el entregable es en ingles)
REGIMENES_EN = {
    "pre_crisis": "Pre-crisis",
    "crisis_gas": "Gas crisis",
    "excepcion_iberica": "Iberian exception",
    "normalizacion": "Normalisation",
}


def precios_demo(anios: range) -> pd.DataFrame:
    """Serie sintetica con forma solar y un nivel distinto por regimen."""
    rng = np.random.default_rng(7)
    frames = []
    for anio in anios:
        idx = pd.date_range(f"{anio}-01-01", f"{anio}-12-31 23:45", freq="15min", tz="Europe/Madrid")
        h = idx.hour + idx.minute / 60
        nivel = {2021: 90, 2022: 180, 2023: 110, 2024: 65, 2025: 60, 2026: 55}.get(anio, 60)
        valle = 45 * np.clip(np.sin(np.pi * (h - 7) / 12), 0, None) * (1 + (anio - 2021) * 0.12)
        pico = 20 * np.exp(-((h - 21) ** 2) / 3)
        p = nivel - valle + pico + rng.normal(0, 8, len(idx))
        frames.append(
            pd.DataFrame({"timestamp_utc": idx.tz_convert("UTC"), "precio_es": p, "resolucion_min": 15})
        )
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    bat = Battery(
        duracion_horas=config["bateria"]["duracion_horas"],
        eficiencia_ida_vuelta=config["bateria"]["eficiencia_ida_vuelta"],
        ciclos_maximos_anuales=config["bateria"]["ciclos_maximos_anuales"],
    )

    if args.demo:
        log.info("MODO DEMO: precios sinteticos, no son datos reales.")
        precios = precios_demo(range(2021, 2027))
        origen = "DEMO · synthetic prices. These are NOT market data."
    else:
        ruta = ROOT / config["datos"]["processed_dir"] / "precios_omie.parquet"
        if not ruta.exists():
            raise SystemExit(f"No existe {ruta}. Ejecuta antes scripts/01_descarga_omie.py, o usa --demo.")
        precios = pd.read_parquet(ruta)
        origen = "OMIE day-ahead market prices, downloaded with scripts/01_descarga_omie.py."

    precios = anotar(precios)
    local = precios["timestamp_utc"].dt.tz_convert("Europe/Madrid")
    precios["anio"] = local.dt.year
    precios["mes"] = local.dt.strftime("%Y-%m")
    precios["hora"] = local.dt.hour
    precios["horas"] = precios["resolucion_min"] / 60.0

    log.info("Calculando spreads diarios...")
    spread2 = daily_spread(precios, 2.0).rename(columns={"spread_2h": "spread"})
    spread2["anio"] = pd.to_datetime(spread2["dia_local"]).dt.year
    spread2["mes"] = pd.to_datetime(spread2["dia_local"]).dt.to_period("M").astype(str)

    log.info("Simulando el despacho de la bateria...")
    ingreso = simulate_price_taker(precios, bat)
    ingreso["anio"] = pd.to_datetime(ingreso["dia"]).dt.year

    # ── Anual
    anual = precios.groupby("anio").apply(
        lambda g: pd.Series(
            {
                "Hours of data": g["horas"].sum(),
                "Mean price (€/MWh)": g["precio_es"].mean(),
                "Median (€/MWh)": g["precio_es"].median(),
                "Std. dev. (€/MWh)": g["precio_es"].std(),
                "Minimum (€/MWh)": g["precio_es"].min(),
                "Maximum (€/MWh)": g["precio_es"].max(),
                "Hours at price ≤ 0": g.loc[g["precio_es"] <= 0, "horas"].sum(),
            }
        ),
        include_groups=False,
    )
    anual["% hours at price ≤ 0"] = anual["Hours at price ≤ 0"] / anual["Hours of data"]
    anual["Mean 2 h spread (€/MWh)"] = spread2.groupby("anio")["spread"].mean()
    anual["2 h battery revenue (€/MW/year)"] = ingreso.groupby("anio")["ingreso_eur_por_mw"].sum()
    anual = anual.reset_index().rename(columns={"anio": "Year"})
    anual["Year"] = anual["Year"].astype(str)

    # ── Mensual
    mensual = precios.groupby("mes").apply(
        lambda g: pd.Series(
            {
                "Mean price (€/MWh)": g["precio_es"].mean(),
                "Hours at price ≤ 0": g.loc[g["precio_es"] <= 0, "horas"].sum(),
            }
        ),
        include_groups=False,
    )
    mensual["Mean 2 h spread (€/MWh)"] = spread2.groupby("mes")["spread"].mean()
    mensual = mensual.reset_index().rename(columns={"mes": "Month"})

    # ── Perfil horario
    perfil = (
        precios.pivot_table(index="hora", columns="anio", values="precio_es", aggfunc="mean")
        .reset_index()
        .rename(columns={"hora": "Hour"})
    )
    perfil.columns = [str(c) for c in perfil.columns]

    # ── Regímenes
    reg = resumen(precios)
    reg["regimen"] = reg["regimen"].map(REGIMENES_EN).fillna(reg["regimen"])
    reg = reg.rename(
        columns={
            "regimen": "Regime",
            "periodos": "Periods",
            "desde": "From",
            "hasta": "To",
            "precio_medio": "Mean price (€/MWh)",
            "precio_max": "Maximum price (€/MWh)",
        }
    )

    # ── Escritura
    wb = Workbook()
    eur = "#,##0.00"
    pct = "0.0%"

    ws = wb.active
    ws.title = "Read me"
    ws.sheet_view.showGridLines = False
    ws.column_dimensions["A"].width = 100
    ws["A1"] = "Project 68 · Exploratory analysis of the Spanish day-ahead market"
    ws["A1"].font = F_TIT
    textos = [
        f"Generated on {date.today():%d/%m/%Y} by scripts/04_exporta_excel.py",
        origen,
        "",
        "This file is an OUTPUT: it is fully regenerated every time the script runs.",
        "Do not edit it by hand: the next run overwrites any change. Save a copy to work on it.",
        "",
        "Raw data live in data/processed/precios_omie.parquet, which is what the analysis and",
        "the models read. Excel only gets the aggregates a person can read: half a million",
        "quarter-hourly rows would not be useful here.",
        "",
        "Figures are values computed in Python, not formulas: traceability lives in the code",
        "and in version control, not in the cells.",
    ]
    for i, t in enumerate(textos):
        ws.cell(row=3 + i, column=1, value=t).font = F_TXT if t else F_NOTA

    ws = wb.create_sheet("By regime")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "The five years of data are not a single market"
    ws["A1"].font = F_TIT
    ws["A2"] = (
        "Training on the whole series mixes regimes that no longer exist. "
        "Only 'Normalisation' represents the market we project to 2031."
    )
    ws["A2"].font = F_NOTA
    tabla(ws, 4, reg, {"Mean price (€/MWh)": eur, "Maximum price (€/MWh)": eur})

    ws = wb.create_sheet("Yearly")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Yearly summary"
    ws["A1"].font = F_TIT
    tabla(
        ws,
        3,
        anual,
        {
            "Mean price (€/MWh)": eur,
            "Median (€/MWh)": eur,
            "Std. dev. (€/MWh)": eur,
            "Minimum (€/MWh)": eur,
            "Maximum (€/MWh)": eur,
            "Hours of data": "#,##0",
            "Hours at price ≤ 0": "#,##0",
            "% hours at price ≤ 0": pct,
            "Mean 2 h spread (€/MWh)": eur,
            "2 h battery revenue (€/MW/year)": "#,##0",
        },
    )

    ws = wb.create_sheet("Monthly")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Monthly series"
    ws["A1"].font = F_TIT
    fin = tabla(
        ws,
        3,
        mensual,
        {"Mean price (€/MWh)": eur, "Hours at price ≤ 0": "#,##0", "Mean 2 h spread (€/MWh)": eur},
    )
    ch = LineChart()
    ch.title = "Mean price and capturable spread (€/MWh)"
    ch.height, ch.width = 9, 24
    ch.add_data(Reference(ws, min_col=2, max_col=2, min_row=3, max_row=fin), titles_from_data=True)
    ch.add_data(Reference(ws, min_col=4, max_col=4, min_row=3, max_row=fin), titles_from_data=True)
    ch.set_categories(Reference(ws, min_col=1, min_row=4, max_row=fin))
    ws.add_chart(ch, "F3")

    ws = wb.create_sheet("Hourly profile")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Mean price by hour of the day"
    ws["A1"].font = F_TIT
    ws["A2"] = "The midday dip that deepens year after year is solar cannibalisation."
    ws["A2"].font = F_NOTA
    fin = tabla(ws, 4, perfil, {c: eur for c in perfil.columns if c != "Hour"})
    ch = LineChart()
    ch.title = "Hourly price profile by year (€/MWh)"
    ch.height, ch.width = 10, 24
    ch.add_data(
        Reference(ws, min_col=2, max_col=len(perfil.columns), min_row=4, max_row=fin),
        titles_from_data=True,
    )
    ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=fin))
    ws.add_chart(ch, openpyxl.utils.get_column_letter(len(perfil.columns) + 2) + "4")

    salida = ROOT / "reports" / f"analisis_exploratorio{'_demo' if args.demo else ''}.xlsx"
    salida.parent.mkdir(parents=True, exist_ok=True)
    wb.save(salida)
    log.info("\nEscrito %s", salida)
    log.info("\n%s", anual.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
