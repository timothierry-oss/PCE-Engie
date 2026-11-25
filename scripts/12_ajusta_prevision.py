"""Etapa 3 del documento BESS-cannibalization: ajusta la prevision de ENGIE con el modelo de impacto.

    uv run python scripts/12_ajusta_prevision.py     # parametros: config/base.yaml, seccion prevision

Lee data/forecast_base_case.xlsx (prevision horaria 2027-2040), la pasa a cuartos de hora, aplica los
modelos de scripts/10_modelo_impacto.py (data/processed/modelo_impacto_<sufijo>.joblib) con varias
potencias constantes X de baterias y vuelve a horas. Correspondencia de columnas y supuestos: cabecera de
pce68.modelo.prevision.

DATOS CONFIDENCIALES de ENGIE: todo se escribe en data/processed/prevision_ajustada/ (ignorado por git),
nunca en reports/. Salidas (textos en ingles):

    prevision_ajustada.parquet  una fila por hora: p_engie y el precio ajustado de cada modelo y GW
    annual_summary.csv          por año, modelo y GW: precio medio, spread diario 2 h, horas de precio <= 0
    coverage.csv                por año: % de horas fuera del rango visto en el entrenamiento (por variable)
    forecast_adjusted.xlsx      resumen y cobertura en Excel
    adjusted_spread.png         spread por año antes y despues, y cobertura del dominio de entrenamiento
    adjusted_profile.png        precio medio por hora del dia, antes y despues, en dos años
"""

from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from openpyxl import Workbook

from pce68.dispatch.battery import Battery
from pce68.excel import F_NOTA, F_TIT
from pce68.excel import tabla as tabla_excel
from pce68.features.metrics import daily_spread
from pce68.modelo.impacto import ajusta_prevision
from pce68.modelo.prevision import (
    TZ,
    VARIABLES_REALES,
    a_cuartos,
    a_horas,
    cobertura,
    lee_prevision,
    referencia_entrenamiento,
    supuestos_fijos,
)

ROOT = Path(__file__).resolve().parents[1]
TINTA, TINTA_2, GRIS = "#0b0b0b", "#52514e", "#898781"
COLOR = {"gbm_delta": "#2a78d6", "gbm_p0sq": "#eb6834", "ridge_delta": "#1baf7a"}
NOMBRE = {"gbm_delta": "Gradient boosting, Δ target", "gbm_p0sq": "Gradient boosting, ratio (p0² weights)",
          "ridge_delta": "Ridge, Δ target"}
AZULES = {5: "#8ab6ea", 10: "#4b8fdd", 17: "#1f5fae"}  # magnitud (GW): una sola tonalidad, claro -> oscuro


def estilo(ax) -> None:
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)
    ax.tick_params(colors=TINTA_2, labelsize=9)


def spread_anual(ts: pd.Series, precio: pd.Series, horas: float) -> pd.Series:
    """Spread diario medio de cada año local (precios horarios)."""
    d = daily_spread(pd.DataFrame({"timestamp_utc": ts, "precio_es": precio.to_numpy(),
                                   "resolucion_min": 60}), horas)
    col = d.columns[1]
    anio = pd.to_datetime(d["dia_local"]).dt.year
    return d.groupby(anio)[col].mean()


def figura_spread(res: pd.DataFrame, cob: pd.DataFrame, gws: list[int], salida: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    ax = ejes[0]
    base = res[(res["model"] == "gbm_delta") & (res["gw"] == gws[0])].set_index("year")
    ax.plot(base.index, base["spread_E"], color=TINTA, linewidth=2.4, label="ENGIE forecast (no extra BESS)")
    for g in gws:
        s = res[(res["model"] == "gbm_delta") & (res["gw"] == g)].set_index("year")
        ax.plot(s.index, s["spread_adj"], color=AZULES.get(g, "#4b8fdd"), marker="o", markersize=4,
                linewidth=1.8, label=f"Adjusted, {g} GW (gradient boosting, Δ target)")
    ax.set_title("Mean daily 2 h spread (€/MWh)", loc="left", fontsize=11, color=TINTA)
    ax.legend(frameon=False, fontsize=8)
    estilo(ax)
    ax = ejes[1]
    ax.bar(cob.index - 0.2, cob["all_in_range_pct"], color="#898781", width=0.4,
           label="All inputs inside the training range")
    ax.bar(cob.index + 0.2, cob["all_in_range_excl_ccgt_pct"], color="#c9c8c2", width=0.4,
           label="Same, ignoring gas (CCGT)")
    ax.set_ylim(0, 100)
    ax.set_title("Hours inside the training range (%)", loc="left", fontsize=11, color=TINTA)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    estilo(ax)
    for a in ejes:
        a.set_xlabel("Year", fontsize=9, color=TINTA_2)
    fig.suptitle("Effect of BESS on the ENGIE base-case forecast, and how far it is from the training data",
                 x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)


def figura_perfil(horario: pd.DataFrame, modelos: list[str], gw: int, anios: list[int], salida: Path) -> None:
    fig, ejes = plt.subplots(1, len(anios), figsize=(5.6 * len(anios), 4.4), sharey=True)
    hora = horario["timestamp_utc"].dt.tz_convert(TZ)
    for ax, a in zip(np.atleast_1d(ejes), anios, strict=True):
        m = (hora.dt.year == a).to_numpy()
        ax.plot(range(24), horario.loc[m, "p_engie"].groupby(hora[m].dt.hour).mean(), color=TINTA,
                linewidth=2.4, label="ENGIE forecast")
        for mod in modelos:
            s = horario.loc[m, f"{mod}_{gw}GW"].groupby(hora[m].dt.hour).mean()
            ax.plot(range(24), s, color=COLOR[mod], linewidth=1.8, label=f"{NOMBRE[mod]}, {gw} GW")
        ax.set_title(str(a), loc="left", fontsize=11, color=TINTA)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour of day (local time)", fontsize=9, color=TINTA_2)
        estilo(ax)
    np.atleast_1d(ejes)[0].set_ylabel("Mean day-ahead price (€/MWh)", fontsize=9, color=TINTA_2)
    np.atleast_1d(ejes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Mean price by hour: ENGIE forecast vs adjusted for {gw} GW of BESS", x=0.01, ha="left",
                 fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    pre = cfg["prevision"]
    proc = ROOT / cfg["datos"]["processed_dir"]
    salida = ROOT / pre["salida"]
    salida.mkdir(parents=True, exist_ok=True)
    horas_bat = Battery(**cfg["bateria"]).duracion_horas
    gws, modelos = [int(g) for g in pre["gw"]], list(pre["modelos"])

    paquete = joblib.load(proc / f"modelo_impacto_{pre['entrenamiento']}.joblib")
    fundamentales = paquete["fundamentales"]
    desde = pd.Timestamp(pre["entrenamiento"][:8], tz="UTC")
    ref = referencia_entrenamiento(proc, desde)

    # entradas que la prevision no trae: valor constante (config) o mediana del entrenamiento
    gas, co2, bombeo = supuestos_fijos(proc, desde, pre["supuestos"])
    print(f"Supuestos fijos (no vienen en la prevision): gas {gas:.1f} €/MWh, CO2 {co2:.1f} €/t, "
          f"consumo de bombeo {bombeo:.0f} MW")

    crudo, avisos = lee_prevision(ROOT / pre["archivo"])
    for a in avisos:
        print("  valor reparado:", a)
    q = a_cuartos(crudo, gas, co2, bombeo)
    print(f"Prevision: {len(crudo):,} horas ({crudo['FechaLocal'].min():%d/%m/%Y}-"
          f"{crudo['FechaLocal'].max():%d/%m/%Y}) -> {len(q):,} cuartos de hora")

    # comprobacion de la cadena: sin baterias la ridge delta no debe cambiar el precio
    r0 = ajusta_prevision(paquete["modelos"]["ridge_delta"], q, 0.0, fundamentales)
    print(f"Comprobacion (ridge delta, 0 GW): diferencia maxima con la prevision = "
          f"{(r0['p_ajustado'] - r0['p_engie']).abs().max():.2e} €/MWh")

    horario = a_horas(q.assign(p_engie=q["p_engie"]), ["p_engie"])
    for mod in modelos:
        for g in gws:
            r = ajusta_prevision(paquete["modelos"][mod], q, float(g), fundamentales)
            h = a_horas(r, ["p_ajustado"]).rename(columns={"p_ajustado": f"{mod}_{g}GW"})
            horario = horario.merge(h, on="timestamp_utc", how="left")
            print(f"  {mod} {g} GW: precio medio {h.iloc[:, 1].mean():.1f} "
                  f"(prevision {horario['p_engie'].mean():.1f})")
    horario.to_parquet(salida / "prevision_ajustada.parquet", index=False)

    # resumen por año
    anio = horario["timestamp_utc"].dt.tz_convert(TZ).dt.year
    sp0 = spread_anual(horario["timestamp_utc"], horario["p_engie"], horas_bat)
    filas = []
    for mod in modelos:
        for g in gws:
            p = horario[f"{mod}_{g}GW"]
            sp = spread_anual(horario["timestamp_utc"], p, horas_bat)
            for a in sorted(anio.unique()):
                m = (anio == a).to_numpy()
                filas.append({
                    "year": a, "model": mod, "gw": g, "price_E": horario.loc[m, "p_engie"].mean(),
                    "price_adj": p[m].mean(), "spread_E": sp0[a], "spread_adj": sp[a],
                    "hours_le_0_E_pct": 100 * (horario.loc[m, "p_engie"] <= 0).mean(),
                    "hours_le_0_adj_pct": 100 * (p[m] <= 0).mean()})
    res = pd.DataFrame(filas)
    res["price_change_pct"] = 100 * (res["price_adj"] / res["price_E"] - 1)
    res["spread_change_pct"] = 100 * (res["spread_adj"] / res["spread_E"] - 1)
    cob = cobertura(a_horas(q, VARIABLES_REALES), ref)
    cols_cob = ["all_in_range_pct", "all_in_range_excl_ccgt_pct"]
    res = res.merge(cob[cols_cob].reset_index(), on="year", how="left")
    res.to_csv(salida / "annual_summary.csv", index=False, float_format="%.2f")
    cob.to_csv(salida / "coverage.csv", float_format="%.1f")

    figura_spread(res, cob, gws, salida / "adjusted_spread.png")
    figura_perfil(horario, modelos, max(gws), [2028, 2035], salida / "adjusted_profile.png")
    nota = ["Written for ENGIE internal use: derived from the confidential base-case forecast.",
            f"Fixed inputs the forecast does not contain: gas {gas:.1f} €/MWh, CO2 {co2:.1f} €/t, "
            f"pumping consumption {bombeo:.0f} MW (training medians).",
            "Constant installed BESS power per scenario; the model was trained on 1-17 GW.",
            "Outside the training range the models extrapolate flat (trees) or linearly (ridge): "
            "see Coverage."]
    wb = Workbook()
    for nombre, df, fmt in (("Summary", res, "0.0"), ("Coverage", cob.reset_index(), "0.0")):
        ws = wb.active if nombre == "Summary" else wb.create_sheet(nombre)
        ws.title = nombre
        ws.sheet_view.showGridLines = False
        ws["A1"], ws["A1"].font = f"ENGIE forecast adjusted for BESS: {nombre}", F_TIT
        for i, t in enumerate(nota):
            ws.cell(row=2 + i, column=1, value=t).font = F_NOTA
        tabla_excel(ws, 3 + len(nota), df, {c: fmt for c in df.columns if df[c].dtype.kind == "f"})
    wb.save(salida / "forecast_adjusted.xlsx")

    pd.set_option("display.width", 220)
    print("\nCobertura del dominio de entrenamiento (% de horas fuera del rango visto):")
    print(cob.to_string())
    ver = res[(res["model"] == modelos[0]) & (res["gw"] == max(gws))]
    print(f"\n{modelos[0]}, {max(gws)} GW:")
    print(ver[["year", "price_E", "price_adj", "price_change_pct", "spread_E", "spread_adj",
               "spread_change_pct", "all_in_range_excl_ccgt_pct"]].round(1).to_string(index=False))
    print(f"\nGuardado en {salida.relative_to(ROOT)}/ (ignorado por git)")


if __name__ == "__main__":
    main()
