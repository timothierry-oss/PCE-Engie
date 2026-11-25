"""LP de baterias en equilibrio con la respuesta del precio, aplicado a la prevision de ENGIE.

    uv run python scripts/14_lp_forecast.py                    # toda la prevision (varios minutos)
    uv run python scripts/14_lp_forecast.py --anios 2027 2028  # solo algunos años (prueba rapida)

Combina las dos piezas de scripts/13_respuesta_precio.py: el LP (energia, potencia y ciclos respetados por
construccion) decide el programa del parque y la respuesta del precio, aprendida de las curvas de OMIE, dice
cuanto se mueve el precio con ese programa (pce68.dispatch.equilibrio.equilibrio_exacto). Un LP por mes y por
potencia X (constante en todo el periodo), con la prevision horaria de ENGIE como precio sin baterias p0.
Correspondencia de columnas y supuestos: cabecera de pce68.modelo.prevision.

DATOS CONFIDENCIALES de ENGIE: todo va a data/processed/prevision_lp/ (ignorado por git), nunca a reports/.
Salidas (textos en ingles):

    prevision_lp.parquet     por hora: p_engie y, por GW, el precio con baterias y la potencia neta del parque
    annual_summary.csv       por año y GW: precio, spread 2 h, ingreso precio aceptante e ingreso en
                             equilibrio, ciclos, brecha; y el modelo anterior (gbm_delta, script 12) si existe
    monthly_diagnostics.csv  brecha de equilibrio de cada LP
    coverage.csv             % de horas fuera del dominio de entrenamiento, por año
    lp_forecast.xlsx         resumen y cobertura
    lp_spread_revenue.png    spread e ingreso por año
    lp_profile.png           precio medio por hora del dia en dos años
"""

from __future__ import annotations

import argparse
import time
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
from pce68.dispatch.equilibrio import equilibrio_exacto, precios_marginales
from pce68.dispatch.lp import plan_lp
from pce68.excel import F_NOTA, F_TIT
from pce68.excel import tabla as tabla_excel
from pce68.features.metrics import daily_spread
from pce68.modelo.impacto import variables
from pce68.modelo.prevision import (
    TZ,
    VARIABLES_REALES,
    a_cuartos,
    a_horario,
    a_horas,
    cobertura,
    lee_prevision,
    referencia_entrenamiento,
    supuestos_fijos,
)

ROOT = Path(__file__).resolve().parents[1]
TINTA, TINTA_2, GRIS = "#0b0b0b", "#52514e", "#898781"
AZULES = {5: "#8ab6ea", 10: "#4b8fdd", 17: "#1f5fae"}  # magnitud (GW): una sola tonalidad, claro -> oscuro
NARANJA = "#eb6834"


def estilo(ax) -> None:
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)
    ax.tick_params(colors=TINTA_2, labelsize=9)


def spread_anual(ts: pd.Series, precio, horas: float) -> pd.Series:
    """Spread diario medio de cada año local (precios horarios)."""
    d = daily_spread(pd.DataFrame({"timestamp_utc": ts, "precio_es": np.asarray(precio),
                                   "resolucion_min": 60}), horas)
    return d.groupby(pd.to_datetime(d["dia_local"]).dt.year)[d.columns[1]].mean()


def corre(z: pd.DataFrame, resp, gws: list[int], paso_mw: float, bat: Battery, sim: dict):
    """Un LP de equilibrio por mes y GW: (potencia neta, precio, brechas mensuales, plan precio aceptante)."""
    p0, h = z["p0"].to_numpy(float), np.ones(len(z))
    local = z["timestamp_utc"].dt.tz_convert(TZ)
    mes, dia = local.dt.strftime("%Y-%m").to_numpy(), local.dt.date.to_numpy()
    kmax = int(round(max(gws) * 1000 / paso_mw))
    medios = (np.arange(kmax) + 0.5) * paso_mw
    d = {g: np.zeros(len(z)) for g in gws}
    precio = {g: np.empty(len(z)) for g in gws}
    pt = np.zeros(len(z))  # plan de 1 MW precio aceptante a p0 (margen por mes, repartido en el mes)
    filas = []
    borde, pen = float(sim["carga_bordes"]), float(sim["penalizacion_eur_mwh"])
    meses = list(pd.unique(mes))
    t0 = time.perf_counter()
    for k, m in enumerate(meses, 1):
        idx = np.flatnonzero(mes == m)
        ndias = len(set(dia[idx]))
        ciclos = bat.ciclos_maximos_anuales * ndias / 365
        ev = resp.evaluador(z.iloc[idx])
        pc_all, pd_all = precios_marginales(p0[idx], ev, medios)
        margen_pt, perfil_pt, _ = plan_lp(p0[idx], h[idx], bat, ciclos, borde, pen)
        pt[idx] = perfil_pt
        for g in gws:
            J = int(round(g * 1000 / paso_mw))
            anchos = np.full(J, paso_mw)
            d[g][idx], _ = equilibrio_exacto(p0[idx], h[idx], pc_all[:, :J], pd_all[:, :J], anchos, bat,
                                             ciclos, borde, pen)
            precio[g][idx] = p0[idx] + ev(d[g][idx])
            mejor = g * 1000 * plan_lp(precio[g][idx], h[idx], bat, ciclos, borde, pen)[0]
            propio = float(-(d[g][idx] * precio[g][idx] * h[idx]).sum())
            filas.append({"month": m, "gw": g, "equilibrium_gap_pct": 100 * (mejor - propio) / mejor
                          if mejor > 1e-9 else 0.0})
        if k % 12 == 0 or k == len(meses):
            print(f"  {k}/{len(meses)} meses, {time.perf_counter() - t0:.0f} s", flush=True)
    return d, precio, pd.DataFrame(filas), pt


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    pre, rs, sim = cfg["prevision"], cfg["respuesta"], cfg["simulacion_lp"]
    ap = argparse.ArgumentParser()
    ap.add_argument("--anios", type=int, nargs="+", help="solo estos años (por defecto, todos)")
    args = ap.parse_args()
    proc = ROOT / cfg["datos"]["processed_dir"]
    salida = ROOT / "data" / "processed" / "prevision_lp"
    salida.mkdir(parents=True, exist_ok=True)
    bat = Battery(**cfg["bateria"])
    gws = [int(g) for g in pre["gw"]]
    paso_mw = float(rs["paso_gw"]) * 1000

    paquete = joblib.load(proc / f"respuesta_precio_{pre['respuesta_modelo']}.joblib")
    resp, fundamentales = paquete["respuesta"], paquete["fundamentales"]
    desde = pd.Timestamp(pre["respuesta_modelo"][:8], tz="UTC")
    gas, co2, bombeo = supuestos_fijos(proc, desde, pre["supuestos"])
    print(f"Supuestos fijos: gas {gas:.1f} €/MWh, CO2 {co2:.1f} €/t, consumo de bombeo {bombeo:.0f} MW")

    crudo, avisos = lee_prevision(ROOT / pre["archivo"])
    for a in avisos:
        print("  valor reparado:", a)
    if args.anios:
        crudo = crudo[crudo["FechaLocal"].dt.year.isin(args.anios)].reset_index(drop=True)
    h = a_horario(crudo, gas, co2, bombeo)
    z = variables(h.rename(columns={"p_engie": "p0"}), fundamentales)
    faltan = set(paquete["columnas"]) - set(z.columns)
    if faltan:
        raise SystemExit(f"faltan variables para la respuesta: {sorted(faltan)}")
    print(f"Prevision: {len(z):,} horas, GW = {gws}, tramos de {paso_mw:.0f} MW", flush=True)

    d, precio, diag, pt = corre(z, resp, gws, paso_mw, bat, sim)

    horario = pd.DataFrame({"timestamp_utc": z["timestamp_utc"], "p_engie": z["p0"]})
    for g in gws:
        horario[f"lp_{g}GW"], horario[f"net_charge_mw_{g}GW"] = precio[g], d[g]
    horario.to_parquet(salida / "prevision_lp.parquet", index=False)
    diag.to_csv(salida / "monthly_diagnostics.csv", index=False, float_format="%.2f")

    # modelo anterior (script 12), si esta calculado para los mismos años
    previo = salida.parent / "prevision_ajustada" / "prevision_ajustada.parquet"
    comp = horario[["timestamp_utc"]].merge(pd.read_parquet(previo), on="timestamp_utc", how="left") \
        if previo.exists() else None

    local = horario["timestamp_utc"].dt.tz_convert(TZ)
    anio = local.dt.year
    sp0 = spread_anual(horario["timestamp_utc"], horario["p_engie"], bat.duracion_horas)
    filas = []
    for g in gws:
        sp = spread_anual(horario["timestamp_utc"], horario[f"lp_{g}GW"], bat.duracion_horas)
        sp_prev = (spread_anual(horario["timestamp_utc"], comp[f"gbm_delta_{g}GW"], bat.duracion_horas)
                   if comp is not None and f"gbm_delta_{g}GW" in comp else None)
        for a in sorted(anio.unique()):
            m = (anio == a).to_numpy()
            dias = local[m].dt.date.nunique()
            ingreso = float(-(d[g][m] * precio[g][m]).sum()) / (g * 1000) / dias
            ingreso_pt = float(-(pt[m] * horario.loc[m, "p_engie"].to_numpy()).sum()) / dias
            meses_a = local[m].dt.strftime("%Y-%m").unique()
            fila = {"year": a, "gw": g, "price_E": horario.loc[m, "p_engie"].mean(),
                    "price_lp": horario.loc[m, f"lp_{g}GW"].mean(), "spread_E": sp0[a], "spread_lp": sp[a],
                    "revenue_price_taker": ingreso_pt, "revenue_equilibrium": ingreso,
                    "cycles_per_day": float(np.clip(-d[g][m], 0, None).sum() / (g * 1000 * bat.duracion_horas)
                                            / dias),
                    "equilibrium_gap_pct_max": float(diag[(diag["gw"] == g) & diag["month"].isin(meses_a)]
                                                     ["equilibrium_gap_pct"].max())}
            if sp_prev is not None:
                fila["price_gbm_delta"] = comp.loc[m, f"gbm_delta_{g}GW"].mean()
                fila["spread_gbm_delta"] = sp_prev[a]
            filas.append(fila)
    res = pd.DataFrame(filas)
    res["price_change_pct"] = 100 * (res["price_lp"] / res["price_E"] - 1)
    res["spread_change_pct"] = 100 * (res["spread_lp"] / res["spread_E"] - 1)
    res["revenue_vs_price_taker_pct"] = 100 * (res["revenue_equilibrium"] / res["revenue_price_taker"] - 1)
    ref = referencia_entrenamiento(proc, desde)
    cob = cobertura(a_horas(a_cuartos(crudo, gas, co2, bombeo), VARIABLES_REALES), ref)
    res = res.merge(cob[["all_in_range_excl_ccgt_pct"]].reset_index(), on="year", how="left")
    res.to_csv(salida / "annual_summary.csv", index=False, float_format="%.2f")
    cob.to_csv(salida / "coverage.csv", float_format="%.1f")

    figura_anual(res, gws, salida / "lp_spread_revenue.png")
    figura_perfil(horario, gws, [a for a in (2028, 2035) if a in set(anio)] or sorted(set(anio))[:1],
                  comp is not None, salida / "lp_profile.png")
    nota = ["Derived from the confidential ENGIE base-case forecast: internal use only.",
            f"Fixed inputs not in the forecast: gas {gas:.1f} €/MWh, CO2 {co2:.1f} €/t, "
            f"pumping {bombeo:.0f} MW.",
            "Constant installed power per scenario. Price response learned from OMIE curves "
            "Oct 2025-Nov 2026.",
            "revenue_price_taker = a battery that does not move prices; revenue_equilibrium = prices react."]
    wb = Workbook()
    for nombre, df in (("Summary", res), ("Coverage", cob.reset_index())):
        ws = wb.active if nombre == "Summary" else wb.create_sheet(nombre)
        ws.title = nombre
        ws.sheet_view.showGridLines = False
        ws["A1"], ws["A1"].font = f"ENGIE forecast with BESS in equilibrium: {nombre}", F_TIT
        for i, txt in enumerate(nota):
            ws.cell(row=2 + i, column=1, value=txt).font = F_NOTA
        tabla_excel(ws, 3 + len(nota), df, {c: "0.0" for c in df.columns if df[c].dtype.kind == "f"})
    wb.save(salida / "lp_forecast.xlsx")

    pd.set_option("display.width", 220)
    print("\nBrecha de equilibrio mensual (%): mediana / maximo por GW")
    print(diag.groupby("gw")["equilibrium_gap_pct"].agg(["median", "max"]).round(1).to_string())
    g = max(gws)
    cols = ["year", "price_E", "price_lp", "spread_E", "spread_lp", "spread_change_pct",
            "revenue_price_taker", "revenue_equilibrium", "cycles_per_day", "equilibrium_gap_pct_max",
            "all_in_range_excl_ccgt_pct"]
    print(f"\n{g} GW:")
    print(res[res["gw"] == g][cols].round(1).to_string(index=False))
    print(f"\nGuardado en {salida.relative_to(ROOT)}/ (ignorado por git)")


def figura_anual(res: pd.DataFrame, gws: list[int], salida: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    g0 = res[res["gw"] == gws[0]].set_index("year")
    ax = ejes[0]
    ax.plot(g0.index, g0["spread_E"], color=TINTA, linewidth=2.4, label="ENGIE forecast (no extra BESS)")
    for g in gws:
        s = res[res["gw"] == g].set_index("year")
        ax.plot(s.index, s["spread_lp"], color=AZULES.get(g, "#4b8fdd"), marker="o", markersize=4,
                linewidth=1.8, label=f"LP + price response, {g} GW")
    if "spread_gbm_delta" in res:
        s = res[res["gw"] == max(gws)].set_index("year")
        ax.plot(s.index, s["spread_gbm_delta"], color=NARANJA, linestyle="--", linewidth=1.6,
                label=f"Previous emulator (gbm Δ), {max(gws)} GW")
    ax.set_title("Mean daily 2 h spread (€/MWh)", loc="left", fontsize=11, color=TINTA)
    ax.legend(frameon=False, fontsize=8)
    estilo(ax)
    ax = ejes[1]
    ax.plot(g0.index, g0["revenue_price_taker"], color=TINTA, linewidth=2.4,
            label="Price taker (prices do not react)")
    for g in gws:
        s = res[res["gw"] == g].set_index("year")
        ax.plot(s.index, s["revenue_equilibrium"], color=AZULES.get(g, "#4b8fdd"), marker="o", markersize=4,
                linewidth=1.8, label=f"Equilibrium, {g} GW")
    ax.set_title("Revenue per MW of battery (€/MW/day)", loc="left", fontsize=11, color=TINTA)
    ax.legend(frameon=False, fontsize=8)
    estilo(ax)
    for a in ejes:
        a.set_xlabel("Year", fontsize=9, color=TINTA_2)
    fig.suptitle("ENGIE base-case forecast with BESS in price equilibrium", x=0.01, ha="left", fontsize=12,
                 color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)


def figura_perfil(horario: pd.DataFrame, gws: list[int], anios: list[int], con_previo: bool,
                  salida: Path) -> None:
    g = max(gws)
    hora = horario["timestamp_utc"].dt.tz_convert(TZ)
    fig, ejes = plt.subplots(1, len(anios), figsize=(5.6 * len(anios), 4.4), sharey=True, squeeze=False)
    previo = None
    if con_previo:
        ruta = ROOT / "data" / "processed" / "prevision_ajustada" / "prevision_ajustada.parquet"
        previo = pd.read_parquet(ruta)
        previo = horario[["timestamp_utc"]].merge(previo, on="timestamp_utc", how="left")
    for ax, a in zip(ejes[0], anios, strict=True):
        m = (hora.dt.year == a).to_numpy()
        def por_hora(x, m=m):
            return x[m].groupby(hora[m].dt.hour).mean()

        ax.plot(range(24), por_hora(horario["p_engie"]), color=TINTA, linewidth=2.4, label="ENGIE forecast")
        ax.plot(range(24), por_hora(horario[f"lp_{g}GW"]), color=AZULES.get(g, "#1f5fae"), linewidth=2,
                label=f"LP + price response, {g} GW")
        if previo is not None and f"gbm_delta_{g}GW" in previo:
            ax.plot(range(24), por_hora(previo[f"gbm_delta_{g}GW"]), color=NARANJA, linestyle="--",
                    linewidth=1.6, label=f"Previous emulator (gbm Δ), {g} GW")
        ax.set_title(str(a), loc="left", fontsize=11, color=TINTA)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour of day (local time)", fontsize=9, color=TINTA_2)
        estilo(ax)
    ejes[0][0].set_ylabel("Mean day-ahead price (€/MWh)", fontsize=9, color=TINTA_2)
    ejes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Mean price by hour with {g} GW of BESS", x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
