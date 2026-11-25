"""Panel p^X para una rejilla de parques de baterias (etapa 1 del documento BESS-cannibalization).

    uv run python scripts/08_simula_rejilla.py                 # parametros: config/base.yaml, rejilla
    uv run python scripts/08_simula_rejilla.py --desde-panel   # rehace resumen y figura sin simular

Una sola simulacion LP del parque mayor de la rejilla (pce68.dispatch.lp, mismos supuestos que
scripts/06_simula_lp.py), con una "foto" del estado en cada potencia: como los tramos no se
reoptimizan, la foto de X GW es exactamente el parque de X GW simulado solo.

Salidas (AAAAMMDD = primer y ultimo dia):

    data/processed/panel_bess_*.parquet  una fila por (cuarto de hora, gw): precio_base (p0),
                                         precio_parque (p^X), mw_parque, carga_mwh, analizado,
                                         y = p^X / p0 (NaN si |p0| < umbral_ratio_eur_mwh)
    reports/rejilla/resumen_gw_*.csv     por potencia: ingreso, precio medio, spread 2 h, y medio
    reports/rejilla/rejilla_*.png        ingreso por MW y spread 2 h frente a la potencia
    reports/rejilla/ratio_hora_*.png     y medio por hora del dia, una linea por potencia

Los dias de los extremos (simulacion_lp.dias_descartados) quedan con analizado = False y no
cuentan en el resumen. Textos de figuras y CSV en ingles, como el entregable.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pce68.data.curvas import preparar_dias
from pce68.dispatch.battery import Battery
from pce68.dispatch.lp import despacho_lp
from pce68.features.metrics import daily_spread

ROOT = Path(__file__).resolve().parents[1]
TZ = "Europe/Madrid"
AZUL = "#2a78d6"
TINTA, TINTA_2, GRIS = "#0b0b0b", "#52514e", "#898781"
# rampa secuencial azul (claro -> oscuro) para ordenar las potencias
RAMPA = ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95",
         "#104281", "#0d366b"]


def ratio(base: np.ndarray, p_parque: np.ndarray, umbral: float) -> np.ndarray:
    """y = p^X / p0 donde p0 != 0 y |p0| >= umbral; NaN en el resto."""
    base, p_parque = np.asarray(base, float), np.asarray(p_parque, float)
    definido = (base != 0) & (np.abs(base) >= umbral)
    return np.divide(p_parque, base, out=np.full(len(base), np.nan), where=definido)


def simula(rej: dict, sim: dict, bat: Battery, todos: list, analizados: list) -> tuple[pd.DataFrame, float]:
    """Corre el LP del parque mayor con fotos; devuelve el panel largo y el ingreso aceptante (€/MW/dia)."""
    gws = sorted(float(g) for g in rej["gw"])
    paso = float(sim["paso_mw"])
    ciclos = bat.ciclos_maximos_anuales * len(todos) / 365
    t0 = time.perf_counter()
    curvas = preparar_dias(ROOT / "data" / "processed" / "curvas", todos)
    base = curvas.precios()
    print(f"Curvas preparadas: {len(curvas.periodos)} periodos, {time.perf_counter() - t0:.0f} s")
    t0 = time.perf_counter()
    r = despacho_lp(curvas, gws[-1] * 1000, bat, paso, ciclos, float(sim["carga_bordes"]),
                    float(sim["penalizacion_eur_mwh"]), fotos_mw=[g * 1000 for g in gws])
    print(f"Simulacion: {len(r.attrs['tramos'])} tramos, {time.perf_counter() - t0:.0f} s")

    per = r[["timestamp_utc", "resolucion_min"]].copy()
    per.attrs = {}  # no arrastrar fotos y planes de r a cada parte del panel
    per["dia"] = per["timestamp_utc"].dt.tz_convert(TZ).dt.date
    per["analizado"] = per["dia"].isin(set(analizados))
    per["precio_base"] = base
    umbral = float(rej["umbral_ratio_eur_mwh"])
    partes = []
    for g in gws:
        f = r.attrs["fotos"][g * 1000]
        pX = f["precio_eur_mwh"].to_numpy(float)
        y = ratio(base, pX, umbral)
        partes.append(per.assign(gw=g, precio_parque=pX, mw_parque=f["delta_mw"].to_numpy(float),
                                 carga_mwh=f["carga_mwh"].to_numpy(float), y=y))
    panel = pd.concat(partes, ignore_index=True)

    h = per["resolucion_min"].to_numpy(float) / 60
    ventana = per["analizado"].to_numpy()
    aceptante = float(-(r.attrs["planes"][0] * base * h)[ventana].sum()) / len(analizados)
    return panel, aceptante


def resumen(panel: pd.DataFrame, bat: Battery, aceptante: float) -> pd.DataFrame:
    """Una fila por potencia (y la referencia sin parque), sobre los dias analizados."""
    a = panel[panel["analizado"]]
    filas = []
    base0 = a[a["gw"] == a["gw"].min()]
    sp0 = daily_spread(base0.rename(columns={"precio_base": "precio_es"}), bat.duracion_horas)
    filas.append({"gw": 0.0, "revenue_eur_mw": np.nan, "revenue_eur_mw_day": aceptante,
                  "mean_price": base0["precio_base"].mean(), "spread_2h": sp0.iloc[:, 1].mean(),
                  "cycles_per_day": np.nan, "mean_y": 1.0, "median_y": 1.0, "share_y_defined": np.nan})
    for g, s in a.groupby("gw"):
        h = s["resolucion_min"].to_numpy(float) / 60
        mw = g * 1000
        ingreso = float(-(s["mw_parque"] * s["precio_parque"] * h).sum())
        dias = s["dia"].nunique()
        sp = daily_spread(s.rename(columns={"precio_parque": "precio_es"}), bat.duracion_horas)
        descarga = float((np.clip(-s["mw_parque"].to_numpy(), 0, None) * h).sum())
        filas.append({
            "gw": g, "revenue_eur_mw": ingreso / mw, "revenue_eur_mw_day": ingreso / mw / dias,
            "mean_price": s["precio_parque"].mean(), "spread_2h": sp.iloc[:, 1].mean(),
            "cycles_per_day": descarga / (mw * bat.duracion_horas) / dias,
            "mean_y": s["y"].mean(), "median_y": s["y"].median(), "share_y_defined": s["y"].notna().mean(),
        })
    return pd.DataFrame(filas)


def estilo(ax) -> None:
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)
    ax.tick_params(colors=TINTA_2, labelsize=9)


def figura_rejilla(res: pd.DataFrame, titulo: str, salida: Path) -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, col, nombre in ((a1, "revenue_eur_mw_day", "Battery revenue (€/MW/day)"),
                            (a2, "spread_2h", "Mean daily 2 h spread (€/MWh)")):
        ax.plot(res["gw"], res[col], color=AZUL, linewidth=2, marker="o", markersize=7,
                markeredgecolor="white", markeredgewidth=1.5)
        for x, v in zip(res["gw"].iloc[[0, -1]], res[col].iloc[[0, -1]], strict=True):
            ax.annotate(f"{v:.0f}", (x, v), textcoords="offset points", xytext=(9, 4), ha="left",
                        fontsize=9, color=TINTA)
        ax.set_ylim(0, res[col].max() * 1.12)
        ax.set_title(nombre, loc="left", fontsize=11, color=TINTA)
        ax.set_xlabel("Installed battery power in Spain (GW)", fontsize=9, color=TINTA_2)
        estilo(ax)
    a1.text(0, -0.2, "0 GW: one price-taking MW on historical prices", transform=a1.transAxes,
            fontsize=8, color=TINTA_2)
    fig.suptitle(f"Spread cannibalisation, LP-dispatched 2 h fleet, {titulo}", x=0.01, ha="left",
                 fontsize=12, color=TINTA)
    fig.tight_layout()
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)


def figura_ratio_hora(panel: pd.DataFrame, titulo: str, salida: Path) -> None:
    a = panel[panel["analizado"]].copy()
    a["hora"] = a["timestamp_utc"].dt.tz_convert(TZ).dt.hour
    m = a.groupby(["gw", "hora"])["y"].mean().unstack(0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if len(m.columns) > len(RAMPA):
        raise ValueError(f"como mucho {len(RAMPA)} potencias en la figura por hora")
    for col, color in zip(m.columns, RAMPA[len(RAMPA) - len(m.columns):], strict=True):
        ax.plot(m.index, m[col], color=color, linewidth=2, label=f"{col:g} GW")
    ax.axhline(1, color=GRIS, linewidth=1)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlabel("Hour of day (local time)", fontsize=9, color=TINTA_2)
    ax.set_title(f"Mean price impact factor y = p^X / p0 by hour, {titulo}", loc="left", fontsize=11,
                 color=TINTA)
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="center left", bbox_to_anchor=(1.01, 0.5),
              title="Battery fleet", title_fontsize=8)
    estilo(ax)
    fig.tight_layout()
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    rej, sim = cfg["rejilla"], cfg["simulacion_lp"]
    p = argparse.ArgumentParser()
    p.add_argument("--desde", type=date.fromisoformat, default=date.fromisoformat(str(rej["desde"])))
    p.add_argument("--hasta", type=date.fromisoformat, default=date.fromisoformat(str(rej["hasta"])))
    p.add_argument("--desde-panel", action="store_true", help="no simular: usar el panel ya guardado")
    args = p.parse_args()
    bat = Battery(**cfg["bateria"])
    todos = [args.desde + timedelta(days=i) for i in range((args.hasta - args.desde).days + 1)]
    k = int(sim["dias_descartados"])
    analizados = todos[k:len(todos) - k]

    suf = f"{args.desde:%Y%m%d}_{args.hasta:%Y%m%d}"
    ruta_panel = ROOT / cfg["datos"]["processed_dir"] / f"panel_bess_{suf}.parquet"
    carpeta = ROOT / "reports" / "rejilla"
    carpeta.mkdir(parents=True, exist_ok=True)
    if args.desde_panel:
        panel = pd.read_parquet(ruta_panel)
        panel["dia"] = pd.to_datetime(panel["dia"]).dt.date
        # el umbral puede haber cambiado en la config: y se recalcula y se guarda
        panel["y"] = ratio(panel["precio_base"], panel["precio_parque"], float(rej["umbral_ratio_eur_mwh"]))
        panel.to_parquet(ruta_panel, index=False)
        aceptante = float(pd.read_csv(carpeta / f"resumen_gw_{suf}.csv").iloc[0]["revenue_eur_mw_day"])
    else:
        panel, aceptante = simula(rej, sim, bat, todos, analizados)
        panel.to_parquet(ruta_panel, index=False)

    res = resumen(panel, bat, aceptante)
    res.to_csv(carpeta / f"resumen_gw_{suf}.csv", index=False, float_format="%.4f")
    titulo = f"{analizados[0]:%d/%m/%Y}–{analizados[-1]:%d/%m/%Y}"
    figura_rejilla(res, titulo, carpeta / f"rejilla_{suf}.png")
    figura_ratio_hora(panel, titulo, carpeta / f"ratio_hora_{suf}.png")

    pd.set_option("display.width", 200)
    print(res.round(3).to_string(index=False))
    print(f"\nPanel: {len(panel)} filas -> {ruta_panel.relative_to(ROOT)}; resumen y figuras en "
          f"{carpeta.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
