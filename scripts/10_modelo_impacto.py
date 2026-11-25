"""Etapa 2 del documento BESS-cannibalization: aprender y = p^X / p0 = f(t, X, z), validado fuera de muestra.

    uv run python scripts/10_modelo_impacto.py                  # parametros: config/base.yaml, modelo
    uv run python scripts/10_modelo_impacto.py --modelos ridge  # solo la regresion ridge (rapido)
    uv run python scripts/10_modelo_impacto.py --objetivos delta  # solo la variante aditiva delta = p^X - p0

Necesita el panel de scripts/08_simula_rejilla.py, data/processed/fundamentales.parquet (07) y
data/processed/combustibles.parquet (09). Modelos y supuestos en pce68.modelo.impacto.

Validacion: cada mes se predice con un modelo entrenado sin ese mes. Salidas (AAAAMMDD = periodo
de la rejilla):

    reports/modelo/metricas_*.csv     errores en y y en precio (p^ = y^ * p0), por modelo
    reports/modelo/por_gw_*.csv       por potencia: precio medio, spread 2 h e ingreso del parque,
                                      simulados, predichos y sin impacto (y = 1)
    reports/modelo/modelo_gw_*.png    spread e ingreso frente a la potencia
    reports/modelo/modelo_hora_*.png  precio medio por hora con 5 y 17 GW
    reports/modelo/coeficientes_ridge_*.csv  coeficientes de la ridge final (efecto por variable)
    data/processed/predicciones_cv_*.parquet predicciones fuera de muestra, fila a fila
    data/processed/modelo_impacto_*.joblib   modelos entrenados con todo el periodo (etapa 3)

Textos de figuras y CSV en ingles, como el entregable.
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

from pce68.dispatch.battery import Battery
from pce68.modelo.impacto import (
    TZ,
    GBMDelta,
    GBMImpacto,
    RidgeDelta,
    RidgeImpacto,
    columnas_z,
    metricas,
    por_gw,
    precio_de,
    tabla_aprendizaje,
    validacion_por_mes,
)

ROOT = Path(__file__).resolve().parents[1]
TINTA, TINTA_2, GRIS = "#0b0b0b", "#52514e", "#898781"
# fuente -> (color, estilo, etiqueta); simulado en tinta (referencia), modelos en orden categorico
ESTILO = {
    "simulated": (TINTA, "-", "Simulated (curve shift)"),
    "ridge_p0sq": ("#2a78d6", "-", "Ridge, weighted by p0² (out of sample)"),
    "gbm_p0sq": ("#eb6834", "-", "Gradient boosting, weighted by p0² (out of sample)"),
    "ridge_delta": ("#7a4fd0", "-", "Ridge, additive target Δ = p^X − p0 (out of sample)"),
    "gbm_delta": ("#c2185b", "-", "Gradient boosting, additive target Δ = p^X − p0 (out of sample)"),
    "ridge": ("#1baf7a", "-", "Ridge, unweighted (out of sample)"),
    "gbm": ("#eda100", "-", "Gradient boosting, unweighted (out of sample)"),
    "no_impact": (GRIS, "--", "No impact (y = 1)"),
}
SUFIJO = {"ninguna": "", "p0_cuadrado": "_p0sq"}


def estilo(ax) -> None:
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)
    ax.tick_params(colors=TINTA_2, labelsize=9)


def figura_gw(pg: pd.DataFrame, fuentes: list[str], titulo: str, salida: Path) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(11, 4.4))
    col_sp = [c for c in pg.columns if c.startswith("spread_")][0]
    for ax, col, nombre in ((ejes[0], col_sp, "Mean daily 2 h spread (€/MWh)"),
                            (ejes[1], "revenue_eur_mw_day", "Fleet revenue (€/MW/day)")):
        for f in fuentes:
            s = pg[pg["source"] == f]
            color, ls, etiqueta = ESTILO[f]
            ax.plot(s["gw"], s[col], color=color, linestyle=ls, linewidth=2, marker="o", markersize=5,
                    label=etiqueta)
        ax.set_title(nombre, loc="left", fontsize=11, color=TINTA)
        ax.set_xlabel("Installed battery power in Spain (GW)", fontsize=9, color=TINTA_2)
        estilo(ax)
    ejes[1].text(0, -0.2, "Revenue: simulated fleet schedule valued at each set of prices",
                 transform=ejes[1].transAxes, fontsize=8, color=TINTA_2)
    ejes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Learned price-impact model vs simulator, {titulo}", x=0.01, ha="left", fontsize=12,
                 color=TINTA)
    fig.tight_layout()
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)


def figura_hora(t: pd.DataFrame, preds: pd.DataFrame, nombres: list[str], gws: list[float], titulo: str,
                salida: Path) -> None:
    hora = t["timestamp_utc"].dt.tz_convert(TZ).dt.hour
    fig, ejes = plt.subplots(1, len(gws), figsize=(5.5 * len(gws), 4.4), sharey=True)
    for ax, g in zip(np.atleast_1d(ejes), gws, strict=True):
        m = (t["gw"] == g).to_numpy()
        series = {"no_impact": t.loc[m, "p0"], "simulated": t.loc[m, "precio_parque"]}
        for n in nombres:
            series[n] = pd.Series(precio_de(t, preds, n)[m], index=t.index[m])
        for f in ("no_impact", "simulated", *nombres):
            color, ls, etiqueta = ESTILO[f]
            media = series[f].groupby(hora[m]).mean()
            ax.plot(media.index, media.values, color=color, linestyle=ls, linewidth=2, label=etiqueta)
        ax.set_title(f"{g:g} GW", loc="left", fontsize=11, color=TINTA)
        ax.set_xticks(range(0, 24, 3))
        ax.set_xlabel("Hour of day (local time)", fontsize=9, color=TINTA_2)
        estilo(ax)
    np.atleast_1d(ejes)[0].set_ylabel("Mean day-ahead price (€/MWh)", fontsize=9, color=TINTA_2)
    np.atleast_1d(ejes)[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Mean price by hour: simulated vs learned model, {titulo}", x=0.01, ha="left",
                 fontsize=12, color=TINTA)
    fig.tight_layout()
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)


def coeficientes_ridge(m: RidgeImpacto) -> pd.DataFrame:
    """Coeficientes de la ridge final: efecto de cada variable de phi(z) sobre y, en x y en x^2."""
    nombres = ([f"hour_{h:02d}" for h in range(24)]
               + [f"p0_spline_{k}" for k in range(m.splines_.n_features_out_)]
               + [f"{c} (standardised)" for c in m.continuas])
    coef = m.ridge_.coef_
    n = len(nombres)
    return pd.DataFrame({"feature": nombres, "coef_x": coef[:n], "coef_x2": coef[n:]})


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    rej, mod = cfg["rejilla"], cfg["modelo"]
    p = argparse.ArgumentParser()
    p.add_argument("--modelos", nargs="+", default=["ridge", "gbm"], choices=["ridge", "gbm"])
    p.add_argument("--ponderaciones", nargs="+", default=list(mod["ponderaciones"]), choices=list(SUFIJO),
                   help="ninguna = minimos cuadrados en y; p0_cuadrado = pesos p0^2 (error en precio)")
    p.add_argument("--objetivos", nargs="+", default=["ratio", "delta"], choices=["ratio", "delta"],
                   help="ratio = y = p^X / p0 (documento del cliente); delta = p^X - p0 en €/MWh")
    args = p.parse_args()
    bat = Battery(**cfg["bateria"])
    periodos = mod.get("periodos") or [{"desde": rej["desde"], "hasta": rej["hasta"]}]
    desdes = [pd.Timestamp(q["desde"]) for q in periodos]
    hastas = [pd.Timestamp(q["hasta"]) for q in periodos]
    suf = f"{min(desdes):%Y%m%d}_{max(hastas):%Y%m%d}"
    proc = ROOT / cfg["datos"]["processed_dir"]
    carpeta = ROOT / "reports" / "modelo"
    carpeta.mkdir(parents=True, exist_ok=True)

    panel = pd.concat([pd.read_parquet(proc / f"panel_bess_{d:%Y%m%d}_{h:%Y%m%d}.parquet")
                       for d, h in zip(desdes, hastas, strict=True)], ignore_index=True)
    panel["dia"] = pd.to_datetime(panel["dia"]).dt.date
    fundamentales = list(mod["fundamentales"])
    t = tabla_aprendizaje(panel, pd.read_parquet(proc / "fundamentales.parquet"),
                          pd.read_parquet(proc / "combustibles.parquet"), fundamentales)
    cols = columnas_z(t)
    print(f"Tabla: {len(t)} filas ({t['dia'].nunique()} dias x {t['gw'].nunique()} potencias), "
          f"{len(cols)} variables z; y definido en {t['y'].notna().mean():.1%} de las filas")
    print(f"  y: mediana {t['y'].median():.3f}, percentil 99 {t['y'].quantile(0.99):.1f}, "
          f"min {t['y'].min():.0f}, max {t['y'].max():.0f}")
    faltan = t[cols].isna().mean()
    if faltan.max() > 0:
        print(f"  variables con huecos: {faltan[faltan > 0].round(4).to_dict()}")

    fabricas = {}
    if "delta" in args.objetivos:  # el delta no necesita ponderar: su error cuadratico ya es el del precio
        if "ridge" in args.modelos:
            fabricas["ridge_delta"] = lambda: RidgeDelta(cols, alpha=float(mod["alpha_ridge"]))
        if "gbm" in args.modelos:
            fabricas["gbm_delta"] = lambda: GBMDelta(cols, **mod["gbm"])
    for pond in args.ponderaciones if "ratio" in args.objetivos else []:
        if "ridge" in args.modelos:
            fabricas[f"ridge{SUFIJO[pond]}"] = (
                lambda pond=pond: RidgeImpacto(cols, alpha=float(mod["alpha_ridge"]), ponderacion=pond))
        if "gbm" in args.modelos:
            fabricas[f"gbm{SUFIJO[pond]}"] = (
                lambda pond=pond: GBMImpacto(cols, ponderacion=pond, **mod["gbm"]))
    nombres = list(fabricas)
    # en las figuras, los modelos ponderados si los hay (los otros tienen errores de cientos de €/MWh)
    en_figura = [n for n in nombres if n.endswith(("_p0sq", "_delta"))] or nombres

    t0 = time.perf_counter()
    preds = validacion_por_mes(t, fabricas)
    print(f"Validacion por mes ({t['mes'].nunique()} meses): {time.perf_counter() - t0:.0f} s")
    preds["y_no_impact"] = 1.0

    met = metricas(t, preds, nombres + ["no_impact"])
    pg = por_gw(t, preds, nombres, bat.duracion_horas)
    por_mes = pd.concat([metricas(t.loc[idx], preds.loc[idx], nombres + ["no_impact"]).assign(month=mes)
                         for mes, idx in t.groupby("mes").groups.items()], ignore_index=True)
    met.to_csv(carpeta / f"metricas_{suf}.csv", index=False, float_format="%.4f")
    por_mes.to_csv(carpeta / f"metricas_por_mes_{suf}.csv", index=False, float_format="%.4f")
    pg.to_csv(carpeta / f"por_gw_{suf}.csv", index=False, float_format="%.3f")
    t[["timestamp_utc", "gw", "p0", "precio_parque", "y"]].join(preds).to_parquet(
        proc / f"predicciones_cv_{suf}.parquet", index=False)

    titulo = f"{t['dia'].min():%d/%m/%Y}–{t['dia'].max():%d/%m/%Y}"
    figura_gw(pg, ["simulated", *en_figura, "no_impact"], titulo, carpeta / f"modelo_gw_{suf}.png")
    gws_fig = [g for g in (5.0, 17.0) if g in set(t["gw"])]
    figura_hora(t, preds, en_figura, gws_fig, titulo, carpeta / f"modelo_hora_{suf}.png")

    # modelos finales, con todo el periodo, para la etapa 3 (pce68.modelo.impacto.ajusta_prevision)
    finales = {n: f().fit(t, t["gw"].to_numpy(), t["y"].to_numpy()) for n, f in fabricas.items()}
    for n, m in finales.items():
        if n.startswith("ridge"):
            coeficientes_ridge(m).to_csv(carpeta / f"coeficientes_{n}_{suf}.csv", index=False,
                                         float_format="%.5f")
    joblib.dump({"modelos": finales, "columnas": cols, "fundamentales": fundamentales, "periodo": suf},
                proc / f"modelo_impacto_{suf}.joblib")

    pd.set_option("display.width", 220)
    print("\nErrores fuera de muestra:")
    print(met.round(3).to_string(index=False))
    print("\nError medio en precio por mes de prueba (€/MWh):")
    print(por_mes.pivot(index="month", columns="model", values="mae_price_eur_mwh").round(1).to_string())
    print("\nPor potencia:")
    print(pg.pivot(index="gw", columns="source").round(1).to_string())
    print(f"\nGuardado en {carpeta.relative_to(ROOT)}/ y {proc.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
