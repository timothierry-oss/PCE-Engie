"""Compara los modelos de impacto (script 10) entrenados con periodos distintos, sobre las mismas filas.

    uv run python scripts/11_compara_modelos.py     # parametros: config/base.yaml, seccion comparacion

No reentrena nada: lee las predicciones fuera de muestra (data/processed/predicciones_cv_<sufijo>.parquet,
una por entrenamiento) y el panel simulado de cada periodo (data/processed/panel_bess_*.parquet, para el
programa del parque). Solo se comparan las filas (cuarto de hora, GW) que estan en TODOS los entrenamientos:
sin esto, un modelo entrenado con mas meses se evaluaria sobre mas datos y la comparacion no seria justa.

Variantes = entrenamiento x modelo (gbm/ridge, delta/p0sq). Referencias: "simulated" (el simulador, la
verdad que se intenta imitar) y "no_impact" (p^X = p0, sin baterias). Salidas en reports/modelo/comparacion/
(textos en ingles, como el entregable):

    summary.csv      por variante: errores de precio, y error de spread e ingreso en toda la rejilla de GW
    by_gw.csv        por GW: precio medio, spread diario e ingreso de cada variante y de las referencias
    by_month.csv     error medio de precio por mes de prueba
    compare_gw.png   spread e ingreso frente a la potencia
    compare_error_gw.png  lo mismo como error frente al simulador (se ven mejor las diferencias)
    compare_month.png     error de precio por mes, un panel por modelo
    model_comparison.xlsx todo lo anterior en un Excel
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from openpyxl import Workbook

from pce68.dispatch.battery import Battery
from pce68.excel import F_NOTA, F_TIT, F_TXT
from pce68.excel import tabla as tabla_excel
from pce68.modelo.impacto import por_gw

ROOT = Path(__file__).resolve().parents[1]
CLAVE = ["timestamp_utc", "gw"]
TINTA, TINTA_2, GRIS = "#0b0b0b", "#52514e", "#898781"
# un color y un marcador por modelo (orden fijo, paleta validada: ver skill dataviz); el entrenamiento
# se distingue por el estilo de linea (continua = el ultimo, mas datos; punteada = el primero)
MODELOS = {
    "gbm_delta": ("#2a78d6", "o", "Gradient boosting, Δ target"),
    "gbm_p0sq": ("#eb6834", "s", "Gradient boosting, ratio (p0² weights)"),
    "ridge_delta": ("#1baf7a", "^", "Ridge, Δ target"),
    "ridge_p0sq": ("#eda100", "D", "Ridge, ratio (p0² weights)"),
}
ESTILOS = [":", "-", "--", "-."]


def estilo(ax) -> None:
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)
    ax.tick_params(colors=TINTA_2, labelsize=9)


def carga(cfg: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Filas comunes (t) y, por entrenamiento, sus predicciones alineadas con t."""
    proc = ROOT / cfg["datos"]["processed_dir"]
    panel = pd.concat([pd.read_parquet(f) for f in sorted(proc.glob("panel_bess_*.parquet"))])
    panel = panel[panel["analizado"]].drop_duplicates(CLAVE)
    panel["dia"] = pd.to_datetime(panel["dia"]).dt.date
    crudo = {}
    for e in cfg["comparacion"]["entrenamientos"]:
        d = pd.read_parquet(proc / f"predicciones_cv_{e['sufijo']}.parquet")
        crudo[e["nombre"]] = d.drop_duplicates(CLAVE)
    comunes = None
    for d in crudo.values():
        k = pd.MultiIndex.from_frame(d[CLAVE])
        comunes = k if comunes is None else comunes.intersection(k)
    t = (pd.DataFrame(index=comunes).reset_index()
         .merge(panel[[*CLAVE, "dia", "resolucion_min", "mw_parque", "precio_parque"]], on=CLAVE, how="left")
         .merge(next(iter(crudo.values()))[[*CLAVE, "p0"]], on=CLAVE, how="left")
         .sort_values(CLAVE).reset_index(drop=True))
    t["mes"] = pd.to_datetime(t["dia"]).dt.strftime("%Y-%m")
    preds = {n: t[CLAVE].merge(d, on=CLAVE, how="left") for n, d in crudo.items()}
    return t, preds


def calcula(t: pd.DataFrame, preds: dict[str, pd.DataFrame], modelos: list[str], horas: float):
    """summary, by_gw y by_month de todas las variantes."""
    columnas = {f"{n} | {m}": preds[n][f"p_{m}"] for n in preds for m in modelos}
    p = pd.DataFrame(columnas)
    p["no_impact"] = t["p0"]
    pg = por_gw(t, p.rename(columns=lambda c: f"p_{c}"), list(columnas), horas)
    spread_col = [c for c in pg.columns if c.startswith("spread_")][0]
    pg = pg.rename(columns={spread_col: "daily_spread_eur_mwh", "mean_price": "mean_price_eur_mwh",
                            "revenue_eur_mw_day": "revenue_eur_mw_day"})
    pg["training"] = pg["source"].str.split(" \\| ").str[0].where(pg["source"].str.contains("\\|"), "-")
    pg["model"] = pg["source"].str.split(" \\| ").str[-1]
    pg = pg[["gw", "training", "model", "mean_price_eur_mwh", "daily_spread_eur_mwh", "revenue_eur_mw_day"]]

    sim = pg[pg["model"] == "simulated"].set_index("gw")
    gmax = pg["gw"].max()
    filas = []
    for variante in [*columnas, "no_impact"]:
        entr, m = (variante.split(" | ") if " | " in variante else ("-", variante))
        precio = t["p0"] if m == "no_impact" else p[variante]
        e = (precio - t["precio_parque"]).to_numpy(float)
        bajo = (t["p0"].abs() < 5).to_numpy()
        x = pg[(pg["training"] == entr) & (pg["model"] == m)].set_index("gw")
        sp, ing = "daily_spread_eur_mwh", "revenue_eur_mw_day"
        rel_sp = float((100 * (x[sp] - sim[sp]).abs() / sim[sp]).mean())
        rel_ing = float((100 * (x[ing] - sim[ing]).abs() / sim[ing]).mean())
        filas.append({
            "training": entr, "model": m, "rows": len(t),
            "price_MAE_eur_mwh": np.abs(e).mean(), "price_RMSE_eur_mwh": np.sqrt((e**2).mean()),
            "price_MAE_p0_below_5": np.abs(e[bajo]).mean(), "price_MAE_p0_above_5": np.abs(e[~bajo]).mean(),
            "spread_error_pct_mean_over_GW": rel_sp,
            "revenue_error_pct_mean_over_GW": rel_ing,
            f"spread_error_{gmax:g}GW_eur_mwh": x.loc[gmax, sp] - sim.loc[gmax, sp],
            f"revenue_error_{gmax:g}GW_eur_mw_day": x.loc[gmax, ing] - sim.loc[gmax, ing],
        })
    resumen = pd.DataFrame(filas).sort_values("price_MAE_eur_mwh").reset_index(drop=True)

    meses = []
    for variante in [*columnas, "no_impact"]:
        entr, m = (variante.split(" | ") if " | " in variante else ("-", variante))
        precio = t["p0"] if m == "no_impact" else p[variante]
        err = (precio - t["precio_parque"]).abs().groupby(t["mes"]).mean()
        err = err.rename_axis("month").reset_index(name="price_MAE_eur_mwh")
        meses.append(err.assign(training=entr, model=m))
    return resumen, pg, pd.concat(meses, ignore_index=True)


def figura_gw(pg: pd.DataFrame, entrenos: list[str], modelos: list[str], salida: Path, error: bool,
              titulo: str) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    sim = pg[pg["model"] == "simulated"].set_index("gw")
    for ax, col, nombre in ((ejes[0], "daily_spread_eur_mwh", "Daily 2 h spread (€/MWh)"),
                            (ejes[1], "revenue_eur_mw_day", "Fleet revenue (€/MW/day)")):
        def serie(df, col=col):
            s = df.set_index("gw")[col]
            return s - sim[col] if error else s

        if error:
            ax.axhline(0, color=TINTA, linewidth=1.6, label="Simulator (reference)")
        else:
            ax.plot(sim.index, sim[col], color=TINTA, linewidth=2.4, label="Simulator (reference)")
        ni = serie(pg[pg["model"] == "no_impact"])
        if error:  # su error (decenas de €) aplasta la escala: se da en una nota
            nota = f"No impact (off scale), at {ni.index.max():g} GW: {ni.iloc[-1]:+.0f}"
            ax.text(0.02, 0.96, nota, transform=ax.transAxes, fontsize=8, color=TINTA_2, va="top")
        else:
            ax.plot(ni.index, ni.values, color=GRIS, linestyle="--", linewidth=1.6,
                    label="No impact (p = p0)")
        for m, (color, marca, etiqueta) in MODELOS.items():
            if m not in modelos:
                continue
            for k, n in enumerate(entrenos):
                s = serie(pg[(pg["training"] == n) & (pg["model"] == m)])
                ax.plot(s.index, s.values, color=color, marker=marca, markersize=5, linewidth=1.8,
                        linestyle=ESTILOS[k % len(ESTILOS)], label=f"{etiqueta}, trained on {n}")
        sufijo = " — error vs simulator" if error else ""
        ax.set_title(nombre + sufijo, loc="left", fontsize=11, color=TINTA)
        ax.set_xlabel("Installed battery power in Spain (GW)", fontsize=9, color=TINTA_2)
        estilo(ax)
    h, lab = ejes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0.19, 1, 0.95))
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)


def figura_mes(mes: pd.DataFrame, entrenos: list[str], modelos: list[str], salida: Path, titulo: str) -> None:
    fig, ejes = plt.subplots(2, 2, figsize=(11.5, 7), sharex=True, sharey=True)
    ni = mes[mes["model"] == "no_impact"].set_index("month")["price_MAE_eur_mwh"]
    for ax, m in zip(ejes.ravel(), modelos, strict=False):
        color, marca, etiqueta = MODELOS[m]
        ax.plot(ni.index, ni.values, color=GRIS, linestyle="--", linewidth=1.6, label="No impact (p = p0)")
        for k, n in enumerate(entrenos):
            s = mes[(mes["training"] == n) & (mes["model"] == m)].set_index("month")["price_MAE_eur_mwh"]
            ax.plot(s.index, s.values, color=color, marker=marca, markersize=5, linewidth=1.8,
                    linestyle=ESTILOS[k % len(ESTILOS)], label=f"trained on {n}")
        ax.set_title(etiqueta, loc="left", fontsize=10, color=TINTA)
        ax.legend(frameon=False, fontsize=8)
        ax.tick_params(axis="x", rotation=45)
        estilo(ax)
    for ax in ejes[:, 0]:
        ax.set_ylabel("Mean absolute price error (€/MWh)", fontsize=9, color=TINTA_2)
    fig.suptitle(titulo, x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)


def exporta_excel(resumen, pg, mes, nota: list[str], salida: Path) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = "Summary"
    ws.sheet_view.showGridLines = False
    ws["A1"], ws["A1"].font = "Model comparison on identical rows", F_TIT
    for i, txt in enumerate(nota):
        ws.cell(row=2 + i, column=1, value=txt).font = F_NOTA
    fila = 3 + len(nota)
    fmt = {c: "0.00" for c in resumen.columns if c not in ("training", "model", "rows")}
    fmt["rows"] = "#,##0"
    tabla_excel(ws, fila, resumen, fmt)
    for nombre, df in (("By GW", pg), ("By month", mes)):
        w = wb.create_sheet(nombre)
        w.sheet_view.showGridLines = False
        tabla_excel(w, 1, df, {c: "0.00" for c in df.columns if df[c].dtype.kind == "f"})
    ws2 = wb.create_sheet("Read me")
    ws2.sheet_view.showGridLines = False
    for i, txt in enumerate(["How to read this workbook", "",
                             "Summary: one row per variant (training period x model), sorted by price MAE.",
                             "By GW: price, daily spread and fleet revenue for each battery power.",
                             "By month: mean absolute price error per test month.", "", *nota]):
        ws2.cell(row=1 + i, column=1, value=txt).font = F_TIT if i == 0 else F_TXT
    ws2.column_dimensions["A"].width = 110
    wb.save(salida)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    com = cfg["comparacion"]
    modelos = list(com["modelos"])
    entrenos = [e["nombre"] for e in com["entrenamientos"]]
    horas = Battery(**cfg["bateria"]).duracion_horas
    carpeta = ROOT / "reports" / "modelo" / "comparacion"
    carpeta.mkdir(parents=True, exist_ok=True)

    t, preds = carga(cfg)
    faltan = t[["precio_parque", "mw_parque", "p0"]].isna().any(axis=1).sum()
    if faltan:
        raise SystemExit(f"{faltan} filas comunes sin panel o sin p0: falta algun panel_bess_*.parquet")
    periodo = f"{t['dia'].min():%d/%m/%Y}–{t['dia'].max():%d/%m/%Y}"
    resumen, pg, mes = calcula(t, preds, modelos, horas)
    resumen.to_csv(carpeta / "summary.csv", index=False, float_format="%.3f")
    pg.to_csv(carpeta / "by_gw.csv", index=False, float_format="%.3f")
    mes.to_csv(carpeta / "by_month.csv", index=False, float_format="%.3f")

    titulo = f"Same rows for every model: {periodo}, {t['dia'].nunique()} days, {len(t):,} quarter-hours"
    figura_gw(pg, entrenos, modelos, carpeta / "compare_gw.png", False, "Models vs simulator. " + titulo)
    figura_gw(pg, entrenos, modelos, carpeta / "compare_error_gw.png", True, "Error vs simulator. " + titulo)
    figura_mes(mes, entrenos, modelos, carpeta / "compare_month.png", "Price error by month. " + titulo)
    nota = [f"Evaluated on the {len(t):,} (quarter-hour, GW) rows common to all training sets: {periodo}.",
            "Out-of-sample: each month is predicted by a model trained without that month.",
            "Errors are against the simulator (the LP fleet on OMIE curves), not against the real market.",
            "Revenue values the simulated fleet schedule at each set of prices."]
    exporta_excel(resumen, pg, mes, nota, carpeta / "model_comparison.xlsx")

    pd.set_option("display.width", 220)
    print(f"{len(t):,} filas comunes, {periodo}, entrenamientos: {', '.join(entrenos)}\n")
    print(resumen.round(2).to_string(index=False))
    print(f"\nGuardado en {carpeta.relative_to(ROOT)}/")


if __name__ == "__main__":
    main()
