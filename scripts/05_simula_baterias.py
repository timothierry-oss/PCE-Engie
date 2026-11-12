"""Simula parques de baterias sobre las curvas de OMIE de un dia y compara antes/despues.

    uv run python scripts/05_simula_baterias.py                       # dia y escenarios de config
    uv run python scripts/05_simula_baterias.py --dia 2025-05-15 --gw 0 5 17 --paso-mw 0

Para cada escenario (GW instalados) añade la carga como demanda y la descarga como oferta
(pce68.dispatch.flota), recalcula el precio de cada periodo y guarda en reports/simulacion/:

    precios_AAAAMMDD.csv     precio y potencia del parque por periodo y escenario
    resumen_AAAAMMDD.csv     spread de 2 h e ingreso por MW de cada escenario
    perfil_AAAAMMDD.png      precio del dia y carga/descarga del parque
    curvas_AAAAMMDD.png      curvas de oferta y demanda antes/despues en una hora de carga
                             y una de descarga
    resumen_AAAAMMDD.png     spread e ingreso por MW segun el tamaño del parque
    simulacion_AAAAMMDD.xlsx las mismas tablas con graficos nativos de Excel

Son SALIDAS: se regeneran enteras cada vez y nunca se editan a mano (para trabajar en
Excel, haz una copia). Sus textos estan en ingles, como el entregable.
Los supuestos estan en pce68.dispatch.flota.

Con --desde-csv no se simula: Excel y graficos se rehacen con los CSV ya guardados.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from openpyxl import Workbook
from openpyxl.chart import LineChart, Reference, ScatterChart, Series

from pce68.data.curvas import PRECIO_MAX, PRECIO_MIN, nombre_fichero, parsear_curvas
from pce68.dispatch.battery import Battery
from pce68.dispatch.flota import ciclo, despacho
from pce68.excel import F_NOTA, F_TIT, F_TXT
from pce68.excel import tabla as tabla_excel
from pce68.features.metrics import daily_spread

ROOT = Path(__file__).resolve().parents[1]
TZ = "Europe/Madrid"

# Tinta y rampa ordinal azul (mas oscuro = mas GW); la referencia sin baterias en gris.
TINTA, TINTA_2, MUDO, REJILLA = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
AZULES = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
NARANJA = "#eb6834"


def cargar_curvas(dia: date) -> pd.DataFrame:
    crudo = ROOT / "data" / "raw" / "curva_pbc" / nombre_fichero(dia)
    procesado = ROOT / "data" / "processed" / "curvas" / f"curvas_{dia:%Y%m%d}.parquet"
    if crudo.exists():
        return parsear_curvas(crudo.read_bytes().decode("latin-1"))
    if procesado.exists():
        return pd.read_parquet(procesado)
    raise SystemExit(f"No hay curvas de {dia}: descargalas con scripts/02_descarga_curvas.py")


def colores(gws: list[float]) -> dict[float, str]:
    con = [g for g in gws if g > 0]
    paso = (len(AZULES) - 1) / max(len(con) - 1, 1)
    return {0: MUDO} | {g: AZULES[round(i * paso)] for i, g in enumerate(con)}


def etiqueta(gw: float) -> str:
    return "no batteries" if gw == 0 else f"{gw:g} GW"


def estilo(ax) -> None:
    ax.grid(True, color=REJILLA, linewidth=0.6)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c3c2b7")
    ax.tick_params(colors=TINTA_2, labelsize=9)


def escalones(g: pd.DataFrame, lado: str, extra_mw: float) -> tuple[np.ndarray, np.ndarray]:
    """Curva acumulada (MW, €/MWh) de un periodo; `extra_mw` = bloque de bateria casado siempre."""
    s = g[g["lado"] == lado]
    p = s["precio"].to_numpy(float)
    q = s["potencia_mw"].to_numpy(float)
    # los flujos casados y la bateria van a los extremos de precio, como en curvas.precios
    if lado == "oferta":
        p = np.where(s["tipologia"].str.startswith("Imp"), PRECIO_MIN, p)
        p, q = np.append(p, PRECIO_MIN), np.append(q, extra_mw)
        orden = np.argsort(p, kind="stable")
    else:
        p = np.where(s["tipologia"].str.startswith("Exp"), PRECIO_MAX, p)
        p, q = np.append(p, PRECIO_MAX), np.append(q, extra_mw)
        orden = np.argsort(-p, kind="stable")
    return np.concatenate([[0.0], np.cumsum(q[orden])]), np.concatenate([[p[orden][0]], p[orden]])


def figura_perfil(tabla: pd.DataFrame, gws, col, dia, salida: Path) -> None:
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, height_ratios=[3, 2])
    x = tabla["hora"].to_numpy()
    for gw in gws:
        ls, lw = ("--", 1.6) if gw == 0 else ("-", 2)
        ax.step(x, tabla[f"precio_{gw:g}gw"], where="post", color=col[gw], ls=ls, lw=lw, label=etiqueta(gw))
        if gw > 0:
            ax2.step(x, tabla[f"mw_{gw:g}gw"] / 1000, where="post", color=col[gw], lw=2, label=etiqueta(gw))
    ax.set_title(f"Spanish day-ahead price, {dia:%d/%m/%Y}: before and after batteries",
                 loc="left", color=TINTA, fontsize=12)
    ax.set_ylabel("€/MWh", color=TINTA_2)
    ax.legend(frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.42, 0.85), labelcolor=TINTA_2)
    for gw in gws:  # etiqueta directa al final de cada linea
        ax.annotate(etiqueta(gw), xy=(x[-1], tabla[f"precio_{gw:g}gw"].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=8, color=TINTA_2)
    ax2.axhline(0, color="#c3c2b7", lw=1)
    ax2.set_ylabel("GW on the grid", color=TINTA_2)
    ax2.set_title("Battery fleet: charging (+, extra demand) and discharging (−, extra supply)",
                  loc="left", color=TINTA_2, fontsize=10)
    ax2.set_xlabel("local hour (start of period)", color=TINTA_2)
    ax2.set_xticks(range(0, 25, 2))
    for a in (ax, ax2):
        estilo(a)
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def figura_curvas(curvas: pd.DataFrame, tabla: pd.DataFrame, gw: float, col, dia, salida: Path) -> None:
    delta = tabla[f"mw_{gw:g}gw"]
    horas = [("carga", int(delta.idxmax())), ("descarga", int(delta.idxmin()))]
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (que, i) in zip(axs, horas, strict=True):
        fila = tabla.loc[i]
        g = curvas[curvas["timestamp_utc"] == fila["timestamp_utc"]]
        g = g[~g["tipologia"].str.fullmatch(r"C\d+") | g["casada"]]  # como curvas.precios por defecto
        carga, descarga = max(fila[f"mw_{gw:g}gw"], 0.0), max(-fila[f"mw_{gw:g}gw"], 0.0)
        cortes = []
        for nombre, extra_d, extra_o, color, ls in (
            (etiqueta(0), 0.0, 0.0, MUDO, "--"),
            (etiqueta(gw), carga, descarga, col[gw], "-"),
        ):
            qo, po = escalones(g, "oferta", extra_o)
            qd, pdm = escalones(g, "demanda", extra_d)
            ax.step(qo / 1000, po, where="pre", color=color, ls=ls, lw=2, label=f"supply, {nombre}")
            ax.step(qd / 1000, pdm, where="pre", color=color, ls=ls, lw=1.2, alpha=0.8,
                    label=f"demand, {nombre}")
            cortes.append(qd[-1] / 1000)
        p0, p1 = round(fila["precio_0gw"], 1) + 0.0, round(fila[f"precio_{gw:g}gw"], 1) + 0.0  # sin "-0.0"
        ax.axhline(p0, color=MUDO, lw=0.8, ls=":")
        ax.axhline(p1, color=col[gw], lw=0.8, ls=":")
        ax.annotate(f"{p0:.1f} → {p1:.1f} €/MWh", xy=(0.02, 0.95), xycoords="axes fraction",
                    color=TINTA, fontsize=10, va="top")
        mw = carga if que == "carga" else descarga
        ax.set_title(f"{fila['hora_txt']}: the fleet {'charges' if que == 'carga' else 'discharges'} "
                     f"{mw / 1000:.1f} GW", loc="left", color=TINTA, fontsize=11)
        ax.set_ylim(-60, max(150, p0, p1) * 1.3)
        ax.set_xlim(0, max(cortes) * 1.05)
        ax.set_xlabel("cumulative power (GW)", color=TINTA_2)
        ax.set_ylabel("€/MWh", color=TINTA_2)
        estilo(ax)
    axs[0].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(0.02, 0.6),
                  labelcolor=TINTA_2)
    fig.suptitle(f"OMIE aggregate curves, {dia:%d/%m/%Y} (complex bids: accepted volume; prices at −500 and "
                 f"{PRECIO_MAX:g} €/MWh off the axis)", x=0.01, ha="left", color=TINTA_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def figura_resumen(resumen: pd.DataFrame, dia, salida: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = resumen["gw"]
    ax.plot(x, resumen["spread_2h"], color=AZULES[2], lw=2, marker="o", ms=8)
    ax.set_title("Daily 2 h spread (€/MWh)", loc="left", color=TINTA, fontsize=11)
    ax2.plot(x, resumen["ingreso_eur_mw"], color=AZULES[2], lw=2, marker="o", ms=8,
             label="fleet that moves the price")
    ax2.axhline(resumen["ingreso_aceptante_eur_mw"].iloc[0], color=NARANJA, lw=1.6, ls="--",
                label="1 MW price-taker (ceiling)")
    ax2.set_title("Daily revenue per installed MW (€/MW)", loc="left", color=TINTA, fontsize=11)
    ax2.legend(frameon=False, fontsize=9, labelcolor=TINTA_2)
    for a in (ax, ax2):
        a.set_xlabel("GW of batteries", color=TINTA_2)
        a.set_xticks(x)
        a.set_ylim(bottom=min(0, a.get_ylim()[0]))
        estilo(a)
    fig.suptitle(f"{dia:%d/%m/%Y}", x=0.01, ha="left", color=TINTA_2, fontsize=9)
    fig.tight_layout()
    fig.savefig(salida, dpi=150)
    plt.close(fig)


def _linea(serie, color: str, ancho_pt: float = 2.0, discontinua: bool = False) -> None:
    serie.graphicalProperties.line.solidFill = color.lstrip("#")
    serie.graphicalProperties.line.width = int(ancho_pt * 12700)  # EMU
    serie.smooth = False
    if discontinua:
        serie.graphicalProperties.line.dashStyle = "dash"


def exporta_excel(tabla: pd.DataFrame, resumen: pd.DataFrame, gws, dia, params: dict, salida: Path) -> None:
    """Libro con los resultados de la simulacion y graficos nativos de Excel."""
    col = colores(gws)
    eur = "#,##0.00"
    wb = Workbook()

    ws = wb.active
    ws.title = "Read me"
    ws.sheet_view.showGridLines = False
    ws["A1"] = f"Battery fleet simulation on the OMIE curves, {dia:%d/%m/%Y}"
    ws["A1"].font = F_TIT
    textos = [
        "Generated by scripts/05_simula_baterias.py. This file is an OUTPUT: it is fully regenerated",
        "every time the script runs. Save a copy under another name to work on it by hand.",
        "",
        "Sheets:",
        "  Prices     price of every hour in every scenario (€/MWh)",
        "  Fleet      fleet power on the grid: + charging (extra demand), − discharging (extra supply)",
        "  Summary    2 h spread and daily revenue per installed MW against GW installed",
        "",
        "Parameters:",
        *[f"  {k}: {v}" for k, v in params.items()],
        "",
        "Main assumptions (details in src/pce68/dispatch/flota.py):",
        "  perfect foresight, at most one cycle per day, fixed interconnection flows,",
        "  other bids (pumped hydro, hydro, solar) do not react to the batteries.",
    ]
    for i, t in enumerate(textos):
        ws.cell(row=3 + i, column=1, value=t).font = F_TXT if t else F_NOTA
    ws.column_dimensions["A"].width = 100

    def hoja_horaria(nombre: str, titulo: str, nota: str, prefijo: str, escala: float, fmt: str,
                     eje_y: str, con_ref: bool):
        ws = wb.create_sheet(nombre)
        ws.sheet_view.showGridLines = False
        ws["A1"], ws["A2"] = titulo, nota
        ws["A1"].font, ws["A2"].font = F_TIT, F_NOTA
        escenarios = [g for g in gws if con_ref or g > 0]
        df = pd.DataFrame({"Hour": tabla["hora_txt"]})
        for g in escenarios:
            df[etiqueta(g)] = tabla[f"{prefijo}_{g:g}gw"] / escala
        fin = tabla_excel(ws, 4, df, {c: fmt for c in df.columns if c != "Hour"})
        ch = LineChart()
        ch.title, ch.height, ch.width = titulo, 10, 24
        ch.y_axis.title, ch.x_axis.title = eje_y, "local hour"
        ch.add_data(Reference(ws, min_col=2, max_col=len(df.columns), min_row=4, max_row=fin),
                    titles_from_data=True)
        ch.set_categories(Reference(ws, min_col=1, min_row=5, max_row=fin))
        for serie, g in zip(ch.series, escenarios, strict=True):
            _linea(serie, col[g], 1.5 if g == 0 else 2.25, discontinua=g == 0)
        ch.x_axis.delete = ch.y_axis.delete = False
        ws.add_chart(ch, "H4")

    hoja_horaria("Prices", "Spanish day-ahead price (€/MWh)",
                 "Dashed grey line: no batteries. Darker = more GW.",
                 "precio", 1, eur, "€/MWh", con_ref=True)
    hoja_horaria("Fleet", "Battery fleet power on the grid (GW)",
                 "Positive = the fleet charges (extra demand); negative = it discharges (extra supply).",
                 "mw", 1000, "0.00", "GW", con_ref=False)

    ws = wb.create_sheet("Summary")
    ws.sheet_view.showGridLines = False
    ws["A1"], ws["A2"] = "Cannibalisation: spread and revenue against GW installed", (
        "Ceiling = 1 MW that does not move the price (pce68.dispatch.battery). Revenue of a single day.")
    ws["A1"].font, ws["A2"].font = F_TIT, F_NOTA
    df = pd.DataFrame({
        "GW installed": resumen["gw"],
        "MW cycling": resumen["mw_activos"],
        "Mean price (€/MWh)": resumen["precio_medio"],
        "Min. price (€/MWh)": resumen["precio_min"],
        "Max. price (€/MWh)": resumen["precio_max"],
        "2 h spread (€/MWh)": resumen["spread_2h"],
        "Revenue (€/MW·day)": resumen["ingreso_eur_mw"],
        "Price-taker ceiling (€/MW·day)": resumen["ingreso_aceptante_eur_mw"],
    })
    formatos = {c: eur for c in df.columns if c != "MW cycling"} | {"MW cycling": "#,##0"}
    fin = tabla_excel(ws, 4, df, formatos)
    x = Reference(ws, min_col=1, min_row=5, max_row=fin)
    for n, (columnas, titulo, eje, ancla) in enumerate([
        ([6], "Daily 2 h spread (€/MWh)", "€/MWh", "J4"),
        ([7, 8], "Daily revenue per installed MW (€/MW)", "€/MW", "J24"),
    ]):
        ch = ScatterChart()
        ch.title, ch.height, ch.width, ch.style = titulo, 9, 18, 13
        ch.x_axis.title, ch.y_axis.title = "GW of batteries", eje
        for j, c in enumerate(columnas):
            serie = Series(Reference(ws, min_col=c, min_row=5, max_row=fin), x, title=df.columns[c - 1])
            techo = n == 1 and j == 1
            _linea(serie, NARANJA if techo else AZULES[2], 1.75 if techo else 2.25, discontinua=techo)
            serie.marker.symbol = "none" if techo else "circle"
            if not techo:
                serie.marker.size = 8
                serie.marker.graphicalProperties.solidFill = AZULES[2].lstrip("#")
                serie.marker.graphicalProperties.line.solidFill = AZULES[2].lstrip("#")
            ch.series.append(serie)
        ch.x_axis.delete = ch.y_axis.delete = False
        if len(columnas) == 1:
            ch.legend = None
        ws.add_chart(ch, ancla)

    wb.save(salida)


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    sim = cfg["simulacion"]
    p = argparse.ArgumentParser()
    p.add_argument("--dia", type=date.fromisoformat, default=date.fromisoformat(str(sim["dia"])))
    p.add_argument("--gw", type=float, nargs="+", default=sim["escenarios_gw"])
    p.add_argument("--paso-mw", type=float, default=sim["paso_mw"],
                   help="tamaño de cada tramo del parque; 0 = todo el parque en las mismas horas")
    p.add_argument("--desde-csv", action="store_true",
                   help="no simular: rehacer Excel y graficos con los CSV de reports/simulacion")
    args = p.parse_args()
    bat = Battery(**cfg["bateria"])
    gws = sorted(set([0.0] + [float(g) for g in args.gw]))
    dia = args.dia

    carpeta = ROOT / "reports" / "simulacion"
    carpeta.mkdir(parents=True, exist_ok=True)
    sufijo = f"{dia:%Y%m%d}"
    curvas = cargar_curvas(dia)
    if args.desde_csv:
        tabla = pd.read_csv(carpeta / f"precios_{sufijo}.csv", parse_dates=["timestamp_utc"])
        resumen = pd.read_csv(carpeta / f"resumen_{sufijo}.csv")
        gws = sorted(resumen["gw"].astype(float))
    else:
        tabla, resumen = simula(curvas, gws, bat, args.paso_mw)
        csv = carpeta / f"precios_{sufijo}.csv"
        tabla.drop(columns=["hora"]).to_csv(csv, index=False, float_format="%.2f")
        resumen.to_csv(carpeta / f"resumen_{sufijo}.csv", index=False, float_format="%.2f")
    local = tabla["timestamp_utc"].dt.tz_convert(TZ)
    tabla["hora"] = local.dt.hour + local.dt.minute / 60
    h = tabla["resolucion_min"].to_numpy(float) / 60

    col = colores(gws)
    # la ultima fila repetida cierra el escalon de la ultima hora en el grafico
    cierre = tabla.iloc[[-1]].assign(hora=tabla["hora"].iloc[-1] + h[-1])
    perfil = pd.concat([tabla, cierre], ignore_index=True)
    figura_perfil(perfil, gws, col, dia, carpeta / f"perfil_{sufijo}.png")
    if max(gws) > 0:
        figura_curvas(curvas, tabla, max(gws), col, dia, carpeta / f"curvas_{sufijo}.png")
    figura_resumen(resumen, dia, carpeta / f"resumen_{sufijo}.png")
    params = {
        "day": f"{dia:%d/%m/%Y}",
        "scenarios (GW)": ", ".join(f"{g:g}" for g in gws),
        "fleet tranche (MW)": "whole fleet at once" if args.paso_mw <= 0 else f"{args.paso_mw:g}",
        "battery": f"{bat.duracion_horas:g} h, round-trip efficiency {bat.eficiencia_ida_vuelta:g}",
    }
    exporta_excel(tabla, resumen, gws, dia, params, carpeta / f"simulacion_{sufijo}.xlsx")
    print(f"\nGuardado en {carpeta.relative_to(ROOT)}/")


def simula(curvas: pd.DataFrame, gws, bat: Battery, paso_mw: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Precios y potencia del parque por periodo, y resumen por escenario."""
    tabla = None
    filas = []
    for gw in gws:
        mw = gw * 1000
        paso = mw if paso_mw <= 0 else paso_mw
        r = despacho(curvas, mw, bat, paso) if mw > 0 else despacho(curvas, 0, bat, 1)
        if tabla is None:
            tabla = r[["timestamp_utc", "resolucion_min"]].copy()
        tabla[f"precio_{gw:g}gw"] = r["precio_eur_mwh"]
        tabla[f"mw_{gw:g}gw"] = r["delta_mw"]

        h = r["resolucion_min"].to_numpy(float) / 60
        spread = daily_spread(r.rename(columns={"precio_eur_mwh": "precio_es"}), bat.duracion_horas)
        ingreso = float((-r["delta_mw"] * r["precio_eur_mwh"] * h).sum() / mw) if mw > 0 else np.nan
        filas.append({
            "gw": gw,
            "mw_activos": sum(r.attrs["mw_activos"].values()),
            "precio_medio": r["precio_eur_mwh"].mean(),
            "precio_min": r["precio_eur_mwh"].min(),
            "precio_max": r["precio_eur_mwh"].max(),
            "spread_2h": spread.iloc[0, 1],
            "ingreso_eur_mw": ingreso,
        })
        print(f"{etiqueta(gw):>13s}: spread 2 h = {filas[-1]['spread_2h']:6.2f} €/MWh, "
              f"ingreso = {ingreso:7.2f} €/MW, MW ciclando = {filas[-1]['mw_activos']:,.0f}")

    precio_ref = tabla["precio_0gw"].to_numpy(float)
    h = tabla["resolucion_min"].to_numpy(float) / 60
    resumen = pd.DataFrame(filas)
    resumen["ingreso_aceptante_eur_mw"] = ciclo(precio_ref, h, bat)[0]
    resumen.loc[resumen["gw"] == 0, "ingreso_eur_mw"] = resumen["ingreso_aceptante_eur_mw"]

    local = tabla["timestamp_utc"].dt.tz_convert(TZ)
    tabla["hora_txt"] = local.dt.strftime("%H:%M")
    tabla["timestamp_local"] = local.dt.strftime("%Y-%m-%d %H:%M")
    tabla["hora"] = local.dt.hour + local.dt.minute / 60
    return tabla, resumen


if __name__ == "__main__":
    main()
