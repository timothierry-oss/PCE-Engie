"""Respuesta del precio a las baterias, aprendida de las curvas, y su validacion contra el simulador.

    uv run python scripts/13_respuesta_precio.py        # parametros: config/base.yaml, seccion respuesta

1. Para cada cuarto de hora de las curvas (oct 2025 - nov 2026) y cada nivel de potencia neta del parque,
   calcula el efecto EXACTO sobre el precio (pce68.modelo.respuesta.tabla_respuesta). Se guarda en cache
   (data/processed/respuesta_dp_*.npz): es lo mas lento.
2. Entrena g(d, z) solo con los periodos ANTERIORES a respuesta.validacion_desde y mide su error estatico
   en los de prueba (¿predice bien el efecto de d MW sobre un periodo que no vio?).
3. Prueba de punta a punta: en los meses de prueba, con SOLO el precio sin baterias p0 y las variables z
   (no las curvas), el LP en equilibrio exacto (pce68.dispatch.equilibrio) planifica el parque y los precios
   reaccionan con g. Se compara con el simulador de curvas (panel de scripts/08_simula_rejilla.py: la
   verdad que se imita) y con el emulador anterior de scripts/10 (gbm_delta).
4. Reentrena con todo el periodo y guarda el modelo para la prevision de ENGIE (script 14).

Salidas (datos publicos de OMIE/ESIOS; textos en ingles) en reports/respuesta/:

    validation_static.csv       error del efecto sobre el precio por nivel de potencia (periodos de prueba)
    validation_end_to_end.csv   por GW: precio medio, spread 2 h, ingreso, ciclos, error de precio, brecha
    validation_gw.png           spread e ingreso frente a la potencia: simulador, sin impacto, LP+respuesta,
                                emulador anterior
    data/processed/respuesta_precio_*.joblib   modelo final (script 14)
"""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from pce68.data.curvas import preparar_dias
from pce68.dispatch.battery import Battery
from pce68.dispatch.equilibrio import equilibrio_exacto, precios_marginales, segmentos
from pce68.dispatch.lp import plan_lp
from pce68.features.metrics import daily_spread
from pce68.modelo.impacto import TZ, columnas_z, tabla_aprendizaje
from pce68.modelo.respuesta import RespuestaPrecio, tabla_respuesta

ROOT = Path(__file__).resolve().parents[1]
TINTA, TINTA_2, GRIS = "#0b0b0b", "#52514e", "#898781"
CLAVE = ["timestamp_utc", "gw"]


def estilo(ax) -> None:
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRIS)
    ax.tick_params(colors=TINTA_2, labelsize=9)


def datos_respuesta(cfg: dict, proc: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """(periodos con p0, niveles en MW, dp): de la cache o, si no existe, de las curvas."""
    rs = cfg["respuesta"]
    niveles = np.array(rs["niveles_gw"], float) * 1000
    suf = f"{pd.Timestamp(rs['desde']):%Y%m%d}_{pd.Timestamp(rs['hasta']):%Y%m%d}"
    cache = proc / f"respuesta_dp_{suf}.npz"
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["niveles"], niveles):
            per = pd.DataFrame({"timestamp_utc": pd.to_datetime(z["ts_ns"], unit="ns", utc=True),
                                "resolucion_min": z["res"], "p0": z["p0"]})
            assert per["timestamp_utc"].dt.year.min() >= 2025, "cache con fechas erroneas: borrarla"
            return per, niveles, z["dp"]
    dias = list(pd.date_range(rs["desde"], rs["hasta"]).date)
    t0 = time.perf_counter()
    curvas = preparar_dias(proc / "curvas", dias)
    print(f"Curvas preparadas: {len(curvas.periodos):,} periodos, {time.perf_counter() - t0:.0f} s",
          flush=True)

    def progreso(i, n, d):
        print(f"  nivel {d / 1000:+g} GW ({i}/{n}), {time.perf_counter() - t0:.0f} s", flush=True)

    p0, dp = tabla_respuesta(curvas, niveles, progreso=progreso)
    per = curvas.periodos[["timestamp_utc", "resolucion_min"]].assign(p0=p0)
    ts_ns = per["timestamp_utc"].dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]")
    np.savez(cache, ts_ns=ts_ns.astype("int64").to_numpy(),
             res=per["resolucion_min"].to_numpy(), p0=p0, dp=dp, niveles=niveles)
    return per, niveles, dp


def tabla_z(per: pd.DataFrame, proc: Path, fundamentales: list[str]) -> pd.DataFrame:
    """Variables z de cada periodo (mismo codigo que el emulador de scripts/10)."""
    falso = per.rename(columns={"p0": "precio_base"}).assign(analizado=True)
    falso["dia"] = falso["timestamp_utc"].dt.tz_convert(TZ).dt.date
    t = tabla_aprendizaje(falso, pd.read_parquet(proc / "fundamentales.parquet"),
                          pd.read_parquet(proc / "combustibles.parquet"), fundamentales)
    assert len(t) == len(per) and (t["timestamp_utc"].to_numpy() == per["timestamp_utc"].to_numpy()).all()
    return t


def simula_parque(t: pd.DataFrame, resp: RespuestaPrecio | None, gw: float, bat: Battery, sim: dict,
                  paso_mw: float) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Equilibrio EXACTO (un LP por mes: presupuesto de 1 ciclo/dia, carga inicial y final al 50 %).

    Devuelve la potencia neta del parque, el precio con el parque y la brecha de equilibrio de cada mes.
    Con `resp=None` el precio no reacciona: es el plan precio aceptante.
    """
    p0, h = t["p0"].to_numpy(float), t["resolucion_min"].to_numpy(float) / 60
    d, precio, brechas = np.zeros(len(t)), np.empty(len(t)), []
    mes = t["mes"].to_numpy()
    for m in pd.unique(mes):
        idx = np.flatnonzero(mes == m)
        ciclos = bat.ciclos_maximos_anuales * t["dia"].iloc[idx].nunique() / 365
        ev = resp.evaluador(t.iloc[idx]) if resp is not None else (lambda d: np.zeros(len(d)))
        anchos, medios = segmentos(gw * 1000, paso_mw)
        pc, pdesc = precios_marginales(p0[idx], ev, medios)
        borde, pen = float(sim["carga_bordes"]), float(sim["penalizacion_eur_mwh"])
        d[idx], _ = equilibrio_exacto(p0[idx], h[idx], pc, pdesc, anchos, bat, ciclos, borde, pen)
        precio[idx] = p0[idx] + ev(d[idx])
        mejor = gw * 1000 * plan_lp(precio[idx], h[idx], bat, ciclos, borde, pen)[0]
        propio = float(-(d[idx] * precio[idx] * h[idx]).sum())
        brechas.append((mejor - propio) / mejor if mejor > 1e-9 else 0.0)
    return d, precio, brechas


def metricas(ts, res_min, precio, d, gw: float, horas_bat: float) -> dict:
    """Precio medio, spread diario 2 h, ingreso (€/MW/dia) y ciclos/dia de un parque de `gw` GW."""
    h = np.asarray(res_min, float) / 60
    dias = pd.Series(ts).dt.tz_convert(TZ).dt.date.nunique()
    sp = daily_spread(pd.DataFrame({"timestamp_utc": ts, "precio_es": np.asarray(precio),
                                    "resolucion_min": np.asarray(res_min)}), horas_bat)
    mw = gw * 1000
    return {"mean_price": float(np.mean(precio)), "spread_2h": float(sp.iloc[:, 1].mean()),
            "revenue_eur_mw_day": float(-(np.asarray(d) * np.asarray(precio) * h).sum() / mw / dias),
            "cycles_per_day": float((np.clip(-np.asarray(d), 0, None) * h).sum() / (mw * horas_bat) / dias)}


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    rs, sim = cfg["respuesta"], cfg["simulacion_lp"]
    proc = ROOT / cfg["datos"]["processed_dir"]
    carpeta = ROOT / "reports" / "respuesta"
    carpeta.mkdir(parents=True, exist_ok=True)
    bat = Battery(**cfg["bateria"])
    fundamentales = list(cfg["modelo"]["fundamentales"])
    suf = f"{pd.Timestamp(rs['desde']):%Y%m%d}_{pd.Timestamp(rs['hasta']):%Y%m%d}"

    per, niveles, dp = datos_respuesta(cfg, proc)
    t = tabla_z(per, proc, fundamentales)
    cols = columnas_z(t)
    print(f"{len(t):,} periodos, {len(niveles)} niveles, {len(cols)} variables z", flush=True)

    # --- entrenamiento con lo anterior a validacion_desde
    prueba = (pd.to_datetime(t["dia"]) >= pd.Timestamp(rs["validacion_desde"])).to_numpy()
    t0 = time.perf_counter()
    resp = RespuestaPrecio(cols, **rs["gbm"]).fit(t[~prueba].reset_index(drop=True), niveles, dp[~prueba])
    print(f"Entrenamiento ({(~prueba).sum():,} periodos de entrenamiento): {time.perf_counter() - t0:.0f} s",
          flush=True)

    # --- error estatico en los periodos de prueba
    tp = t[prueba].reset_index(drop=True)
    ev = resp.evaluador(tp)
    filas = []
    for j, d in enumerate(niveles):
        if d == 0:
            continue
        real, pred = dp[prueba][:, j], ev(np.full(len(tp), d))
        filas.append({"level_gw": d / 1000, "true_mean_eur_mwh": real.mean(),
                      "pred_mean_eur_mwh": pred.mean(), "MAE_model": np.abs(pred - real).mean(),
                      "MAE_no_response": np.abs(real).mean(),
                      "median_abs_err_model": np.median(np.abs(pred - real))})
    est = pd.DataFrame(filas)
    est.to_csv(carpeta / "validation_static.csv", index=False, float_format="%.3f")
    print("\nError estatico del efecto sobre el precio (periodos de prueba, €/MWh):")
    print(est.round(2).to_string(index=False), flush=True)

    # --- de punta a punta frente al simulador de curvas
    panel = pd.concat([pd.read_parquet(f) for f in sorted(proc.glob("panel_bess_*.parquet"))])
    panel = panel[panel["analizado"]].drop_duplicates(CLAVE)
    emu = pd.read_parquet(proc / "predicciones_cv_20251001_20261112.parquet")[[*CLAVE, "p_gbm_delta"]]
    filas, curvas_gw = [], {}
    for gw in rs["validacion_gw"]:
        t0 = time.perf_counter()
        d, precio, brechas = simula_parque(tp, resp, float(gw), bat, sim, float(rs["paso_gw"]) * 1000)
        # precio aceptante: el mismo parque sin efecto en el precio
        d_pt, _, _ = simula_parque(tp, None, float(gw), bat, sim, float(rs["paso_gw"]) * 1000)
        tr = panel[panel["gw"] == gw].merge(
            pd.DataFrame({"timestamp_utc": tp["timestamp_utc"], "p0": tp["p0"], "d": d, "p_lp": precio,
                          "d_pt": d_pt}),
            on="timestamp_utc", how="inner").merge(emu, on=CLAVE, how="left")
        ts, rm = tr["timestamp_utc"], tr["resolucion_min"]
        fuentes = {
            "simulator": metricas(ts, rm, tr["precio_parque"], tr["mw_parque"], gw, bat.duracion_horas),
            "price_taker": metricas(ts, rm, tr["p0"], tr["d_pt"], gw, bat.duracion_horas),
            "lp_response": metricas(ts, rm, tr["p_lp"], tr["d"], gw, bat.duracion_horas),
        }
        # el emulador no tiene programa propio: su ingreso valora el del simulador (como en script 10)
        fuentes["emulator_gbm_delta"] = metricas(ts, rm, tr["p_gbm_delta"], tr["mw_parque"], gw,
                                                 bat.duracion_horas)
        err = {"simulator": 0.0, "price_taker": (tr["p0"] - tr["precio_parque"]).abs().mean(),
               "lp_response": (tr["p_lp"] - tr["precio_parque"]).abs().mean(),
               "emulator_gbm_delta": (tr["p_gbm_delta"] - tr["precio_parque"]).abs().mean()}
        for nombre, m in fuentes.items():
            brecha = 100 * float(np.max(brechas)) if nombre == "lp_response" else np.nan
            filas.append({"gw": gw, "source": nombre, **m, "price_MAE_vs_simulator": err[nombre],
                          "equilibrium_gap_pct": brecha})
        print(f"  {gw} GW listo ({time.perf_counter() - t0:.0f} s), brecha de equilibrio maxima "
              f"{100 * max(brechas):.1f} %", flush=True)
        curvas_gw[gw] = tr
    e2e = pd.DataFrame(filas)
    e2e.to_csv(carpeta / "validation_end_to_end.csv", index=False, float_format="%.3f")
    pd.set_option("display.width", 220)
    print("\nPunta a punta, meses de prueba:")
    print(e2e.round(2).to_string(index=False), flush=True)

    figura(e2e, carpeta / "validation_gw.png", f"{tp['dia'].min():%d/%m}–{tp['dia'].max():%d/%m/%Y}")

    # --- modelo final con todo el periodo
    t0 = time.perf_counter()
    final = RespuestaPrecio(cols, **rs["gbm"]).fit(t, niveles, dp)
    joblib.dump({"respuesta": final, "columnas": cols, "fundamentales": fundamentales, "periodo": suf,
                 "niveles_mw": niveles}, proc / f"respuesta_precio_{suf}.joblib")
    print(f"\nModelo final entrenado y guardado ({time.perf_counter() - t0:.0f} s): "
          f"data/processed/respuesta_precio_{suf}.joblib")


def figura(e2e: pd.DataFrame, salida: Path, periodo: str) -> None:
    estilos = {"simulator": (TINTA, "-", None, "Simulator (reference)"),
               "price_taker": (GRIS, "--", None, "Price taker (prices do not react)"),
               "lp_response": ("#2a78d6", "-", "o", "LP + learned price response"),
               "emulator_gbm_delta": ("#eb6834", "-", "s", "Previous emulator (gbm Δ)")}
    fig, ejes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, col, nombre in ((ejes[0], "spread_2h", "Daily 2 h spread (€/MWh)"),
                            (ejes[1], "revenue_eur_mw_day", "Fleet revenue (€/MW/day)")):
        for f, (color, ls, marca, etiqueta) in estilos.items():
            s = e2e[e2e["source"] == f]
            ax.plot(s["gw"], s[col], color=color, linestyle=ls, marker=marca, markersize=5, linewidth=2,
                    label=etiqueta)
        ax.set_title(nombre, loc="left", fontsize=11, color=TINTA)
        ax.set_xlabel("Installed battery power in Spain (GW)", fontsize=9, color=TINTA_2)
        estilo(ax)
    ejes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Held-out months {periodo}: LP with learned price response vs curve simulator",
                 x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(salida, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
