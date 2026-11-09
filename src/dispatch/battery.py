"""Arbitraje de una bateria en el mercado diario, como precio aceptante.

Supuestos (todos revisables, estan aqui y no repartidos por el codigo):

* 1 MW de potencia; el ingreso sale en €/MW y escala linealmente.
* Precio aceptante: la bateria no mueve el precio. Con 0,1 GW instalados hoy
  es razonable; con los 17 GW del PNIEC a 2030 no lo sera, y el ingreso de
  esta simulacion sera un techo.
* Prevision perfecta de los precios del dia (se conocen tras la casacion, a las
  13:00 de D-1). Otro techo: en la realidad se oferta sin conocerlos.
* Como mucho un ciclo completo al dia, cargando antes de descargar. Con 2 h de
  duracion y la curva del pato española (valle solar, pico de tarde) casi todo
  el valor esta en ese ciclo.
* Eficiencia de ida y vuelta aplicada en la carga: para descargar E MWh hay
  que comprar E/eficiencia MWh, siempre a 1 MW como maximo.
* Tope de ciclos anual: si el año tiene mas dias rentables que ciclos
  permitidos, se quedan los mejores dias.
* Solo energia del diario. Sin servicios de ajuste, sin degradacion, sin
  peajes ni costes de operacion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TZ = "Europe/Madrid"


@dataclass(frozen=True)
class Battery:
    duracion_horas: float
    eficiencia_ida_vuelta: float
    ciclos_maximos_anuales: int

    def __post_init__(self) -> None:
        if not 0 < self.eficiencia_ida_vuelta <= 1:
            raise ValueError("eficiencia_ida_vuelta debe estar en (0, 1]")
        if self.duracion_horas <= 0:
            raise ValueError("duracion_horas debe ser positiva")


def _coste_extremo(p: np.ndarray, h: np.ndarray, energia: float, caros: bool) -> float:
    """€ de mover `energia` MWh a 1 MW en los periodos mas caros (o baratos). NaN si no cabe."""
    if h.sum() < energia - 1e-9:
        return np.nan
    orden = np.argsort(-p if caros else p, kind="stable")
    p, h = p[orden], h[orden]
    tomado = np.clip(energia - (np.cumsum(h) - h), 0, h)
    return float((p * tomado).sum())


def ciclo_optimo(p: np.ndarray, h: np.ndarray, bat: Battery) -> float:
    """Mejor margen (€/MW) de un ciclo en un dia: carga en [0, k), descarga en [k, n).

    Para cada corte k se compra en los periodos mas baratos antes de k y se
    vende en los mas caros desde k. Es exacto para un ciclo con carga y
    descarga a potencia nominal. Devuelve 0 si ningun ciclo gana dinero.
    """
    e_venta = bat.duracion_horas
    e_compra = e_venta / bat.eficiencia_ida_vuelta
    mejor = 0.0
    for k in range(1, len(p)):
        compra = _coste_extremo(p[:k], h[:k], e_compra, caros=False)
        if np.isnan(compra):
            continue
        venta = _coste_extremo(p[k:], h[k:], e_venta, caros=True)
        if np.isnan(venta):
            break  # cada k posterior deja aun menos periodos para descargar
        mejor = max(mejor, venta - compra)
    return mejor


def simulate_price_taker(precios: pd.DataFrame, bat: Battery) -> pd.DataFrame:
    """Ingreso diario de arbitraje de 1 MW de bateria.

    `precios` necesita timestamp_utc, precio_es (€/MWh) y resolucion_min.
    Devuelve una fila por dia local: dia, ingreso_eur_por_mw, ciclos (0 o 1).
    """
    dia = precios["timestamp_utc"].dt.tz_convert(TZ).dt.date
    filas = []
    for d, g in precios.groupby(dia, sort=True):
        g = g.sort_values("timestamp_utc")
        margen = ciclo_optimo(g["precio_es"].to_numpy(float), g["resolucion_min"].to_numpy(float) / 60.0, bat)
        filas.append((d, margen))
    out = pd.DataFrame(filas, columns=["dia", "ingreso_eur_por_mw"])
    out["ciclos"] = (out["ingreso_eur_por_mw"] > 0).astype(int)

    # Tope anual: fuera de los mejores dias, la bateria no cicla.
    anio = pd.to_datetime(out["dia"]).dt.year
    rango = out.groupby(anio)["ingreso_eur_por_mw"].rank(method="first", ascending=False)
    fuera = rango > bat.ciclos_maximos_anuales
    out.loc[fuera, ["ingreso_eur_por_mw", "ciclos"]] = 0
    return out
