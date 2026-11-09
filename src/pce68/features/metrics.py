"""Metricas diarias sobre la serie de precios."""

from __future__ import annotations

import numpy as np
import pandas as pd

TZ = "Europe/Madrid"


def _dias(precios: pd.DataFrame):
    """Itera (dia_local, precios del dia, horas de cada periodo)."""
    dia = precios["timestamp_utc"].dt.tz_convert(TZ).dt.date
    for d, g in precios.groupby(dia, sort=True):
        g = g.sort_values("timestamp_utc")
        yield d, g["precio_es"].to_numpy(float), (g["resolucion_min"].to_numpy(float) / 60.0)


def _media_extremo(p: np.ndarray, h: np.ndarray, horas: float, caros: bool) -> float:
    """Precio medio de las `horas` mas caras (o mas baratas) del dia, ponderado por tiempo."""
    orden = np.argsort(-p if caros else p, kind="stable")
    p, h = p[orden], h[orden]
    acum = np.cumsum(h)
    tomado = np.clip(horas - (acum - h), 0, h)  # cuanto de cada periodo entra en la ventana
    return float((p * tomado).sum() / horas)


def daily_spread(precios: pd.DataFrame, horas: float) -> pd.DataFrame:
    """Spread capturable de cada dia: media de las `horas` mas caras menos la de las mas baratas.

    Es la metrica de referencia del arbitraje con una bateria de `horas` de
    duracion: bruta, sin eficiencia ni orden temporal (cargar antes de
    descargar). El ingreso realista sale de `pce68.dispatch.battery`.

    Devuelve columnas `dia_local` y `spread_<horas>h` (p. ej. `spread_2h`).
    Los dias con menos de 2·`horas` de datos quedan en NaN.
    """
    col = f"spread_{horas:g}h"
    filas = []
    for d, p, h in _dias(precios):
        if h.sum() < 2 * horas:
            s = np.nan
        else:
            s = _media_extremo(p, h, horas, caros=True) - _media_extremo(p, h, horas, caros=False)
        filas.append((d, s))
    return pd.DataFrame(filas, columns=["dia_local", col])
