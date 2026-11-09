"""Parque de baterias que mueve el precio: carga en el valle solar y descarga en el pico.

Se parte de las curvas agregadas de un dia (pce68.data.curvas): la carga se añade como
demanda y la descarga como oferta, y el precio se recalcula en el cruce de las curvas.

Supuestos (todos revisables, estan aqui y no repartidos por el codigo):

* El parque se reparte en tramos de `paso_mw`. Cada tramo, uno tras otro, ve los precios
  que dejan los anteriores y hace su mejor ciclo del dia: carga en los periodos mas
  baratos y descarga despues en los mas caros. Es como si las baterias compitieran: las
  ultimas en llegar se van a horas que las primeras aun no han aplastado. Con
  `paso_mw` igual a la potencia total, todo el parque carga y descarga en las mismas
  horas, elegidas con los precios sin baterias (la version mas simple).
* Un tramo cuyo mejor ciclo ya no gana dinero se queda parado.
* Mismos supuestos de ciclo que pce68.dispatch.battery: prevision perfecta, un ciclo al
  dia como mucho, potencia nominal, eficiencia aplicada en la carga (para descargar
  E MWh se compran E/eficiencia MWh).
* La carga entra como compra a precio maximo y la descarga como venta a precio minimo:
  se casan siempre (ver pce68.data.curvas.precios).
* Los flujos de interconexion casados quedan fijos, y el resto de ofertas no cambia: ni
  el bombeo ni la hidraulica ni el solar reaccionan a la llegada de las baterias. Con
  varios GW esto exagera el efecto de las baterias sobre el precio.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pce68.data.curvas import precios
from pce68.dispatch.battery import Battery

TZ = "Europe/Madrid"


def _reparto(p: np.ndarray, h: np.ndarray, energia: float, caros: bool) -> np.ndarray:
    """MWh que mueve 1 MW en cada periodo para cubrir `energia` en los mas caros (o baratos)."""
    orden = np.argsort(-p if caros else p, kind="stable")
    tomado = np.zeros_like(p)
    tomado[orden] = np.clip(energia - (np.cumsum(h[orden]) - h[orden]), 0, h[orden])
    return tomado


def ciclo(p: np.ndarray, h: np.ndarray, bat: Battery) -> tuple[float, np.ndarray]:
    """Mejor ciclo de 1 MW en un dia y su perfil de potencia en la red.

    Prueba cada corte k: carga en [0, k) en los periodos mas baratos y descarga en [k, n)
    en los mas caros. Devuelve el margen (€/MW) y el perfil en MW por periodo: positivo
    = carga (demanda extra), negativo = descarga (oferta extra). Si ningun ciclo gana
    dinero, margen 0 y perfil nulo.
    """
    e_venta = bat.duracion_horas
    e_compra = e_venta / bat.eficiencia_ida_vuelta
    mejor, perfil = 0.0, np.zeros_like(p)
    for k in range(1, len(p)):
        if h[:k].sum() < e_compra - 1e-9:
            continue
        if h[k:].sum() < e_venta - 1e-9:
            break  # cada k posterior deja aun menos periodos para descargar
        compra = _reparto(p[:k], h[:k], e_compra, caros=False)
        venta = _reparto(p[k:], h[k:], e_venta, caros=True)
        margen = float((p[k:] * venta).sum() - (p[:k] * compra).sum())
        if margen > mejor:
            mejor, perfil = margen, np.concatenate([compra, -venta]) / h
    return mejor, perfil


def despacho(curvas: pd.DataFrame, potencia_mw: float, bat: Battery, paso_mw: float) -> pd.DataFrame:
    """Reparte un parque de `potencia_mw` en tramos y devuelve su efecto en la red.

    Devuelve timestamp_utc, resolucion_min, delta_mw (positivo = carga, negativo =
    descarga, lo que espera curvas.precios) y precio_eur_mwh con el parque.
    Ademas `attrs["mw_activos"]` guarda los MW que llegaron a ciclar.
    """
    if paso_mw <= 0:
        raise ValueError("paso_mw debe ser positivo")
    periodos = curvas[["timestamp_utc", "resolucion_min"]].drop_duplicates().sort_values("timestamp_utc")
    ts = pd.DatetimeIndex(periodos["timestamp_utc"])
    h = periodos["resolucion_min"].to_numpy(float) / 60.0
    dia = ts.tz_convert(TZ).date
    delta = np.zeros(len(ts))
    activos = dict.fromkeys(np.unique(dia), 0.0)

    restante = float(potencia_mw)
    while restante > 1e-9:
        q = min(paso_mw, restante)
        restante -= q
        p = precios(curvas, pd.Series(delta, index=ts))["precio_eur_mwh"].to_numpy(float)
        for d in activos:
            m = dia == d
            margen, perfil = ciclo(p[m], h[m], bat)
            if margen > 0:
                delta[m] += q * perfil
                activos[d] += q

    out = periodos.reset_index(drop=True).assign(delta_mw=delta)
    out["precio_eur_mwh"] = precios(curvas, pd.Series(delta, index=ts))["precio_eur_mwh"].to_numpy(float)
    out.attrs["mw_activos"] = activos
    return out
