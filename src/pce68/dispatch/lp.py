"""Parque de baterias planificado por programacion lineal (LP) sobre todo el periodo.

A diferencia de pce68.dispatch.flota (como mucho un ciclo al dia, dia a dia), aqui cada tramo
resuelve un LP sobre todos los periodos a la vez, con un estado de carga continuo: puede hacer
varios ciclos al dia, ciclos parciales o guardar energia de un dia para otro.

LP de 1 MW (el plan de un tramo de q MW es q veces este, porque el tramo es precio aceptante):

    max  sum_t (d_t - c_t) p_t h_t  -  pen * sum_t (c_t + d_t) h_t
    s_t = s_{t-1} + eta * c_t * h_t - d_t * h_t          (perdidas en la carga, como battery.py)
    0 <= c_t <= 1 ;  0 <= d_t <= 1 ;  0 <= s_t <= D ;  s_0 = s_T = carga_bordes * D
    sum_t d_t h_t <= ciclos * D

c = carga y d = descarga (MW), s = energia almacenada al final de cada periodo (MWh),
D = duracion_horas, eta = eficiencia_ida_vuelta. Se resuelve con HiGHS (scipy.optimize.linprog).

Supuestos (revisables; valores en config/base.yaml, seccion simulacion_lp):

* Prevision perfecta sobre todo el periodo: la bateria conoce todos los precios de antemano.
  Sobrestima el ingreso y el efecto en el precio (en la realidad se oferta el dia anterior).
* Bordes: carga fija (50 %) al empezar y al acabar, para que el LP no vacie la bateria "gratis"
  al final. Los dias de los extremos se quitan de los resultados (scripts/06_simula_lp.py).
* Presupuesto de ciclos sobre el periodo: energia descargada <= ciclos * D (1 ciclo al dia de media).
* Penalizacion minima de la carga y la descarga: evita que el LP cargue y descargue a la vez,
  lo que solo le interesaria con precios negativos.
* Tramos de paso_mw colocados uno tras otro: cada uno ve los precios que dejan los anteriores y
  ninguno se reoptimiza despues (como flota.despacho). Cada tramo ignora su propio efecto en el precio.
* El resto, como en pce68.data.curvas: las demas ofertas y los flujos de interconexion no cambian.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linprog

from pce68.data.curvas import precios
from pce68.dispatch.battery import Battery


def plan_lp(
    p: np.ndarray, h: np.ndarray, bat: Battery, ciclos: float, carga_bordes: float = 0.5,
    penalizacion: float = 1e-3,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mejor plan de 1 MW sobre todo el periodo, a precios `p` fijos.

    Devuelve (margen, perfil, carga): margen en €/MW sin la penalizacion; perfil en MW por
    periodo (positivo = carga, demanda extra; negativo = descarga); carga = energia almacenada
    al final de cada periodo (MWh por MW).
    """
    p, h = np.asarray(p, float), np.asarray(h, float)
    n = len(p)
    dur, eta = bat.duracion_horas, bat.eficiencia_ida_vuelta
    s0 = carga_bordes * dur

    # variables [c (n), d (n), s (n)]; linprog minimiza el coste = -ingreso + penalizacion
    coste = np.concatenate([h * (p + penalizacion), h * (penalizacion - p), np.zeros(n)])
    hd = sparse.diags(h)
    balance = sparse.hstack([-eta * hd, hd, sparse.identity(n) - sparse.eye(n, k=-1)], format="csr")
    b_balance = np.zeros(n)
    b_balance[0] = s0  # s_0 del enunciado: la carga antes del primer periodo
    ciclos_fila = sparse.hstack(
        [sparse.csr_matrix((1, n)), sparse.csr_matrix(h.reshape(1, -1)), sparse.csr_matrix((1, n))],
        format="csr",
    )
    limites = [(0.0, 1.0)] * (2 * n) + [(0.0, dur)] * (n - 1) + [(s0, s0)]
    r = linprog(coste, A_ub=ciclos_fila, b_ub=[ciclos * dur], A_eq=balance, b_eq=b_balance,
                bounds=limites, method="highs")
    if r.status != 0:
        raise RuntimeError(f"el LP no tiene solucion: {r.message}")
    perfil = r.x[:n] - r.x[n:2 * n]
    perfil[np.abs(perfil) < 1e-9] = 0.0  # ruido numerico del solver
    margen = float(-(perfil * p * h).sum())
    return margen, perfil, r.x[2 * n:]


def despacho_lp(
    curvas: pd.DataFrame, potencia_mw: float, bat: Battery, paso_mw: float, ciclos: float,
    carga_bordes: float = 0.5, penalizacion: float = 1e-3,
) -> pd.DataFrame:
    """Reparte un parque de `potencia_mw` en tramos de `paso_mw`, cada uno planificado con `plan_lp`.

    Devuelve timestamp_utc, resolucion_min, delta_mw (positivo = carga, como espera
    curvas.precios), carga_mwh (energia almacenada en todo el parque al final de cada periodo)
    y precio_eur_mwh con el parque. Ademas:
    attrs["tramos"]: tabla por tramo (tramo, mw, activo, margen_visto_eur_mw: el margen con los
    precios que vio al colocarse) y attrs["planes"]: perfil de cada tramo (MW por MW instalado).
    """
    if paso_mw <= 0:
        raise ValueError("paso_mw debe ser positivo")
    periodos = curvas[["timestamp_utc", "resolucion_min"]].drop_duplicates().sort_values("timestamp_utc")
    ts = pd.DatetimeIndex(periodos["timestamp_utc"])
    h = periodos["resolucion_min"].to_numpy(float) / 60.0
    delta, carga = np.zeros(len(ts)), np.zeros(len(ts))
    p = precios(curvas, pd.Series(delta, index=ts))["precio_eur_mwh"].to_numpy(float)

    planes, tramos = [], []
    restante = float(potencia_mw)
    while restante > 1e-9:
        q = min(paso_mw, restante)
        restante -= q
        margen, perfil, s = plan_lp(p, h, bat, ciclos, carga_bordes, penalizacion)
        activo = margen > 0
        if activo:
            delta += q * perfil
            carga += q * s
            p = precios(curvas, pd.Series(delta, index=ts))["precio_eur_mwh"].to_numpy(float)
        else:  # parado: guarda su carga de los bordes todo el periodo
            perfil = np.zeros(len(ts))
            carga += q * carga_bordes * bat.duracion_horas
        planes.append(perfil)
        tramos.append({"tramo": len(tramos) + 1, "mw": q, "activo": activo, "margen_visto_eur_mw": margen})

    out = periodos.reset_index(drop=True).assign(delta_mw=delta, carga_mwh=carga, precio_eur_mwh=p)
    out.attrs["tramos"] = pd.DataFrame(tramos)
    out.attrs["planes"] = np.array(planes)
    return out
