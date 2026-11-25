"""Parque de baterias en equilibrio con el precio: LP (restricciones fisicas) + respuesta del precio.

`plan_lp` (pce68.dispatch.lp) da el mejor plan de una bateria a precios FIJOS. Aqui los precios dependen
del propio parque: p_t = p0_t + g_t(d_t), con d_t la potencia neta del parque (MW, positivo = carga) y g la
respuesta del precio (pce68.modelo.respuesta, o cualquier funcion d -> €/MWh). El equilibrio, con las
baterias como precio aceptante frente al precio que deja el conjunto, es el punto fijo

    d = X * plan_lp(p0 + g(d))

que se busca promediando los planes sucesivos (metodo de las medias sucesivas, paso 1/k): el promedio
de planes factibles es factible, asi que el plan final cumple energia, potencia y ciclos. Sin amortiguar,
el LP salta de un extremo a otro entre iteraciones.

Diagnostico de convergencia: `brecha_rel` = (ingreso del mejor plan al precio final - ingreso del plan final)
/ ingreso del mejor plan. 0 = nadie quiere cambiar de plan = equilibrio.

Supuestos (los de pce68.dispatch.lp, mas): prevision perfecta; la respuesta solo depende de la potencia del
propio periodo; el parque es un solo agente aceptante de precio (no se internaliza su propio efecto en el
precio: las baterias reales tampoco coordinan sus ofertas).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from pce68.dispatch.battery import Battery
from pce68.dispatch.lp import plan_lp


@dataclass
class Equilibrio:
    d_mw: np.ndarray                       # potencia neta del parque por periodo (positivo = carga)
    precio: np.ndarray                     # p0 + g(d): precio con el parque
    ingreso_eur: float                     # -sum d * precio * h
    brecha_rel: float                      # ver cabecera
    historial: list[float] = field(default_factory=list)  # cambio maximo de d / potencia en cada iteracion


def equilibrio_parque(p0, h, evaluador, potencia_mw: float, bat: Battery, ciclos: float,
                      carga_bordes: float = 0.5, penalizacion: float = 1e-3,
                      iteraciones: int = 12) -> Equilibrio:
    """Plan del parque de `potencia_mw` en equilibrio con la respuesta `evaluador` (ver cabecera)."""
    p0, h = np.asarray(p0, float), np.asarray(h, float)
    d = np.zeros(len(p0))
    hist = []
    for k in range(1, iteraciones + 1):
        _, perfil, _ = plan_lp(p0 + evaluador(d), h, bat, ciclos, carga_bordes, penalizacion)
        nuevo = potencia_mw * perfil
        paso = nuevo - d
        d = d + paso / k
        hist.append(float(np.abs(paso).max() / potencia_mw / k))
    precio = p0 + evaluador(d)
    margen_mejor, perfil, _ = plan_lp(precio, h, bat, ciclos, carga_bordes, penalizacion)
    mejor = potencia_mw * margen_mejor
    propio = float(-(d * precio * h).sum())
    brecha = (mejor - propio) / mejor if mejor > 1e-9 else 0.0
    return Equilibrio(d, precio, propio, float(brecha), hist)


def segmentos(potencia_mw: float, paso_mw: float) -> tuple[np.ndarray, np.ndarray]:
    """Tramos de potencia (anchos y puntos medios, MW) de 0 a `potencia_mw`, de a lo sumo `paso_mw`."""
    n = max(1, int(np.ceil(potencia_mw / paso_mw - 1e-9)))
    bordes = np.linspace(0.0, potencia_mw, n + 1)
    return np.diff(bordes), (bordes[:-1] + bordes[1:]) / 2


def equilibrio_exacto(p0, h, precio_carga: np.ndarray, precio_descarga: np.ndarray, anchos: np.ndarray,
                      bat: Battery, ciclos: float, carga_bordes: float = 0.5, penalizacion: float = 1e-3
                      ) -> tuple[np.ndarray, float]:
    """Equilibrio en UN solo LP (sin iterar), para una respuesta del precio creciente en la potencia neta.

    Si las baterias son precio aceptante frente al precio p(d) = p0 + g(d) que deja el conjunto y g es
    creciente, el equilibrio es el minimo de  Phi(d) = sum_t h_t [p0_t d_t + integral_0^{d_t} g_t]  sobre los
    planes factibles (juego de potencial): la derivada de Phi en d_t es p(d_t). Con g constante a trozos,
    Phi es lineal a trozos y convexa, asi que es un LP: la carga de cada periodo se divide en tramos j (ancho
    `anchos[j]`, MW) cuyo coste marginal es el precio que deja ese tramo, `precio_carga[t, j]` (no decrece
    con j), y lo mismo con la descarga, `precio_descarga[t, j]` (no crece con j). La convexidad llena los
    tramos en orden sin imponerlo. Mismas restricciones de energia, potencia y ciclos que pce68.dispatch.lp,
    para un parque de sum(anchos) MW.

    Devuelve (d_mw por periodo, positivo = carga; valor del LP). El ingreso real se calcula con el precio
    p0 + g(d), no con el valor del LP (que es el potencial, no el ingreso).
    """
    p0, h = np.asarray(p0, float), np.asarray(h, float)
    n, J = len(p0), len(anchos)
    X = float(np.sum(anchos))
    dur, eta = bat.duracion_horas, bat.eficiencia_ida_vuelta
    s0 = carga_bordes * dur * X
    # variables: [x (n*J, carga por tramo), y (n*J, descarga por tramo), s (n)]
    hj = np.repeat(h, J).reshape(n, J)
    coste = np.concatenate([(hj * (precio_carga + penalizacion)).ravel(),
                            (hj * (penalizacion - precio_descarga)).ravel(), np.zeros(n)])
    suma_j = sparse.kron(sparse.identity(n), np.ones((1, J)), format="csr")   # (n, n*J): suma de tramos
    hd = sparse.diags(h)
    balance = sparse.hstack([-eta * hd @ suma_j, hd @ suma_j,
                             sparse.identity(n) - sparse.eye(n, k=-1)], format="csr")
    b_balance = np.zeros(n)
    b_balance[0] = s0
    ciclos_fila = sparse.hstack([sparse.csr_matrix((1, n * J)), sparse.csr_matrix(hj.reshape(1, -1)),
                                 sparse.csr_matrix((1, n))], format="csr")
    anchos_n = np.tile(np.asarray(anchos, float), n)
    limites = ([(0.0, a) for a in anchos_n] * 2 + [(0.0, dur * X)] * (n - 1) + [(s0, s0)])
    r = linprog(coste, A_ub=ciclos_fila, b_ub=[ciclos * dur * X], A_eq=balance, b_eq=b_balance,
                bounds=limites, method="highs")
    if r.status != 0:
        raise RuntimeError(f"el LP de equilibrio no tiene solucion: {r.message}")
    x, y = r.x[:n * J].reshape(n, J), r.x[n * J:2 * n * J].reshape(n, J)
    d = x.sum(axis=1) - y.sum(axis=1)
    d[np.abs(d) < 1e-9] = 0.0
    return d, float(r.fun)


def precios_marginales(p0, evaluador, medios_mw) -> tuple[np.ndarray, np.ndarray]:
    """Precios (n, J) que deja cada tramo de carga y de descarga, para `equilibrio_exacto`.

    `evaluador`: d_mw (una por periodo) -> efecto sobre el precio. Se fuerza la monotonia (la carga no
    abarata con el tramo, la descarga no encarece) para absorber el ruido numerico del modelo.
    """
    p0 = np.asarray(p0, float)
    n = len(p0)
    carga = np.stack([p0 + evaluador(np.full(n, float(m))) for m in medios_mw], axis=1)
    descarga = np.stack([p0 + evaluador(np.full(n, -float(m))) for m in medios_mw], axis=1)
    return np.maximum.accumulate(carga, axis=1), np.minimum.accumulate(descarga, axis=1)
