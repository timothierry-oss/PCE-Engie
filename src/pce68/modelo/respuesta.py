"""Respuesta del precio a la potencia neta de las baterias, aprendida de las curvas de OMIE.

Idea: el LP (pce68.dispatch.lp) decide CUANDO cargar y descargar y respeta las restricciones fisicas
(energia, potencia, ciclos); lo que no puede hacer sin curvas es saber CUANTO se mueve el precio. Esa
parte se aprende aqui, una vez, de las curvas del pasado, y se aplica a una prevision de precios sin curvas
(pce68.dispatch.equilibrio las combina).

Para cada periodo i de unas curvas y cada nivel d (MW netos: positivo = carga = demanda extra, negativo =
descarga = oferta extra) se calcula el efecto EXACTO sobre el precio, delta_i(d) = precio_i(d) - precio_i(0),
con `CurvasPreparadas.precios` (mismo supuesto que el simulador: el resto de ofertas y los flujos no cambian).
Un HistGradientBoostingRegressor aprende delta = g(d, z) con las variables z de pce68.modelo.impacto,
CRECIENTE en d (mas demanda extra nunca baja el precio). `evaluador` devuelve g(d, z) - g(0, z): cero sin
baterias por construccion.

Supuestos y limites (revisables):

* La respuesta es continua y lineal a trozos entre los niveles de entrenamiento (ver `evaluador`).
* La respuesta es ESTATICA: depende de la potencia neta del propio periodo, no de las horas vecinas (igual
  que en las curvas: el precio de un periodo solo depende de su propio delta).
* Se aprende solo de octubre de 2025 a noviembre de 2026 (periodos en cuartos de hora con curvas): fuera de
  ese dominio de z el arbol extrapola en plano.
* El nivel d es el de TODO el parque: las baterias no se distinguen entre si (todas precio aceptante frente
  al precio que deja el conjunto).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from pce68.data.curvas import CurvasPreparadas


def tabla_respuesta(curvas: CurvasPreparadas, niveles_mw, p0: np.ndarray | None = None,
                    progreso=None) -> tuple[np.ndarray, np.ndarray]:
    """(p0, dp): precio sin baterias (n,) y efecto exacto sobre el precio de cada nivel (n, L)."""
    p0 = curvas.precios() if p0 is None else np.asarray(p0, float)
    n = len(p0)
    dp = np.zeros((n, len(niveles_mw)))
    for j, d in enumerate(niveles_mw):
        if d != 0:
            dp[:, j] = curvas.precios(np.full(n, float(d))) - p0
        if progreso:
            progreso(j + 1, len(niveles_mw), d)
    return p0, dp


class RespuestaPrecio:
    """delta precio = g(d, z), creciente en d (HistGradientBoosting)."""

    def __init__(self, columnas: list[str], **parametros):
        self.columnas, self.parametros = list(columnas), parametros

    def _base(self, Z: pd.DataFrame) -> np.ndarray:
        return Z[self.columnas].to_numpy(float)

    def fit(self, Z: pd.DataFrame, niveles_mw, dp: np.ndarray) -> RespuestaPrecio:
        base = self._base(Z)
        n, k = dp.shape
        self.niveles_ = np.asarray(niveles_mw, float)
        X = np.hstack([np.repeat(self.niveles_, n)[:, None], np.tile(base, (k, 1))])
        self.gbm_ = HistGradientBoostingRegressor(
            random_state=0, early_stopping=False, monotonic_cst=[1] + [0] * base.shape[1], **self.parametros)
        self.gbm_.fit(X, dp.T.reshape(-1))
        return self

    def evaluador(self, Z: pd.DataFrame):
        """Funcion d_mw (una por fila de Z) -> efecto sobre el precio (€/MWh); es 0 si d = 0.

        El arbol responde a escalones; aqui se evalua en los niveles de entrenamiento y se INTERPOLA
        linealmente entre ellos: la respuesta es continua y creciente, como la de las curvas reales (miles de
        ofertas pequeñas). Con escalones el equilibrio caeria en los saltos, donde el precio no esta definido.
        Fuera de [min, max] de los niveles se mantiene el valor del extremo.
        """
        base = self._base(Z)
        n, niv = len(base), self.niveles_
        G = np.stack([self.gbm_.predict(np.hstack([np.full((n, 1), lv), base])) for lv in niv], axis=1)
        i0 = int(np.argmin(np.abs(niv)))
        G = G - G[:, i0:i0 + 1]                                      # cero sin baterias
        G[:, i0:] = np.maximum.accumulate(G[:, i0:], axis=1)         # monotonia: ruido numerico del arbol
        G[:, :i0 + 1] = np.minimum.accumulate(G[:, :i0 + 1][:, ::-1], axis=1)[:, ::-1]
        filas = np.arange(n)

        def evalua(d_mw) -> np.ndarray:
            d = np.clip(np.broadcast_to(np.asarray(d_mw, float), (n,)), niv[0], niv[-1])
            k = np.clip(np.searchsorted(niv, d, side="right") - 1, 0, len(niv) - 2)
            w = (d - niv[k]) / (niv[k + 1] - niv[k])
            return G[filas, k] * (1 - w) + G[filas, k + 1] * w

        return evalua
