"""Aproximacion rapida del simulador de impacto de las baterias (etapas 2 y 3 del documento del cliente).

Documento: docs/BESS-cannibalization.md. Notacion del documento:

    p0_{d,t}   precio historico del cuarto de hora t del dia d
    p^X_{d,t}  precio simulado con X GW de baterias (scripts/08_simula_rejilla.py)
    y_{d,t}(X) = p^X / p0, solo donde p0 != 0
    y = f(t, X, z) + e          (etapa 2: z = fundamentales observables, sin identificador de dia)
    p'_{s,t} = f^(t, X_s, z^E) * p^E_{s,t}     (etapa 3: prevision de ENGIE ajustada)

Variables z (`variables`): hora y cuarto de hora, dia de la semana, fin de semana, estacion (seno y
coseno del dia del año), p0 y los precios una hora antes y despues, forma del dia (media, minimo,
maximo y rango de p0 en el dia local), fundamentales ESIOS (lista en config/base.yaml, modelo),
cuota renovable (eolica + solar + hidraulica) / demanda, gas MIBGAS-ES y CO2.

Modelos (misma interfaz: fit(Z, x_gw, y) y predict(Z, x_gw)):

* RidgeImpacto: y = 1 + x * a(z) + x^2 * b(z), con x = X / 10 GW y a, b lineales en phi(z) (hora
  en indicadoras, p0 en splines, el resto estandarizado). Sin baterias da y = 1 por construccion y
  cada coeficiente se lee como el efecto de una variable por GW: es el modelo "transparente".
* GBMImpacto: HistGradientBoostingRegressor sobre (X, z), mas flexible y sin forma impuesta en X.

Supuestos (revisables):

* Se entrena con todos los p0 != 0, como dice el documento. Con p0 cerca de 0 los cocientes son
  enormes (cientos o miles); los p0 = 0 no entran (y no existe) y ahi la prediccion de precio es
  0 * y = 0 sea cual sea X.
* Ponderacion (`ponderacion`): "ninguna" = minimos cuadrados en y, dominados por esos cocientes
  enormes. "p0_cuadrado" = peso w = p0^2 en cada cuarto de hora. Como (y^ p0 - p^X)^2 =
  p0^2 (y^ - y)^2, es exactamente minimizar el error cuadratico en PRECIO, sin cambiar la formula
  del documento. Los pesos se normalizan a media 1 (w / media(w)), para que alpha_ridge tenga el
  mismo sentido con y sin ponderacion. Limite: con p0 ~ 0 el modelo solo puede dar p^ ~ 0, no la
  subida de precio que provoca la carga de las baterias a mediodia.
* Sin festivos: solo se distingue el fin de semana.
* Gas del dia (MIBGAS-ES); CO2 mensual (ESIOS 1391, ver config combustibles).
* Fundamentales: valores reales de ESIOS, no previsiones. En la etapa 3 hay que dar los z^E de la
  prevision de ENGIE con las mismas columnas y unidades (MW medios del cuarto de hora).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import SplineTransformer

from pce68.features.metrics import daily_spread

TZ = "Europe/Madrid"
ESCALA_GW = 10.0
RENOVABLES = ("eolica", "solar_fv", "solar_termica", "hidraulica")


# ── Variables


def variables(df: pd.DataFrame, fundamentales: list[str]) -> pd.DataFrame:
    """Variables z de cada cuarto de hora.

    `df`: una fila por timestamp_utc (sin repetir), con p0 (€/MWh), las columnas `fundamentales`,
    gas_mibgas_es y co2_eur_t. Vale para el historico (p0 = precio_base) y para una prevision
    (p0 = p^E). Devuelve las variables en el mismo orden de filas.
    """
    df = df.sort_values("timestamp_utc").reset_index(drop=True)
    local = df["timestamp_utc"].dt.tz_convert(TZ)
    z = pd.DataFrame({"timestamp_utc": df["timestamp_utc"]})
    z["hora"] = local.dt.hour
    z["cuarto"] = local.dt.hour * 4 + local.dt.minute // 15
    z["dia_semana"] = local.dt.dayofweek
    z["fin_de_semana"] = (z["dia_semana"] >= 5).astype(float)
    angulo = 2 * np.pi * (local.dt.dayofyear - 1) / 365.25
    z["estacion_sin"], z["estacion_cos"] = np.sin(angulo), np.cos(angulo)

    p0 = df.set_index("timestamp_utc")["p0"]
    z["p0"] = p0.to_numpy(float)
    for nombre, dt in (("p0_hora_antes", -1), ("p0_hora_despues", 1)):
        vecino = p0.reindex(df["timestamp_utc"] + pd.Timedelta(hours=dt)).to_numpy(float)
        z[nombre] = np.where(np.isnan(vecino), z["p0"], vecino)
    dia = local.dt.date
    g = z.groupby(dia)["p0"]
    z["p0_dia_media"], z["p0_dia_min"] = g.transform("mean"), g.transform("min")
    z["p0_dia_max"] = g.transform("max")
    z["p0_dia_rango"] = z["p0_dia_max"] - z["p0_dia_min"]

    for c in fundamentales:
        z[c] = df[c].to_numpy(float)
    ren = [c for c in RENOVABLES if c in df]
    if ren and "demanda" in df:
        z["cuota_renovable"] = df[ren].sum(axis=1, min_count=1) / df["demanda"]
    z["gas"] = df["gas_mibgas_es"].to_numpy(float)
    z["co2"] = df["co2_eur_t"].to_numpy(float)
    return z


def tabla_aprendizaje(panel: pd.DataFrame, fund: pd.DataFrame, comb: pd.DataFrame,
                      fundamentales: list[str]) -> pd.DataFrame:
    """Panel de la rejilla (solo dias analizados) con las variables z de cada cuarto de hora."""
    panel = panel[panel["analizado"]]
    base = (panel.drop_duplicates("timestamp_utc")[["timestamp_utc", "precio_base"]]
            .rename(columns={"precio_base": "p0"}))
    base = base.merge(fund[["timestamp_utc", *fundamentales]], on="timestamp_utc", how="left")
    base["dia"] = base["timestamp_utc"].dt.tz_convert(TZ).dt.date
    comb = comb.assign(dia=pd.to_datetime(comb["dia"]).dt.date)
    base = base.merge(comb[["dia", "gas_mibgas_es", "co2_eur_t"]], on="dia", how="left")
    z = variables(base, fundamentales)
    t = panel.merge(z.drop(columns="p0"), on="timestamp_utc", how="left")
    t["p0"] = t["precio_base"]
    t["mes"] = pd.to_datetime(t["dia"]).dt.strftime("%Y-%m")
    return t.reset_index(drop=True)


def columnas_z(t: pd.DataFrame) -> list[str]:
    """Columnas de variables z de una tabla de `tabla_aprendizaje` o de `variables`."""
    fuera = {"timestamp_utc", "dia", "mes", "gw", "precio_base", "precio_parque", "mw_parque",
             "carga_mwh", "analizado", "resolucion_min", "y"}
    return [c for c in t.columns if c not in fuera]


# ── Modelos

PONDERACIONES = ("ninguna", "p0_cuadrado")


def pesos(p0, ponderacion: str) -> np.ndarray | None:
    """Peso de cada cuarto de hora en el ajuste: None (todos iguales) o p0^2 normalizado a media 1."""
    if ponderacion == "ninguna":
        return None
    if ponderacion == "p0_cuadrado":
        w = np.asarray(p0, float) ** 2
        return w / w.mean()
    raise ValueError(f"ponderacion debe ser una de {PONDERACIONES}, no {ponderacion!r}")


class RidgeImpacto:
    """y = 1 + x a(z) + x^2 b(z), con x = GW / 10 y a, b lineales en phi(z) (ver modulo)."""

    nombre = "ridge"

    def __init__(self, columnas: list[str], alpha: float = 10.0, ponderacion: str = "ninguna"):
        self.columnas, self.alpha, self.ponderacion = columnas, alpha, ponderacion
        self.continuas = [c for c in columnas if c not in ("hora", "cuarto", "dia_semana", "p0")]

    def _phi(self, Z: pd.DataFrame, ajustar: bool) -> np.ndarray:
        num = Z[self.continuas].to_numpy(float)
        p0 = Z[["p0"]].to_numpy(float)
        if ajustar:
            self.mediana_ = np.nanmedian(num, axis=0)
            num = np.where(np.isnan(num), self.mediana_, num)
            self.media_, self.desv_ = num.mean(axis=0), num.std(axis=0) + 1e-9
            self.splines_ = SplineTransformer(n_knots=6, degree=3, knots="quantile",
                                              extrapolation="linear").fit(p0)
        num = np.where(np.isnan(num), self.mediana_, num)
        num = (num - self.media_) / self.desv_
        horas = np.eye(24)[Z["hora"].to_numpy(int)]
        return np.hstack([horas, self.splines_.transform(p0), num])

    def _diseno(self, phi: np.ndarray, x_gw) -> np.ndarray:
        x = np.broadcast_to(np.asarray(x_gw, float) / ESCALA_GW, (len(phi),))[:, None]
        return np.hstack([phi * x, phi * x**2])

    def fit(self, Z: pd.DataFrame, x_gw, y) -> RidgeImpacto:
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        phi = self._phi(Z.loc[ok], ajustar=True)
        x = np.asarray(x_gw, float)[ok] if np.ndim(x_gw) else x_gw
        w = pesos(Z.loc[ok, "p0"], self.ponderacion)
        self.ridge_ = Ridge(alpha=self.alpha, fit_intercept=False).fit(self._diseno(phi, x), y[ok] - 1,
                                                                       sample_weight=w)
        return self

    def predict(self, Z: pd.DataFrame, x_gw) -> np.ndarray:
        return 1 + self._diseno(self._phi(Z, ajustar=False), x_gw) @ self.ridge_.coef_


class GBMImpacto:
    """HistGradientBoostingRegressor sobre (X, z)."""

    nombre = "gbm"

    def __init__(self, columnas: list[str], ponderacion: str = "ninguna", **parametros):
        self.columnas, self.ponderacion, self.parametros = columnas, ponderacion, parametros

    def _matriz(self, Z: pd.DataFrame, x_gw) -> np.ndarray:
        x = np.broadcast_to(np.asarray(x_gw, float), (len(Z),))[:, None]
        return np.hstack([x, Z[self.columnas].to_numpy(float)])

    def fit(self, Z: pd.DataFrame, x_gw, y) -> GBMImpacto:
        y = np.asarray(y, float)
        ok = np.isfinite(y)
        x = np.asarray(x_gw, float)[ok] if np.ndim(x_gw) else x_gw
        self.gbm_ = HistGradientBoostingRegressor(random_state=0, early_stopping=False, **self.parametros)
        w = pesos(Z.loc[ok, "p0"], self.ponderacion)
        self.gbm_.fit(self._matriz(Z.loc[ok], x), y[ok], sample_weight=w)
        return self

    def predict(self, Z: pd.DataFrame, x_gw) -> np.ndarray:
        return self.gbm_.predict(self._matriz(Z, x_gw))


# ── Variante aditiva: delta = p^X - p0 en €/MWh (en lugar del cociente y)
#
# Mismas variables z y mismos hiperparametros; cambia el objetivo. Ventajas: existe tambien con
# p0 = 0 (y negativo), no explota con p0 ~ 0 y el error cuadratico en delta ES el error en precio,
# asi que no hace falta ponderar por p0^2. Sin baterias da delta = 0 (ridge, por construccion).
# fit() calcula delta = precio_parque - p0 a partir de Z (la tabla de aprendizaje); el argumento y
# (cociente) se ignora, solo existe para mantener la interfaz comun. predict() devuelve delta.


class RidgeDelta(RidgeImpacto):
    """delta = x a(z) + x^2 b(z), con x = GW / 10 y a, b lineales en phi(z)."""

    nombre = "ridge_delta"
    objetivo = "delta"

    def fit(self, Z: pd.DataFrame, x_gw, y=None) -> RidgeDelta:
        delta = Z["precio_parque"].to_numpy(float) - Z["p0"].to_numpy(float)
        phi = self._phi(Z, ajustar=True)
        self.ridge_ = Ridge(alpha=self.alpha, fit_intercept=False).fit(self._diseno(phi, x_gw), delta)
        return self

    def predict(self, Z: pd.DataFrame, x_gw) -> np.ndarray:
        return self._diseno(self._phi(Z, ajustar=False), x_gw) @ self.ridge_.coef_


class GBMDelta(GBMImpacto):
    """HistGradientBoostingRegressor sobre (X, z) con objetivo delta = p^X - p0."""

    nombre = "gbm_delta"
    objetivo = "delta"

    def fit(self, Z: pd.DataFrame, x_gw, y=None) -> GBMDelta:
        delta = Z["precio_parque"].to_numpy(float) - Z["p0"].to_numpy(float)
        self.gbm_ = HistGradientBoostingRegressor(random_state=0, early_stopping=False, **self.parametros)
        self.gbm_.fit(self._matriz(Z, x_gw), delta)
        return self


def precio_predicho(modelo, Z: pd.DataFrame, x_gw) -> np.ndarray:
    """Precio p^X predicho: p0 + delta^ (modelos delta) o y^ * p0 (modelos de cociente)."""
    pred = modelo.predict(Z, x_gw)
    p0 = Z["p0"].to_numpy(float)
    return p0 + pred if getattr(modelo, "objetivo", "ratio") == "delta" else pred * p0


# ── Validacion y metricas


def validacion_por_mes(t: pd.DataFrame, fabricas: dict) -> pd.DataFrame:
    """Predicciones fuera de muestra: cada mes se predice con un modelo entrenado sin ese mes.

    `fabricas`: {nombre: funcion sin argumentos que devuelve un modelo nuevo}. Devuelve una columna
    y_<nombre> por modelo (todas las filas, tambien las de p0 = 0) y la columna fold (mes de prueba).
    """
    out = pd.DataFrame(index=t.index)
    out["fold"] = t["mes"]
    meses = t["mes"].to_numpy()
    k = len(np.unique(meses))
    for nombre, fabrica in fabricas.items():
        pred = np.full(len(t), np.nan)
        precio = np.full(len(t), np.nan)
        for ent, prueba in GroupKFold(n_splits=k).split(t, groups=meses):
            m = fabrica().fit(t.iloc[ent], t["gw"].to_numpy()[ent], t["y"].to_numpy()[ent])
            x = t["gw"].to_numpy()[prueba]
            pred[prueba] = m.predict(t.iloc[prueba], x)
            precio[prueba] = precio_predicho(m, t.iloc[prueba], x)
        p0 = t["p0"].to_numpy(float)
        # y_<nombre>: cociente predicho (en modelos delta, el implicito, NaN donde p0 = 0)
        # p_<nombre>: precio predicho p^X (siempre definido)
        if getattr(fabrica(), "objetivo", "ratio") == "delta":
            pred = np.where(p0 != 0, precio / np.where(p0 != 0, p0, 1), np.nan)
        out[f"y_{nombre}"] = pred
        out[f"p_{nombre}"] = precio
    return out


def precio_de(t: pd.DataFrame, preds: pd.DataFrame, nombre: str) -> np.ndarray:
    """Precio predicho por un modelo: columna p_<nombre> si existe, y_<nombre> * p0 si no."""
    if f"p_{nombre}" in preds:
        return preds[f"p_{nombre}"].to_numpy(float)
    return preds[f"y_{nombre}"].to_numpy(float) * t["p0"].to_numpy(float)


def metricas(t: pd.DataFrame, preds: pd.DataFrame, nombres: list[str]) -> pd.DataFrame:
    """Errores de cada modelo: en y (donde existe) y en precio p^ = y^ * p0 (todas las filas)."""
    filas = []
    y = t["y"].to_numpy(float)
    ok = np.isfinite(y)
    pX, p0 = t["precio_parque"].to_numpy(float), t["p0"].to_numpy(float)
    bajo = np.abs(p0) < 5

    def media(v):
        return float(np.mean(v)) if len(v) else np.nan

    for n in nombres:
        yh = preds[f"y_{n}"].to_numpy(float)
        e_y, e_p = yh[ok] - y[ok], precio_de(t, preds, n) - pX
        filas.append({
            "model": n,
            "rmse_y": float(np.sqrt(np.mean(e_y**2))), "mae_y": float(np.mean(np.abs(e_y))),
            "median_abs_err_y": float(np.median(np.abs(e_y))),
            "mae_price_eur_mwh": float(np.mean(np.abs(e_p))),
            "rmse_price_eur_mwh": float(np.sqrt(np.mean(e_p**2))),
            "mae_price_p0_below_5": media(np.abs(e_p[bajo])),
            "mae_price_p0_above_5": media(np.abs(e_p[~bajo])),
        })
    return pd.DataFrame(filas)


def por_gw(t: pd.DataFrame, preds: pd.DataFrame, nombres: list[str], horas_spread: float) -> pd.DataFrame:
    """Por potencia: precio medio, spread diario y ingreso del parque, simulados y predichos.

    El ingreso predicho valora el programa simulado del parque (mw_parque) a los precios predichos:
    mide si el modelo conserva el valor de arbitraje que ve ENGIE, no solo el precio medio.
    """
    h = t["resolucion_min"].to_numpy(float) / 60
    filas = []
    for g, idx in t.groupby("gw").groups.items():
        s = t.loc[idx]
        precios = {"simulated": s["precio_parque"].to_numpy(float), "no_impact": s["p0"].to_numpy(float)}
        for n in nombres:
            precios[n] = precio_de(t, preds, n)[t.index.get_indexer(idx)]
        dias = s["dia"].nunique()
        for fuente, p in precios.items():
            sp = daily_spread(pd.DataFrame({"timestamp_utc": s["timestamp_utc"], "precio_es": p,
                                            "resolucion_min": s["resolucion_min"]}), horas_spread)
            ingreso = float(-(s["mw_parque"].to_numpy(float) * p * h[idx]).sum())
            filas.append({"gw": g, "source": fuente, "mean_price": float(np.mean(p)),
                          f"spread_{horas_spread:g}h": float(sp.iloc[:, 1].mean()),
                          "revenue_eur_mw_day": ingreso / (g * 1000) / dias})
    return pd.DataFrame(filas)


# ── Etapa 3: prevision de ENGIE


def ajusta_prevision(modelo, prevision: pd.DataFrame, gw, fundamentales: list[str]) -> pd.DataFrame:
    """p' = f^(t, X_s, z^E) * p^E para una prevision de ENGIE.

    `prevision`: una fila por cuarto de hora con timestamp_utc, p_engie (€/MWh), las columnas
    `fundamentales`, gas_mibgas_es y co2_eur_t (z^E, coherentes con p_engie). `gw`: potencia
    instalada X_s, un numero o una serie alineada con `prevision` (trayectoria PNIEC, por ejemplo).
    Devuelve timestamp_utc, gw, p_engie, y_hat y p_ajustado.
    """
    gw = np.broadcast_to(np.asarray(gw, float), (len(prevision),))
    df = (prevision.rename(columns={"p_engie": "p0"}).assign(_gw=gw)
          .sort_values("timestamp_utc").reset_index(drop=True))
    z = variables(df, fundamentales)
    x = df["_gw"].to_numpy(float)
    p = precio_predicho(modelo, z, x)
    p0 = z["p0"].to_numpy(float)
    # y_hat: cociente (modelos de cociente) o el implicito p'/p0, NaN donde p0 = 0 (modelos delta)
    if getattr(modelo, "objetivo", "ratio") == "delta":
        y_hat = p / np.where(p0 != 0, p0, np.nan)
    else:
        y_hat = modelo.predict(z, x)
    return pd.DataFrame({"timestamp_utc": z["timestamp_utc"], "gw": x, "p_engie": z["p0"],
                         "y_hat": y_hat, "p_ajustado": p})
