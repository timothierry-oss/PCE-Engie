import numpy as np
import pandas as pd
import pytest

from pce68.modelo.impacto import (
    GBMDelta,
    GBMImpacto,
    RidgeDelta,
    RidgeImpacto,
    ajusta_prevision,
    columnas_z,
    metricas,
    precio_predicho,
    validacion_por_mes,
    variables,
)

FUND = ["demanda", "eolica", "solar_fv"]


def base(dias: int = 4, desde: str = "2026-03-02") -> pd.DataFrame:
    """Cuartos de hora sinteticos: p0 con valle a mediodia, solar a mediodia."""
    inicio = pd.Timestamp(desde, tz="Europe/Madrid")
    ts = pd.date_range(inicio, periods=96 * dias, freq="15min").tz_convert("UTC")
    hora = ts.tz_convert("Europe/Madrid").hour + ts.tz_convert("Europe/Madrid").minute / 60
    solar = np.clip(np.sin((hora - 7) / 12 * np.pi), 0, None) * 15000
    return pd.DataFrame({
        "timestamp_utc": ts, "p0": 100 - solar / 200, "demanda": 28000.0, "eolica": 6000.0,
        "solar_fv": solar, "gas_mibgas_es": 40.0, "co2_eur_t": 75.0,
    })


def tabla(dias_por_mes: int = 3) -> pd.DataFrame:
    """Tabla como la de tabla_aprendizaje: tres meses, tres potencias, y = 1 + 0,02 X solar / 15 GW."""
    partes = []
    for mes in ("2026-01-05", "2026-02-02", "2026-03-02"):
        z = variables(base(dias_por_mes, mes), FUND)
        for g in (1.0, 5.0, 17.0):
            y = 1 + 0.02 * g * z["solar_fv"].to_numpy() / 15000
            partes.append(z.assign(gw=g, y=y, precio_base=z["p0"], precio_parque=y * z["p0"],
                                   dia=z["timestamp_utc"].dt.tz_convert("Europe/Madrid").dt.date,
                                   mes=mes[:7]))
    return pd.concat(partes, ignore_index=True)


def test_variables_forma_del_dia_y_vecinos():
    z = variables(base(2), FUND)
    assert len(z) == 192 and z["hora"].between(0, 23).all()
    assert z["cuarto"].max() == 95
    # el precio de una hora antes es el de 4 cuartos antes (y el propio en el primer cuarto)
    np.testing.assert_allclose(z["p0_hora_antes"].to_numpy()[4:], z["p0"].to_numpy()[:-4])
    assert z["p0_hora_antes"].iloc[0] == z["p0"].iloc[0]
    assert (z["p0_dia_rango"] == z["p0_dia_max"] - z["p0_dia_min"]).all()
    np.testing.assert_allclose(z["cuota_renovable"], (z["eolica"] + z["solar_fv"]) / z["demanda"])


def test_ridge_sin_baterias_da_uno_y_aprende_el_efecto():
    t = tabla()
    cols = columnas_z(t)
    m = RidgeImpacto(cols, alpha=1e-3).fit(t, t["gw"].to_numpy(), t["y"].to_numpy())
    np.testing.assert_allclose(m.predict(t, 0.0), 1.0)
    pred = m.predict(t, t["gw"].to_numpy())
    assert np.abs(pred - t["y"].to_numpy()).max() < 0.05


def test_ignora_y_no_definido():
    t = tabla()
    t.loc[::7, "y"] = np.nan
    cols = columnas_z(t)
    for m in (RidgeImpacto(cols), GBMImpacto(cols, max_iter=20)):
        pred = m.fit(t, t["gw"].to_numpy(), t["y"].to_numpy()).predict(t, t["gw"].to_numpy())
        assert np.isfinite(pred).all()


def test_validacion_por_mes_no_usa_el_mes_de_prueba():
    t = tabla()
    vistos = []

    class Espia(RidgeImpacto):
        def fit(self, Z, x, y):
            vistos.append(set(Z["mes"]))
            return super().fit(Z, x, y)

    preds = validacion_por_mes(t, {"espia": lambda: Espia(columnas_z(t))})
    assert len(vistos) == 3 and all(len(v) == 2 for v in vistos)
    assert preds["y_espia"].notna().all()
    met = metricas(t, preds.assign(y_no_impact=1.0), ["espia", "no_impact"])
    assert set(met["model"]) == {"espia", "no_impact"}


def test_ajusta_prevision_multiplica_y_respeta_la_trayectoria_desordenada():
    t = tabla()
    m = RidgeImpacto(columnas_z(t), alpha=1e-3).fit(t, t["gw"].to_numpy(), t["y"].to_numpy())
    prev = base(2, "2027-03-01").rename(columns={"p0": "p_engie"})
    gw = np.where(np.arange(len(prev)) < 96, 0.0, 17.0)  # 0 GW el primer dia, 17 GW el segundo
    orden = np.random.default_rng(0).permutation(len(prev))
    r = ajusta_prevision(m, prev.iloc[orden], gw[orden], FUND)
    assert r["timestamp_utc"].is_monotonic_increasing
    np.testing.assert_allclose(r["gw"].to_numpy(), gw)
    np.testing.assert_allclose(r["y_hat"].to_numpy()[:96], 1.0)
    np.testing.assert_allclose(r["p_ajustado"], r["y_hat"] * r["p_engie"])
    assert r["y_hat"].to_numpy()[96:].max() == pytest.approx(1 + 0.02 * 17, abs=0.05)


def test_ponderacion_p0_cuadrado_es_el_error_en_precio():
    """Ridge con pesos p0^2 en y == ridge sobre (p^X - p0) con el diseño multiplicado por p0."""
    from sklearn.linear_model import Ridge

    from pce68.modelo.impacto import pesos

    t = tabla()
    rng = np.random.default_rng(1)
    t["y"] = t["y"] + rng.normal(0, 0.05, len(t))
    t["precio_parque"] = t["y"] * t["p0"]
    m = RidgeImpacto(columnas_z(t), alpha=3.0, ponderacion="p0_cuadrado")
    m.fit(t, t["gw"].to_numpy(), t["y"].to_numpy())
    p0 = t["p0"].to_numpy()
    s = np.sqrt(p0**2 / np.mean(p0**2))  # raiz de los pesos normalizados
    D = m._diseno(m._phi(t, ajustar=False), t["gw"].to_numpy())
    directo = Ridge(alpha=3.0, fit_intercept=False).fit(D * s[:, None], (t["y"].to_numpy() - 1) * s)
    np.testing.assert_allclose(m.ridge_.coef_, directo.coef_, rtol=1e-6, atol=1e-8)
    # el peso de un cuarto de hora con p0 = 0,01 es despreciable frente a uno con p0 = 100
    w = pesos(np.array([0.01, 100.0]), "p0_cuadrado")
    assert w[0] / w[1] == pytest.approx(1e-8)
    assert pesos(p0, "ninguna") is None
    with pytest.raises(ValueError):
        pesos(p0, "otra")


def test_gbm_acepta_la_ponderacion():
    t = tabla()
    cols = columnas_z(t)
    m = GBMImpacto(cols, ponderacion="p0_cuadrado", max_iter=20).fit(t, t["gw"].to_numpy(), t["y"].to_numpy())
    assert np.isfinite(m.predict(t, t["gw"].to_numpy())).all()


def test_modelos_delta_aprenden_el_efecto_aditivo_incluso_con_p0_cero():
    t = tabla()
    t.loc[::5, "p0"] = 0.0  # con p0 = 0 el cociente no existe, pero delta si
    t["precio_parque"] = t["p0"] + 0.02 * t["gw"] * t["solar_fv"] / 15000 * 50
    t["y"] = t["precio_parque"] / t["p0"].replace(0, np.nan)
    cols = columnas_z(t)
    x = t["gw"].to_numpy()
    for m in (RidgeDelta(cols, alpha=1e-3), GBMDelta(cols, max_iter=50)):
        m.fit(t, x, t["y"].to_numpy())
        p = precio_predicho(m, t, x)
        assert np.isfinite(p).all()
        assert np.abs(p - t["precio_parque"].to_numpy()).mean() < 2.0
    # ridge delta: sin baterias, delta = 0 por construccion
    np.testing.assert_allclose(RidgeDelta(cols).fit(t, x).predict(t, 0.0), 0.0)


def test_validacion_y_prevision_con_modelo_delta():
    t = tabla()
    cols = columnas_z(t)
    preds = validacion_por_mes(t, {"gbm_delta": lambda: GBMDelta(cols, max_iter=20)})
    np.testing.assert_allclose(preds["p_gbm_delta"] / t["p0"], preds["y_gbm_delta"])
    met = metricas(t, preds.assign(y_no_impact=1.0), ["gbm_delta", "no_impact"])
    assert set(met["model"]) == {"gbm_delta", "no_impact"}
    m = RidgeDelta(cols, alpha=1e-3).fit(t, t["gw"].to_numpy())
    prev = base(2, "2027-03-01").rename(columns={"p0": "p_engie"})
    r = ajusta_prevision(m, prev, 17.0, FUND)
    assert (r["p_ajustado"] - r["p_engie"]).abs().max() > 0  # hay efecto con 17 GW
