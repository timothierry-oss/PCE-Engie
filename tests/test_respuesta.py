import numpy as np
import pandas as pd

from pce68.modelo.respuesta import RespuestaPrecio


def datos(n=400):
    """z = una variable (a); efecto exacto del nivel d: pendiente a * d (en €/MWh por MW)."""
    rng = np.random.default_rng(0)
    Z = pd.DataFrame({"a": rng.uniform(0.005, 0.02, n)})
    niveles = np.array([-8000.0, -3000.0, -1000.0, 0.0, 1000.0, 3000.0, 8000.0])
    dp = Z["a"].to_numpy()[:, None] * niveles[None, :]
    return Z, niveles, dp


def test_evaluador_cero_continuo_y_creciente():
    Z, niveles, dp = datos()
    ev = RespuestaPrecio(["a"], max_iter=60, min_samples_leaf=10).fit(Z, niveles, dp).evaluador(Z)
    n = len(Z)
    assert np.abs(ev(np.zeros(n))).max() < 1e-9                        # sin baterias, sin efecto
    d = np.linspace(-9000, 9000, 181)
    valores = np.stack([ev(np.full(n, x)) for x in d])                  # (181, n)
    assert (np.diff(valores, axis=0) >= -1e-9).all()                    # creciente en d
    assert np.abs(np.diff(valores, axis=0)).max() < 0.2 * np.abs(valores).max()  # sin saltos grandes
    np.testing.assert_allclose(ev(np.full(n, 20000.0)), ev(np.full(n, 8000.0)))  # plano fuera del rango


def test_evaluador_aprende_el_efecto():
    Z, niveles, dp = datos()
    ev = RespuestaPrecio(["a"], max_iter=200, min_samples_leaf=5).fit(Z, niveles, dp).evaluador(Z)
    real = Z["a"].to_numpy() * 2000.0
    assert np.abs(ev(np.full(len(Z), 2000.0)) - real).mean() < 0.15 * real.mean()
