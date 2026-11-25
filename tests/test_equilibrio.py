import numpy as np
import pytest

from pce68.dispatch.battery import Battery
from pce68.dispatch.equilibrio import (
    equilibrio_exacto,
    equilibrio_parque,
    precios_marginales,
    segmentos,
)
from pce68.dispatch.lp import plan_lp

BAT = Battery(duracion_horas=2, eficiencia_ida_vuelta=0.85, ciclos_maximos_anuales=365)


def dias(n=4):
    """Precios horarios: barato de 10 a 15 h, caro de 19 a 22 h."""
    h = np.arange(24)
    p = np.where((h >= 10) & (h < 15), 20.0, np.where((h >= 19) & (h < 22), 120.0, 60.0))
    return np.tile(p, n)


def test_sin_respuesta_es_el_plan_precio_aceptante():
    p0, h = dias(), np.ones(96)
    eq = equilibrio_parque(p0, h, lambda d: np.zeros_like(d), 1000.0, BAT, ciclos=4.0, iteraciones=1)
    _, perfil, _ = plan_lp(p0, h, BAT, 4.0)
    np.testing.assert_allclose(eq.d_mw, 1000.0 * perfil, atol=1e-6)
    np.testing.assert_allclose(eq.precio, p0)


def test_la_respuesta_reduce_el_spread_y_el_plan_es_factible():
    p0, h = dias(), np.ones(96)
    pend = 0.01  # €/MWh por MW neto: 1000 MW mueven el precio 10 €/MWh
    ev = lambda d: pend * d  # noqa: E731
    eq = equilibrio_parque(p0, h, ev, 5000.0, BAT, ciclos=4.0, iteraciones=40)
    assert np.abs(eq.d_mw).max() <= 5000.0 + 1e-6                     # potencia
    assert -eq.d_mw[eq.d_mw < 0].sum() <= 4.0 * 2 * 5000.0 + 1e-6      # ciclos: descarga <= ciclos * D * X
    cargado = eq.d_mw > 0
    assert eq.precio[cargado].mean() > p0[cargado].mean()              # cargar sube el precio
    descarga = eq.d_mw < 0
    assert eq.precio[descarga].mean() < p0[descarga].mean()            # descargar lo baja
    # el spread entre el momento de carga y el de descarga se estrecha
    assert (eq.precio[eq.d_mw < 0].mean() - eq.precio[eq.d_mw > 0].mean()
            < p0[eq.d_mw < 0].mean() - p0[eq.d_mw > 0].mean())
    assert eq.brecha_rel < 0.05                                        # equilibrio aproximado
    assert eq.ingreso_eur == pytest.approx(-(eq.d_mw * eq.precio).sum())


def test_mas_respuesta_menos_ingreso():
    p0, h = dias(), np.ones(96)
    ing = [equilibrio_parque(p0, h, lambda d, s=s: s * d, 5000.0, BAT, 4.0, iteraciones=30).ingreso_eur
           for s in (0.0, 0.005, 0.02)]
    assert ing[0] > ing[1] > ing[2] > 0


def test_exacto_sin_respuesta_es_el_plan_precio_aceptante():
    p0, h = dias(), np.ones(96)
    anchos, medios = segmentos(5000.0, 1000.0)
    pc, pd_ = precios_marginales(p0, lambda d: np.zeros_like(d), medios)
    d, _ = equilibrio_exacto(p0, h, pc, pd_, anchos, BAT, ciclos=4.0)
    margen, _, _ = plan_lp(p0, h, BAT, 4.0)
    assert -(d * p0 * h).sum() == pytest.approx(5000.0 * margen, rel=1e-6)


def test_exacto_coincide_con_las_medias_sucesivas_en_el_caso_lineal():
    p0, h = dias(), np.ones(96)
    ev = lambda d: 0.01 * d  # noqa: E731
    anchos, medios = segmentos(5000.0, 100.0)  # tramos finos: la respuesta lineal queda bien aproximada
    pc, pd_ = precios_marginales(p0, ev, medios)
    d, _ = equilibrio_exacto(p0, h, pc, pd_, anchos, BAT, ciclos=4.0)
    iter_ = equilibrio_parque(p0, h, ev, 5000.0, BAT, 4.0, iteraciones=300)
    ingreso = -(d * (p0 + ev(d)) * h).sum()
    assert ingreso == pytest.approx(iter_.ingreso_eur, rel=0.03)
    assert np.abs(d - iter_.d_mw).max() < 0.1 * 5000.0
    assert np.abs(d).max() <= 5000.0 + 1e-6
    assert -d[d < 0].sum() <= 4.0 * 2 * 5000.0 + 1e-6   # presupuesto de ciclos
    # es un equilibrio: ninguna bateria gana apreciablemente cambiando de plan al precio que deja el parque
    precio = p0 + ev(d)
    mejor = 5000.0 * plan_lp(precio, h, BAT, 4.0)[0]
    assert (mejor - (-(d * precio * h).sum())) / mejor < 0.05
