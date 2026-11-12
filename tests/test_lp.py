from datetime import date

import numpy as np
import pytest

from pce68.data.curvas import parsear_curvas, precios
from pce68.dispatch.battery import Battery
from pce68.dispatch.lp import despacho_lp, plan_lp

BAT = Battery(duracion_horas=2, eficiencia_ida_vuelta=0.85, ciclos_maximos_anuales=365)


def _reconstruye_carga(perfil, h, bat, s0):
    c, d = np.clip(perfil, 0, None), np.clip(-perfil, 0, None)
    return s0 + np.cumsum(bat.eficiencia_ida_vuelta * c * h - d * h)


def test_plan_respeta_balance_bordes_limites_y_presupuesto():
    rng = np.random.default_rng(0)
    p = rng.uniform(-20, 250, 192)  # dos dias en cuartos de hora, con algun precio negativo
    h = np.full(192, 0.25)
    margen, perfil, s = plan_lp(p, h, BAT, ciclos=2)
    assert margen > 0
    assert perfil.max() <= 1 + 1e-9 and perfil.min() >= -1 - 1e-9
    assert s.min() >= -1e-9 and s.max() <= BAT.duracion_horas + 1e-9
    assert s[-1] == pytest.approx(1.0)  # 50 % de 2 MWh al final
    # la carga devuelta sale del perfil: no hay carga y descarga a la vez (que el perfil esconderia)
    np.testing.assert_allclose(_reconstruye_carga(perfil, h, BAT, 1.0), s, atol=1e-6)
    assert (np.clip(-perfil, 0, None) * h).sum() <= 2 * BAT.duracion_horas + 1e-6
    assert margen == pytest.approx(-(perfil * p * h).sum())


def test_carga_antes_de_medianoche_y_descarga_el_dia_siguiente():
    p = np.full(48, 50.0)
    p[23], p[31] = 5.0, 120.0  # 23 h del primer dia barata, 7 h del segundo cara
    margen, perfil, _ = plan_lp(p, np.ones(48), BAT, ciclos=2)
    assert perfil[23] == pytest.approx(1.0) and perfil[31] == pytest.approx(-1.0)
    # vende 1 MWh a 120; compra 1 MWh a 5 (guarda 0,85) y repone 0,15 MWh comprando a 50
    assert margen == pytest.approx(120 - 5 - 50 * 0.15 / 0.85, abs=1e-6)


def test_el_presupuesto_de_ciclos_limita_la_descarga():
    p = np.tile([0.0, 100.0], 24)  # precios que invitan a ciclar cada hora
    h = np.ones(48)
    _, perfil1, _ = plan_lp(p, h, BAT, ciclos=1)
    _, perfil5, _ = plan_lp(p, h, BAT, ciclos=5)
    descarga = lambda x: (np.clip(-x, 0, None) * h).sum()  # noqa: E731
    assert descarga(perfil1) == pytest.approx(BAT.duracion_horas)
    assert descarga(perfil5) > descarga(perfil1)


def test_con_precios_planos_no_hace_nada():
    margen, perfil, s = plan_lp(np.full(96, 50.0), np.full(96, 0.25), BAT, ciclos=1)
    assert margen == pytest.approx(0.0) and not perfil.any()
    np.testing.assert_allclose(s, 1.0)


# ── Parque sobre unas curvas sinteticas (las de tests/test_flota.py)


def curva_pbc(dia: date, demanda_mw: list[float]) -> str:
    """Un dia de 24 h: oferta en escalones de 1000 MW a 0, 10, 20... €/MWh y demanda inelastica."""
    cab = f"OMIE;;;{dia:%d/%m/%Y};\n\nPeriodo;Fecha;Pais;Unidad;Tipo;Potencia;Precio;OC;Tipologia;\n"
    lineas = []
    for k, dem in enumerate(demanda_mw, start=1):
        lineas += [f"{k};{dia:%d/%m/%Y};MI;;V;1000,0;{10 * i},00;O;S;" for i in range(10)]
        lineas.append(f"{k};{dia:%d/%m/%Y};MI;;C;{dem:.1f};4000,00;O;S;".replace(".", ","))
    return cab + "\n".join(lineas) + "\n"


DEMANDA = [3500.0] * 10 + [1500.0] * 6 + [3500.0] * 3 + [6500.0] * 3 + [3500.0] * 2


def test_despacho_lp_carga_en_el_valle_descarga_en_el_pico():
    c = parsear_curvas(curva_pbc(date(2025, 5, 15), DEMANDA))
    base = precios(c)["precio_eur_mwh"].to_numpy()
    r = despacho_lp(c, 1000, BAT, paso_mw=250, ciclos=1)
    d = r["delta_mw"].to_numpy()
    assert (d[10:16] >= 0).all() and d[10:16].sum() > 0  # carga a mediodia
    assert d[19:22].sum() < 0  # descarga en el pico
    assert 0.85 * d[d > 0].sum() == pytest.approx(-d[d < 0].sum())  # bordes iguales: balance cerrado
    assert r["carga_mwh"].iloc[-1] == pytest.approx(1000 * 2 * 0.5)
    p = r["precio_eur_mwh"].to_numpy()
    assert p[19:22].max() <= base[19:22].max() and p[10:16].min() >= base[10:16].min()
    tramos = r.attrs["tramos"]
    assert len(tramos) == 4 and tramos["activo"].all()
    assert r.attrs["planes"].shape == (4, 24)


def test_despacho_lp_sin_parque_deja_el_precio_base():
    c = parsear_curvas(curva_pbc(date(2025, 5, 15), DEMANDA))
    r = despacho_lp(c, 0, BAT, paso_mw=250, ciclos=1)
    assert not r["delta_mw"].any()
    np.testing.assert_array_equal(r["precio_eur_mwh"].to_numpy(), precios(c)["precio_eur_mwh"].to_numpy())
