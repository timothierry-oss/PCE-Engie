from datetime import date

import numpy as np
import pytest

from pce68.data.curvas import parsear_curvas, precios
from pce68.dispatch.battery import Battery, ciclo_optimo
from pce68.dispatch.flota import ciclo, despacho

BAT = Battery(duracion_horas=2, eficiencia_ida_vuelta=0.8, ciclos_maximos_anuales=365)


def curva_pbc(dia: date, demanda_mw: list[float]) -> str:
    """Un dia de 24 h: oferta en escalones de 1000 MW a 0, 10, 20... €/MWh y demanda inelastica."""
    cab = f"OMIE;;;{dia:%d/%m/%Y};\n\nPeriodo;Fecha;Pais;Unidad;Tipo;Potencia;Precio;OC;Tipologia;\n"
    lineas = []
    for k, dem in enumerate(demanda_mw, start=1):
        lineas += [f"{k};{dia:%d/%m/%Y};MI;;V;1000,0;{10 * i},00;O;S;" for i in range(10)]
        lineas.append(f"{k};{dia:%d/%m/%Y};MI;;C;{dem:.1f};4000,00;O;S;".replace(".", ","))
    return cab + "\n".join(lineas) + "\n"


# valle a mediodia (demanda 1500 MW -> 10 €/MWh), pico de tarde (6500 MW -> 60 €/MWh)
DEMANDA = [3500.0] * 10 + [1500.0] * 6 + [3500.0] * 3 + [6500.0] * 3 + [3500.0] * 2


def test_ciclo_tiene_el_margen_de_battery_y_un_perfil_coherente():
    p = np.array([30, 10, 5, 5, 20, 60, 80, 70, 30], float)
    h = np.ones_like(p)
    margen, perfil = ciclo(p, h, BAT)
    assert margen == pytest.approx(ciclo_optimo(p, h, BAT))
    assert perfil.max() <= 1 + 1e-9 and perfil.min() >= -1 - 1e-9
    assert perfil[perfil > 0].sum() == pytest.approx(2 / 0.8)  # compra E/eficiencia
    assert -perfil[perfil < 0].sum() == pytest.approx(2)  # vende E
    assert np.flatnonzero(perfil > 0).max() < np.flatnonzero(perfil < 0).min()  # carga antes


def test_ciclo_sin_beneficio_no_mueve_nada():
    margen, perfil = ciclo(np.full(6, 40.0), np.ones(6), BAT)
    assert margen == 0 and not perfil.any()


def test_despacho_carga_en_el_valle_descarga_en_el_pico_y_cierra_el_spread():
    c = parsear_curvas(curva_pbc(date(2025, 5, 15), DEMANDA))
    base = precios(c)["precio_eur_mwh"].to_numpy()
    assert base[12] == 10 and base[20] == 60

    r = despacho(c, 1000, BAT, paso_mw=250)
    d = r["delta_mw"].to_numpy()
    assert (d[10:16] >= 0).all() and d[10:16].sum() > 0  # carga a mediodia
    assert (d[19:22] <= 0).all() and d[19:22].sum() < 0  # descarga en el pico
    assert d[d > 0].sum() * 0.8 == pytest.approx(-d[d < 0].sum())  # balance de energia
    p = r["precio_eur_mwh"].to_numpy()
    assert p[19:22].max() <= base[19:22].max() and p[10:16].min() >= base[10:16].min()
    assert sum(r.attrs["mw_activos"].values()) == 1000


def test_despacho_sin_parque_deja_el_precio_base():
    c = parsear_curvas(curva_pbc(date(2025, 5, 15), DEMANDA))
    r = despacho(c, 0, BAT, paso_mw=100)
    assert not r["delta_mw"].any()
    assert np.array_equal(r["precio_eur_mwh"].to_numpy(), precios(c)["precio_eur_mwh"].to_numpy())
