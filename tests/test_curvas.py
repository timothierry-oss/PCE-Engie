from datetime import date

import numpy as np
import pandas as pd
import pytest

from pce68.data.curvas import cargar, parsear_curvas, precio_equilibrio, precios

TZ = "Europe/Madrid"

# Escalones repetidos en cada periodo: (tipo, MW, €/MWh, ofertada/casada, tipologia).
# La oferta cubre la demanda de 2000 MW al precio de 50; la compleja a 90 (80 MW ofertados,
# 50 casados) solo hace falta si se anade demanda. Las filas "C" simples duplican filas
# ofertadas y no deben contar; la "C" de la compleja si se guarda (columna casada).
FILAS = [
    ("V", 100.0, 10.0, "O", "S"),
    ("V", 1991.1, 50.0, "O", "S"),
    ("V", 80.0, 90.0, "O", "C02"),
    ("V", 50.0, 90.0, "C", "C02"),
    ("C", 2000.0, 4000.0, "O", "S"),
    ("V", 100.0, 10.0, "C", "S"),
    ("C", 2000.0, 4000.0, "C", "S"),
    ("V", 9.0, 0.0, "C", "Imp FR"),
]


def num(x: float, decimales: int) -> str:
    """1991.1 -> '1.991,1', como en los ficheros de OMIE."""
    s = f"{x:,.{decimales}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def etiqueta_hq(k: int) -> str:
    """Periodo k (desde 1) -> 'HxQy', como en los ficheros cuarto-horarios de OMIE."""
    return f"H{(k - 1) // 4 + 1}Q{(k - 1) % 4 + 1}"


def curva_pbc(dia: date, n: int, filas=FILAS, etiqueta=str) -> str:
    cab = (
        f"OMIE - Mercado de electricidad;Fecha Emisi\xf3n :14/05/2025 - 13:30;;{dia:%d/%m/%Y};"
        "Mercado diario;;;;\n\n"
        "Periodo;Fecha;Pais;Unidad;Tipo Oferta;Potencia Compra/Venta;Precio Compra/Venta;"
        "Ofertada (O)/Casada (C);Tipolog\xeda de Oferta;\n"
    )
    lineas = [
        f"{etiqueta(k)};{dia:%d/%m/%Y};MI;;{tipo};{num(mw, 1)};{num(precio, 2)};{oc};{tip};"
        for k in range(1, n + 1)
        for tipo, mw, precio, oc, tip in filas
    ]
    return cab + "\n".join(lineas) + "\n"


# ── Lectura


@pytest.mark.parametrize(
    ("dia", "n", "resol"),
    [
        (date(2025, 5, 15), 24, 60),
        (date(2025, 3, 30), 23, 60),  # cambio a horario de verano
        (date(2025, 10, 26), 25, 60),  # cambio a horario de invierno
        (date(2025, 10, 15), 96, 15),  # diario cuarto-horario
        (date(2025, 10, 26), 100, 15),
    ],
)
def test_parsear_curvas_cubre_el_dia_local(dia, n, resol):
    c = parsear_curvas(curva_pbc(dia, n))
    ts = c["timestamp_utc"].drop_duplicates().reset_index(drop=True)
    assert len(ts) == n and (c["resolucion_min"] == resol).all()
    assert ts.iloc[0] == pd.Timestamp(dia).tz_localize(TZ)
    assert (ts.dt.tz_convert(TZ).dt.date == dia).all()
    assert ts.diff().dropna().eq(pd.Timedelta(minutes=resol)).all()


@pytest.mark.parametrize(
    ("dia", "n"),
    [(date(2026, 9, 13), 96), (date(2026, 3, 29), 92), (date(2026, 10, 25), 100)],
)
def test_parsear_curvas_etiquetas_hq_cuarto_horarias(dia, n):
    c = parsear_curvas(curva_pbc(dia, n, etiqueta=etiqueta_hq))
    ts = c["timestamp_utc"].drop_duplicates().reset_index(drop=True)
    assert len(ts) == n and (c["resolucion_min"] == 15).all()
    assert ts.iloc[0] == pd.Timestamp(dia).tz_localize(TZ)
    assert ts.iloc[6] == ts.iloc[0] + pd.Timedelta(minutes=90)  # H2Q3 = periodo 7
    assert ts.diff().dropna().eq(pd.Timedelta(minutes=15)).all()


def test_parsear_curvas_rechaza_etiquetas_desconocidas():
    with pytest.raises(ValueError):
        parsear_curvas(curva_pbc(date(2026, 9, 13), 96, etiqueta=lambda k: f"H{k}X"))


def test_parsear_curvas_contenido():
    c = parsear_curvas(curva_pbc(date(2025, 5, 15), 24))
    primero = c[c["timestamp_utc"] == c["timestamp_utc"].min()]
    oferta = primero[primero["lado"] == "oferta"].set_index(["precio", "tipologia", "casada"])["potencia_mw"]
    # el punto de millar y la coma decimal se leen bien; las casadas simples no se suman dos veces
    assert oferta[(50.0, "S", False)] == pytest.approx(1991.1)
    assert oferta[(10.0, "S", False)] == pytest.approx(100.0)
    assert (10.0, "S", True) not in oferta.index
    assert oferta[(90.0, "C02", False)] == pytest.approx(80.0)  # compleja ofertada
    assert oferta[(90.0, "C02", True)] == pytest.approx(50.0)  # y su parte casada
    assert oferta[(0.0, "Imp FR", True)] == pytest.approx(9.0)  # el flujo casado si se conserva
    demanda = primero[primero["lado"] == "demanda"]
    assert demanda["potencia_mw"].tolist() == [2000.0] and demanda["precio"].tolist() == [4000.0]


def test_parsear_curvas_usa_es_si_hay_separacion_y_si_no_mi():
    # periodo 2 con separacion: la curva ES (oferta a 5) y la PT (oferta a 70), sin MI
    texto = curva_pbc(date(2025, 5, 15), 24)
    es = "2;15/05/2025;ES;;V;3.000,0;5,00;O;S;\n2;15/05/2025;ES;;C;1.000,0;4.000,00;O;S;"
    pt = "2;15/05/2025;PT;;V;3.000,0;70,00;O;S;\n2;15/05/2025;PT;;C;1.000,0;4.000,00;O;S;"
    lineas = [ln for ln in texto.splitlines() if not ln.startswith("2;")] + [es, pt]
    c = parsear_curvas("\n".join(lineas) + "\n")
    p = precios(c)["precio_eur_mwh"].tolist()
    assert p[1] == 5.0  # la curva ES; sumar ES y PT daria otro precio
    assert p[0] == p[2] == 50.0  # el resto, MI


def test_parsear_curvas_rechaza_periodos_que_no_encajan():
    with pytest.raises(ValueError):
        parsear_curvas(curva_pbc(date(2025, 5, 15), 20))


def test_cargar_une_y_filtra_dias(tmp_path):
    for d in (date(2025, 5, 15), date(2025, 5, 16)):
        parsear_curvas(curva_pbc(d, 24)).to_parquet(tmp_path / f"curvas_{d:%Y%m%d}.parquet", index=False)
    assert cargar(tmp_path)["timestamp_utc"].nunique() == 48
    assert cargar(tmp_path, [date(2025, 5, 16)])["timestamp_utc"].nunique() == 24
    with pytest.raises(FileNotFoundError):
        cargar(tmp_path, [date(2020, 1, 1)])


# ── Cruce de curvas


def test_equilibrio_con_demanda_inelastica():
    assert precio_equilibrio([10, 50], [100, 100], [1000], [150]) == 50


def test_equilibrio_en_un_escalon_de_demanda_parcialmente_casado():
    # 60 MW pagan 100 y 60 MW pagan 30; hay 100 MW a 10 y 100 MW a 50: se casan 100 MW y el precio es 30
    assert precio_equilibrio([10, 50], [100, 100], [100, 30], [60, 60]) == 30


def test_equilibrio_en_un_escalon_de_oferta_parcialmente_casado():
    assert precio_equilibrio([10, 50], [100, 100], [60, 20], [130, 100]) == 50


def test_equilibrio_si_la_oferta_no_alcanza_marca_la_demanda():
    assert precio_equilibrio([10], [100], [1000], [150]) == 1000
    assert precio_equilibrio([10], [100], [1e6], [150]) == 4000  # acotado al precio maximo


# ── Precios con la curva de un dia


def test_precios_base_y_efecto_de_la_bateria():
    c = parsear_curvas(curva_pbc(date(2025, 5, 15), 24))
    ts = c["timestamp_utc"].drop_duplicates()
    assert precios(c)["precio_eur_mwh"].eq(50.0).all()

    # oferta hasta 50: 9 + 100 + 1991,1 = 2100,1 MW; la compleja a 90 anade 50 (casada) u 80 (ofertada)
    carga = pd.Series(120.0, index=ts)  # 2120 MW de demanda: basta con lo casado de la compleja
    assert precios(c, carga)["precio_eur_mwh"].eq(90.0).all()
    assert precios(c, carga, complejas="no")["precio_eur_mwh"].eq(4000.0).all()  # la oferta no alcanza
    carga = pd.Series(160.0, index=ts)  # 2160 MW: hace falta lo ofertado, no solo lo casado
    assert precios(c, carga)["precio_eur_mwh"].eq(4000.0).all()
    assert precios(c, carga, complejas="ofertadas")["precio_eur_mwh"].eq(90.0).all()
    with pytest.raises(ValueError):
        precios(c, complejas=True)

    descarga = pd.Series(-1950.0, index=ts)  # oferta extra: desplaza el cruce hacia abajo
    assert precios(c, descarga)["precio_eur_mwh"].eq(10.0).all()


def test_precios_solo_cambia_en_los_periodos_con_delta():
    c = parsear_curvas(curva_pbc(date(2025, 5, 15), 24))
    ts = c["timestamp_utc"].drop_duplicates()
    r = precios(c, pd.Series({ts.iloc[3]: 120.0}))
    assert r["precio_eur_mwh"].tolist() == [50.0] * 3 + [90.0] + [50.0] * 20
    assert np.all(r["timestamp_utc"].to_numpy() == ts.to_numpy())
