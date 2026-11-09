from datetime import date

import numpy as np
import pandas as pd
import pytest

from pce68.data.omie import parsear
from pce68.dispatch.battery import Battery, ciclo_optimo, simulate_price_taker
from pce68.features.metrics import daily_spread
from pce68.features.regimenes import anotar, resumen

TZ = "Europe/Madrid"


def marginalpdbc(dia: date, precios: list[float]) -> str:
    filas = [f"{dia:%Y;%m;%d};{i};{p + 1:.2f};{p:.2f};" for i, p in enumerate(precios, start=1)]
    return "MARGINALPDBC;\n" + "\n".join(filas) + "\n*\n"


def serie(dia: str, precios: list[float], resol: int = 60) -> pd.DataFrame:
    t0 = pd.Timestamp(dia).tz_localize(TZ).tz_convert("UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": t0 + pd.to_timedelta(np.arange(len(precios)) * resol, unit="min"),
            "precio_es": np.asarray(precios, float),
            "resolucion_min": resol,
        }
    )


# ── OMIE


@pytest.mark.parametrize(
    ("dia", "n", "resol"),
    [
        (date(2024, 1, 15), 24, 60),
        (date(2024, 3, 31), 23, 60),  # cambio a horario de verano
        (date(2024, 10, 27), 25, 60),  # cambio a horario de invierno
        (date(2025, 10, 15), 96, 15),  # diario cuarto-horario
        (date(2025, 10, 26), 100, 15),
    ],
)
def test_parsear_cubre_el_dia_local(dia, n, resol):
    df = parsear(marginalpdbc(dia, list(range(n))))
    assert len(df) == n and (df["resolucion_min"] == resol).all()
    local = df["timestamp_utc"].dt.tz_convert(TZ)
    assert local.iloc[0] == pd.Timestamp(dia).tz_localize(TZ)
    assert (local.dt.date == dia).all()
    assert df["timestamp_utc"].diff().dropna().eq(pd.Timedelta(minutes=resol)).all()
    # España es la ultima columna, Portugal la anterior
    assert df["precio_es"].tolist() == [float(i) for i in range(n)]
    assert df["precio_pt"].tolist() == [float(i + 1) for i in range(n)]


def test_parsear_rechaza_periodos_que_no_encajan():
    with pytest.raises(ValueError):
        parsear(marginalpdbc(date(2024, 1, 15), [50.0] * 23))


# ── Spread


def test_spread_horario():
    p = [10, 0, 50, 40, 100, 90] + [60] * 18
    s = daily_spread(serie("2024-05-01", p), 2.0)
    assert s.columns.tolist() == ["dia_local", "spread_2h"]
    assert s["spread_2h"].iloc[0] == pytest.approx((100 + 90) / 2 - (0 + 10) / 2)


def test_spread_igual_en_cuartos_que_en_horas():
    p = np.r_[np.full(8, 20.0), np.full(8, 80.0), np.full(8, 50.0)]
    h = daily_spread(serie("2024-05-01", p.tolist()), 2.0)["spread_2h"].iloc[0]
    q = daily_spread(serie("2024-05-01", np.repeat(p, 4).tolist(), 15), 2.0)["spread_2h"].iloc[0]
    assert h == pytest.approx(q) == pytest.approx(60.0)


# ── Bateria


def test_ciclo_respeta_el_orden_temporal():
    # Lo caro va antes que lo barato: sin poder descargar antes de cargar no hay margen.
    bat = Battery(1, 1.0, 365)
    p = np.array([100.0, 0.0])
    assert ciclo_optimo(p, np.ones(2), bat) == 0.0
    assert ciclo_optimo(p[::-1].copy(), np.ones(2), bat) == pytest.approx(100.0)


def test_ciclo_con_eficiencia():
    bat = Battery(2, 0.8, 365)
    # Para vender 2 MWh a 100 hay que comprar 2.5 MWh: 2 h a 0 y media hora a 10.
    p = np.array([0.0, 0.0, 10.0, 10.0, 100.0, 100.0])
    assert ciclo_optimo(p, np.ones(6), bat) == pytest.approx(200 - 0.5 * 10)


def test_simulacion_aplica_el_tope_de_ciclos():
    dias = pd.concat(
        [serie(f"2024-05-0{i}", [0.0] * 12 + [10.0 * i] * 12) for i in range(1, 4)], ignore_index=True
    )
    r = simulate_price_taker(dias, Battery(2, 1.0, 2))
    assert r["ciclos"].tolist() == [0, 1, 1]
    assert r["ingreso_eur_por_mw"].tolist() == pytest.approx([0, 40, 60])


# ── Regímenes


def test_regimenes_por_fecha_local():
    # 30/06/2021 22:30 UTC ya es 01/07 en Madrid: crisis_gas
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(
                ["2021-06-30 21:30", "2021-06-30 22:30", "2022-06-15 00:00", "2024-01-01 00:00"], utc=True
            ),
            "precio_es": [1.0, 2.0, 3.0, 4.0],
            "resolucion_min": 60,
        }
    )
    assert anotar(df)["regimen"].astype(str).tolist() == [
        "pre_crisis",
        "crisis_gas",
        "excepcion_iberica",
        "normalizacion",
    ]
    r = resumen(df)
    assert r["regimen"].tolist() == ["pre_crisis", "crisis_gas", "excepcion_iberica", "normalizacion"]
    assert r.columns.tolist() == ["regimen", "periodos", "desde", "hasta", "precio_medio", "precio_max"]
