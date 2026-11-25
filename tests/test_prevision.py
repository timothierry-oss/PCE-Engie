import numpy as np
import pandas as pd

from pce68.modelo.prevision import (
    COLUMNAS_FICHERO,
    VARIABLES_REALES,
    a_cuartos,
    a_horas,
    cobertura,
    lee_prevision,
)


def fichero(tmp_path, n=6, inicio="2027-03-28 00:00"):
    """Excel sintetico con el cambio de hora de marzo (el 28/03/2027 a las 02:00 salta a las 03:00)."""
    salto = pd.Timestamp("2027-03-28 02:00")
    horas = [h for h in pd.date_range(inicio, periods=n + 1, freq="h") if h != salto][:n]
    df = pd.DataFrame({c: np.ones(n) for c in COLUMNAS_FICHERO})
    df["FechaLocal"] = horas
    df["PRECIO_ES"] = np.arange(n) * 10.0
    df["DEMANDA_ES"], df["IMPORT_ES"] = -30000.0, 2000.0
    df["EXPORT_ES"] = pd.Series([-3000.0] * n, dtype=object)
    df.loc[1, "EXPORT_ES"] = "-4011,2vsc"
    ruta = tmp_path / "f.xlsx"
    df.to_excel(ruta, index=False)
    return ruta


def test_lee_repara_el_texto_y_avisa(tmp_path):
    df, avisos = lee_prevision(fichero(tmp_path))
    assert df.loc[1, "EXPORT_ES"] == -4011.2 and df["EXPORT_ES"].dtype.kind == "f"
    assert len(avisos) == 1 and "EXPORT_ES" in avisos[0]


def test_a_cuartos_signos_horas_y_cambio_de_hora(tmp_path):
    df, _ = lee_prevision(fichero(tmp_path))
    q = a_cuartos(df, gas=40.0, co2=75.0, bombeo_consumo=-300.0)
    assert len(q) == 4 * len(df) and q["timestamp_utc"].is_monotonic_increasing
    assert q["timestamp_utc"].is_unique
    assert (q["demanda"] == 30000.0).all()                              # signo cambiado
    assert q.loc[0, "intercambios"] == 2000.0 - 3000.0                  # saldo = IMPORT + EXPORT (negativo)
    assert (q["gas_mibgas_es"] == 40.0).all() and (q["bombeo_consumo"] == -300.0).all()
    # el cuarto de hora siguiente a la 01:00 local (00:00 UTC) es la 03:00 local: salta 2 h locales = 1 h UTC
    paso = q["timestamp_utc"].diff().dropna()
    assert paso.max() == pd.Timedelta(minutes=15)                       # UTC continuo, sin huecos
    media = a_horas(q.assign(x=q["p_engie"]), ["x"])
    assert len(media) == len(df) and np.allclose(media["x"], df["PRECIO_ES"])


def test_cobertura_cuenta_horas_fuera_de_rango():
    ts = pd.date_range("2027-01-01", periods=4, freq="h", tz="UTC")
    ref = pd.DataFrame({c: [0.0, 10.0] for c in VARIABLES_REALES})
    prev = pd.DataFrame({c: [5.0, 5.0, 5.0, 5.0] for c in VARIABLES_REALES}).assign(timestamp_utc=ts)
    prev.loc[0, "solar_fv"] = 99.0
    t = cobertura(prev, ref)
    assert t.loc[2027, "solar_fv"] == 25.0 and t.loc[2027, "all_in_range_pct"] == 75.0
    prev.loc[1, "ciclo_combinado"] = 99.0  # fuera de rango solo en el ciclo combinado
    t = cobertura(prev, ref)
    assert t.loc[2027, "all_in_range_pct"] == 50.0 and t.loc[2027, "all_in_range_excl_ccgt_pct"] == 75.0
