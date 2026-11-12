from datetime import date

import pandas as pd
import pytest

from pce68.data import esios

DIA = date(2026, 9, 1)  # medianoche local = 2026-08-31 22:00 UTC (CEST)


class Respuesta:
    def __init__(self, datos):
        self.datos = datos

    def raise_for_status(self):
        pass

    def json(self):
        return self.datos


class Sesion:
    """Sustituye a requests.Session: devuelve siempre los mismos valores, sin red."""

    def __init__(self, datos):
        self.datos = datos

    def get(self, url, params=None, timeout=None):
        return Respuesta(self.datos)


def _valores(inicio_utc: str, n: int, paso: pd.Timedelta, geo: str, valor=lambda i: float(i)):
    t0 = pd.Timestamp(inicio_utc)
    fmt = "%Y-%m-%dT%H:%M:%SZ"
    return [{"value": valor(i), "datetime_utc": (t0 + i * paso).strftime(fmt), "geo_name": geo}
            for i in range(n)]


def test_horario_se_repite_en_cuartos_y_se_queda_la_peninsula():
    h = pd.Timedelta(hours=1)
    vals = _valores("2026-08-31T00:00:00Z", 72, h, "Peninsular") + \
        _valores("2026-08-31T00:00:00Z", 72, h, "Baleares", valor=lambda i: -1.0)
    s = Sesion({"indicator": {"name": "Demanda real", "values": vals}})
    nombre, serie = esios.indicador(s, 1293, DIA, DIA)
    assert nombre == "Demanda real"
    assert len(serie) == 96 and serie.notna().all()
    assert serie.index[0] == pd.Timestamp("2026-08-31T22:00:00Z")
    assert list(serie.iloc[:5]) == [22.0, 22.0, 22.0, 22.0, 23.0]  # hora 22 UTC en sus 4 cuartos
    assert (serie >= 0).all()  # nada de Baleares


def test_cinco_minutos_se_promedian_por_cuarto():
    vals = _valores("2026-08-31T22:00:00Z", 288, pd.Timedelta(minutes=5), "Peninsular")
    _, serie = esios.indicador(Sesion({"indicator": {"name": "Eolica", "values": vals}}), 551, DIA, DIA)
    assert len(serie) == 96
    assert serie.iloc[0] == pytest.approx(1.0) and serie.iloc[1] == pytest.approx(4.0)


def test_zona_espana_si_no_hay_peninsular_y_error_si_no_hay_ninguna():
    v = pd.DataFrame({"geo_name": ["España", "Portugal", "Francia"], "value": [1.0, 2.0, 3.0]})
    assert list(esios._zona(v)["value"]) == [1.0]
    with pytest.raises(ValueError):
        esios._zona(pd.DataFrame({"geo_name": ["Portugal", "Francia"], "value": [2.0, 3.0]}))


def test_buscar_ignora_acentos_y_mayusculas():
    cat = {"indicators": [{"id": 1293, "name": "Demanda real"},
                          {"id": 551, "name": "Generación T.Real eólica"}]}
    assert esios.buscar(Sesion(cat), "EOLICA") == [(551, "Generación T.Real eólica")]


def test_sin_token_se_para_con_un_mensaje(monkeypatch, tmp_path):
    monkeypatch.delenv("ESIOS_TOKEN", raising=False)
    monkeypatch.setattr(esios, "ENV", tmp_path / ".env")
    with pytest.raises(SystemExit):
        esios.token()


def test_token_de_env_y_la_variable_de_entorno_manda(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text('OTRA=1\nESIOS_TOKEN="abc123"\n', encoding="utf-8")
    monkeypatch.setattr(esios, "ENV", env)
    monkeypatch.delenv("ESIOS_TOKEN", raising=False)
    assert esios.token() == "abc123"
    monkeypatch.setenv("ESIOS_TOKEN", "desde_el_entorno")
    assert esios.token() == "desde_el_entorno"
