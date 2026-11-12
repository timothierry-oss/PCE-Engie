"""Fundamentales del mercado desde la API ESIOS de REE: demanda y generacion por tecnologia.

    uv run python scripts/07_descarga_fundamentales.py

Token personal (gratuito, se pide a consultasios@ree.es): se lee de la variable de entorno
ESIOS_TOKEN o, si no existe, de la linea ESIOS_TOKEN=... del fichero .env en la raiz del proyecto.
.env esta en .gitignore: el token no debe subirse nunca a GitHub, el repositorio es publico.

Peticion: GET https://api.esios.ree.es/indicators/{id}?start_date=...&end_date=..., con la
cabecera x-api-key. La respuesta trae indicator.name e indicator.values: una fila por instante
(datetime_utc) y por zona (geo_name).

Supuestos (revisables):

* Zona: si el indicador trae varias, se usa la peninsular (geo_name con "pen"); si no la hay, la
  de España. Es la zona de la curva ES/MI de OMIE, salvo Portugal.
* Cada instante marca el inicio de su periodo. Los valores horarios se repiten en sus cuatro
  cuartos de hora; los de 5 o 10 minutos se promedian por cuarto de hora.
* Se pide un dia de margen a cada lado y se recorta despues en UTC: asi da igual que ESIOS lea
  las fechas en hora local o en UTC.
* Peticiones de 30 dias como maximo (las largas pueden agotar el tiempo de espera).
"""

from __future__ import annotations

import os
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ENV = Path(__file__).resolve().parents[3] / ".env"
URL = "https://api.esios.ree.es"
TZ = "Europe/Madrid"
DIAS_POR_PETICION = 30
ZONAS = ("pen", "espana")  # por orden de preferencia, sin acentos


def token() -> str:
    t = os.environ.get("ESIOS_TOKEN", "").strip()
    if not t and ENV.exists():
        for linea in ENV.read_text(encoding="utf-8").splitlines():
            clave, _, valor = linea.partition("=")
            if clave.strip() == "ESIOS_TOKEN":
                t = valor.strip().strip("'\"")
    if not t:
        raise SystemExit('Falta el token: pon ESIOS_TOKEN=... en el fichero .env o export ESIOS_TOKEN="..."')
    return t


def sesion():
    import requests

    s = requests.Session()
    s.headers.update({
        "Accept": "application/json; application/vnd.esios-api-v1+json",
        "Content-Type": "application/json",
        "x-api-key": token(),
    })
    return s


def _sin_acentos(texto: str) -> str:
    return unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode().lower()


def buscar(s, texto: str) -> list[tuple[int, str]]:
    """Indicadores del catalogo cuyo nombre contiene `texto` (sin distinguir acentos ni mayusculas)."""
    r = s.get(f"{URL}/indicators", timeout=60)
    r.raise_for_status()
    clave = _sin_acentos(texto)
    catalogo = r.json()["indicators"]
    return sorted((int(i["id"]), i["name"]) for i in catalogo if clave in _sin_acentos(i["name"]))


def _zona(v: pd.DataFrame) -> pd.DataFrame:
    if "geo_name" not in v or v["geo_name"].nunique() <= 1:
        return v
    nombres = v["geo_name"].map(_sin_acentos)
    for clave in ZONAS:
        elegidas = nombres.str.contains(clave)
        if elegidas.any():
            return v[elegidas]
    raise ValueError(f"ninguna zona peninsular o española entre {sorted(v['geo_name'].unique())}")


def a_cuartos(v: pd.DataFrame, inicio: pd.Timestamp, fin: pd.Timestamp) -> pd.Series:
    """Filas (datetime_utc, value) -> media por cuarto de hora en [inicio, fin), indice UTC."""
    ts = pd.to_datetime(v["datetime_utc"], utc=True)
    serie = v["value"].astype(float).groupby(ts).mean().sort_index()
    rejilla = pd.date_range(inicio, fin, freq="15min", inclusive="left", name="timestamp_utc")
    paso = serie.index.to_series().diff().median() if len(serie) > 1 else pd.Timedelta(minutes=15)
    if paso >= pd.Timedelta(minutes=60):
        return serie.reindex(rejilla, method="ffill", limit=3)
    return serie.resample("15min").mean().reindex(rejilla)


def indicador(s, ident: int, desde: date, hasta: date) -> tuple[str, pd.Series]:
    """Nombre oficial y serie por cuarto de hora (UTC) de los dias locales desde..hasta, incluidos."""
    inicio = pd.Timestamp(desde).tz_localize(TZ).tz_convert("UTC")
    fin = pd.Timestamp(hasta + timedelta(days=1)).tz_localize(TZ).tz_convert("UTC")
    trozos, nombre = [], None
    d, ultimo = desde - timedelta(days=1), hasta + timedelta(days=1)
    while d <= ultimo:
        h = min(d + timedelta(days=DIAS_POR_PETICION - 1), ultimo)
        params = {"start_date": f"{d:%Y-%m-%d}T00:00", "end_date": f"{h:%Y-%m-%d}T23:59"}
        r = s.get(f"{URL}/indicators/{ident}", params=params, timeout=60)
        r.raise_for_status()
        ind = r.json()["indicator"]
        nombre = ind["name"]
        if ind["values"]:
            trozos.append(pd.DataFrame(ind["values"]))
        d = h + timedelta(days=1)
    if not trozos:
        raise ValueError(f"el indicador {ident} no devuelve valores entre {desde} y {hasta}")
    v = _zona(pd.concat(trozos, ignore_index=True))
    return nombre, a_cuartos(v, inicio, fin).rename(str(ident))
