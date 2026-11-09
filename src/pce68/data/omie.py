"""Precios marginales del mercado diario de OMIE (ficheros marginalpdbc).

Cada dia OMIE publica marginalpdbc_AAAAMMDD.1:

    MARGINALPDBC;
    2024;01;01;1;63.33;63.33;
    ...
    *

Columnas: año; mes; dia; periodo; precio Portugal; precio España (€/MWh).
El periodo cuenta desde la medianoche local, empezando en 1. Hasta el
30/09/2025 hay 24 periodos horarios (23 o 25 en los cambios de hora); desde el
01/10/2025 el diario casa en cuartos de hora: 96 periodos (92 o 100).

La hora UTC se calcula sumando duraciones a la medianoche local, que nunca es
ambigua, asi los cambios de hora salen bien sin tratarlos aparte.
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from datetime import date
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

TZ = "Europe/Madrid"
URL_BASE = "https://www.omie.es/es/file-download?parents=marginalpdbc&filename="
URL = URL_BASE + "marginalpdbc_{:%Y%m%d}.1"
# OMIE solo sirve dia a dia los ficheros recientes; los años cerrados los agrupa
# en un zip anual con los mismos marginalpdbc_AAAAMMDD.1 dentro.
URL_ANUAL = URL_BASE + "marginalpdbc_{}.zip"


def nombre_fichero(dia: date) -> str:
    return f"marginalpdbc_{dia:%Y%m%d}.1"


def parsear(texto: str) -> pd.DataFrame:
    """Convierte un marginalpdbc en filas timestamp_utc, precio_es, precio_pt, resolucion_min."""
    lineas = [ln for ln in texto.splitlines() if ln[:1].isdigit()]
    if not lineas:
        raise ValueError("fichero marginalpdbc sin datos")
    df = pd.read_csv(
        io.StringIO("\n".join(lineas)),
        sep=";",
        header=None,
        usecols=range(6),
        names=["anio", "mes", "dia", "periodo", "precio_pt", "precio_es"],
    )
    if df[["anio", "mes", "dia"]].nunique().max() > 1:
        raise ValueError("un marginalpdbc debe traer un solo dia")

    a, m, d = df.loc[0, ["anio", "mes", "dia"]]
    dia = pd.Timestamp(year=int(a), month=int(m), day=int(d))
    inicio = dia.tz_localize(TZ)
    fin = (dia + pd.Timedelta(days=1)).tz_localize(TZ)  # medianoche local siguiente: dias de 23, 24 o 25 h
    horas_dia = (fin.tz_convert("UTC") - inicio.tz_convert("UTC")) / pd.Timedelta(hours=1)
    n = len(df)
    resol = 60.0 * horas_dia / n
    if resol not in (15.0, 60.0):
        raise ValueError(f"{a}-{m:02d}-{d:02d}: {n} periodos no encajan con un dia de {horas_dia:g} h")

    t0 = inicio.tz_convert("UTC")
    return pd.DataFrame(
        {
            "timestamp_utc": t0 + pd.to_timedelta((df["periodo"] - 1) * resol, unit="min"),
            "precio_es": df["precio_es"].astype(float),
            "precio_pt": df["precio_pt"].astype(float),
            "resolucion_min": int(resol),
        }
    )


def _descargar_anual(s, anio: int, destino: Path) -> int:
    """Baja el zip anual y extrae sus marginalpdbc a `destino`. Devuelve cuantos extrajo."""
    r = s.get(URL_ANUAL.format(anio), timeout=120)
    r.raise_for_status()
    n = 0
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for nombre in z.namelist():
            base = Path(nombre).name
            if base.startswith("marginalpdbc_") and base.endswith(".1"):
                (destino / base).write_bytes(z.read(nombre))
                n += 1
    return n


def descargar(dias: list[date], destino: Path, pausa_s: float = 0.2) -> list[date]:
    """Descarga los ficheros que falten en `destino`. Devuelve los dias que fallaron.

    Primero prueba el fichero diario; si OMIE ya no lo sirve suelto (años
    antiguos), prueba una vez el zip anual de ese año.
    """
    import requests

    destino.mkdir(parents=True, exist_ok=True)
    fallidos = []
    anios_probados: set[int] = set()
    with requests.Session() as s:
        for dia in dias:
            ruta = destino / nombre_fichero(dia)
            if ruta.exists() and ruta.stat().st_size > 0:
                continue
            try:
                r = s.get(URL.format(dia), timeout=30)
                r.raise_for_status()
                parsear(r.text)  # no guardar paginas de error con nombre de fichero valido
                ruta.write_text(r.text, encoding="latin-1")
                time.sleep(pausa_s)
                continue
            except Exception as e:  # noqa: BLE001 - se informa y se sigue con el resto
                error = e
            if dia.year not in anios_probados:
                anios_probados.add(dia.year)
                try:
                    n = _descargar_anual(s, dia.year, destino)
                    log.info("%d: zip anual con %d dias", dia.year, n)
                except Exception as e:  # noqa: BLE001
                    log.warning("%d: tampoco hay zip anual (%s)", dia.year, e)
                if ruta.exists() and ruta.stat().st_size > 0:
                    continue
            log.warning("%s: %s", dia, error)
            fallidos.append(dia)
    return fallidos


def consolidar(origen: Path) -> pd.DataFrame:
    """Une todos los marginalpdbc de `origen` en una serie ordenada y sin duplicados."""
    frames = [parsear(f.read_text(encoding="latin-1")) for f in sorted(origen.glob("marginalpdbc_*.1"))]
    if not frames:
        raise FileNotFoundError(f"no hay ficheros marginalpdbc en {origen}")
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates("timestamp_utc", keep="last").sort_values("timestamp_utc", ignore_index=True)