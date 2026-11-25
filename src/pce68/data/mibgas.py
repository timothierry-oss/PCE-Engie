"""Precio del gas en España: indice diario MIBGAS-ES del mercado organizado de gas (MIBGAS).

    uv run python scripts/09_descarga_combustibles.py

MIBGAS publica un Excel por año, de acceso libre (MIBGAS_Data_AAAA.xlsx, unos 3 MB). La hoja
"MIBGAS Indexes" trae una fila por dia de entrega ("Delivery day") y varios indices en €/MWh.

Supuestos (revisables):

* Se usa la columna "MIBGAS-ES Index": indice de referencia del gas con entrega en el PVB
  (red española) para ese dia. El ultimo dia del fichero aun no tiene indice (NaN).
* El dia de entrega del gas (de 6 h a 6 h) se asigna entero al dia local del mercado electrico.
"""

from __future__ import annotations

import io

import pandas as pd

URL = "https://www.mibgas.es/en/file-access/MIBGAS_Data_{anio}.xlsx?path=AGNO_{anio}/XLS"
HOJA = "MIBGAS Indexes"
COLUMNA = "MIBGAS-ES Index"


def leer_indices(contenido: bytes) -> pd.DataFrame:
    """Excel anual de MIBGAS -> columnas dia (date) y gas_mibgas_es (€/MWh), sin dias vacios."""
    df = pd.read_excel(io.BytesIO(contenido), sheet_name=HOJA)
    col = [c for c in df.columns if str(c).replace("\n", " ").startswith(COLUMNA)]
    if not col:
        raise ValueError(f"la hoja {HOJA!r} no tiene la columna {COLUMNA!r}: {list(df.columns)}")
    out = pd.DataFrame({
        "dia": pd.to_datetime(df.iloc[:, 0], errors="coerce").dt.date,
        "gas_mibgas_es": pd.to_numeric(df[col[0]], errors="coerce"),
    }).dropna()
    return out.drop_duplicates("dia").sort_values("dia", ignore_index=True)
