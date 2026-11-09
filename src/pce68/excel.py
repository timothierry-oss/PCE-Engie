"""Tablas con formato para los Excel de reports/ (los usan los scripts 04 y 05)."""

from __future__ import annotations

import numpy as np
import openpyxl.styles
import openpyxl.utils
import pandas as pd

AZUL = "1F3864"
F_TIT = openpyxl.styles.Font(name="Arial", size=14, bold=True, color=AZUL)
F_HDR = openpyxl.styles.Font(name="Arial", size=10, bold=True)
F_TXT = openpyxl.styles.Font(name="Arial", size=10)
F_NOTA = openpyxl.styles.Font(name="Arial", size=9, italic=True, color="666666")
FILL_HDR = openpyxl.styles.PatternFill("solid", fgColor="DCE6F1")


def celda(v):
    """Convierte un valor de pandas/numpy en algo que openpyxl escriba bien.

    Los NaN tienen que salir como celda vacia: openpyxl los escribe literalmente
    y Excel abre el fichero como danado. En columnas de tipo objeto llegan como
    float de Python, no de numpy, asi que se comprueba cualquier escalar.
    """
    if v is None or (np.isscalar(v) and pd.isna(v)) or v is pd.NaT:
        return None
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, pd.Timestamp):
        return v.tz_localize(None) if v.tzinfo else v
    return v


def tabla(ws, fila0, df: pd.DataFrame, formatos: dict[str, str] | None = None) -> int:
    """Escribe un DataFrame como tabla con cabecera. Devuelve la ultima fila."""
    for j, col in enumerate(df.columns, start=1):
        c = ws.cell(row=fila0, column=j, value=str(col))
        c.font, c.fill = F_HDR, FILL_HDR
        c.alignment = openpyxl.styles.Alignment(wrap_text=True, vertical="center")
        ws.column_dimensions[openpyxl.utils.get_column_letter(j)].width = max(12, min(26, len(str(col)) + 4))

    for i, (_, r) in enumerate(df.iterrows(), start=fila0 + 1):
        for j, col in enumerate(df.columns, start=1):
            c = ws.cell(row=i, column=j, value=celda(r[col]))
            c.font = F_TXT
            if formatos and col in formatos:
                c.number_format = formatos[col]
    return fila0 + len(df)
