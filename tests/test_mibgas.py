import io
from datetime import date, datetime

import pytest
from openpyxl import Workbook

from pce68.data.mibgas import leer_indices


def excel(cabecera: list, filas: list) -> bytes:
    wb = Workbook()
    wb.active.title = "Disclaimer"
    ws = wb.create_sheet("MIBGAS Indexes")
    ws.append(cabecera)
    for f in filas:
        ws.append(f)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


CAB = ["Delivery day", "MIBGAS PVB Last Price Index Day-Ahead\n[EUR/MWh]", "MIBGAS-ES Index\n[EUR/MWh]"]


def test_lee_el_indice_es_ordenado_y_sin_dias_vacios():
    contenido = excel(CAB, [
        [datetime(2026, 11, 12), 86.4, None],  # ultimo dia: aun sin indice
        [datetime(2026, 11, 11), 85.3, 84.24],
        [datetime(2026, 11, 10), 82.0, 84.15],
    ])
    g = leer_indices(contenido)
    assert list(g["dia"]) == [date(2026, 11, 10), date(2026, 11, 11)]
    assert list(g["gas_mibgas_es"]) == [84.15, 84.24]


def test_sin_la_columna_falla_con_un_mensaje_claro():
    with pytest.raises(ValueError, match="MIBGAS-ES Index"):
        leer_indices(excel(CAB[:2], [[datetime(2026, 11, 11), 85.3]]))
