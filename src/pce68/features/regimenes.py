"""Regimenes del mercado diario español.

Cinco años de precios no son un unico mercado. Entrenar sobre toda la serie
mezcla situaciones que ya no existen, asi que cada periodo se etiqueta con el
regimen al que pertenece y los modelos filtran por esa etiqueta.

Las fronteras son fechas de politica o de mercado, no salen de los datos:

    pre_crisis         hasta el 30/06/2021. Gas TTF en el rango historico.
    crisis_gas         01/07/2021 - 14/06/2022. El TTF se dispara en el verano
                       de 2021 (inventarios minimos, China al GNL, Rusia sin
                       ofertas spot) y la invasion de Ucrania (24/02/2022) lo
                       agrava. El ciclo combinado marca precio casi todas las horas.
    excepcion_iberica  15/06/2022 - 31/12/2023. Tope al gas de las centrales
                       que casan en el diario (RDL 10/2022). El precio de OMIE
                       no incluye la compensacion al gas, que se paga aparte:
                       la serie esta artificialmente baja respecto al coste real.
    normalizacion      desde el 01/01/2024. Gas de vuelta a 30-40 €/MWh y la
                       solar creciendo mas que la demanda: horas a cero o
                       negativas (115 en 2023, 819 en 2024) y curva del pato.
                       Es el mercado que se proyecta.

Eventos puntuales como el cero electrico del 28/04/2025 no son un regimen:
cambian la operacion (mas ciclos combinados en restricciones tecnicas) pero
apenas el precio del diario.
"""

from __future__ import annotations

import pandas as pd

TZ = "Europe/Madrid"

# (nombre, primer dia local incluido). Cada regimen dura hasta el inicio del siguiente.
REGIMENES: list[tuple[str, str]] = [
    ("pre_crisis", "1900-01-01"),
    ("crisis_gas", "2021-07-01"),
    ("excepcion_iberica", "2022-06-15"),
    ("normalizacion", "2024-01-01"),
]


def anotar(precios: pd.DataFrame) -> pd.DataFrame:
    """Devuelve una copia de `precios` con la columna `regimen`."""
    out = precios.copy()
    local = out["timestamp_utc"].dt.tz_convert(TZ)
    dia = local.dt.tz_localize(None).dt.normalize()
    cortes = [pd.Timestamp(d) for _, d in REGIMENES[1:]]
    nombres = [n for n, _ in REGIMENES]
    idx = pd.Series(0, index=out.index)
    for corte in cortes:
        idx += (dia >= corte).astype(int)
    out["regimen"] = pd.Categorical.from_codes(idx.to_numpy(), categories=nombres, ordered=True)
    return out


def resumen(precios: pd.DataFrame) -> pd.DataFrame:
    """Una fila por regimen presente en los datos, en orden cronologico.

    Columnas: regimen, periodos, desde, hasta (dias locales), precio_medio, precio_max.
    El precio medio pondera por la duracion de cada periodo, para que las horas
    y los cuartos de hora cuenten lo mismo por unidad de tiempo.
    """
    if "regimen" not in precios:
        precios = anotar(precios)
    df = precios.assign(
        _dia=precios["timestamp_utc"].dt.tz_convert(TZ).dt.date,
        _peso=precios["resolucion_min"] / 60.0,
    )
    df["_pxh"] = df["precio_es"] * df["_peso"]
    g = df.groupby("regimen", observed=True)
    out = pd.DataFrame(
        {
            "periodos": g.size(),
            "desde": g["_dia"].min(),
            "hasta": g["_dia"].max(),
            "precio_medio": g["_pxh"].sum() / g["_peso"].sum(),
            "precio_max": g["precio_es"].max(),
        }
    )
    return out.reset_index().assign(regimen=lambda d: d["regimen"].astype(str))
