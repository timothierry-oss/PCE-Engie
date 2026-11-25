"""Prevision de ENGIE (forecast_base_case.xlsx) como entrada de `impacto.ajusta_prevision` (etapa 3).

DATOS CONFIDENCIALES de ENGIE: nada de lo que salga de aqui va a reports/ ni al repositorio; los scripts
escriben en data/processed/ (ignorado por git).

Formato del fichero (verificado el 22/11/2026): una fila por hora, 01/01/2027-31/12/2040, FechaLocal sin
zona horaria (hora local de Madrid: 14 saltos de 2 h y 14 repetidas = cambios de hora), PRECIO_ES en
€/MWh y el resto en MW medios de la hora.

Correspondencia con las variables del modelo (SUPUESTOS, a confirmar con ENGIE):

    demanda            = -DEMANDA_ES           la prevision da la demanda con signo negativo
    eolica, solar_fv, solar_termica, nuclear, ciclo_combinado = EOLICA_ES, SOLAR_PV_ES, SOLAR_TERM_ES,
                         NUCLEAR_ES, CCGT_ES
    hidraulica         = HIDRO_ES              (sin FLUY_ES: asi la mediana se parece a la de ESIOS)
    cogeneracion       = COGEN_ES              (sin OTROS_ES, por la misma razon)
    intercambios       = IMPORT_ES + EXPORT_ES EXPORT_ES viene negativo: da el saldo, como ESIOS 553
    bombeo_turbinacion = P_BOMBEO_ES
    demanda_residual   = demanda - eolica - solar_fv

Entradas que la prevision NO trae (bombeo_consumo, gas, CO2): se fijan en un valor constante, por defecto
la mediana del periodo de entrenamiento. Es un supuesto fuerte: el modelo no puede ver cambios del gas o
del CO2 entre 2027 y 2040.

La prevision es horaria y el modelo se entreno en cuartos de hora: cada hora se repite en 4 cuartos
(precio constante dentro de la hora, como el mercado diario de OMIE antes del 01/10/2025) y el resultado
se vuelve a promediar por hora.

Limite principal: la prevision sale del dominio de entrenamiento (solar y eolica mucho mayores, nuclear
que desaparece). Los arboles extrapolan en plano, la ridge en recta: `cobertura` mide cuantas horas caen
dentro del rango visto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Madrid"
SALTO = pd.Timedelta(minutes=15)
COLUMNAS_FICHERO = ["FechaLocal", "PRECIO_ES", "NUCLEAR_ES", "CCGT_ES", "EOLICA_ES", "SOLAR_PV_ES",
                    "SOLAR_TERM_ES", "HIDRO_ES", "FLUY_ES", "IMPORT_ES", "COGEN_ES", "OTROS_ES",
                    "P_BOMBEO_ES", "DEMANDA_ES", "EXPORT_ES"]
# variables con informacion real en la prevision (las otras se fijan): las que se miran en `cobertura`
VARIABLES_REALES = ["p_engie", "demanda", "eolica", "solar_fv", "solar_termica", "hidraulica", "nuclear",
                    "ciclo_combinado", "cogeneracion", "intercambios", "bombeo_turbinacion"]


def lee_prevision(ruta) -> tuple[pd.DataFrame, list[str]]:
    """Lee el Excel y fuerza las columnas a numero. Devuelve (tabla, avisos de valores reparados).

    Un valor con texto (p. ej. '-4011,2vsc' en EXPORT_ES) se repara quitando el texto y cambiando la coma
    decimal: es una suposicion y se avisa. Si no se puede reparar queda en NaN.
    """
    df = pd.read_excel(ruta)
    falta = set(COLUMNAS_FICHERO) - set(df.columns)
    if falta:
        raise ValueError(f"faltan columnas en {ruta}: {sorted(falta)}")
    avisos = []
    for c in COLUMNAS_FICHERO[1:]:
        n = pd.to_numeric(df[c], errors="coerce")
        malos = n.isna() & df[c].notna()
        if malos.any():
            texto = (df.loc[malos, c].astype(str).str.replace(",", ".", regex=False)
                     .str.extract(r"(-?\d+(?:\.\d+)?)")[0])
            n.loc[malos] = pd.to_numeric(texto, errors="coerce")
            for i in df.index[malos]:
                avisos.append(f"fila {i + 2}, {c}: {df.at[i, c]!r} -> {n.at[i]}")
        df[c] = n
    return df[COLUMNAS_FICHERO], avisos


def a_horario(df: pd.DataFrame, gas: float, co2: float, bombeo_consumo: float) -> pd.DataFrame:
    """Prevision horaria -> una fila por hora con las columnas que espera `impacto.variables`.

    Devuelve timestamp_utc, p_engie, las `fundamentales` del modelo, gas_mibgas_es y co2_eur_t.
    """
    local = df["FechaLocal"].dt.tz_localize(TZ, ambiguous="infer", nonexistent="shift_forward")
    h = pd.DataFrame({"timestamp_utc": local.dt.tz_convert("UTC"), "p_engie": df["PRECIO_ES"]})
    h["demanda"] = -df["DEMANDA_ES"]
    h["eolica"], h["solar_fv"], h["solar_termica"] = df["EOLICA_ES"], df["SOLAR_PV_ES"], df["SOLAR_TERM_ES"]
    h["hidraulica"], h["nuclear"], h["ciclo_combinado"] = df["HIDRO_ES"], df["NUCLEAR_ES"], df["CCGT_ES"]
    h["cogeneracion"] = df["COGEN_ES"]
    h["intercambios"] = df["IMPORT_ES"] + df["EXPORT_ES"]
    h["bombeo_turbinacion"] = df["P_BOMBEO_ES"]
    h["bombeo_consumo"] = bombeo_consumo
    h["demanda_residual"] = h["demanda"] - h["eolica"] - h["solar_fv"]
    h["gas_mibgas_es"], h["co2_eur_t"] = gas, co2
    return h


def a_cuartos(df: pd.DataFrame, gas: float, co2: float, bombeo_consumo: float) -> pd.DataFrame:
    """Como `a_horario`, con cada hora repetida en 4 cuartos de hora (el modelo se entreno en cuartos)."""
    h = a_horario(df, gas, co2, bombeo_consumo)
    q = h.loc[h.index.repeat(4)].reset_index(drop=True)
    q["timestamp_utc"] = q["timestamp_utc"] + SALTO * np.tile(np.arange(4), len(h))
    return q


def a_horas(resultado: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Promedia los 4 cuartos de cada hora (UTC) de `resultado` en las `columnas`."""
    hora = resultado["timestamp_utc"].dt.floor("h")
    return resultado.groupby(hora)[columnas].mean().rename_axis("timestamp_utc").reset_index()


def cobertura(prev: pd.DataFrame, referencia: pd.DataFrame) -> pd.DataFrame:
    """Por año: parte (%) de horas fuera del rango [min, max] visto en `referencia`, por variable.

    `prev` y `referencia` con las columnas de VARIABLES_REALES; `prev` con timestamp_utc.
    `all_in_range_pct` es la parte de horas con TODAS las variables dentro del rango.
    """
    lo, hi = referencia[VARIABLES_REALES].min(), referencia[VARIABLES_REALES].max()
    fuera = (prev[VARIABLES_REALES] < lo) | (prev[VARIABLES_REALES] > hi)
    anio = prev["timestamp_utc"].dt.tz_convert(TZ).dt.year.rename("year")
    tabla = 100 * fuera.groupby(anio).mean()
    tabla["all_in_range_pct"] = 100 * (~fuera.any(axis=1)).groupby(anio).mean()
    # el ciclo combinado de la prevision esta casi siempre a 0 MW y el entrenamiento nunca lo vio por debajo
    # de ~1,8 GW: se da tambien el indicador sin el, para no confundir ese cambio de regimen con el resto
    resto = fuera.drop(columns="ciclo_combinado")
    tabla["all_in_range_excl_ccgt_pct"] = 100 * (~resto.any(axis=1)).groupby(anio).mean()
    return tabla.round(1)


def referencia_entrenamiento(proc: Path, desde: pd.Timestamp) -> pd.DataFrame:
    """Valores vistos en el entrenamiento, con los nombres de VARIABLES_REALES (rangos de `cobertura`)."""
    f = pd.read_parquet(proc / "fundamentales.parquet")
    f = f[f["timestamp_utc"] >= desde]
    p = pd.read_parquet(proc / "precios_omie.parquet")
    p = p[p["timestamp_utc"] >= desde]
    cols = [c for c in VARIABLES_REALES if c != "p_engie"]
    precio = p["precio_es"].rename("p_engie").reset_index(drop=True)
    return pd.concat([f[cols].reset_index(drop=True), precio], axis=1)


def supuestos_fijos(proc: Path, desde: pd.Timestamp, sup: dict) -> tuple[float, float, float]:
    """(gas €/MWh, CO2 €/t, consumo de bombeo MW) de las entradas que la prevision no trae.

    Valor de `sup` (config prevision.supuestos) o, si es None, la mediana del periodo de entrenamiento.
    """
    comb = pd.read_parquet(proc / "combustibles.parquet")
    comb = comb[pd.to_datetime(comb["dia"]) >= desde.tz_localize(None)]
    fund = pd.read_parquet(proc / "fundamentales.parquet")
    gas = sup["gas_eur_mwh"] if sup["gas_eur_mwh"] is not None else float(comb["gas_mibgas_es"].median())
    co2 = sup["co2_eur_t"] if sup["co2_eur_t"] is not None else float(comb["co2_eur_t"].median())
    bombeo = (sup["bombeo_consumo_mw"] if sup["bombeo_consumo_mw"] is not None
              else float(fund.loc[fund["timestamp_utc"] >= desde, "bombeo_consumo"].median()))
    return float(gas), float(co2), float(bombeo)
