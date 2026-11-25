"""Precios diarios del gas (MIBGAS) y del CO2 (ESIOS) para las variables del modelo.

    uv run python scripts/09_descarga_combustibles.py      # fechas en config/base.yaml, combustibles

Gas: indice MIBGAS-ES (pce68.data.mibgas), un Excel por año guardado en data/raw/mibgas/ (el del
año en curso se vuelve a bajar siempre). CO2: indicador ESIOS de la config (1391, "Precio de
derechos de emision de despacho de los TNP"), un valor al mes, que se repite en todos los dias
del mes. Salida: data/processed/combustibles.parquet, una fila por dia local (dia, gas_mibgas_es
en €/MWh, co2_eur_t en €/t).
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import requests
import yaml

from pce68.data import esios, mibgas

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "mibgas"


def gas(desde: date, hasta: date) -> pd.DataFrame:
    RAW.mkdir(parents=True, exist_ok=True)
    partes = []
    with requests.Session() as s:
        for anio in range(desde.year, hasta.year + 1):
            ruta = RAW / f"MIBGAS_Data_{anio}.xlsx"
            if not ruta.exists() or anio >= date.today().year:
                r = s.get(mibgas.URL.format(anio=anio), timeout=120)
                r.raise_for_status()
                ruta.write_bytes(r.content)
            g = mibgas.leer_indices(ruta.read_bytes())
            print(f"MIBGAS {anio}: {len(g)} dias, {g['dia'].min()} a {g['dia'].max()}")
            partes.append(g)
    g = pd.concat(partes, ignore_index=True)
    return g[(g["dia"] >= desde) & (g["dia"] <= hasta)]


def co2(ident: int, desde: date, hasta: date) -> pd.DataFrame:
    nombre, serie = esios.indicador(esios.sesion(), ident, desde, hasta)
    v = serie.dropna()
    print(f"ESIOS {ident}: {nombre} | {len(v)} valores")
    mensual = v.groupby(v.index.tz_convert(esios.TZ).strftime("%Y-%m")).first()
    dias = pd.date_range(desde, hasta, freq="D")
    out = pd.DataFrame({"dia": dias.date, "co2_eur_t": dias.strftime("%Y-%m").map(mensual)})
    # el mes en curso puede no estar publicado: se repite el ultimo valor conocido
    out["co2_eur_t"] = out["co2_eur_t"].ffill()
    return out


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    com = cfg["combustibles"]
    p = argparse.ArgumentParser()
    p.add_argument("--desde", type=date.fromisoformat, default=date.fromisoformat(str(com["desde"])))
    p.add_argument("--hasta", type=date.fromisoformat,
                   default=date.fromisoformat(str(com["hasta"])) if com.get("hasta") else date.today())
    args = p.parse_args()

    df = co2(int(com["co2_esios"]), args.desde, args.hasta).merge(gas(args.desde, args.hasta), on="dia",
                                                                   how="left")
    salida = ROOT / cfg["datos"]["processed_dir"] / "combustibles.parquet"
    df[["dia", "gas_mibgas_es", "co2_eur_t"]].to_parquet(salida, index=False)
    print(f"\n{len(df)} dias -> {salida.relative_to(ROOT)}; sin gas: {df['gas_mibgas_es'].isna().sum()}, "
          f"sin CO2: {df['co2_eur_t'].isna().sum()}")
    print(df.assign(mes=pd.to_datetime(df["dia"]).dt.strftime("%Y-%m")).groupby("mes")
          [["gas_mibgas_es", "co2_eur_t"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()
