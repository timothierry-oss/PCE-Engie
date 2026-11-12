"""Descarga los fundamentales del mercado (demanda, generacion por tecnologia) desde ESIOS.

    export ESIOS_TOKEN="..."      # en el terminal, nunca en un fichero del repositorio
    uv run python scripts/07_descarga_fundamentales.py
    uv run python scripts/07_descarga_fundamentales.py --buscar "eolica"   # buscar identificadores

Los indicadores (columna: identificador ESIOS) y las fechas estan en config/base.yaml, seccion
fundamentales. Para cada uno se imprime el nombre oficial que devuelve ESIOS: si no corresponde a
la columna, hay que corregir el identificador. Salida: data/processed/fundamentales.parquet, una
fila por cuarto de hora (timestamp_utc) y una columna por indicador, mas demanda_residual
(demanda - eolica - solar_fv) si estan las tres. Los supuestos estan en pce68.data.esios.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from pce68.data import esios

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    fun = cfg["fundamentales"]
    p = argparse.ArgumentParser()
    p.add_argument("--desde", type=date.fromisoformat, default=date.fromisoformat(str(fun["desde"])))
    p.add_argument("--hasta", type=date.fromisoformat, default=date.fromisoformat(str(fun["hasta"])))
    p.add_argument("--buscar", metavar="TEXTO", help="lista los indicadores cuyo nombre contiene TEXTO")
    args = p.parse_args()

    s = esios.sesion()
    if args.buscar:
        encontrados = esios.buscar(s, args.buscar)
        for ident, nombre in encontrados:
            print(f"{ident:>6}  {nombre}")
        print(f"\n{len(encontrados)} indicadores con '{args.buscar}'")
        return

    series, fallidos = {}, []
    for col, ident in fun["indicadores"].items():
        try:
            nombre, serie = esios.indicador(s, int(ident), args.desde, args.hasta)
        except Exception as e:  # noqa: BLE001 - se informa y se sigue con el resto
            print(f"{col:18s} {ident:>6}: ERROR {e}")
            fallidos.append(col)
            continue
        series[col] = serie
        print(f"{col:18s} {ident:>6}: {nombre} | {serie.notna().sum()}/{len(serie)} cuartos de hora con dato")
    if not series:
        raise SystemExit("Nada descargado")

    df = pd.DataFrame(series)
    if {"demanda", "eolica", "solar_fv"} <= set(df):
        df["demanda_residual"] = df["demanda"] - df["eolica"] - df["solar_fv"]
    salida = ROOT / cfg["datos"]["processed_dir"] / "fundamentales.parquet"
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.reset_index().to_parquet(salida, index=False)
    print(f"\n{len(df)} cuartos de hora, {len(series)} indicadores -> {salida.relative_to(ROOT)}")
    if fallidos:
        print(f"Fallidos: {', '.join(fallidos)}")


if __name__ == "__main__":
    main()
