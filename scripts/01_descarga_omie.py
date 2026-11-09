"""Descarga los precios del mercado diario de OMIE y los guarda en parquet.

    uv run python scripts/01_descarga_omie.py     # o `make data`

Solo baja los dias que no esten ya en data/raw/omie, asi que relanzarlo es
barato. Despues consolida todo en data/processed/precios_omie.parquet, que es
de donde leen el analisis y los modelos.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from pce68.data.omie import consolidar, descargar

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "base.yaml").read_text())
    datos = config["datos"]
    desde = pd.Timestamp(datos["desde"]).date()
    hasta = pd.Timestamp(datos["hasta"]).date() if datos.get("hasta") else date.today() - timedelta(days=1)
    dias = [d.date() for d in pd.date_range(desde, hasta, freq="D")]

    raw = ROOT / datos["raw_dir"]
    log.info("Descargando %d dias (%s a %s) en %s ...", len(dias), desde, hasta, raw)
    fallidos = descargar(dias, raw)
    if fallidos:
        log.warning("Fallaron %d dias; relanza el script para reintentarlos.", len(fallidos))

    precios = consolidar(raw)
    salida = ROOT / datos["processed_dir"] / "precios_omie.parquet"
    salida.parent.mkdir(parents=True, exist_ok=True)
    precios.to_parquet(salida, index=False)
    log.info(
        "Escrito %s: %d periodos, %s a %s",
        salida,
        len(precios),
        precios["timestamp_utc"].min(),
        precios["timestamp_utc"].max(),
    )


if __name__ == "__main__":
    main()
