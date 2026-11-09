"""Descarga las curvas agregadas de oferta y demanda de OMIE y las reduce a parquet.

    uv run python scripts/02_descarga_curvas.py --cada 10
    uv run python scripts/02_descarga_curvas.py --desde 2024-01-01 --hasta 2025-12-31 --cada 1

Cada dia se baja (unos 4 MB en horario, unos 20 MB en cuarto-horario), se reduce a los
escalones de la curva y se guarda en data/processed/curvas/curvas_AAAAMMDD.parquet. El
fichero crudo se descarta, salvo con --guardar-raw. Si ya esta en data/raw/curva_pbc se
usa sin volver a bajarlo. Los dias ya procesados se saltan, asi que se puede relanzar.
"""

from __future__ import annotations

import argparse
import time
from datetime import date, timedelta
from pathlib import Path

import requests

from pce68.data.curvas import URL, nombre_fichero, parsear_curvas

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "curva_pbc"
SALIDA = ROOT / "data" / "processed" / "curvas"
MIN_BYTES = 100_000  # una pagina de error pesa mucho menos que una curva real


def leer(dia: date, sesion: requests.Session, guardar_raw: bool) -> str:
    ruta = RAW / nombre_fichero(dia)
    if ruta.exists() and ruta.stat().st_size > MIN_BYTES:
        return ruta.read_bytes().decode("latin-1")
    r = sesion.get(URL.format(dia), timeout=120)
    r.raise_for_status()
    if len(r.content) < MIN_BYTES:
        raise ValueError(f"fichero sospechosamente pequeno ({len(r.content)} bytes)")
    if guardar_raw:
        RAW.mkdir(parents=True, exist_ok=True)
        ruta.write_bytes(r.content)
    return r.content.decode("latin-1")


def main() -> None:
    hoy = date.today()
    p = argparse.ArgumentParser()
    p.add_argument("--desde", type=date.fromisoformat, default=hoy.replace(year=hoy.year - 6))
    p.add_argument("--hasta", type=date.fromisoformat, default=hoy)
    p.add_argument("--cada", type=int, default=10, help="1 = todos los dias, 10 = un dia de cada diez")
    p.add_argument("--guardar-raw", action="store_true", help="conserva los ficheros crudos")
    args = p.parse_args()

    SALIDA.mkdir(parents=True, exist_ok=True)
    hechos, fallidos = 0, []
    dia = args.desde
    with requests.Session() as s:
        while dia <= args.hasta:
            destino = SALIDA / f"curvas_{dia:%Y%m%d}.parquet"
            if not destino.exists():
                try:
                    curva = parsear_curvas(leer(dia, s, args.guardar_raw))
                    curva.to_parquet(destino, index=False)
                    hechos += 1
                    print(f"{dia}: {len(curva)} escalones, {destino.stat().st_size / 1e3:.0f} KB")
                    time.sleep(0.5)
                except Exception as e:  # noqa: BLE001 - se informa y se sigue con el resto
                    print(f"{dia}: ERROR {e}")
                    fallidos.append(dia)
            dia += timedelta(days=args.cada)
    print(f"\nTerminado: {hechos} dias nuevos en {SALIDA}, {len(fallidos)} fallidos.")


if __name__ == "__main__":
    main()
