"""Comprueba la lectura de las curvas: su cruce debe dar el precio que publico OMIE.

    uv run python scripts/03_valida_curvas.py --dia 2025-05-15

Para cada periodo del dia compara el precio donde se cruzan la oferta y la demanda con
el precio de data/raw/omie/marginalpdbc_AAAAMMDD.1. Se hace con y sin los escalones de
ofertas complejas. En los periodos en que España y Portugal casaron a precios distintos
se usa la curva ES (ver pce68.data.curvas); el resumen se da para todo el dia y para los
periodos sin separacion.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd

from pce68.data import omie
from pce68.data.curvas import nombre_fichero, parsear_curvas, precios

ROOT = Path(__file__).resolve().parents[1]


def cargar_curvas(dia: date) -> pd.DataFrame:
    crudo = ROOT / "data" / "raw" / "curva_pbc" / nombre_fichero(dia)
    procesado = ROOT / "data" / "processed" / "curvas" / f"curvas_{dia:%Y%m%d}.parquet"
    if crudo.exists():
        return parsear_curvas(crudo.read_bytes().decode("latin-1"))
    if procesado.exists():
        return pd.read_parquet(procesado)
    raise SystemExit(f"No hay curvas de {dia}: descargalas con scripts/02_descarga_curvas.py")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dia", type=date.fromisoformat, default=date(2025, 5, 15))
    dia = p.parse_args().dia

    ruta_omie = ROOT / "data" / "raw" / "omie" / omie.nombre_fichero(dia)
    if not ruta_omie.exists():
        raise SystemExit(f"No esta {ruta_omie}: ejecuta antes scripts/01_descarga_omie.py")
    publicado = omie.parsear(ruta_omie.read_text(encoding="latin-1"))
    publicado = publicado[["timestamp_utc", "precio_es", "precio_pt"]]

    curvas = cargar_curvas(dia)
    con = precios(curvas, complejas=True).rename(columns={"precio_eur_mwh": "cruce"})
    sin = precios(curvas).rename(columns={"precio_eur_mwh": "cruce_sin_complejas"})
    t = publicado.merge(con, on="timestamp_utc").merge(sin, on="timestamp_utc")
    t["dif"] = t["cruce"] - t["precio_es"]
    t["dif_sin_complejas"] = t["cruce_sin_complejas"] - t["precio_es"]
    t["iberico"] = t["precio_es"] == t["precio_pt"]
    t.insert(0, "hora", t["timestamp_utc"].dt.tz_convert(omie.TZ).dt.strftime("%H:%M"))

    pd.set_option("display.width", 200)
    print(t.drop(columns="timestamp_utc").round(2).to_string(index=False))

    print(f"\n{dia}: {len(t)} periodos, {t['iberico'].sum()} sin separacion Espana-Portugal")
    grupos = (("todos", t), ("sin separacion", t[t["iberico"]]), ("con separacion", t[~t["iberico"]]))
    for grupo, sub in grupos:
        for nombre in ("dif", "dif_sin_complejas"):
            d = sub[nombre].abs()
            if len(d):
                print(f"  {grupo:15s} {nombre:18s} mediana |dif| = {d.median():.3f} €/MWh, "
                      f"max = {d.max():.2f}, dentro de 0,01: {(d < 0.01).sum()}/{len(d)}")


if __name__ == "__main__":
    main()
