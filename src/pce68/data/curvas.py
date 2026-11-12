"""Curvas agregadas de oferta y demanda del mercado diario de OMIE (curva_pbc).

Cada dia OMIE publica curva_pbc_AAAAMMDD.1, con una linea por escalon de la curva:

    OMIE - Mercado de electricidad;Fecha Emision :14/05/2025 - 13:30;;15/05/2025;...

    Periodo;Fecha;Pais;Unidad;Tipo Oferta;Potencia;Precio;Ofertada (O)/Casada (C);Tipologia;
    1;15/05/2025;MI;;V;9,5;-500,00;O;S;
    1;15/05/2025;MI;;C;6,8;1.500,00;O;S;

Como se leen las columnas (la interpretacion sale del formato; scripts/03_valida_curvas.py
la comprueba contra el precio que publico OMIE):

* Tipo Oferta: V = venta (curva de oferta), C = compra (curva de demanda).
* Potencia en MW y precio en €/MWh, con coma decimal y punto de millar ("1.991,1").
* O = ofertada, C = casada. Las filas casadas repiten parte de las ofertadas. Se guardan
  las O, los flujos de interconexion y, de las ofertas complejas, tambien las casadas
  (columna `casada`), porque `precios` las necesita (ver supuestos).
* Tipologia: S = oferta simple; C01, C02, ... = ofertas con condiciones complejas (el
  algoritmo puede aceptarlas o rechazarlas enteras, no siguen la curva); "Imp XX" /
  "Exp XX" = flujo de interconexion ya casado (no es una oferta mas). Sus MW son reales:
  con ellos el cruce cae sobre el precio publicado.
* Periodo: un numero (1, 2, ...) en los ficheros horarios; en los cuarto-horarios (desde el
  01/10/2025) una etiqueta HxQy, hora x y cuarto y (H1Q1 ... H24Q4), que es el periodo
  (x - 1) * 4 + y (comprobado en curva_pbc_20260913.1: 96 etiquetas, H1Q1 a H24Q4).
* Pais: MI = mercado iberico. Si España y Portugal casan a precios distintos, ese
  periodo no trae MI sino una curva ES y otra PT, cada una con su flujo de interconexion
  (p. ej. "Exp ES" en la curva ES). Se usa ES si existe y si no MI; PT se descarta.

La hora UTC se calcula igual que en pce68.data.omie: medianoche local mas la duracion
de los periodos, asi los cambios de hora y el paso a cuartos de hora salen bien.

Supuestos de `precios` (revisables):

* Los flujos de interconexion casados (Imp/Exp) se quedan fijos: no reaccionan al nuevo
  precio. Con una bateria grande esto es una simplificacion.
* La demanda extra (carga de bateria) entra como compra al precio maximo y la oferta
  extra (descarga) como venta al precio minimo: se casan siempre.
* Ofertas complejas (complejas="casadas", por defecto): entran solo por el volumen que el
  algoritmo de OMIE caso, a su precio; las simples, por todo lo ofertado. Es la construccion
  que reproduce el precio publicado (|dif| maximo; periodos a menos de 0,01 €/MWh):
      15/05/2025 (horario)       0,98 €/MWh; 17/24   (igual que sin complejas)
      02/09/2026 (cuarto-hora)   1,23 €/MWh; 82/96   (sin complejas: 103 €/MWh; ofertadas: 51)
      13/09/2026 (cuarto-hora)   1,38 €/MWh; 84/96   (sin complejas: 15 €/MWh; ofertadas: 45)
  Con baterias, el volumen casado de las complejas se queda fijo, como el resto de ofertas.
  "ofertadas" usa todo lo ofertado (simples y complejas) y "no" ignora las complejas.
* La curva es la del precio español: con separacion de mercados el flujo ES-PT queda
  fijo, como los demas flujos.
"""

from __future__ import annotations

import io
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

TZ = "Europe/Madrid"
URL = "https://www.omie.es/es/file-download?parents=curva_pbc&filename=curva_pbc_{:%Y%m%d}.1"
PRECIO_MIN = -500.0
PRECIO_MAX = 4000.0
_INFINITO = 1e6  # para flujos casados que no dependen del precio


def nombre_fichero(dia: date) -> str:
    return f"curva_pbc_{dia:%Y%m%d}.1"


def _es_dato(linea: str) -> bool:
    return linea[:1].isdigit() or (linea[:1] == "H" and linea[1:2].isdigit())


def _numero_periodo(p: pd.Series) -> pd.Series:
    """'7' (horario) o 'H2Q3' (cuarto-horario: hora 2, cuarto 3, es decir periodo 7) -> numero."""
    hq = p.str.extract(r"^H(\d+)Q([1-4])$").astype(float)
    num = (hq[0] - 1) * 4 + hq[1]
    num = num.fillna(pd.to_numeric(p, errors="coerce"))
    if num.isna().any():
        raise ValueError(f"periodos no reconocidos: {sorted(p[num.isna()].unique())[:5]}")
    return num.astype(int)


def parsear_curvas(texto: str) -> pd.DataFrame:
    """Convierte un curva_pbc en los escalones de la curva de cada periodo.

    Devuelve timestamp_utc, resolucion_min, lado ("oferta" o "demanda"), precio (€/MWh),
    potencia_mw, tipologia y casada, con un escalon por (periodo, lado, precio, tipologia,
    casada). Se quedan las filas ofertadas, los flujos de interconexion casados y las filas
    casadas de las ofertas complejas, de la curva ES del periodo si la hay y si no de la MI.
    """
    lineas = [ln for ln in texto.splitlines() if _es_dato(ln)]
    if not lineas:
        raise ValueError("fichero curva_pbc sin datos")
    df = pd.read_csv(
        io.StringIO("\n".join(lineas)),
        sep=";",
        header=None,
        usecols=range(9),
        names=["periodo", "fecha", "pais", "unidad", "tipo", "potencia_mw", "precio", "oc", "tipologia"],
        decimal=",",
        thousands=".",
        dtype={"periodo": str, "fecha": str, "pais": str, "tipo": str, "oc": str, "tipologia": str},
    )
    df["periodo"] = _numero_periodo(df["periodo"].str.strip())
    fechas = df["fecha"].unique()
    if len(fechas) != 1:
        raise ValueError("un curva_pbc debe traer un solo dia")

    dia = pd.to_datetime(fechas[0], format="%d/%m/%Y")
    inicio = dia.tz_localize(TZ)
    fin = (dia + pd.Timedelta(days=1)).tz_localize(TZ)  # dias de 23, 24 o 25 h
    horas_dia = (fin.tz_convert("UTC") - inicio.tz_convert("UTC")) / pd.Timedelta(hours=1)
    n = int(df["periodo"].max())
    resol = 60.0 * horas_dia / n
    if resol not in (15.0, 60.0):
        raise ValueError(f"{dia:%Y-%m-%d}: {n} periodos no encajan con un dia de {horas_dia:g} h")

    # ES donde España casa sola; MI donde el mercado es uno; PT nunca
    con_es = df.loc[df["pais"] == "ES", "periodo"].unique()
    df = df[np.where(df["periodo"].isin(con_es), df["pais"] == "ES", df["pais"] == "MI")]

    tipologia = df["tipologia"].fillna("")
    flujo = tipologia.str.startswith(("Imp", "Exp"))
    compleja_casada = (df["oc"] == "C") & tipologia.str.fullmatch(r"C\d+")
    df = df.loc[(df["oc"] == "O") | flujo | compleja_casada].assign(
        tipologia=tipologia,
        casada=lambda d: d["oc"] == "C",
        lado=lambda d: np.where(d["tipo"] == "C", "demanda", "oferta"),
    )
    claves = ["periodo", "lado", "precio", "tipologia", "casada"]
    curva = df.groupby(claves, as_index=False)["potencia_mw"].sum()

    t0 = inicio.tz_convert("UTC")
    curva.insert(0, "timestamp_utc", t0 + pd.to_timedelta((curva["periodo"] - 1) * resol, unit="min"))
    curva["resolucion_min"] = int(resol)
    return curva.drop(columns="periodo").sort_values(["timestamp_utc", "lado", "precio"], ignore_index=True)


def cargar(carpeta: Path, dias: list[date] | None = None) -> pd.DataFrame:
    """Une los curvas_AAAAMMDD.parquet de `carpeta` (o solo los de `dias`)."""
    ficheros = sorted(carpeta.glob("curvas_*.parquet"))
    if dias is not None:
        pedidos = {f"curvas_{d:%Y%m%d}.parquet" for d in dias}
        ficheros = [f for f in ficheros if f.name in pedidos]
    if not ficheros:
        raise FileNotFoundError(f"no hay curvas en {carpeta}")
    return pd.concat([pd.read_parquet(f) for f in ficheros], ignore_index=True)


def precio_equilibrio(
    p_oferta: np.ndarray, q_oferta: np.ndarray, p_demanda: np.ndarray, q_demanda: np.ndarray
) -> float:
    """Precio al que se cruzan una curva de oferta y una de demanda en escalones.

    Es el menor precio p tal que la oferta a precio <= p cubre toda la demanda que paga
    estrictamente mas que p. Si el cruce cae en un escalon de demanda parcialmente casado
    (o en uno de oferta), ese escalon marca el precio. Si la oferta no alcanza, el precio
    lo marca la puja de demanda mas alta, y el resultado se acota a [PRECIO_MIN, PRECIO_MAX].
    """
    p_oferta, q_oferta = np.asarray(p_oferta, float), np.asarray(q_oferta, float)
    p_demanda, q_demanda = np.asarray(p_demanda, float), np.asarray(q_demanda, float)
    candidatos = np.unique(np.concatenate([p_oferta, p_demanda]))

    orden = np.argsort(p_oferta, kind="stable")
    acum_oferta = np.concatenate([[0.0], np.cumsum(q_oferta[orden])])
    oferta = acum_oferta[np.searchsorted(p_oferta[orden], candidatos, side="right")]

    orden = np.argsort(p_demanda, kind="stable")
    acum_demanda = np.concatenate([[0.0], np.cumsum(q_demanda[orden])])
    # demanda que paga estrictamente mas que cada candidato
    demanda = acum_demanda[-1] - acum_demanda[np.searchsorted(p_demanda[orden], candidatos, side="right")]

    cubre = np.flatnonzero(oferta >= demanda - 1e-6)  # siempre hay alguno: al mayor precio no queda demanda
    return float(np.clip(candidatos[cubre[0]], PRECIO_MIN, PRECIO_MAX))


def precios(
    curvas: pd.DataFrame, delta_mw: pd.Series | None = None, complejas: str = "casadas"
) -> pd.DataFrame:
    """Precio de equilibrio de cada periodo, con la demanda opcionalmente desplazada.

    `curvas` es la salida de `parsear_curvas` (o de `cargar`). `delta_mw` es una serie
    indexada por timestamp_utc: positiva = demanda extra (carga de bateria, en MW de
    la red), negativa = oferta extra (descarga). `complejas` elige como entran las ofertas
    complejas: "casadas" (por defecto, su volumen casado), "ofertadas" o "no" (ver supuestos).

    Devuelve timestamp_utc y precio_eur_mwh.
    """
    compleja = curvas["tipologia"].str.fullmatch(r"C\d+")
    if complejas == "casadas":
        curvas = curvas[~compleja | curvas["casada"]]
    elif complejas == "ofertadas":
        curvas = curvas[~compleja | ~curvas["casada"]]
    elif complejas == "no":
        curvas = curvas[~compleja]
    else:
        raise ValueError(f"complejas debe ser 'casadas', 'ofertadas' o 'no', no {complejas!r}")
    delta = {} if delta_mw is None else delta_mw.to_dict()

    filas = []
    for ts, g in curvas.groupby("timestamp_utc", sort=True):
        of = g[g["lado"] == "oferta"]
        de = g[g["lado"] == "demanda"]
        p_of = np.where(of["tipologia"].str.startswith("Imp"), -_INFINITO, of["precio"].to_numpy(float))
        p_de = np.where(de["tipologia"].str.startswith("Exp"), _INFINITO, de["precio"].to_numpy(float))
        q_of, q_de = of["potencia_mw"].to_numpy(float), de["potencia_mw"].to_numpy(float)

        extra = float(delta.get(ts, 0.0))
        if extra > 0:
            p_de, q_de = np.append(p_de, PRECIO_MAX), np.append(q_de, extra)
        elif extra < 0:
            p_of, q_of = np.append(p_of, PRECIO_MIN), np.append(q_of, -extra)
        filas.append((ts, precio_equilibrio(p_of, q_of, p_de, q_de)))
    return pd.DataFrame(filas, columns=["timestamp_utc", "precio_eur_mwh"])
