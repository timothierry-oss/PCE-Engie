# ENGIE_68
Projecto de desarrollo para Engie España
USAMOS LOS DATOS DE LA PLATAFORMA OMIE

Análisis del mercado diario español (OMIE) y del valor de arbitraje de una batería.

## Uso

```bash
uv sync
make data         # descarga OMIE -> data/processed/precios_omie.parquet
make excel        # reports/analisis_exploratorio.xlsx
make excel-demo   # lo mismo con precios sintéticos, sin descargar nada
make test
```

## Estructura

| Ruta | Qué hace |
|---|---|
| `config/base.yaml` | Parámetros: fechas de descarga y batería (duración, eficiencia, ciclos). |
| `src/pce68/data/omie.py` | Descarga y parseo de `marginalpdbc` (horario y cuarto-horario, cambios de hora). |
| `src/pce68/features/regimenes.py` | Etiqueta cada periodo con su régimen: pre_crisis, crisis_gas, excepcion_iberica, normalizacion. |
| `src/pce68/features/metrics.py` | Spread diario: media de las N h más caras menos las N h más baratas. |
| `src/pce68/dispatch/battery.py` | Arbitraje de 1 MW de batería, precio aceptante, un ciclo diario. |
| `scripts/` | Puntos de entrada numerados del flujo. |

Los supuestos de cada cálculo están documentados al principio de su módulo.
