# ENGIE_68
Projet de développement pour ENGIE España
NOUS UTILISONS LES DONNÉES DE LA PLATEFORME OMIE

Analyse du marché journalier espagnol (OMIE) et de la valeur d'arbitrage d'une batterie.

## Utilisation

```bash
uv sync
make data         # télécharge OMIE -> data/processed/precios_omie.parquet
make excel        # reports/analisis_exploratorio.xlsx
make excel-demo   # la même chose avec des prix synthétiques, sans rien télécharger
make test
```

## Ouvrir les Excel des résultats

Depuis la racine du projet. Sur macOS (`-a "Microsoft Excel"` force Excel même si les `.xlsx` sont associés à une autre application) :

```bash
open -a "Microsoft Excel" reports/analisis_exploratorio.xlsx                            # analyse exploratoire (script 04)
open -a "Microsoft Excel" reports/simulacion/simulacion_20250515.xlsx                    # parc par tranches, un jour (script 05)
open -a "Microsoft Excel" reports/simulacion_lp/simulacion_lp_20260901_20260914.xlsx     # parc de 5 GW par LP, 1-14/09/2026 (script 06)
```

Sous Windows (PowerShell) : `start reports\simulacion_lp\simulacion_lp_20260901_20260914.xlsx` (même chose pour les autres).

Ce sont des SORTIES des scripts : elles sont entièrement régénérées à chaque exécution. Pour travailler à la main, fais une copie ; ferme Excel avant de relancer un script. Pour refaire les Excel et les figures sans resimuler :

```bash
uv run python scripts/05_simula_baterias.py --desde-csv
uv run python scripts/06_simula_lp.py --desde-csv
```

## Structure

| Chemin | Rôle |
|---|---|
| `config/base.yaml` | Paramètres : dates de téléchargement et batterie (durée, rendement, cycles). |
| `src/pce68/data/omie.py` | Téléchargement et lecture de `marginalpdbc` (horaire et quart d'heure, changements d'heure). |
| `src/pce68/features/regimenes.py` | Étiquette chaque période avec son régime : pre_crisis, crisis_gas, excepcion_iberica, normalizacion. |
| `src/pce68/features/metrics.py` | Spread journalier : moyenne des N heures les plus chères moins celle des N heures les moins chères. |
| `src/pce68/dispatch/battery.py` | Arbitrage d'une batterie de 1 MW, preneuse de prix, un cycle par jour. |
| `scripts/` | Points d'entrée numérotés de la chaîne de traitement. |

Les hypothèses de chaque calcul sont documentées au début de son module.
