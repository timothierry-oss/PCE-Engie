# CLAUDE.md — Projet 68 ENGIE (pce68)

Ce fichier donne à Claude Code le contexte du projet. Mets-le à jour quand l'état change.

## Le projet

- **Cadre** : Projet Commande Entreprise (PCE), semestre 7, formation ingénieur généraliste FISE, IMT Atlantique, 2026-2027. Projet n° 68.
- **Partenaire** : ENGIE España, équipe Data, AI & Advanced Analytics de ENGIE Supply & Energy Management (SEM) Iberia. Le client supervise le groupe.
- **Titre** : *Forecasting Price Cannibalisation in the Spanish Power Market under Large-Scale Battery Deployment*.
- **Question** : comment un déploiement massif de batteries change-t-il les prix du marché espagnol (cannibalisation des spreads, revenus d'une batterie), selon différents scénarios de transition énergétique ?
- **Livrables** (pondération) : revue de littérature et analyse de marché (15 %) ; collecte de données et analyse exploratoire (20 %) ; prévision de revenus et modélisation de scénarios (35 %) ; analyse des résultats, recommandations et rapport final (15 %) ; **code reproductible** documenté, Python et/ou SQL (15 %).
- **Données fournies par le partenaire** : historique des prix et volumes du marché espagnol (public, avec un délai de 90 jours selon le cahier des charges ; les fichiers OMIE téléchargés au 12/10/2026 allaient pourtant jusqu'au 11/10/2026), caractéristiques techniques des batteries, documentation sur les marchés.

## L'utilisateur

Tim, étudiant en première année à IMT Atlantique (Nantes), spécialisation Énergie et Environnement prévue en deuxième année. Il travaille sur un **Mac** (Intel, x86_64) avec VS Code et `uv`. Réponds **en français**, explique les choix et les commandes simplement, et dis clairement ce qui est vérifié et ce qui est supposé.

## Idée de modélisation en cours

Tim veut une première simulation de l'impact des batteries :

1. Prendre les **courbes agrégées d'offre et de demande** de OMIE (mercado diario) comme référence : l'intersection donne le prix de chaque heure.
2. Modifier ces courbes pour simuler un parc de batteries : **demande supplémentaire** pendant le creux solaire de midi (la batterie charge, E/η pour décharger E) et **offre supplémentaire** pendant le pic du soir (elle décharge).
3. Recalculer les prix et **tracer les courbes avant/après**. Comparer le spread et le revenu par MW selon la taille du parc (scénarios envisagés : 0, 2, 5, 10 et 17 GW ; 17 GW est le chiffre du PNIEC 2030 cité dans le commentaire de `battery.py`).

Limites à garder en tête : prix aceptante dans `battery.py` (plafond de revenu), prévision parfaite, un cycle par jour, flux d'interconnexion supposés fixes, ni pompage hydraulique ni écrêtement du solaire dans la réponse du marché.

## Structure du code

```
config/base.yaml                 paramètres : dates de téléchargement, batterie (durée, rendement, cycles)
src/pce68/data/omie.py           téléchargement et lecture de marginalpdbc (prix horaires puis quart d'heure)
src/pce68/data/curvas.py         lecture des courbes curva_pbc et prix à l'intersection offre/demande
src/pce68/data/esios.py          fondamentaux ESIOS (demande, production par filière), token dans .env ou ESIOS_TOKEN
src/pce68/data/mibgas.py         prix du gaz : indice journalier MIBGAS-ES (Excel annuel de MIBGAS)
src/pce68/features/regimenes.py  régimes : pre_crisis, crisis_gas, excepcion_iberica, normalizacion
src/pce68/features/metrics.py    spread quotidien (N heures les plus chères moins N les moins chères)
src/pce68/dispatch/battery.py    arbitrage d'une batterie de 1 MW, un cycle par jour, prix aceptante
src/pce68/excel.py               mise en forme des tableaux Excel (utilisée par 04 et 05)
src/pce68/dispatch/flota.py      parc de batteries ajouté aux courbes OMIE (charge = demande, décharge = offre)
src/pce68/dispatch/lp.py         parc planifié par LP sur toute la période (état de charge continu, budget de cycles)
scripts/01_descarga_omie.py      télécharge les prix OMIE -> data/processed/precios_omie.parquet
scripts/02_descarga_curvas.py    télécharge les courbes, les réduit en parquet (data/processed/curvas/)
scripts/03_valida_curvas.py      compare l'intersection des courbes au prix publié par OMIE
scripts/04_exporta_excel.py      génère reports/analisis_exploratorio.xlsx (SORTIE, jamais éditée à la main)
scripts/05_simula_baterias.py    simulation avant/après -> reports/simulacion/ (CSV, PNG, xlsx : SORTIES)
scripts/06_simula_lp.py          simulation LP sur plusieurs jours -> reports/simulacion_lp/ (CSV, PNG : SORTIES)
scripts/07_descarga_fundamentales.py  ESIOS -> data/processed/fundamentales.parquet (indicateurs dans config/base.yaml)
scripts/08_simula_rejilla.py     grille de parcs (1 à 17 GW), une seule LP avec « photos » -> panel p^X
                                 data/processed/panel_bess_*.parquet + reports/rejilla/ (SORTIES)
scripts/09_descarga_combustibles.py  gaz MIBGAS + CO2 ESIOS 1391 -> data/processed/combustibles.parquet
src/pce68/modelo/impacto.py      étapes 2-3 du document client : variables z, ridge et gradient boosting de
                                 y = p^X/p0, validation par mois, ajusta_prevision (p' = ŷ · p^E)
scripts/10_modelo_impacto.py     apprend f(t, X, z) sur le panel -> reports/modelo/ + modèle .joblib (SORTIES)
scripts/11_compara_modelos.py    compare les modèles de 10 (entraînements x ridge/gbm x delta/p0sq) sur les MÊMES lignes
                                 -> reports/modelo/comparacion/ (CSV, PNG, model_comparison.xlsx : SORTIES)
src/pce68/modelo/respuesta.py    réponse du prix à la puissance nette du parc (g(d, z), croissante, interpolée), apprise des courbes
src/pce68/dispatch/equilibrio.py LP en équilibre avec la réponse du prix : equilibre_exacto (1 LP, potentiel convexe) et
                                 equilibrio_parque (itératif, ne converge pas avec une réponse en escaliers : référence)
scripts/13_respuesta_precio.py   calcule la réponse exacte sur les courbes, entraîne g, valide contre le simulateur -> reports/respuesta/
src/pce68/modelo/prevision.py    prévision ENGIE (forecast_base_case.xlsx) -> entrée de ajusta_prevision, couverture du domaine
scripts/14_lp_forecast.py        LP + réponse du prix sur la prévision ENGIE 2027-2040 -> data/processed/prevision_lp/ (CONFIDENTIEL)
scripts/12_ajusta_prevision.py   applique le modèle à la prévision ENGIE 2027-2040 -> data/processed/prevision_ajustada/
                                 (CONFIDENTIEL : jamais dans reports/, dossier ignoré par git)
tests/                           pytest (test_pce68.py, test_curvas.py, test_flota.py, test_lp.py, test_esios.py,
                                 test_mibgas.py, test_modelo.py, test_prevision.py, test_equilibrio.py, test_respuesta.py)
```

Les dossiers `src/data`, `src/dispatch`, `src/features` (hors `src/pce68`) sont d'anciennes copies : **le code utilisé est dans `src/pce68`**. Ne les modifie pas ; propose de les supprimer.

## Commandes (macOS, sans `make`)

```
uv sync                                          # installer les dépendances (crée .venv)
uv run python -m pytest -q                       # tests (utiliser python -m pytest, voir pièges)
uv run python scripts/01_descarga_omie.py        # prix OMIE
uv run python scripts/02_descarga_curvas.py --cada 10
uv run python scripts/03_valida_curvas.py --dia 2025-05-15
uv run python scripts/04_exporta_excel.py [--demo]
uv run python scripts/05_simula_baterias.py [--dia 2025-05-15 --gw 0 2 5 10 17 --paso-mw 250]   # ~8 min
uv run python scripts/05_simula_baterias.py --desde-csv   # refait Excel et PNG depuis les CSV, sans resimuler
uv run python scripts/02_descarga_curvas.py --desde 2026-09-01 --hasta 2026-09-14 --cada 1
uv run python scripts/06_simula_lp.py            # paramètres : config/base.yaml, section simulacion_lp (~2 min)
uv run python scripts/06_simula_lp.py --desde-csv   # refait l'Excel et les PNG depuis les CSV, sans resimuler
open reports/simulacion_lp/simulacion_lp_20260901_20260914.xlsx
uv run python scripts/07_descarga_fundamentales.py [--buscar "eolica"]   # token ESIOS lu dans .env (ou ESIOS_TOKEN)
uv run python scripts/03_valida_curvas.py --dia 2024-01-17 2026-08-12 --csv reports/validacion/validacion_curvas.csv
uv run python scripts/08_simula_rejilla.py      # grille de GW sur 2026 (config : section rejilla)
uv run python scripts/08_simula_rejilla.py --desde-panel   # refait résumé et figures sans resimuler
uv run python scripts/09_descarga_combustibles.py
uv run python scripts/10_modelo_impacto.py [--modelos ridge]   # config : section modelo (~ quelques min)
uv run python scripts/08_simula_rejilla.py --desde 2025-10-01 --hasta 2025-12-31   # panel T4 2025 (quarts d'heure)
uv run python scripts/11_compara_modelos.py     # compare les entraînements (config : section comparacion)
open reports/modelo/comparacion/model_comparison.xlsx
uv run python scripts/13_respuesta_precio.py    # réponse du prix + validation (~4 min ; cache data/processed/respuesta_dp_*.npz)
uv run python scripts/14_lp_forecast.py [--anios 2027 2028]   # ~6 min pour 14 ans ; sorties dans data/, pas dans reports/
uv run python scripts/12_ajusta_prevision.py   # prévision ENGIE ajustée (config : section prevision) ; sorties dans data/
open reports/analisis_exploratorio.xlsx
```

Conventions : code et commentaires en espagnol dans les modules existants (les noms de variables aussi : `precios`, `dia_local`...) ; garde cette langue dans le code, explique en français à Tim. **Les sorties (textes des Excel, figures) sont en anglais**, comme le livrable pour ENGIE. Lint : `uv run ruff check .` (ligne maximale 110). Les hypothèses vont dans l'en-tête des modules ou dans `config/base.yaml`, pas dispersées dans le code.

## Format des courbes OMIE (`curva_pbc_AAAAMMDD.1`)

- Téléchargement : `https://www.omie.es/es/file-download?parents=curva_pbc&filename=curva_pbc_AAAAMMDD.1` ; un fichier par jour, environ 4 Mo en horaire (avant le 01/10/2025) et 18 à 20 Mo en quart d'heure. L'archive en ligne commence le **01/01/2023** (404 sur toute l'année 2022, vérifié le 12/11/2026).
- CSV en `;`, encodage latin-1, 3 lignes d'en-tête, nombres à l'espagnole (`1.991,1`).
- Colonnes : période ; date ; pays (`MI`, `ES`, `PT`) ; unité (vide) ; type (`V` vente, `C` achat) ; MW ; €/MWh ; `O` offerte / `C` casée ; typologie.
- **Période** : un nombre (`1`…`24`) en horaire ; en quart d'heure, une étiquette **`HxQy`** (`H1Q1`…`H24Q4`, `H25Qy` les jours de 25 h), soit la période (x − 1) × 4 + y. Vérifié sur `curva_pbc_20260913.1` ; `curvas.py` gère les deux depuis le 10/11/2026.
- Typologie : `S` offre simple ; `C01`, `C02`, `C04` offres complexes ; `Imp XX` / `Exp XX` flux d'interconnexion casés. Leurs MW sont réels (souvent 1000 MW = capacité saturée) : avec eux l'intersection retombe sur le prix publié, et la méthode « solde des volumes casés » donne le même résultat (vérifié le 15/05/2025).
- Les lignes `C` répètent une partie des lignes `O` : la courbe se construit avec les `O`.
- **Quand l'Espagne et le Portugal ont le même prix, le fichier ne contient que `MI`. Quand les prix divergent (typiquement 9h à 19h au printemps), il contient des lignes séparées `ES` et `PT`** (vérifié sur la période 12 du 15/05/2025).

## État de la validation

`curvas.py` utilise la courbe `ES` si elle existe, sinon `MI` (jamais `PT`). Par défaut (`complejas="casadas"`, depuis le 10/11/2026), les offres simples entrent pour tout le volume offert et les offres complexes **seulement pour leur volume casé** (lignes `C` de typologie `C0x`, gardées dans la colonne `casada`). Écart au prix publié (maximum ; périodes à moins de 0,01 €/MWh) :

| Jour | `casadas` (défaut) | `no` (ancien défaut) | `ofertadas` |
|---|---|---|---|
| 15/05/2025 (horaire) | 0,98 ; 17/24 | 0,98 ; 17/24 | 31,5 ; 5/24 |
| 02/09/2026 (quart d'heure) | 1,23 ; 82/96 | 103,3 ; 0/96 | 51,4 ; 1/96 |
| 13/09/2026 (quart d'heure) | 1,38 ; 84/96 | 15,4 ; 45/96 | 44,9 ; 9/96 |

En quart d'heure, ignorer les offres complexes n'est plus acceptable (jusqu'à 103 €/MWh d'écart).

**Ancien format (jusqu'au 16/03/2025 inclus ; le 01/04/2025 est déjà au nouveau)** : en-tête `Hora`, **pas de colonne typologie**, donc ni offres complexes ni flux identifiables. Jusqu'à 13 GW de vente offerte sous le prix ne sont pas casés (complexes rejetées) : avec toutes les offres `O`, l'intersection tombait 10 à 100 €/MWh sous le prix publié. Depuis le 12/11/2026, `curvas._curva_sin_tipologia` prend le casé dans la zone casée et l'offert au-delà.

Revalidation du 12/11/2026 sur 23 jours, 2023 à 2026, horaire et quart d'heure (`reports/validacion/validacion_curvas.csv`) : écart médian 0 partout ; maximum ≤ 1,7 €/MWh sauf 2,75 (16/03/2025), 4,87 (01/04/2025) et 4,67 (12/08/2026).

## À faire en priorité

Faits : correction ES/MI de `curvas.py`, offres complexes au volume casé, format `HxQy`, ancien format sans typologie, revalidation sur 23 jours (12/11/2026), LP 5 GW sur 2026, fondamentaux ESIOS 01/01-12/11/2026, gaz MIBGAS et CO2 2023-2026, grille de parcs 1-17 GW sur 2026 (`08_simula_rejilla.py`).

Suite (document client `docs/BESS-cannibalization.md`) :

1. Seuil du ratio y = p^X / p0 quand p0 est proche de 0 (`rejilla.umbral_ratio_eur_mwh`, 1 €/MWh par défaut) : à discuter avec ENGIE.
2. Étendre l'historique aux courbes de 2023-2025 (horaire avant le 01/10/2025 : question à poser à ENGIE).
3. Fondamentaux ESIOS sur 2023-2025 ; vérifier les identifiants ESIOS (« SIN COMPROBAR » dans la config) ; CO2 journalier (EUA) à obtenir.
4. Étape 2 du document : modèle f(t, X, z) (ridge ou GAM, validation par blocs de jours), puis étape 3 (application à la prévision ENGIE).
5. Limites à discuter dans le rapport : la flexibilité existante (pompage, hydraulique, écrêtement) ne réagit pas, d'où un effet des batteries surestimé ; flux d'interconnexion fixes ; prévision parfaite.

## Pièges connus (Mac)

- **Chemin du projet : `/Users/tim/Projets_Claude/ENGIE_68`** (déplacé le 20/11/2026 hors de `~/Documents`, qui est synchronisé par iCloud). Là-bas, macOS « allégeait » des fichiers (drapeau `dataless`, visible avec `ls -lO`), d'où des `TimeoutError: [Errno 60]` ou « Need authenticator » à la lecture. Le nouveau dossier est hors iCloud (0 fichier `dataless` vérifié) ; garde-le hors de `Documents`, `Bureau` et `iCloud Drive`. Le venv reste hors du projet : `export UV_PROJECT_ENVIRONMENT="$HOME/.venvs/pce68"` (à mettre dans `~/.zshrc`), puis `uv sync --all-groups`. Le `.venv` du projet ne sert plus.
- `ModuleNotFoundError: No module named 'pce68'` : sur le Mac de Tim, `.venv/lib/python3.13/site-packages/_editable_impl_pce68.pth` et le dossier `pce68-0.1.0.dist-info` portent le flag macOS `hidden`, que Python 3.13 semble ignorer. Correctif : `chflags -R nohidden .venv`. Contournement : `PYTHONPATH=src uv run ...`. Si le problème revient après un `uv sync`, relancer le correctif.
- `uv run pytest` peut lancer un `pytest` hors du `.venv` : utiliser `uv run python -m pytest`, après `uv sync --all-groups`.
- `.vscode/settings.json` contient `"python-envs.defaultEnvManager": "ms-python.python:system"` : dans VS Code, choisir l'interpréteur `.venv/bin/python` (Cmd + Maj + P, « Python: Select Interpreter »).
- Fermer l'Excel avant de relancer l'export.

## Règles de travail

- Le dépôt GitHub (timothierry-oss/PCE-Engie) est **public** : n'y mets ni données confidentielles ENGIE, ni documents du client, ni coordonnées de personnes. Les dossiers `data/` et `docs/` sont ignorés par git (`.gitignore`) ; garde-les ainsi.
- Le token ESIOS est dans `.env` (racine du projet, ignoré par git) : il peut y rester, mais **ne le commite jamais** (avant un commit, vérifier avec `git status` que `.env` n'y figure pas).
- Les documents de référence (slides du client, PDF sur la crise du 28/04/2025, courbe en canard, etc.) peuvent être placés dans `docs/` pour lecture, sans les commiter.
- Avant de modifier un fichier, dis ce que tu vas changer. N'écrase pas `reports/*.xlsx` à la main : ils sont régénérés par le script.
- Pas de résultat inventé : si une donnée manque, demande-la ou indique que c'est une hypothèse.
