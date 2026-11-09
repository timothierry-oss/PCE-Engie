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
src/pce68/features/regimenes.py  régimes : pre_crisis, crisis_gas, excepcion_iberica, normalizacion
src/pce68/features/metrics.py    spread quotidien (N heures les plus chères moins N les moins chères)
src/pce68/dispatch/battery.py    arbitrage d'une batterie de 1 MW, un cycle par jour, prix aceptante
src/pce68/excel.py               mise en forme des tableaux Excel (utilisée par 04 et 05)
src/pce68/dispatch/flota.py      parc de batteries ajouté aux courbes OMIE (charge = demande, décharge = offre)
scripts/01_descarga_omie.py      télécharge les prix OMIE -> data/processed/precios_omie.parquet
scripts/02_descarga_curvas.py    télécharge les courbes, les réduit en parquet (data/processed/curvas/)
scripts/03_valida_curvas.py      compare l'intersection des courbes au prix publié par OMIE
scripts/04_exporta_excel.py      génère reports/analisis_exploratorio.xlsx (SORTIE, jamais éditée à la main)
scripts/05_simula_baterias.py    simulation avant/après -> reports/simulacion/ (CSV, PNG, xlsx : SORTIES)
tests/                           pytest (test_pce68.py, test_curvas.py, test_flota.py)
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
open reports/analisis_exploratorio.xlsx
```

Conventions : code et commentaires en espagnol dans les modules existants (les noms de variables aussi : `precios`, `dia_local`...) ; garde cette langue dans le code, explique en français à Tim. **Les sorties (textes des Excel, figures) sont en anglais**, comme le livrable pour ENGIE. Lint : `uv run ruff check .` (ligne maximale 110). Les hypothèses vont dans l'en-tête des modules ou dans `config/base.yaml`, pas dispersées dans le code.

## Format des courbes OMIE (`curva_pbc_AAAAMMDD.1`)

- Téléchargement : `https://www.omie.es/es/file-download?parents=curva_pbc&filename=curva_pbc_AAAAMMDD.1` ; un fichier par jour, environ 4 Mo en horaire (avant le 01/10/2025) et 18 à 20 Mo en quart d'heure. L'archive publique couvre les 6 dernières années.
- CSV en `;`, encodage latin-1, 3 lignes d'en-tête, nombres à l'espagnole (`1.991,1`).
- Colonnes : période ; date ; pays (`MI`, `ES`, `PT`) ; unité (vide) ; type (`V` vente, `C` achat) ; MW ; €/MWh ; `O` offerte / `C` casée ; typologie.
- Typologie : `S` offre simple ; `C01`, `C02`, `C04` offres complexes ; `Imp XX` / `Exp XX` flux d'interconnexion casés. Leurs MW sont réels (souvent 1000 MW = capacité saturée) : avec eux l'intersection retombe sur le prix publié, et la méthode « solde des volumes casés » donne le même résultat (vérifié le 15/05/2025).
- Les lignes `C` répètent une partie des lignes `O` : la courbe se construit avec les `O`.
- **Quand l'Espagne et le Portugal ont le même prix, le fichier ne contient que `MI`. Quand les prix divergent (typiquement 9h à 19h au printemps), il contient des lignes séparées `ES` et `PT`** (vérifié sur la période 12 du 15/05/2025).

## État de la validation (15/05/2025)

`curvas.py` utilise la courbe `ES` si elle existe, sinon `MI` (jamais `PT`), et ignore par défaut les offres complexes (`complejas=False`). Résultat sur les 24 heures : intersection exacte à 17 heures, écart maximal 0,98 €/MWh (les 11 heures avec séparation sont toutes exactes). Avec les offres complexes, l'écart monte à 31 €/MWh.

## À faire en priorité

Faits : correction ES/MI de `curvas.py`, offres complexes exclues par défaut, flux d'interconnexion vérifiés, première simulation (`05_simula_baterias.py`, un seul jour : 15/05/2025).

1. Revalider `curvas.py` sur plusieurs jours (printemps, été, hiver, avant et après le 01/10/2025, donc aussi en quart d'heure).
2. Télécharger les courbes (`02_descarga_curvas.py --cada 10`) et lancer la simulation sur ces jours ; agréger les revenus par saison et par année.
3. Accélérer `flota.despacho` (environ 8 min pour un jour et cinq scénarios, en partie à cause du disque lent).
4. Limites à discuter dans le rapport : la flexibilité existante (pompage, hydraulique, écrêtement) ne réagit pas, d'où un effet des batteries surestimé ; flux d'interconnexion fixes.

## Pièges connus (Mac)

- `ModuleNotFoundError: No module named 'pce68'` : sur le Mac de Tim, `.venv/lib/python3.13/site-packages/_editable_impl_pce68.pth` et le dossier `pce68-0.1.0.dist-info` portent le flag macOS `hidden`, que Python 3.13 semble ignorer. Correctif : `chflags -R nohidden .venv`. Contournement : `PYTHONPATH=src uv run ...`. Si le problème revient après un `uv sync`, relancer le correctif.
- `uv run pytest` peut lancer un `pytest` hors du `.venv` : utiliser `uv run python -m pytest`, après `uv sync --all-groups`.
- `.vscode/settings.json` contient `"python-envs.defaultEnvManager": "ms-python.python:system"` : dans VS Code, choisir l'interpréteur `.venv/bin/python` (Cmd + Maj + P, « Python: Select Interpreter »).
- Fermer l'Excel avant de relancer l'export.

## Règles de travail

- Le dépôt GitHub (timothierry-oss/PCE-Engie) est **public** : n'y mets ni données confidentielles ENGIE, ni documents du client, ni coordonnées de personnes. Les dossiers `data/` et `docs/` sont ignorés par git (`.gitignore`) ; garde-les ainsi.
- Les documents de référence (slides du client, PDF sur la crise du 28/04/2025, courbe en canard, etc.) peuvent être placés dans `docs/` pour lecture, sans les commiter.
- Avant de modifier un fichier, dis ce que tu vas changer. N'écrase pas `reports/*.xlsx` à la main : ils sont régénérés par le script.
- Pas de résultat inventé : si une donnée manque, demande-la ou indique que c'est une hypothèse.
