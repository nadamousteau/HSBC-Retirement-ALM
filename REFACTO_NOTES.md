# REFACTO_NOTES — Audit préalable (Étape 0)

Ce document fige la baseline du code avant la refactorisation V2. Il répond aux
quatre questions demandées par l'encadrant avant toute modification.

---

## 1. Localisation actuelle des appels `np.random.*`

### `np.random.seed(...)` (API legacy, globale)

| Fichier | Ligne | Appel | Rôle |
|---|---|---|---|
| `ALM_modulaire/main.py` | 265 | `np.random.seed(settings.DECUMULATION_SEED)` | Reseed **global** avant génération des rendements de la phase retraite. `DECUMULATION_SEED=None` par défaut → seed OS. |
| `ALM_modulaire/src/economics/generators.py` | 80 | `np.random.seed(42)` | Re-seed **à chaud** dans la branche backtest, quand les données historiques sont plus courtes que `idx_split` (génère le résidu manquant). |
| `ALM_modulaire/src/economics/generators.py` | 88 | `np.random.seed(42)` | Re-seed si le chargement CSV historique échoue (fallback BS pur sur toute la zone backtest). |
| `ALM_modulaire/src/economics/generators.py` | 96 | `np.random.seed(42)` | Re-seed quand `asset_equity`/`asset_bond` sont `None` (fallback BS pur). |
| `ALM_modulaire/src/economics/generators.py` | 109 | `np.random.seed(None)` | Reseed **OS** avant la branche forecast → divergence stochastique souhaitée entre simulations (= non-reproductible d'un run à l'autre). |
| `ALM_modulaire/src/economics/gse.py` | 71 | `np.random.seed(seed)` | Re-seed global dans `MarkovRegimeSwitching.simulate_regimes()` (tirage Markov). |
| `ALM_modulaire/src/economics/gse.py` | 228 | `np.random.seed(seed)` | Re-seed global au début de `EnhancedGSE.generate_scenarios()` (utilisé par Faleh). |
| `ALM_modulaire/src/economics/inflation_vasicek.py` | 61 | `np.random.seed(seed)` | Re-seed global dans `VasicekInflation.simulate()` (inflation accum + décum). |

### `np.random.default_rng(...)` (API moderne, Generator)

| Fichier | Ligne | Appel | Rôle |
|---|---|---|---|
| `ALM_modulaire/src/economics/nelson_siegel_var.py` | 190 | `rng = np.random.default_rng()` | Fallback interne de `simulate_ns_factors` quand `rng` n'est pas fourni (non-reproductible). |
| `ALM_modulaire/src/economics/nelson_siegel_var.py` | 286 | `rng = np.random.default_rng(seed)` | Créé dans `simulate_gbi_monte_carlo` à partir de `settings.GBI_SEED` (= 42 par défaut), propagé à `simulate_ns_factors`. |

### Synthèse

- **8 appels `np.random.seed()` globaux** disséminés (dont 6 hors `main.py`) → fuite
  d'état entre modules, non-déterminisme si un module oublie de re-seeder.
- **2 appels `default_rng`** locaux (NS-VAR(1)) → seul composant qui utilise déjà
  un `Generator` isolé.
- **4 seeds différents côté `settings.py`** : `INFLATION_SEED=42`, `GBI_SEED=42`,
  `FALEH_SEED=42`, `DECUMULATION_SEED=None`. Pas de point d'entrée unique.
- **Pas de seed explicite** pour `shocks.ajouter_chocs_merton` ni pour
  `shocks.injecter_crise_localisee` (ils appellent `np.random.binomial`,
  `np.random.normal` directement sans seed, héritent du seed global laissé par
  le dernier re-seed effectué ailleurs).
- **Pas de seed explicite** non plus pour `generer_rendements_correles_base`
  (utilisé nulle part dans le pipeline main mais exporté).

---

## 2. Durée totale effective des scénarios économiques générés

Valeurs issues de `config/settings.py` avec les valeurs par défaut du repo :

| Paramètre | Valeur |
|---|---|
| `NB_ANNEES_ACCUMULATION` | 40 |
| `NB_ANNEES_DECUMULATION` | 25 |
| `NB_PAS_PAR_AN` | 12 |
| `NB_PERIODES_TOTAL` (dérivé) | **480 mois** |

### Phases générées aujourd'hui

| Phase | Générateur | Durée | Seed |
|---|---|---|---|
| Accumulation — rendements EQ/BD | `generer_rendements_avec_backetest` | 480 mois (backtest + forecast) | 42 backtest, **None** forecast (divergence) |
| Accumulation — inflation | `generer_inflation_vasicek` | 480 mois | `INFLATION_SEED=42` |
| Accumulation — GBI (NS-VAR(1)) | `simulate_gbi_monte_carlo` | `nb_forecast = 480 - max(idx_split,1)` mois sur 360 maturités | `GBI_SEED=42` |
| Accumulation — Faleh GSE | `EnhancedGSE.generate_scenarios_simple` | 480 mois | `FALEH_SEED=42` |
| Décumulation — rendements EQ/BD | Black-Scholes inline dans `main.py` | 300 mois | `DECUMULATION_SEED=None` (reseed OS) |
| Décumulation — inflation | `generer_inflation_vasicek` | 300 mois | `DECUMULATION_SEED=None` |
| Décumulation — GBI | *(non générée : fallback Fixed-Mix)* | — | — |

### Durée totale effective

- Accumulation : **480 mois** de scénarios.
- Décumulation : **300 mois** régénérés **séparément** (pas une continuation
  temporelle — rupture de trajectoire au mois 481 puisque re-seed explicite).
- **Cumul `accum + decum` = 780 mois**, mais **jamais générés en une seule
  passe**. Les deux trajectoires ne se raccordent pas : distribution OK,
  trajectoire d'une sim donnée = discontinue.

Cette rupture est le point que la Tâche 3 doit corriger.

---

## 3. Flux de `mat_cap`, `capital_depart_decum`, `dernier_contexte` dans `main.py`

Pseudo-code condensé du flux actuel (20 lignes) :

```
strategies_run = STRATEGIES_A_COMPARER if MODE_COMPARAISON else [METHODE_DEFAUT]
resultats_comparaison = {}
dernier_contexte = {}

for strat in strategies_run:                          # boucle séquentielle
    settings.METHODE = strat                          # mutation globale !
    if strat == "GBI":
        mat_cap, courbe_investi, hist_apport, hist_dd, hist_salaire, _, inflation_factor = run_simulation_gbi(...)
    else:
        strategy = build_strategy(strat)              # TargetDate / FixedMix / Faleh
        mat_cap, courbe_investi, hist_apport, hist_dd, hist_salaire, inflation_factor = run_simulation(strategy, ...)

    resultats_comparaison[strat] = mat_cap            # OK : dict par strat
    dernier_contexte = {                              # ECRASE à chaque itération
        "courbe_investi": courbe_investi,
        "hist_apport":    hist_apport,
        "hist_salaire":   hist_salaire,
        "mat_cap":        mat_cap,
        "inflation_factor": inflation_factor,
    }
    print_kpis(strat, mat_cap, ...)                   # reporting accum OK

# Après la boucle : dernier_contexte == contexte de la DERNIÈRE strat de la liste
mat_cap          = dernier_contexte["mat_cap"]        # = celui de la dernière strat
inflation_factor = dernier_contexte["inflation_factor"]
# Plots solo utilisent mat_cap (biais de dernière strat)
# Plots comparatifs itèrent resultats_comparaison (OK)

if SIMULER_DECUMULATION:
    capital_depart_decum = mat_cap[-1, :]              # ← BUG : capital de la
                                                      #   dernière strat d'accum,
                                                      #   indépendamment de la
                                                      #   strat de décumulation choisie
    run_decumulation(strategy_decum, ..., capital_depart_decum, ...)
```

### Bugs identifiés par cette trace

- **B1** — `dernier_contexte` est monovaleur : le reporting solo et les plots
  crise utilisent toujours la **dernière** stratégie de la liste (biais d'ordre
  des strats dans `STRATEGIES_A_COMPARER`). Réordonner la liste change les
  graphes solo.
- **B2** — `capital_depart_decum = mat_cap[-1, :]` utilise le `mat_cap` de la
  dernière stratégie d'accum, pas celui de la stratégie d'accum dont on veut
  enchaîner la décumulation. Le couplage accum→décum est donc **incorrect** dès
  qu'il y a ≥ 2 stratégies dans la liste.
- **B3** — `settings.METHODE = strat` mute une variable globale pour aiguiller
  des modules sous-jacents ; effet de bord caché, non thread-safe.
- **B4** — Seule **une** stratégie de décumulation est exécutée (la dernière
  strat d'accum → `STRATEGIE_DECUMULATION`), alors que la comparaison honnête
  demande un couple (accum, décum) par stratégie d'accumulation.

Tâches 4 et 5 corrigent B1/B2. Tâche 6 réduit la surface de B3 côté
décumulation. B3 côté accumulation (`settings.METHODE`) n'est pas dans le
périmètre des 7 tâches et reste en l'état.

---

## 4. Stratégies disponibles en décumulation — fallback GBI

Le bloc `main.py:294-321` aiguille `strat_decum_name` vers une stratégie
d'investissement :

| `STRATEGIE_DECUMULATION` | Branche prise | Stratégie effective |
|---|---|---|
| `"TARGET_DATE"` | `TargetDateStrategy()` | Target Date avec profil décum |
| `"FIXED_MIX"` | `FixedMixStrategy(profil_decum.fixed_allocation)` | Fixed-Mix allocation du profil décum |
| `"FALEH"` | `FalehStrategy(...)` + `initialize_tree(dates_decum)` | Faleh avec arbre reconstruit sur 300 mois |
| `"GBI"` | **fallback silencieux** → `FixedMixStrategy(profil_decum.fixed_allocation)` avec print « GBI non applicable en décumulation » et renommage en `f"FIXED_MIX (fallback GBI, {pct}%)"` | Fixed-Mix déguisé |
| *(autre)* | **fallback silencieux** → `FixedMixStrategy(profil_decum.fixed_allocation)` sans message | Fixed-Mix déguisé |

### Problèmes

- Le fallback GBI est un **bug silencieux** : l'utilisateur demande GBI, le
  système exécute Fixed-Mix et affiche un `Note :` noyé dans les logs. La
  métrique affichée porte GBI dans son titre mais provient d'un autre moteur.
- Le fallback `else` final accepte **n'importe quelle valeur non listée** pour
  `STRATEGIE_DECUMULATION` (ex : typo) et redescend sur Fixed-Mix — toujours
  silencieusement.
- Aucune validation au chargement de `settings.py` : `STRATEGIES_A_COMPARER`
  peut contenir une chaîne inventée, le crash n'intervient que tardivement
  dans la boucle.

Tâche 6 supprime ces fallbacks et valide les chaînes via un `Enum` strict.

---

## 5. Périmètre hors refacto (rappel)

- Calibrations (Vasicek, NS-VAR(1), RSLN, BS) : **intouchées**.
- Sémantique `liabilities/` vs `assets/` : **intouchée**, même si
  `src/liabilities/contributions.py` contient des actifs. Une session dédiée
  sera ouverte avec l'utilisateur pour cette refonte.
- Stratégies GBI, Faleh, FixedMix, TargetDate : logique interne inchangée.
  Seules les signatures évoluent (ajout de `rng`) dans la Tâche 2.
- Données d'entrée `data/inputs/*` : inchangées.

Cette baseline est la référence de non-régression pour les 7 tâches qui
suivent.

---

## Vague 2 — Tâche A — Audit

Grep récursif `settings\.METHODE\b` sur `ALM_modulaire/` (hors `__pycache__`)
avant modifications :

### Écritures (mutations d'exécution)

| Fichier | Ligne | Appel | Nature |
|---|---|---|---|
| `ALM_modulaire/src/pipeline/accumulation.py` | 54 | `settings.METHODE = strat_actuelle` | **WRITE** — mutation globale dans la boucle de stratégies. Non thread-safe, dépend de l'ordre d'itération. |

### Lectures

| Fichier | Ligne | Appel | Nature |
|---|---|---|---|
| `ALM_modulaire/src/analytics/plotting.py` | 92 | `f"Évolution du Capital Accumulé - {settings.METHODE}"` | **READ** dans `plot_capital` — titre dynamique. Lit la valeur mutée par `accumulation.py:54`. |
| `ALM_modulaire/src/pipeline/accumulation.py` | 8 | docstring `(settings.METHODE) sont préservés` | **READ** (commentaire). |

### Hors scope (lectures de `METHODE_DEFAUT`, constante)

| Fichier | Ligne | Rôle |
|---|---|---|
| `ALM_modulaire/main.py` | 108 | `settings.METHODE_DEFAUT` — choix solo en l'absence de `MODE_COMPARAISON`. |
| `ALM_modulaire/src/pipeline/reporting.py` | 12, 29 | Docstring/commentaire `METHODE_DEFAUT`. |
| `ALM_modulaire/src/strategies/enums.py` | 55 | Message d'erreur de validation `METHODE_DEFAUT`. |

`METHODE_DEFAUT` est une constante de configuration, lue mais jamais mutée : elle
reste en place. Seul l'attribut `METHODE` (créé dynamiquement par la mutation)
disparaît du code.

### Plan d'action Tâche A

1. Suppression de `settings.METHODE = strat_actuelle` dans `accumulation.py:54`.
2. `plot_capital` et 4 autres plots solo reçoivent un paramètre obligatoire
   `strategy_name: str` — source unique du titre.
3. `reporting.py` propage `solo_strat_name` (= `METHODE_DEFAUT` si présente,
   sinon première stratégie du dict) à chaque plot solo.
4. `decumulation.py` propage `f"{strat_accum} → {strat_decum}"` aux plots
   retraite.
5. Mise à jour de la docstring de `accumulation.py` pour supprimer la mention
   de la mutation globale.

---

## Vague 2 — Tâche B — Audit corrélation inflation/bonds

### État avant Tâche B

| Phase | Générateur EQ/BD | Générateur inflation | Couplage |
|---|---|---|---|
| Backtest (mois 0..idx_split) | `generer_rendements_avec_backetest` (CSV historique) | `generer_inflation_vasicek` (Vasicek indépendant) | **aucun** |
| Forecast (mois idx_split..N) | `generer_rendements_avec_backetest` (BS divergent) | `generer_inflation_vasicek` (Vasicek indépendant) | **aucun** |

Inflation et bonds tirés sur deux RNG distincts (`rng_bundle["equity_bonds"]`
vs `rng_bundle["inflation"]`). Pas de matrice de covariance jointe.
Conséquence : un scénario d'hyperinflation forecast peut coïncider avec un
scénario de hausse des prix obligataires — incohérence économique connue.

### Calibration empirique cible

Études de référence sur bonds nominaux US 1990-2024 :

| Source | Période | rho(bonds, infl) |
|---|---|---|
| Brixton & Booth (2023) | 1990-2022 | -0.27 |
| Campbell-Sunderam-Viceira (2017) | 1962-2014 | -0.25 à -0.35 |
| Vanguard MAR (2024) | 2000-2024 | -0.31 |

Valeur retenue : **rho_bond_infl = -0.30** dans `INFLATION_BONDS_CORRELATION`.
Hypothèse d'orthogonalité equity-inflation maintenue (`corr_eq_infl = 0.0`,
typique en ALM long terme).

### Architecture cible (Tâche B)

| Phase | Générateur EQ/BD | Générateur inflation | Couplage |
|---|---|---|---|
| Backtest | `generer_rendements_backtest` (CSV) | `generer_inflation_vasicek` indépendant | aucun (path historique déterministe) |
| Forecast | `generer_scenarios_marche_correles` (joint) | `generer_scenarios_marche_correles` (joint) | **Cholesky 3×3 sur (eps_e, eps_b, eps_i)** |

Continuité : `inflation_init = inflation_bt[-1, :]` (per-sim) est passé au
générateur joint pour qu'aucun saut artificiel n'apparaisse au pivot.

### Bundle RNG

Ajout d'une clé **`market_correlated`** EN FIN du tuple `_BUNDLE_KEYS` pour
préserver la SeedSequence enfant des streams existants
(`equity_bonds`, `inflation`, `gbi_ns`, `faleh_gse`).

### Plan d'action Tâche B

1. `config/settings.py` — `INFLATION_BONDS_CORRELATION = -0.30` (déjà ajouté).
2. `src/economics/generators.py` — trois nouvelles fonctions :
   - `generer_rendements_backtest(...)` — extrait la portion historique pure.
   - `generer_rendements_forecast(...)` — forecast EQ/BD seul (pour modules sans
     besoin d'inflation corrélée).
   - `generer_scenarios_marche_correles(...)` — joint EQ/BD/inflation Cholesky.
   Suppression de `generer_rendements_avec_backetest` (pas de shim, pas de
   fonction conservée "au cas où").
3. `src/utils/rng.py` — ajout `"market_correlated"` à `_BUNDLE_KEYS`.
4. `main.py` — refonte du bloc § 2 : backtest indépendant + forecast joint.
5. `tests/test_inflation_bonds_corr.py` — 4 tests pytest indépendants
   (corrélation forte, marges cohérentes, corrélation nulle, continuité init).

---

## Vague 2 — Tâche C — Audit séparation passif / actif

### État avant Tâche C

Trois pathologies couplées dans le code initial :

**P1 — Mauvais étiquetage `liabilities/contributions.py`.**
Le fichier `src/liabilities/contributions.py` contenait `HumanCapitalCurve` (diffusion
salariale stochastique) + 3 fonctions d'apport exponentiel
(`precalculer_parametres_apport_exponentiel`, `calculer_apport_exponentiel`,
`estimer_salaire_saturation`). Ce sont des **flux d'actifs** (revenus du travail
+ versements à la stratégie d'investissement), classés par erreur dans le package
`liabilities/`. Conséquence : le grep "qu'est-ce qu'un passif dans cette base ?"
remontait du code d'actifs ; impossible de localiser le passif client (objectif
revenu retraite) car il n'était matérialisé nulle part.

**P2 — Passif client encodé en deux scalaires plats dans `settings.py`.**
Le passif client était entièrement résumé par :
```python
RETRAIT_MENSUEL_REEL = 2000        # € constants
TAUX_ACTUALISATION_PLANCHER = 0.02 # taux réel
```
Sans typage, sans encapsulation, sans validation, sans calcul de valeur actuelle,
sans distinction REAL/NOMINAL, sans horizon (FIXED ou actuariel). Trois lectures
indépendantes de `settings.RETRAIT_MENSUEL_REEL` dispersées dans
`src/engine/decumulation_core.py` et dans une heuristique dans
`src/strategies/faleh_strategy.py` (`_estimate_target_wealth`).

**P3 — `FalehStrategy._estimate_target_wealth()` invente la cible.**
Cette méthode privée calculait `target_wealth = 0.80 × FV_apports_constants` à
partir d'un taux constant `mu_e` et d'un apport constant — heuristique sans
fondement actuariel, indépendante du passif client réel. Aucun lien avec le
revenu retraite désiré, le mode REAL/NOMINAL, ni l'horizon de retraite. La
stratégie d'investissement optimisait donc un objectif fantôme.

### Conséquences observables

| Symptôme | Trace dans le code |
|---|---|
| Pas de calcul de VA du passif | Aucune fonction `present_value(liability)` dans la base |
| Pas de funded ratio | Notion absente, KPIs solo n'incluent que la distribution de capital |
| GBI sans goal externe | `gpi` (Goal Price Index) construit en interne par stress des courbes NS-VAR(1) sans ancrage sur un capital cible client |
| Décumulation aveugle au mode | `RETRAIT_MENSUEL_REEL` traité comme nominal au mois 1 (pas de capitalisation par `(1+pi)^accum`) |
| Horizon décumulation arbitraire | `NB_ANNEES_DECUMULATION = 25` choisi visuellement, sans cohérence avec `e_60 ≈ 21.6 ans` (TH 00-02) |

### Architecture cible (Tâche C)

```
src/
├── assets/                      ← NOUVEAU package
│   ├── __init__.py
│   ├── human_capital.py         ← HumanCapitalCurve (déplacé verbatim)
│   └── contribution_policy.py   ← 3 fonctions d'apport (déplacées verbatim)
└── liabilities/                 ← purement "passif" désormais
    ├── __init__.py              ← réexporte les 4 modules
    ├── retirement_objective.py  ← NOUVEAU : RetirementLiability + builder
    ├── liability_valuation.py   ← NOUVEAU : funded_ratio()
    ├── mortality.py             ← TH 00-02 (déjà présent)
    └── goal_price_index.py      ← GPI NS-VAR(1) (déjà présent)
```

`src/liabilities/contributions.py` est **supprimé sans shim** (suppression dure :
les 4 imports en aval sont mis à jour). C'est l'invariant explicité par le
cahier des charges Tâche C.

### Conception de `RetirementLiability`

Frozen dataclass à 8 champs (immutabilité = aucune mutation accidentelle d'un
passif client après construction) :

| Champ | Type | Rôle |
|---|---|---|
| `target_income_monthly` | `float` | Revenu mensuel cible (en € du mode choisi) |
| `income_mode` | `"REAL"\|"NOMINAL"` | REAL → capitalisé par `(1+pi)^accum` au pivot |
| `age_retirement` | `int` | Pour le facteur actuariel `ä^(12)_x` |
| `horizon_mode` | `"FIXED"\|"ACTUARIAL"` | Annuité certaine vs vie-contingente |
| `horizon_years_fixed` | `int` | Si FIXED uniquement |
| `discount_rate` | `float` | Taux d'actualisation annuel |
| `inflation_expected` | `float` | Pour la capitalisation REAL → NOMINAL |
| `nb_annees_accumulation` | `int` | Idem (durée du facteur de capitalisation) |

Trois méthodes publiques :

- `expected_duration_years()` — scalaire en mode FIXED, somme des probabilités
  de survie tronquées à `_DEFAULT_TRUNC_AGE = 110` en mode ACTUARIAL.
- `monthly_income_nominal_at_retirement()` — capitalise par
  `(1 + inflation_expected)^nb_annees_accumulation` si REAL, identité si NOMINAL.
- `required_capital_at_retirement()` (alias `present_value_at_retirement`) —
  calcule la VA en mensualités :
    - FIXED : `income_m * (1 - (1+r_m)^(-n_months)) / r_m`
    - ACTUARIAL : `income_m * 12 * annuity_factor_monthly(age_retraite, r, loading=0)`

`build_liability_from_settings(settings_mod)` est l'unique point de
construction : il lit `LIABILITY_*` + `AGE_DEPART_RETRAITE` +
`NB_ANNEES_ACCUMULATION` du module `settings`, ce qui garantit la traçabilité
(une seule source de vérité, pas de constructeurs alternatifs dispersés).

### Choix de design discutés

**D1 — `annuity_strategy.py` n'est PAS branché sur le passif.**
La stratégie "Annuity" achète une rente viagère sur le marché ; son prix est dicté
par l'assureur (mortalité table assureur, marges, frais), pas par le passif
client. Brancher l'annuité sur `liability.required_capital_at_retirement()`
fusionnerait pricing assureur et VA actuarielle interne, perdant l'écart
client/marché. Resté isolé, conformément au cahier des charges.

**D2 — GBI : `liability` passé mais floor relatif maintenu.**
Un floor pur "Goal-anchored" (`floor_pct × goal_amount × beta_t`) bloque le
portefeuille à 100% obligataire dès t=0 dans la calibration de référence
(`CAPITAL_INITIAL = 5_000` vs `goal_amount ≈ 1_080_511 €` avec `beta_0 ≈ 0.20` →
floor ≈ 191k >> 5k). Solution conservative pour cette tâche :

- `liability` est passé à `run_simulation_gbi` (signature mise à jour),
- `goal_amount = liability.required_capital_at_retirement()` est loggé au
  démarrage (visible dans la console),
- la mécanique de floor relative à la trajectoire (drawdown depuis le pic) est
  **inchangée**.

L'évolution future cohérente serait `floor_pct × (PV_apports_futurs + goal_amount × beta_t)` :
le floor rejoint asymptotiquement la cible quand les apports futurs s'amenuisent.
Documentée dans `gbi_core.py` mais non implémentée dans cette tâche pour rester
strictement non-régressive sur la KPI GBI.

**D3 — `FalehStrategy` : suppression dure de `_estimate_target_wealth`.**
La nouvelle signature est `FalehStrategy(..., liability=None, target_wealth=None, ...)`
avec priorité `target_wealth > liability`. Si **aucun** des deux n'est fourni,
`ValueError` immédiate. Il n'y a plus de fallback heuristique silencieux : la
stratégie ne peut plus inventer une cible.

**D4 — Validation horizon.**
`validate_settings` impose `NB_ANNEES_DECUMULATION ≥ int(expected_duration_years()) + 5`.
Avec la calibration par défaut (REAL, ACTUARIAL, age=60, e_60≈21.6) le seuil
minimal est 26. `NB_ANNEES_DECUMULATION` est bumpé de 25 → 30 dans
`settings.py`. Cette marge de 5 ans absorbe la queue de la distribution de durée
de vie (P95 de la loi de survie au-delà de l'espérance).

### Plan d'action Tâche C

1. **Créer `src/assets/`** : `__init__.py`, `human_capital.py`, `contribution_policy.py`
   (déplacements verbatim, pas de modification de code).
2. **Créer `src/liabilities/retirement_objective.py`** : dataclass + builder.
3. **Créer `src/liabilities/liability_valuation.py`** : `funded_ratio(capital, liability)`.
4. **Mettre à jour `src/liabilities/__init__.py`** : réexports.
5. **Supprimer** `src/liabilities/contributions.py` (`rm`, pas de shim).
6. **Refondre `config/settings.py`** : supprimer 2 clés plates, ajouter bloc §7c
   `LIABILITY_*` (6 clés), bumper `NB_ANNEES_DECUMULATION` à 30.
7. **Mettre à jour les imports** : `engine/core.py` et `engine/gbi_core.py`
   passent à `from src.assets import contribution_policy as contributions`.
8. **Brancher `FalehStrategy.__init__`** : argument `liability=None`, suppression
   de `_estimate_target_wealth()` et de l'attribut `target_wealth` calculé en
   interne.
9. **Brancher `engine/decumulation_core.run_decumulation`** : argument
   `liability` obligatoire, lecture via
   `liability.monthly_income_nominal_at_retirement()` et `liability.discount_rate`.
   Renommer `retrait_mensuel_reel → retrait_mensuel_plancher` (la variable n'est
   plus "réelle" : elle est nominale au pivot après capitalisation REAL).
10. **Brancher `engine/gbi_core.run_simulation_gbi`** : argument `liability=None`,
    log de `goal_amount` au démarrage.
11. **Propager `liability` dans le pipeline** : `pipeline/accumulation.py`,
    `pipeline/decumulation.py`, `main.py`.
12. **Étendre `validate_settings`** dans `src/strategies/enums.py` :
    `_validate_liability_settings` valide les 6 clés + cohérence horizon.
13. **Tests** : `tests/test_retirement_liability.py` (25 tests, 6 classes).
14. **Vérification end-to-end** : `pytest` + `python -X utf8 main.py`.

### Vérifications

- **Tests** : 29 PASSED en 2.24s (4 hérités de Tâche B + 25 nouveaux).
  Pas de warning, pas de régression sur l'inflation corrélée.
- **`main.py` end-to-end** : pipeline complet OK. Affichage de la nouvelle ligne
  `PASSIF CLIENT : revenu=2000€/mois (REAL), horizon=22.9ans (ACTUARIAL),
  discount=2.00%, VA_retraite=1,080,511€` avant l'accumulation.
- **Log GBI** : `goal_amount = 1,080,511 €` affiché en début de
  `run_simulation_gbi`.

### KPI deltas Tâche B → Tâche C

| Strat | KPI | Tâche B | Tâche C | Lecture |
|---|---|---|---|---|
| FALEH | Sortino accum | 3.38 | 3.18 | Léger recul attendu : la cible est désormais issue de la VA actuarielle (1,08 M€) au lieu de l'heuristique 80% × FV apports. La stratégie travaille sur un objectif plus exigeant. |
| FALEH | P&L réel P5 décum | -2,227 € | positif | L'alerte "P&L réel négatif au 5e percentile" disparaît. Le retrait plancher capitalisé est cohérent avec le revenu cible REAL. |
| FIXED_MIX | tous KPIs | inchangés | inchangés | Strictement invariant (la stratégie ne lit pas le passif). |
| TARGET_DATE | tous KPIs | inchangés | inchangés | Strictement invariant. |
| GBI | tous KPIs | inchangés | inchangés | `liability` passé mais non utilisé dans la mécanique de floor (D2). |

### Bénéfice structurel

- **Une seule source de vérité** pour le passif client : `RetirementLiability`
  construit depuis `LIABILITY_*` dans `settings.py`. Aucun autre module ne
  réinvente le passif.
- **Étiquetage correct** : `assets/` contient les flux d'actifs, `liabilities/`
  contient le passif client + les outils de valorisation et la mortalité.
- **Échec précoce** : `validate_settings` capture les incohérences (clé
  manquante, mauvais type, mauvais mode, horizon insuffisant) avant tout calcul.
- **Pas de fallback silencieux** : `FalehStrategy` lève `ValueError` si la
  cible n'est pas explicitée ; `decumulation_core` lève `TypeError` si
  `liability` n'est pas passé (argument obligatoire).
- **Funded ratio** disponible (`funded_ratio(capital, liability)`) pour les
  futures KPIs et plots — non utilisé dans cette tâche mais prêt à l'emploi.

