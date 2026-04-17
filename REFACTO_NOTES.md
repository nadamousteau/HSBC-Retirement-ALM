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
