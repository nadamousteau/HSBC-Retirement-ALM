# REFACTO CHANGELOG

Refonte en 7 tâches du pipeline ALM retraite. Aucun commit git effectué :
toutes les modifications sont locales, à valider et committer par l'utilisateur.

## T1 — Découpage de main.py en pipelines
**Fichiers créés**
- `ALM_modulaire/src/pipeline/accumulation.py` — `run_accumulation(...)`
- `ALM_modulaire/src/pipeline/reporting.py` — `run_reporting(...)`
- `ALM_modulaire/src/pipeline/decumulation.py` — `run_decumulation_phase(...)`
- `ALM_modulaire/src/pipeline/__init__.py`

**Fichier modifié**
- `ALM_modulaire/main.py` — réduit à l'orchestration pure.

**Test** : diff sorties console vs baseline = 0 hors timings.

## T2 — RNG master via `settings.GLOBAL_SEED`
**Fichiers créés**
- `ALM_modulaire/src/utils/rng.py` — `make_rng_bundle(global_seed)` (SeedSequence.spawn)
- `ALM_modulaire/src/utils/__init__.py`

**Fichiers modifiés**
- `config/settings.py` — ajout `GLOBAL_SEED = 42`, dépréciation
  `INFLATION_SEED` / `GBI_SEED` / `FALEH_SEED` / `DECUMULATION_SEED`.
- `src/economics/generators.py` — `rng` au lieu de `np.random.seed()` global.
- `src/economics/inflation_vasicek.py` — `rng` au lieu de `seed`.
- `src/economics/gse.py` — `rng` dans `MarkovRegimeSwitching` et `EnhancedGSE`.
- `src/economics/nelson_siegel_var.py` — `simulate_gbi_monte_carlo(rng=...)`.
- `src/economics/shocks.py` — `ajouter_chocs_merton(rng=...)`.
- `src/strategies/faleh_strategy.py` — stocke `self.rng`.
- `src/pipeline/accumulation.py`, `src/pipeline/decumulation.py` — propagent le bundle.

**Tests**
- Deux runs avec `GLOBAL_SEED=42` → identiques hors timings.
- `GLOBAL_SEED=43` → cascade complète (>140 lignes diff).

## T3 — Trajectoire économique unifiée accum + décum
**Fichier modifié**
- `ALM_modulaire/main.py` — génère `r_eq_full`, `r_bd_full`, `inflation_full`
  sur `NB_MOIS_TOTAL_PIPELINE` puis slice `[0:480]` (accum) / `[480:780]` (décum).
- `config/settings.py` — ajout `NB_MOIS_TOTAL_PIPELINE`.
- `src/utils/rng.py` — bundle simplifié (4 clés, plus besoin de RNG séparés accum/décum).

**Test** : continuité statistique au pivot mois 480-481 vérifiée
(mean/std stationnaires autour du pivot, pas de rupture de distribution).

## T4 — Fix `dernier_contexte` → sélection explicite
**Fichier modifié**
- `ALM_modulaire/src/pipeline/reporting.py` — nouvelle fonction `_select_solo_context()`
  qui privilégie `settings.METHODE_DEFAUT` plutôt que la dernière stratégie itérée.

**Test** : sortie bit-identique (plots solo désactivés en mode comparaison).

## T5 — Capital de départ décumulation par stratégie accum
**Fichiers modifiés**
- `ALM_modulaire/src/pipeline/decumulation.py` — refactor `run_decumulation_phase`
  en deux helpers (`_build_decumulation_strategy`, `_run_one_decumulation`) et
  boucle sur `contextes_accum.items()` quand `DECUMULATION_PAR_STRATEGIE_ACCUM=True`.
- `config/settings.py` — ajout `DECUMULATION_PAR_STRATEGIE_ACCUM = True`.

**Test** : 4 décumulations exécutées (GBI/FALEH/FIXED_MIX/TARGET_DATE), chacune
partant de son propre `mat_cap[-1, :]`. Reproductibilité inter-runs confirmée.

## T6 — Enums stricts par phase, suppression du fallback silencieux
**Fichiers créés**
- `ALM_modulaire/src/strategies/enums.py` — `AccumulationStrategy`,
  `DecumulationStrategy`, `validate_settings(...)`.

**Fichiers modifiés**
- `src/strategies/__init__.py` — export des enums et validateur.
- `src/pipeline/accumulation.py` — `else: raise NotImplementedError(...)`.
- `src/pipeline/decumulation.py` — suppression du fallback GBI → FIXED_MIX,
  remplacé par `raise NotImplementedError(...)`.
- `main.py` — `validate_strategy_settings(settings)` en tout début de `main()`.

**Tests** : validation lève bien `ValueError` sur `STRATEGIE_DECUMULATION='GBI'`,
sur `METHODE_DEFAUT='BAR'`, sur `STRATEGIES_A_COMPARER=['FOO']`.
Run nominal identique hors timings.

## T7 — AnnuityStrategy (rente viagère TH 00-02)
**Fichiers créés**
- `ALM_modulaire/src/liabilities/mortality.py` — table `_TH_00_02_LX`
  (approximation par checkpoints interpolés), `survival_probability(...)`,
  `annuity_factor_monthly(age, rate, loading)`.
- `ALM_modulaire/src/strategies/annuity_strategy.py` — `run_annuity_decumulation(...)`.

**Fichiers modifiés**
- `config/settings.py` — `ANNUITY_TECHNICAL_RATE = 0.01`, `ANNUITY_INSURER_LOADING = 0.05`,
  commentaire à jour pour `STRATEGIE_DECUMULATION` (choix élargi, GBI exclu).
- `src/strategies/enums.py` — `DecumulationStrategy.ANNUITY = "ANNUITY"`.
- `src/pipeline/decumulation.py` — dispatch ANNUITY dans `_build_decumulation_strategy`
  (retourne `(None, "ANNUITY")`) et `_run_one_decumulation` (appelle
  `run_annuity_decumulation` au lieu de `run_decumulation`).

**Tests**
- `annuity_factor_monthly(60, 0.01, 0.05) ≈ 258` (17.3 ans de paiements actualisés).
- Mensualité linéaire en capital : 1M EUR → 3 870 EUR/mois nominal.
- `proba_ruine = 0.0`, `capital_residuel = 0.0` (par construction).
- Run complet en mode `STRATEGIE_DECUMULATION = "ANNUITY"` : 4 décums
  (une par accum), résultats cohérents (taux de remplacement nominal ~80%).
- Run nominal `STRATEGIE_DECUMULATION = "TARGET_DATE"` bit-identique hors timings.

## T8 — Plots ALM : scénario représentatif (sim-médoïde L2)
**Problème signalé** : en mode comparaison, les trajectoires "médianes"
des 4 stratégies convergeaient toutes en fin de simulation (capitaux P50
à ~1M EUR ±5%). Cause : sélection d'une sim unique sur sa valeur finale
(`idx_p50 = argsort(final)[500]`) → biais d'endpoint mécanique.

**Premier fix écarté** : percentile P50 recalculé à chaque pas (courbe
lissée). Corrige le biais mais efface la volatilité réelle du marché —
l'utilisateur voyait "des droites".

**Fix retenu** : scénario représentatif (sim-médoïde L2). Pour chaque
matrice de trajectoires, on sélectionne la simulation dont la trajectoire
minimise la distance L2 à la trajectoire P50. Référence ALM : scénarios
CTE, LDI, Solvency II. La sim choisie est un vrai chemin de marché
(crises, rebonds conservés) ET statistiquement central.

**Vérification numérique** (1000 sims, 481 pas, log-rend σ=0.04) :
- Volatilité log-rend sim représentative : 0.0416 (≈ σ réel)
- Volatilité log-rend P50 lissée         : 0.0070 (artificiellement lisse, 6× plus faible)
- Valeur finale sim / P50                : 150 497 / 146 918 (centralité préservée)

**Fichier modifié**
- `src/analytics/plotting.py` — ajout helper `_representative_sim_index(mat)`.
  6 fonctions mises à jour (`plot_capital`, `plot_zoom_crise_capital`,
  `plot_zoom_crise_rendements`, `plot_retraite_capital`,
  `plot_taux_remplacement`, `plot_comparaison_capital`). Enveloppe P5-P95
  conservée en fill/lignes fines, trajectoire centrale = sim représentative
  avec label "Scénario représentatif".

**Test** : `python main.py` passe ; 4 accum × 4 décum ; les plots affichent
désormais des chemins avec dynamique réelle (volatilité, drawdowns)
différents entre stratégies et sans convergence mécanique en fin de run.

## Invariants respectés sur l'ensemble
- Tous les paramètres client restent dans `config/settings.py`.
- Aucun raccourci ni version simplifiée : pipeline complet conservé.
- Pas de workaround : les root causes (seed global, trajectoires séparées,
  `dernier_contexte` implicite, fallback silencieux) ont été corrigées.
- Bit-identité vérifiée sur T1, T2 (même seed), T4, T6, T7 (branche par défaut).
- `python main.py` réussit en mode comparaison à la fin de chaque tâche.
- Zéro commit git : toutes les modifications sont locales.
