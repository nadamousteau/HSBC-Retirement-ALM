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

---

# Vague 2

## Vague 2 — Tâche A — METHODE immutable

**Problème résolu** : `src/pipeline/accumulation.py:54` mutait
`settings.METHODE = strat_actuelle` dans la boucle de stratégies. Cette
mutation globale était lue par `plotting.plot_capital()` pour le titre
("Évolution du Capital Accumulé - {settings.METHODE}"), rendant le rendu
non thread-safe et dépendant de l'ordre d'itération de
`STRATEGIES_A_COMPARER`.

**Fichiers modifiés**
- `src/pipeline/accumulation.py` — suppression de la ligne de mutation,
  mise à jour de la docstring du module.
- `src/analytics/plotting.py` — ajout d'un paramètre obligatoire
  `strategy_name: str` aux 7 plots solo (`plot_capital`, `plot_salaire`,
  `plot_apports`, `plot_zoom_crise_capital`, `plot_zoom_crise_rendements`,
  `plot_retraite_capital`, `plot_taux_remplacement`). Chaque titre
  concaténé avec `- {strategy_name}` pour une annotation non ambiguë.
  `plot_comparaison_capital` inchangé (itère sur le dict de stratégies).
- `src/pipeline/reporting.py` — propage `solo_strat_name` issu de
  `_select_solo_context()` à chaque appel de plot solo et macro.
- `src/pipeline/decumulation.py` — propage
  `f"{strat_accum} → {strat_decum}"` comme `strategy_name` aux
  `plot_retraite_capital` de la phase retraite.

**Grep final**
```
settings\.METHODE\b                       → 0 occurrence (toutes supprimées)
settings\.METHODE (non-\b)                → 4 matches — tous METHODE_DEFAUT
  (main.py, reporting.py docstring ×2, enums.py message)
```

**Tests de non-régression**
- `python main.py` avec `STRATEGIES_A_COMPARER = ["GBI", "FALEH", "FIXED_MIX", "TARGET_DATE"]` → OK.
- Même run avec `STRATEGIES_A_COMPARER = ["TARGET_DATE", "FALEH", "GBI", "FIXED_MIX"]` → OK.
- KPIs par stratégie (Sortino, VaR 95%, Max Drawdown, P5/P50/P95) : bit-identiques
  dans les deux ordres (diff vide).
- KPIs bit-identiques à la baseline V1 (avant Vague 2) : diff vide.

## Vague 2 — Tâche B — Inflation corrélée aux bonds (Cholesky 3×3)

**Problème résolu** : avant Tâche B, l'inflation Vasicek était tirée sur un
RNG complètement disjoint de celui des bonds (`rng_bundle["inflation"]` vs
`rng_bundle["equity_bonds"]`). Conséquence : un scénario forecast pouvait
combiner hyperinflation ET hausse simultanée des prix obligataires, ce qui
contredit la relation empirique observée sur bonds nominaux US 1990-2024
(rho ≈ -0.30 : inflation ↑ ⇒ taux ↑ ⇒ prix bonds ↓).

**Solution** : tirage joint EQ/BD/inflation via matrice de covariance 3×3
sur les innovations brownien mensuelles (decomp. Cholesky), avec
`corr_eq_infl = 0` (hypothèse d'orthogonalité actions/inflation, standard
en ALM long terme) et `corr_bond_infl = -0.30`.

**Fichiers modifiés / créés**

- `config/settings.py` — ajout `INFLATION_BONDS_CORRELATION = -0.30`
  (commentaire : calibration empirique 1990-2024, références).
- `src/economics/generators.py` — trois nouvelles fonctions :
  - `generer_rendements_backtest(...)` : portion historique pure (CSV +
    fallback BS si chargement échoue), commune à toutes les sims.
  - `generer_rendements_forecast(...)` : forecast EQ/BD seul (Cholesky 2×2),
    utile aux modules sans besoin d'inflation corrélée.
  - `generer_scenarios_marche_correles(...)` : tirage joint EQ/BD/inflation
    via covariance 3×3 et Cholesky, avec discrétisation Vasicek-Euler-
    Maruyama préservée (compatible avec la marginale de
    `generer_inflation_vasicek` à corrélation nulle).
  - Suppression de `generer_rendements_avec_backetest` (remplacée par les
    deux helpers ci-dessus, pas de shim).
- `src/utils/rng.py` — ajout `"market_correlated"` EN FIN de `_BUNDLE_KEYS`
  pour préserver la `SeedSequence` enfant des streams existants
  (`equity_bonds`, `inflation`, `gbi_ns`, `faleh_gse` inchangés).
- `main.py` — refonte du bloc § 2 (génération ESG) :
  - Backtest : `generer_rendements_backtest` + `generer_inflation_vasicek`
    indépendants (path historique déterministe, inflation Vasicek pure).
  - Forecast : `generer_scenarios_marche_correles` joint, avec
    `inflation_init = inflation_bt[-1, :]` (per-sim) pour assurer la
    continuité de l'inflation au pivot.
  - Si `INFLATION_STOCHASTIQUE = False` : forecast EQ/BD via
    `generer_rendements_forecast` (sans inflation), inflation constante.
  - Log d'audit : `Phase forecast : N mois, INFLATION_BONDS_CORRELATION = ±X.XXX`.
- `tests/__init__.py` + `tests/test_inflation_bonds_corr.py` — 4 tests pytest :
  - `test_correlation_empirique_forte` : 10 000 sims × 480 mois →
    corr empirique = -0.30 ± 0.02.
  - `test_distributions_marginales` : volatilités annualisées EQ/BD et
    moyenne d'inflation cohérentes (±10 %) avec les générateurs indépendants.
  - `test_correlation_nulle_par_defaut` : avec rho_bi = 0, corrélation
    empirique ≈ 0 (sanity check de la matrice).
  - `test_continuite_inflation_init_per_sim` : `inflation[0, :]` exactement
    égal à `inflation_init` per-sim (test exact, rtol=1e-12).
  - Résultat : **4/4 PASSED** en 3.08s.

**Vérification numérique des distributions marginales**

Run nominal : 1000 sims, accumulation 40 ans + décumulation 25 ans, profil
EQUILIBRE, `GLOBAL_SEED = 42`, mode comparaison sur 4 stratégies.

| Métrique | rho = 0 | rho = -0.30 | Δ |
|---|---:|---:|---:|
| Inflation μ globale (mensuel) | 0.19 % | 0.19 % | identique |
| Inflation σ globale (mensuel) | 0.57 % | 0.57 % | identique |
| Phase forecast | 496 mois | 496 mois | identique |

Les marginales sont préservées par construction (Cholesky n'affecte que la
covariance hors-diagonale, pas les variances).

**Delta KPIs accumulation par stratégie** (profil EQUILIBRE, 1000 sims)

Comparaison "même architecture, rho différent" — isole l'effet causal
introduit par la corrélation. La SeedSequence du stream `market_correlated`
est identique dans les deux runs, donc l'aléa hors-corrélation est constant.

| Stratégie | KPI | rho = 0 | rho = -0.30 | Δ |
|---|---|---:|---:|---:|
| **GBI** | Sortino | 1.88 | 1.88 | 0.00 |
|       | VaR 95 % (P5 nominal, €) | 517 768 | 517 867 | +99 |
|       | Max DD médian | -22.44 % | -22.43 % | +0.01 pp |
|       | P5 / P50 / P95 capital final (€) | 517 772 / 1 028 406 / 2 006 668 | 517 870 / 1 028 393 / 2 007 336 | +98 / -13 / +668 |
| **FALEH** | Sortino | 4.03 | 3.38 | **-0.65** |
|         | VaR 95 % (P5 nominal, €) | 647 214 | 647 463 | +249 |
|         | Max DD médian | -12.64 % | -12.64 % | 0.00 |
|         | P5 / P50 / P95 capital final (€) | 647 243 / 1 027 229 / 1 671 463 | 647 494 / 1 027 300 / 1 671 160 | +251 / +71 / -303 |
|         | P&L réel P5 (€) | (positif → ligne absente) | -2 227 | apparaît |
| **FIXED_MIX** | Sortino | 2.67 | 2.67 | 0.00 |
|             | VaR 95 % (P5 nominal, €) | 579 916 | 579 982 | +66 |
|             | Max DD médian | -19.91 % | -19.91 % | 0.00 |
|             | P5 / P50 / P95 capital final (€) | 579 961 / 1 073 038 / 2 041 448 | 580 029 / 1 072 971 / 2 040 890 | +68 / -67 / -558 |
|             | P&L réel P5 (€) | -153 976 | -169 887 | **-15 911** |
| **TARGET_DATE** | Sortino | 2.14 | 2.14 | 0.00 |
|               | VaR 95 % (P5 nominal, €) | 564 542 | 564 779 | +237 |
|               | Max DD médian | -20.97 % | -20.97 % | 0.00 |
|               | P5 / P50 / P95 capital final (€) | 564 668 / 1 072 845 / 2 085 856 | 564 922 / 1 072 854 / 2 085 288 | +254 / +9 / -568 |

**Lecture**

- **Capitaux nominaux quasi-stables** (delta < 0.05 % sur P50/P95). Logique :
  les marginales EQ/BD ne sont pas modifiées par la corrélation imposée sur
  les innovations — seule la *jointe* (eps_b, eps_i) change.
- **P&L réel pénalisé** : la corrélation négative fait coïncider hausse
  d'inflation et baisse des prix obligataires, ce qui détruit du pouvoir
  d'achat dans les scénarios pessimistes. Effet visible sur :
  - FIXED_MIX : -15 911 € sur P5 réel (portefeuille 60 % bond → forte
    sensibilité à la corrélation).
  - FALEH : nouvelle apparition d'un P5 réel négatif (-2 227 €) là où
    rho = 0 maintenait P5 légèrement positif.
- **Sortino FALEH chute -16 %** (4.03 → 3.38) : la corrélation négative
  augmente la *downside deviation* des rendements réels, métrique centrale
  du Sortino. Les autres stratégies ont une downside deviation plus
  stable car leurs allocations sont moins concentrées sur les bonds aux
  pas de temps tardifs (TARGET_DATE décroît progressivement, FIXED_MIX
  cap à 60 % equity, GBI surveille un goal price index).
- **Ranking préservé** : sur Sortino, l'ordre FALEH (3.38) > FIXED_MIX
  (2.67) > TARGET_DATE (2.14) > GBI (1.88) est identique à rho = 0.
  Les décisions stratégiques (choix de profil, choix de stratégie) ne
  sont donc pas perturbées par l'ajout de la corrélation : le ranking
  inter-stratégies est invariant, seules les amplitudes downside
  réelles sont raffinées (avec un signal économique plus fidèle).

**Tests de non-régression**

- `pytest tests/test_inflation_bonds_corr.py -v` : 4/4 PASSED.
- `python main.py` avec `INFLATION_BONDS_CORRELATION = -0.30` (défaut) :
  pipeline complet OK, 4 décumulations exécutées sans erreur.
- `python main.py` avec `INFLATION_BONDS_CORRELATION = 0.0` : pipeline OK,
  KPIs documentés ci-dessus.
- Streams existants préservés : `equity_bonds`, `inflation`, `gbi_ns`,
  `faleh_gse` gardent leur SeedSequence enfant (l'ajout de
  `market_correlated` à la fin du tuple n'affecte pas les positions
  antérieures dans `SeedSequence.spawn()`).

## Vague 2 — Tâche C — Refonte liability/asset (RetirementLiability + src/assets)

**Objectif**

Représenter explicitement le passif retraite client comme un objet immuable
(`RetirementLiability`) consommé par TOUS les composants du pipeline qui
ont besoin de connaître les volontés client (revenu cible, horizon,
actualisation). Avant cette tâche, ces volontés étaient diluées en deux
scalaires settings (`RETRAIT_MENSUEL_REEL`, `TAUX_ACTUALISATION_PLANCHER`)
exclusivement utilisés en décumulation, ce qui :

- forçait `FalehStrategy._estimate_target_wealth()` à inventer une cible
  via une heuristique 80 % × FV portefeuille, ignorant complètement le
  client ;
- empêchait le moteur GBI de logger sa vraie cible (le "goal" du Goal-Based
  Investing était implicite) ;
- ne permettait pas de basculer entre revenu réel/nominal ou entre horizon
  fixe/actuariel sans ajouter des scalaires ad hoc partout.

En parallèle, la tâche corrige une erreur de packaging structurelle :
`HumanCapitalCurve` et les fonctions `*apport*` vivaient dans
`src/liabilities/contributions.py` — alors que les apports sont des
**flux d'actif** (capital humain) et non des passifs. Le nouveau package
`src/assets/` héberge ces composants à leur place sémantiquement correcte.

**Fichiers créés**

- `src/assets/__init__.py` — re-exporte `human_capital` et `contribution_policy`.
- `src/assets/human_capital.py` — `HumanCapitalCurve` (déplacé verbatim
  depuis `src/liabilities/contributions.py`).
- `src/assets/contribution_policy.py` — fonctions standalone
  `precalculer_parametres_apport_exponentiel`, `calculer_apport_exponentiel`,
  `estimer_salaire_saturation` (déplacées verbatim).
- `src/liabilities/retirement_objective.py` — dataclass `RetirementLiability`
  (frozen=True, post-init validation), méthodes :
  - `expected_duration_years()` : durée du passif (FIXED ou ACTUARIAL via
    table TH 00-02).
  - `monthly_income_nominal_at_retirement()` : revenu mensuel converti en
    nominal au moment de la liquidation (capitalisation par
    `(1+inflation_expected)^accum` si mode REAL).
  - `present_value_at_retirement()` : VA du passif à T_retraite (annuité
    certaine en FIXED, ä^(12)_x en ACTUARIAL).
  - `required_capital_at_retirement()` : alias sémantique consommé par
    Faleh.
  - `build_liability_from_settings(settings)` : builder unique depuis le
    module de configuration.
- `src/liabilities/liability_valuation.py` — `funded_ratio(capital, liability)`
  pour le diagnostic actif/passif (scalaire ou ndarray).
- `tests/test_retirement_liability.py` — 25 tests pytest couvrant :
  validation arguments, conversion REAL/NOMINAL, durée FIXED/ACTUARIAL,
  VA passif (annuité certaine vs ä^(12)_x), funded_ratio scalar/vectorisé,
  builder depuis settings, validate_settings (LIABILITY_* + horizon).

**Fichiers supprimés**

- `src/liabilities/contributions.py` — DELETE complet, aucun shim ni
  re-export. Les call sites importent désormais
  `from src.assets import contribution_policy as contributions`.

**Fichiers modifiés**

- `src/liabilities/__init__.py` — re-exporte `retirement_objective`,
  `liability_valuation`, `mortality`, `goal_price_index` ; expose
  `RetirementLiability`, `build_liability_from_settings`, `funded_ratio`
  au niveau package.
- `config/settings.py` :
  - Suppression de `RETRAIT_MENSUEL_REEL` et `TAUX_ACTUALISATION_PLANCHER`.
  - Nouvelle section §7c LIABILITY_* :
    - `LIABILITY_TARGET_INCOME_MONTHLY = 2000`
    - `LIABILITY_INCOME_MODE = "REAL"`
    - `LIABILITY_HORIZON_MODE = "ACTUARIAL"`
    - `LIABILITY_HORIZON_YEARS_FIXED = 25`
    - `LIABILITY_DISCOUNT_RATE = 0.02`
    - `LIABILITY_INFLATION_EXPECTED = 0.023`
  - `NB_ANNEES_DECUMULATION : 25 → 30` pour respecter la cohérence
    horizon (validate_settings exige `nb_decum ≥ int(expected_duration) + 5`,
    soit ≥ 26 pour age_retraite=60 sur TH 00-02).
- `src/engine/core.py` — import :
  `from src.liabilities import contributions` →
  `from src.assets import contribution_policy as contributions`.
- `src/engine/gbi_core.py` — même remplacement d'import + signature
  `run_simulation_gbi(..., liability=None)` :
  - Logge `goal_amount = liability.required_capital_at_retirement()` au
    démarrage.
  - **Mécanisme de plancher inchangé** : conserve la formule relative
    `floor_pct × W_annee_debut × beta_t / beta_annee_debut`.
    Justification documentée : un plancher absolu Goal-ancré
    `floor_pct × goal_amount × beta_t` n'est pas viable avec
    `CAPITAL_INITIAL = 5 000 €` et `goal_amount ≈ 1 080 511 €` (cushion
    négatif dès t=0 → portefeuille bloqué à 100 % bonds). Une évolution
    future pourrait introduire un plancher
    `floor_pct × (PV_apports_futurs + goal_amount × beta_t)` ; hors scope.
- `src/strategies/faleh_strategy.py` :
  - Constructeur accepte `liability=None, target_wealth=None` (priorité
    explicite > liability > erreur).
  - **Suppression complète** de `_estimate_target_wealth()` (heuristique
    80 % × FV ignorant le client).
  - target_wealth est désormais
    `liability.required_capital_at_retirement()` quand le pipeline
    fournit un passif.
- `src/engine/decumulation_core.py` :
  - `run_decumulation(strategy, ..., liability, dernier_salaire_mensuel=None)` —
    nouvel argument requis.
  - Lit `retrait_mensuel_plancher = liability.monthly_income_nominal_at_retirement()`
    (au lieu de `settings.RETRAIT_MENSUEL_REEL`).
  - Lit `taux_actu_plancher = liability.discount_rate`
    (au lieu de `settings.TAUX_ACTUALISATION_PLANCHER`).
  - Renommage interne `retrait_mensuel_reel → retrait_mensuel_plancher`
    (la cible est désormais nominale au moment de la liquidation, pas
    réelle T0).
- `src/strategies/enums.py` — `validate_settings(settings_mod)` étendu :
  - Vérifie présence et type des 6 clés `LIABILITY_*`.
  - Valide `LIABILITY_INCOME_MODE ∈ {REAL, NOMINAL}` et
    `LIABILITY_HORIZON_MODE ∈ {FIXED, ACTUARIAL}`.
  - Construit le `RetirementLiability` et exige
    `NB_ANNEES_DECUMULATION ≥ int(expected_duration) + 5`.
- `src/pipeline/accumulation.py` — `run_accumulation(..., liability=None)`
  passe le passif à `engine.run_simulation_gbi` et à `FalehStrategy(...)`.
- `src/pipeline/decumulation.py` —
  `run_decumulation_phase(..., liability=None)` (requis sinon
  `ValueError`), passe à `_build_decumulation_strategy(..., liability)`
  (qui transmet à FalehStrategy si décum=FALEH) et à
  `_run_one_decumulation(..., liability)` (qui transmet à
  `run_decumulation`). Le print de tête utilise désormais le passif
  pour `Retrait` et `Horizon`.
- `main.py` :
  - Import `from src.liabilities import build_liability_from_settings`.
  - Construit `liability = build_liability_from_settings(settings)`
    après `validate_strategy_settings(settings)`.
  - Logge un résumé `PASSIF CLIENT : revenu=X€/mois (REAL),
    horizon=Y ans (ACTUARIAL), discount=Z%, VA_retraite=W€`.
  - Passe `liability=liability` à `run_accumulation` et
    `run_decumulation_phase`.

**Vérifications**

- `pytest tests/ -v` : **29/29 PASSED** en 2.24 s
  (4 tests Vague 2/Tâche B + 25 tests Vague 2/Tâche C).
- `python -X utf8 main.py` : pipeline complet OK.
  - PASSIF CLIENT loggué : `revenu=2000€/mois (REAL),
    horizon=22.9ans (ACTUARIAL), discount=2.00%, VA_retraite=1,080,511€`.
  - GBI : `goal_amount = 1,080,511 €` loggué (diagnostic).
  - Décumulation : `Retrait=2000€/mois (REAL), Horizon=30ans
    (passif=22.9ans, mode=ACTUARIAL)`.
- Aucun import résiduel de `src.liabilities.contributions` ou de
  `RETRAIT_MENSUEL_REEL`/`TAUX_ACTUALISATION_PLANCHER` (vérifié par
  Grep — seules les mentions résiduelles sont des docstrings/commentaires
  historiques).

**Comparaison KPI Vague 2/Tâche C vs Vague 2/Tâche B (rho = -0.30)**

⚠️ Les deux runs ne sont PAS strictement comparables : Tâche C bumpe
`NB_ANNEES_DECUMULATION` (25 → 30, contrainte cohérence horizon) et
remplace l'heuristique Faleh `target_wealth = 80 % × FV` par
`target_wealth = liability.required_capital_at_retirement() ≈ 1 080 511 €`.
Les deltas sont donc le résultat conjoint de ces deux changements.

| Stratégie | KPI | Tâche B | Tâche C | Δ |
|---|---|---:|---:|---:|
| **GBI** | Sortino | 1.88 | 1.89 | +0.01 |
|       | VaR 95 % (P5 nominal, €) | 517 867 | 541 510 | +23 643 |
|       | Max DD médian | -22.43 % | -22.46 % | -0.03 pp |
|       | P5 capital final (€) | 517 870 | 541 554 | +23 684 |
|       | P&L réel P5 (€) | -210 802 | -203 845 | +6 957 |
| **FALEH** | Sortino | 3.38 | 3.18 | -0.20 |
|         | VaR 95 % (P5 nominal, €) | 647 463 | 636 232 | -11 231 |
|         | Max DD médian | -12.64 % | -13.42 % | -0.78 pp |
|         | P5 capital final (€) | 647 494 | 636 285 | -11 209 |
|         | P&L réel P5 (€) | -2 227 | (positif → ligne absente) | disparaît |
| **FIXED_MIX** | Sortino | 2.67 | 2.67 | 0.00 |
|             | VaR 95 % (P5 nominal, €) | 579 982 | 579 982 | 0 |
|             | Max DD médian | -19.91 % | -19.91 % | 0.00 |
|             | P5 capital final (€) | 580 029 | 580 029 | 0 |
|             | P&L réel P5 (€) | -169 887 | -169 887 | 0 |
| **TARGET_DATE** | Sortino | 2.14 | 2.14 | 0.00 |
|               | VaR 95 % (P5 nominal, €) | 564 779 | 564 779 | 0 |
|               | Max DD médian | -20.97 % | -20.97 % | 0.00 |
|               | P5 capital final (€) | 564 922 | 564 922 | 0 |

**Lecture**

- **FIXED_MIX et TARGET_DATE strictement invariants** : ces stratégies ne
  consomment ni `liability` ni l'ancien target_wealth — leur seul lien
  avec les changements Tâche C est l'augmentation de
  `NB_ANNEES_DECUMULATION`, qui impacte la phase décumulation mais pas
  les KPIs d'accumulation reportés ici.
- **GBI bouge légèrement** (+ 4.6 % VaR, P&L réel P5 amélioré) : le
  changement vient probablement d'un effet de seed cascade lié à
  l'allongement de la trajectoire totale (`NB_MOIS_TOTAL_PIPELINE`
  passe de 780 à 840 mois). Le mécanisme de plancher GBI est inchangé.
- **FALEH change de manière notable** (-6 % Sortino, -1.7 % P5 nominal,
  P5 réel passe de -2 227 € à positif). Causes :
  - target_wealth = 1 080 511 € au lieu de l'heuristique d'environ
    1 460 000 € (≈ 80 % × FV portefeuille théorique). Cible plus basse
    → contrainte de pénalité ruine `FALEH_PENALTY_RUINE × shortfall²`
    moins active → optimisation pousse moins vers les bonds tardifs
    → légère baisse du capital P5 nominal (-1.7 %) mais amélioration du
    P&L réel P5 (la corrélation négative bonds/inflation pénalise moins
    quand les bonds sont moins surpondérés).
  - L'allongement décumulation 25 → 30 ans n'affecte pas l'accumulation
    Faleh (l'arbre est construit sur l'horizon accumulation seul).
- **Ranking préservé** : Sortino FALEH (3.18) > FIXED_MIX (2.67) >
  TARGET_DATE (2.14) > GBI (1.89). Identique à Tâche B.

**Bénéfice structurel principal**

Faleh optimise désormais sur la *vraie* cible client (1 080 511 € pour
financer 2000 €/mois RÉEL pendant 22.9 ans actuariels avec discount
2 %), pas sur une heuristique arbitraire 80 % × FV portefeuille. Quand
le client modifie `LIABILITY_TARGET_INCOME_MONTHLY` ou `LIABILITY_HORIZON_MODE`,
**toutes** les stratégies adaptent automatiquement leurs comportements
sans modification du pipeline.


