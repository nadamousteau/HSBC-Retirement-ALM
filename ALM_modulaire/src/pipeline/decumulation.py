"""
PIPELINE — PHASE DE DÉCUMULATION
==================================
Encapsule la boucle de décumulation (section 7 de l'ancien main.py).

Tâche 5 : la décumulation peut tourner par stratégie d'accumulation.
Lorsque `settings.DECUMULATION_PAR_STRATEGIE_ACCUM = True`, chaque contexte
d'accumulation fournit son propre capital de départ et sa propre exécution
de décumulation. Sinon, comportement T1 (capital de départ = dernière
stratégie du dict).

Les scénarios de marché et d'inflation de la phase retraite sont identiques
pour toutes les stratégies d'accumulation (slice unifié de la trajectoire
produite en T3), donc la stratégie de décumulation et son éventuel arbre
Faleh sont construits une seule fois en amont de la boucle.
"""

from config import settings
from src import strategies
from src.engine.decumulation_core import run_decumulation, print_decumulation_report
from src.analytics import plotting


def _build_decumulation_strategy(market_params, dates_decum, rng_bundle, liability):
    """
    Construit la stratégie de décumulation demandée par `settings`.

    Pré-condition (T6) : `settings.STRATEGIE_DECUMULATION` a déjà été
    validée par `strategies.validate_settings()` comme appartenant à
    `DecumulationStrategy`. Aucun fallback silencieux n'est plus effectué :
    si une branche manque ici, on lève explicitement.

    Cas `ANNUITY` : retourne `(None, "ANNUITY")`. La rente viagère n'utilise
    pas de stratégie d'allocation — son traitement est court-circuité en
    amont par `_run_one_decumulation`.

    `liability` est injecté dans FalehStrategy (Vague 2 / Tâche C) pour
    fixer target_wealth. Les autres stratégies l'ignorent.
    """
    mu_e, sigma_e, mu_b, sigma_b, corr_eb = market_params
    profil_decum = settings.PROFILS[settings.PROFIL_DECUMULATION]
    strat_decum_name = settings.STRATEGIE_DECUMULATION

    if strat_decum_name == "TARGET_DATE":
        return strategies.TargetDateStrategy(), strat_decum_name

    if strat_decum_name == "FIXED_MIX":
        return strategies.FixedMixStrategy(
            target_equity_pct=profil_decum["fixed_allocation"]
        ), strat_decum_name

    if strat_decum_name == "FALEH":
        from src.strategies.faleh_strategy import FalehStrategy
        rng_faleh = rng_bundle["faleh_gse"] if rng_bundle is not None else None
        strat = FalehStrategy(
            mu_e, sigma_e, mu_b, sigma_b, corr_eb,
            liability=liability, rng=rng_faleh,
        )
        strat.initialize_tree(dates_decum)
        return strat, strat_decum_name

    if strat_decum_name == "ANNUITY":
        return None, strat_decum_name

    raise NotImplementedError(
        f"Stratégie de décumulation '{strat_decum_name}' non implémentée "
        f"dans _build_decumulation_strategy."
    )


def _run_one_decumulation(strategy_decum, strat_decum_name, strat_accum_name,
                          ctx_accum, r_eq_decum, r_bd_decum, inflation_decum,
                          liability):
    """
    Exécute une décumulation à partir d'un contexte d'accumulation donné,
    puis déclenche reporting console et plots selon les flags settings.

    Dispatch :
        - ANNUITY → `run_annuity_decumulation` (pas de marché, rente viagère)
        - autres  → `run_decumulation` (stratégie d'allocation + retraits)

    `liability` (Vague 2 / Tâche C) est passé à `run_decumulation` qui en
    extrait le retrait plancher et le taux d'actualisation.
    """
    capital_depart = ctx_accum["mat_cap"][-1, :]
    hist_salaire = ctx_accum["hist_salaire"]
    dernier_salaire = hist_salaire[-1] if len(hist_salaire) > 0 else None

    if strat_decum_name == "ANNUITY":
        from src.strategies.annuity_strategy import run_annuity_decumulation
        age_retraite = settings.AGE_DEPART + settings.NB_ANNEES_ACCUMULATION
        nb_mois_decum = settings.NB_ANNEES_DECUMULATION * 12
        result = run_annuity_decumulation(
            capital_initial_par_sim=capital_depart,
            inflation=inflation_decum,
            age_retraite_years=age_retraite,
            rate_annual=settings.ANNUITY_TECHNICAL_RATE,
            loading=settings.ANNUITY_INSURER_LOADING,
            nb_mois=nb_mois_decum,
            dernier_salaire_mensuel=dernier_salaire,
        )
    else:
        result = run_decumulation(
            strategy=strategy_decum,
            r_eq=r_eq_decum,
            r_bd=r_bd_decum,
            capital_initial_par_sim=capital_depart,
            inflation=inflation_decum,
            liability=liability,
            dernier_salaire_mensuel=dernier_salaire,
        )

    if getattr(settings, 'PRINT_DECUMULATION_METRICS', False):
        print_decumulation_report(
            result["metrics"],
            strategy_name=f"{strat_accum_name} → {strat_decum_name} / Profil {settings.PROFIL_DECUMULATION}"
        )

    plot_strategy_name = f"{strat_accum_name} → {strat_decum_name}"
    if getattr(settings, 'PLOT_DECUMULATION_CAPITAL', False):
        plotting.plot_retraite_capital(
            result["mat_capital"], strategy_name=plot_strategy_name, reel=False,
            inflation_factor=result["inflation_factor"]
        )
    if getattr(settings, 'PLOT_DECUMULATION_CAPITAL_REEL', False):
        plotting.plot_retraite_capital(
            result["mat_capital"], strategy_name=plot_strategy_name, reel=True,
            inflation_factor=result["inflation_factor"]
        )

    result["strat_decum_name"] = strat_decum_name
    result["strat_accum_name"] = strat_accum_name
    return result


def run_decumulation_phase(contextes_accum, economic_scenarios_decum,
                            inflation_decum, dates_decum, market_params,
                            rng_bundle=None, liability=None):
    """
    Exécute la phase de décumulation.

    Deux modes selon `settings.DECUMULATION_PAR_STRATEGIE_ACCUM` :
        - True  : une décumulation par stratégie d'accumulation, chacune
                  partant de son propre capital final.
        - False : une seule décumulation, basée sur la dernière stratégie
                  d'accumulation du dict (comportement T1).

    Args:
        contextes_accum          : dict {strat_name: contexte_strat} produit
                                   par run_accumulation().
        economic_scenarios_decum : tuple (r_eq_decum, r_bd_decum).
        inflation_decum          : ndarray (nb_mois_decum, nb_sims).
        dates_decum              : DatetimeIndex de la phase retraite.
        market_params            : tuple (mu_e, sigma_e, mu_b, sigma_b,
                                   corr_eb) — nécessaire à Faleh.
        rng_bundle               : dict produit par `make_rng_bundle()` —
                                   fournit le Generator Faleh.
        liability                : `RetirementLiability` — passif client
                                   (Vague 2 / Tâche C). Injecté dans
                                   FalehStrategy ET dans `run_decumulation`
                                   pour fixer retrait plancher et taux
                                   d'actualisation.

    Returns:
        dict {strat_accum_name: result_decum} — un résultat par stratégie
        d'accumulation exécutée (ou un unique résultat si
        DECUMULATION_PAR_STRATEGIE_ACCUM = False).
    """
    if not contextes_accum:
        return None
    if liability is None:
        raise ValueError(
            "run_decumulation_phase : argument `liability` requis "
            "(cf. src/liabilities/retirement_objective.py)."
        )

    r_eq_decum, r_bd_decum = economic_scenarios_decum

    strategy_decum, strat_decum_name = _build_decumulation_strategy(
        market_params, dates_decum, rng_bundle, liability
    )

    print(f"DÉCUMULATION : Stratégie={strat_decum_name}, "
          f"Profil={settings.PROFIL_DECUMULATION}, "
          f"Retrait={liability.target_income_monthly:.0f}€/mois ({liability.income_mode}), "
          f"Horizon={settings.NB_ANNEES_DECUMULATION}ans "
          f"(passif={liability.expected_duration_years():.1f}ans, mode={liability.horizon_mode}), "
          f"Distribution surplus={settings.TAUX_DISTRIBUTION_SURPLUS*100:.0f}%")

    par_strategie = getattr(settings, 'DECUMULATION_PAR_STRATEGIE_ACCUM', False)

    if par_strategie:
        results = {}
        for strat_accum_name, ctx in contextes_accum.items():
            print(f"\n── Décumulation à partir de l'accumulation : {strat_accum_name} ──")
            results[strat_accum_name] = _run_one_decumulation(
                strategy_decum, strat_decum_name, strat_accum_name, ctx,
                r_eq_decum, r_bd_decum, inflation_decum, liability,
            )
        return results

    last_strat_name = next(reversed(contextes_accum))
    last_ctx = contextes_accum[last_strat_name]
    return {
        last_strat_name: _run_one_decumulation(
            strategy_decum, strat_decum_name, last_strat_name, last_ctx,
            r_eq_decum, r_bd_decum, inflation_decum, liability,
        )
    }
