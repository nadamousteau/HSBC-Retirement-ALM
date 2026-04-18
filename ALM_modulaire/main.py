#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Importation des modules du projet
from config import settings, profiles
from data import loader
from src import economics
from src.economics.yield_curve import YieldCurveBuilder
from src.liabilities.goal_price_index import GoalPriceIndex
from src.economics.nelson_siegel_var import simulate_gbi_monte_carlo
from src.economics.generators import (
    generer_inflation_vasicek,
    generer_rendements_backtest,
    generer_rendements_forecast,
    generer_scenarios_marche_correles,
)
from src.pipeline import run_accumulation, run_reporting, run_decumulation_phase
from src.strategies import validate_settings as validate_strategy_settings
from src.utils import make_rng_bundle
from src.liabilities import build_liability_from_settings


def main():
    # =========================================================================
    # 1. INITIALISATION & CONFIGURATION
    # =========================================================================
    validate_strategy_settings(settings)

    mu_e, sigma_e, mu_b, sigma_b, corr_eb = loader.load_market_parameters()
    market_params = (mu_e, sigma_e, mu_b, sigma_b, corr_eb)

    # Bundle RNG dérivé de GLOBAL_SEED — un Generator par composant stochastique.
    rng_bundle = make_rng_bundle(getattr(settings, "GLOBAL_SEED", None))

    # Passif client (Vague 2 / Tâche C). Source unique de vérité pour
    # le revenu cible, l'horizon et l'actualisation. Consommé par
    # FalehStrategy (target_wealth), GBI (goal_amount diagnostic) et
    # decumulation_core (retrait plancher + taux d'actualisation).
    liability = build_liability_from_settings(settings)
    print(f"PASSIF CLIENT : "
          f"revenu={liability.target_income_monthly:.0f}€/mois ({liability.income_mode}), "
          f"horizon={liability.expected_duration_years():.1f}ans ({liability.horizon_mode}), "
          f"discount={liability.discount_rate*100:.2f}%, "
          f"VA_retraite={liability.required_capital_at_retirement():,.0f}€")

    # Bornes temporelles : la trajectoire économique est générée en une seule
    # passe sur l'horizon total (accum + décum), puis slicée pour chaque phase.
    nb_mois_accum = settings.NB_ANNEES_ACCUMULATION * 12
    nb_mois_decum = settings.NB_ANNEES_DECUMULATION * 12
    nb_mois_total = settings.NB_MOIS_TOTAL_PIPELINE
    slice_accum = slice(0, nb_mois_accum)
    slice_decum = slice(nb_mois_accum, nb_mois_total)

    # =========================================================================
    # 2. GÉNÉRATION DES SCÉNARIOS ÉCONOMIQUES (ESG) - 1 TRAJECTOIRE UNIFIÉE
    # =========================================================================
    # Architecture (Vague 2 — Tâche B) :
    #   • Backtest (mois 0..idx_split)  : rendements historiques (CSV) +
    #                                     inflation Vasicek INDÉPENDANTE.
    #   • Forecast (mois idx_split..N)  : tirage JOINT EQ/BD/inflation via
    #                                     matrice de covariance 3×3
    #                                     (rho_bond_infl = INFLATION_BONDS_CORRELATION).
    # La continuité de l'inflation au pivot est assurée en passant
    # `inflation_init = inflation_bt[-1, :]` (vecteur per-sim) au générateur
    # corrélé, qui démarre la chaîne Vasicek forecast à cette valeur.
    dates_full = pd.date_range(
        start=settings.DATE_DEBUT_T0, periods=nb_mois_total, freq='ME'
    )
    dates_accum = dates_full[slice_accum]
    dates_decum = dates_full[slice_decum]

    # --- Backtest : rendements EQ/BD historiques (path déterministe) ---------
    r_eq_bt, r_bd_bt, idx_split = generer_rendements_backtest(
        mu_e, sigma_e, mu_b, sigma_b, corr_eb,
        dates_full, settings.DATE_PIVOT_BACKTEST, settings.NB_SIMULATIONS,
        asset_equity=profiles.Equity, asset_bond=profiles.Bond,
        rng=rng_bundle["equity_bonds"],
    )

    # --- Backtest : inflation Vasicek indépendante ----------------------------
    inflation_stochastique = getattr(settings, 'INFLATION_STOCHASTIQUE', False)
    if inflation_stochastique:
        if idx_split > 0:
            inflation_bt = generer_inflation_vasicek(
                nb_periodes=idx_split,
                nb_sims=settings.NB_SIMULATIONS,
                kappa=getattr(settings, 'INFLATION_KAPPA', 0.15),
                theta=getattr(settings, 'INFLATION_THETA', 0.023),
                sigma=getattr(settings, 'INFLATION_SIGMA', 0.011),
                rng=rng_bundle["inflation"],
            )
        else:
            inflation_bt = np.empty((0, settings.NB_SIMULATIONS))
    else:
        inflation_bt = np.full(
            (idx_split, settings.NB_SIMULATIONS), settings.TAUX_INFLATION
        )

    # --- Forecast : tirage JOINT EQ/BD/inflation via Cholesky 3×3 -----------
    nb_mois_forecast = nb_mois_total - idx_split
    rho_bi = getattr(settings, 'INFLATION_BONDS_CORRELATION', -0.30)
    print(f"Phase forecast : {nb_mois_forecast} mois, "
          f"INFLATION_BONDS_CORRELATION = {rho_bi:+.3f}")

    if nb_mois_forecast > 0:
        if inflation_stochastique:
            # Continuité : la chaîne Vasicek forecast démarre au dernier état
            # backtest (per-sim) pour qu'aucun saut artificiel ne soit injecté
            # au pivot. Si le backtest est vide (idx_split == 0), on laisse
            # generer_scenarios_marche_correles initialiser à theta_m.
            infl_init_fc = inflation_bt[-1, :] if idx_split > 0 else None
            r_eq_fc, r_bd_fc, inflation_fc = generer_scenarios_marche_correles(
                mu_e, sigma_e, mu_b, sigma_b, corr_eb,
                getattr(settings, 'INFLATION_KAPPA', 0.15),
                getattr(settings, 'INFLATION_THETA', 0.023),
                getattr(settings, 'INFLATION_SIGMA', 0.011),
                rho_bi,
                nb_mois_forecast, settings.NB_SIMULATIONS,
                rng=rng_bundle["market_correlated"],
                inflation_init=infl_init_fc,
            )
        else:
            # Inflation constante : pas de besoin de tirage joint, EQ/BD
            # divergents standards.
            r_eq_fc, r_bd_fc = generer_rendements_forecast(
                mu_e, sigma_e, mu_b, sigma_b, corr_eb,
                nb_mois_forecast, settings.NB_SIMULATIONS,
                rng=rng_bundle["market_correlated"],
            )
            inflation_fc = np.full(
                (nb_mois_forecast, settings.NB_SIMULATIONS),
                settings.TAUX_INFLATION,
            )
    else:
        r_eq_fc = np.empty((0, settings.NB_SIMULATIONS))
        r_bd_fc = np.empty((0, settings.NB_SIMULATIONS))
        inflation_fc = np.empty((0, settings.NB_SIMULATIONS))

    # --- Concaténation backtest + forecast -----------------------------------
    r_eq_full = np.vstack([r_eq_bt, r_eq_fc])
    r_bd_full = np.vstack([r_bd_bt, r_bd_fc])
    inflation_full = np.vstack([inflation_bt, inflation_fc])

    if inflation_stochastique:
        print(f"Inflation Vasicek générée : shape={inflation_full.shape}, "
              f"μ={np.mean(inflation_full)*100:.2f}%, σ={np.std(inflation_full)*100:.2f}%")
    else:
        print(f" Inflation constante : {settings.TAUX_INFLATION*100:.2f}%")

    # Injection des chocs (indépendant des stratégies)
    if getattr(settings, 'SIMULER_CRISE_MERTON', False):
        r_eq_full, r_bd_full = economics.shocks.ajouter_chocs_merton(
            r_eq_full, r_bd_full, nb_mois_total, settings.NB_SIMULATIONS,
            rng=rng_bundle["equity_bonds"],
        )

    if getattr(settings, 'SIMULER_CRISE_LOCALISEE', False):
        date_crise_ts = pd.Timestamp(settings.DATE_CRISE)
        date_pivot_ts = pd.Timestamp(settings.DATE_PIVOT_BACKTEST)

        if date_crise_ts > date_pivot_ts:
            r_eq_full, r_bd_full = economics.shocks.injecter_crise_localisee(
                r_eq_full, r_bd_full, dates_full, settings.DATE_CRISE, settings.PARAMS_CRISE_DETAIL
            )
        else:
            print(f"ATTENTION : La date de crise ({settings.DATE_CRISE}) précède ou égale la date pivot ({settings.DATE_PIVOT_BACKTEST}). Le choc a été ignoré.")

    # Slices accum / décum — continuité temporelle garantie (même stream).
    r_eq_accum = r_eq_full[slice_accum]
    r_bd_accum = r_bd_full[slice_accum]
    r_eq_decum = r_eq_full[slice_decum]
    r_bd_decum = r_bd_full[slice_decum]
    inflation_accum = inflation_full[slice_accum]
    inflation_decum = inflation_full[slice_decum]

    # =========================================================================
    # 2b. INITIALISATION GBI (historique + Nelson-Siegel VAR(1) forecast)
    # =========================================================================
    gpi = None
    gbi_tensor_accum = None

    mode_comparaison = getattr(settings, 'MODE_COMPARAISON', False)
    strategies_run = settings.STRATEGIES_A_COMPARER if mode_comparaison else [settings.METHODE_DEFAUT]

    if "GBI" in strategies_run:
        # Courbe des taux historique pour le backtest
        yc = YieldCurveBuilder().load_from_csv(settings.CSV_YIELD_CURVE)
        gpi = GoalPriceIndex(
            yield_curve=yc,
            retirement_date=settings.DATE_RETRAITE_GBI
        )

        # Forecast NS-VAR(1) sur l'horizon COMPLET (accum + décum). GBI n'est
        # utilisé qu'en accumulation mais on génère sur le pipeline entier pour
        # garantir une continuité stochastique du stream NS.
        nb_forecast_full = max(0, nb_mois_total - max(idx_split, 1))
        if nb_forecast_full > 0:
            gbi_tensor_full, gbi_factors, gbi_tau_grid = simulate_gbi_monte_carlo(
                nb_sims=settings.NB_SIMULATIONS,
                nb_months=nb_forecast_full,
                rng=rng_bundle["gbi_ns"],
            )
            # Portion destinée à l'accumulation (idx_split..nb_mois_accum)
            nb_forecast_accum = max(0, nb_mois_accum - max(idx_split, 1))
            gbi_tensor_accum = gbi_tensor_full[:, :nb_forecast_accum, :]

        print(f"GBI : Courbe historique ({len(yc.dates)} obs) + "
              f"NS-VAR(1) forecast ({nb_forecast_full} mois, "
              f"{settings.NB_SIMULATIONS} sims x 360 maturites), "
              f"retraite {settings.DATE_RETRAITE_GBI}, "
              f"plancher {settings.FLOOR_PERCENT_GBI*100:.0f}%")

    # =========================================================================
    # 3-4. BOUCLE D'ÉVALUATION DES STRATÉGIES (ACCUMULATION)
    # =========================================================================
    contextes_par_strat = run_accumulation(
        strategies_run=strategies_run,
        economic_scenarios=(r_eq_accum, r_bd_accum),
        market_params=market_params,
        gpi=gpi,
        gbi_tensor=gbi_tensor_accum,
        dates=dates_accum,
        idx_split=idx_split,
        inflation_tensor=inflation_accum,
        rng_bundle=rng_bundle,
        liability=liability,
    )

    # =========================================================================
    # 5-6. VISUALISATION & SYNTHÈSE POST-ACCUMULATION
    # =========================================================================
    run_reporting(contextes_par_strat, dates_accum)

    # =========================================================================
    # 7. PHASE DE DÉCUMULATION (Retraite)
    # =========================================================================
    if getattr(settings, 'SIMULER_DECUMULATION', False):
        print(f"\nDÉCUMULATION : {nb_mois_decum} mois — slice continu de la trajectoire unifiée")

        run_decumulation_phase(
            contextes_accum=contextes_par_strat,
            economic_scenarios_decum=(r_eq_decum, r_bd_decum),
            inflation_decum=inflation_decum,
            dates_decum=dates_decum,
            market_params=market_params,
            rng_bundle=rng_bundle,
            liability=liability,
        )

    return contextes_par_strat


if __name__ == "__main__":
    main()
