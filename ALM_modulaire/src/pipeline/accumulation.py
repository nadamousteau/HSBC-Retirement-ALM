"""
PIPELINE — PHASE D'ACCUMULATION
================================
Orchestration de la boucle Monte Carlo pour chaque stratégie d'accumulation.

Extrait de main.py (sections 3 / 4 / 4.5) pour rendre main() purement
orchestrateur. Aucun changement de logique : les prints, l'ordre d'appel aux
moteurs et les mutations globales (settings.METHODE) sont préservés à
l'identique pour garantir la bit-identité des sorties.
"""

import numpy as np

from config import settings, profiles
from src import engine, strategies, analytics


def run_accumulation(strategies_run, economic_scenarios, market_params,
                     gpi, gbi_tensor, dates, idx_split, inflation_tensor,
                     rng_bundle=None):
    """
    Exécute la boucle d'accumulation pour chaque stratégie de la liste.

    Args:
        strategies_run     : list[str] — stratégies à évaluer
                             (ex. ["GBI", "FALEH", "FIXED_MIX", "TARGET_DATE"])
        economic_scenarios : tuple (r_eq, r_bd) — rendements mensuels
                             (nb_periodes, nb_sims)
        market_params      : tuple (mu_e, sigma_e, mu_b, sigma_b, corr_eb) —
                             paramètres annuels issus du loader ; nécessaires
                             pour instancier FalehStrategy.
        gpi                : GoalPriceIndex ou None — utilisé uniquement si
                             "GBI" est dans strategies_run
        gbi_tensor         : ndarray (nb_sims, nb_forecast, 360) ou None
        dates              : DatetimeIndex — dates mensuelles de l'accumulation
        idx_split          : int — indice backtest/forecast
        inflation_tensor   : ndarray (nb_periodes, nb_sims) — taux mensuels
        rng_bundle         : dict produit par `make_rng_bundle()` — fournit
                             les Generator dédiés (notamment "faleh_gse").

    Returns:
        dict {strat_name: contexte_strat} où contexte_strat contient
            mat_cap, courbe_investi, hist_apport, hist_salaire, hist_dd,
            inflation_factor, capitaux_finaux, total_investi, kpis,
            tri_median, capital_p5_reel, gain_p5_reel.
    """
    r_eq, r_bd = economic_scenarios
    mu_e, sigma_e, mu_b, sigma_b, corr_eb = market_params

    contextes_par_strat = {}

    for strat_actuelle in strategies_run:
        # Écrasement local pour garantir l'aiguillage dans les modules sous-jacents
        settings.METHODE = strat_actuelle

        # ── Dispatch vers le bon moteur ──────────────────────────────────────
        if strat_actuelle == "GBI":
            mat_cap, courbe_investi, hist_apport, hist_dd, hist_salaire, _, inflation_factor = engine.run_simulation_gbi(
                gpi, gbi_tensor, r_eq, r_bd, dates, idx_split, inflation=inflation_tensor
            )
        else:
            if strat_actuelle == "TARGET_DATE":
                strategy = strategies.TargetDateStrategy()
            elif strat_actuelle == "FIXED_MIX":
                strategy = strategies.FixedMixStrategy(target_equity_pct=profiles.fixed_allocation)
            elif strat_actuelle == "FALEH":
                from src.strategies.faleh_strategy import FalehStrategy
                rng_faleh = rng_bundle["faleh_gse"] if rng_bundle is not None else None
                strategy = FalehStrategy(mu_e, sigma_e, mu_b, sigma_b, corr_eb, rng=rng_faleh)
                strategy.initialize_tree(dates)
            else:
                raise NotImplementedError(
                    f"Stratégie d'accumulation '{strat_actuelle}' non implémentée "
                    f"dans run_accumulation."
                )

            # Exécution du moteur standard (non-GBI)
            mat_cap, courbe_investi, hist_apport, hist_dd, hist_salaire, inflation_factor = engine.run_simulation(
                strategy, r_eq, r_bd, dates, inflation=inflation_tensor
            )

        # =====================================================================
        # 4. POST-TRAITEMENT & ANALYTICS
        # =====================================================================
        capitaux_finaux = mat_cap[-1, :]
        total_investi = courbe_investi[-1]

        idx_sorted = np.argsort(capitaux_finaux)
        idx_p50 = idx_sorted[int(settings.NB_SIMULATIONS * 0.50)]

        # Metrics accumulation
        tri_median = analytics.metrics.calculer_tri_annualise(
            settings.CAPITAL_INITIAL, hist_apport, capitaux_finaux[idx_p50]
        )
        kpis = analytics.metrics.calcul_kpi_complets(capitaux_finaux, total_investi, mat_cap)

        # Capital réel en P5 (prend en compte l'inflation stochastique)
        idx_p5 = idx_sorted[int(settings.NB_SIMULATIONS * 0.05)]
        capital_p5_reel = capitaux_finaux[idx_p5] / inflation_factor[-1, idx_p5]
        gain_p5_reel = capital_p5_reel - total_investi

        contextes_par_strat[strat_actuelle] = {
            "mat_cap": mat_cap,
            "courbe_investi": courbe_investi,
            "hist_apport": hist_apport,
            "hist_salaire": hist_salaire,
            "hist_dd": hist_dd,
            "inflation_factor": inflation_factor,
            "capitaux_finaux": capitaux_finaux,
            "total_investi": total_investi,
            "kpis": kpis,
            "tri_median": tri_median,
            "capital_p5_reel": capital_p5_reel,
            "gain_p5_reel": gain_p5_reel,
        }

        # =====================================================================
        # 4.5. REPORTING CONSOLE (par stratégie)
        # =====================================================================
        if getattr(settings, 'PRINT_PERFORMANCE_GLOBALE', False) or getattr(settings, 'PRINT_METRIQUES_RISQUE', False):
            print("\n" + "=" * 80)
            print(f"ANALYSE QUANTITATIVE - STRATÉGIE : {strat_actuelle}")
            print("=" * 80)

            if getattr(settings, 'PRINT_PERFORMANCE_GLOBALE', False):
                print("\n[ PERFORMANCE GLOBALE ]")
                print(f"  • TRI médian                : {tri_median:>15.2f} %/an")
                print(f"  • Dispersion (P95-P5)       : {kpis['dispersion']:>15,.0f} €")

            if getattr(settings, 'PRINT_METRIQUES_RISQUE', False):
                max_dd_median = np.median([np.min(hist_dd[:, sim]) for sim in range(settings.NB_SIMULATIONS)])
                print("\n[ RISQUE & DOWNSIDE ]")
                print(f"  • Shortfall Risk (< Capital): {kpis['shortfall_prob']*100:>15.2f} %")
                print(f"  • VaR 95% (Capital P5 nominal): {kpis['var_95']:>15,.0f} €")
                print(f"  • Max Drawdown médian       : {max_dd_median*100:>15.2f} %")
                print(f"  • Max Underwater            : {kpis['max_underwater']:>15.1f} années")
                print(f"  • Sortino Ratio             : {kpis['sortino']:>15.2f}")

                if gain_p5_reel < 0:
                    print(f"\n[ ALERTE INFLATION ]")
                    print(f"  • P&L réel P5 (Worst Case)  : {gain_p5_reel:>+16,.0f} € (Destruction de pouvoir d'achat)")
            print("=" * 80 + "\n")

    return contextes_par_strat
