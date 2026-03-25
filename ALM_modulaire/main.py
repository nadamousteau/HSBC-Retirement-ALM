#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Importation des modules du projet
from config import settings, profiles
from data import loader
from src import economics, liabilities, strategies, engine, analytics
from src.analytics import plotting
from src.economics.yield_curve import YieldCurveBuilder
from src.liabilities.goal_price_index import GoalPriceIndex
from src.economics.nelson_siegel_var import simulate_gbi_monte_carlo
from src.economics.generators import generer_inflation_vasicek


def main():
    # =========================================================================
    # 1. INITIALISATION & CONFIGURATION
    # =========================================================================
    mu_e, sigma_e, mu_b, sigma_b, corr_eb = loader.load_market_parameters()

    # =========================================================================
    # 2. GÉNÉRATION DES SCÉNARIOS ÉCONOMIQUES (ESG) - EXÉCUTÉ UNE SEULE FOIS
    # =========================================================================
    dates = pd.date_range(start=settings.DATE_DEBUT_T0, periods=settings.NB_PERIODES_TOTAL, freq='ME')

    #Génération unifiée : Historique avant la date pivot, Stochastique après pour TOUTES les stratégies
    r_eq, r_bd, idx_split = economics.generators.generer_rendements_avec_backetest(
        mu_e, sigma_e, mu_b, sigma_b, corr_eb, dates, settings.DATE_PIVOT_BACKTEST, settings.NB_SIMULATIONS,
        asset_equity=profiles.Equity, asset_bond=profiles.Bond
    )
    
    # =========================================================================
    # 2a. GÉNÉRATION DE L'INFLATION VASICEK (Stochastique)
    # =========================================================================
    if getattr(settings, 'INFLATION_STOCHASTIQUE', False):
        inflation_tensor = generer_inflation_vasicek(
            nb_periodes=settings.NB_PERIODES_TOTAL,
            nb_sims=settings.NB_SIMULATIONS,
            kappa=getattr(settings, 'INFLATION_KAPPA', 0.15),
            theta=getattr(settings, 'INFLATION_THETA', 0.023),
            sigma=getattr(settings, 'INFLATION_SIGMA', 0.011),
            seed=getattr(settings, 'INFLATION_SEED', 42)
        )
        print(f"Inflation Vasicek générée : shape={inflation_tensor.shape}, "
              f"μ={np.mean(inflation_tensor)*100:.2f}%, σ={np.std(inflation_tensor)*100:.2f}%")
    else:
        # Inflation constante (fallback)
        inflation_tensor = np.ones((settings.NB_PERIODES_TOTAL, settings.NB_SIMULATIONS)) * settings.TAUX_INFLATION
        print(f" Inflation constante : {settings.TAUX_INFLATION*100:.2f}%")
    
    # Injection des chocs (Indépendant de la stratégie)
    if getattr(settings, 'SIMULER_CRISE_MERTON', False):
        r_eq, r_bd = economics.shocks.ajouter_chocs_merton(
            r_eq, r_bd, settings.NB_PERIODES_TOTAL, settings.NB_SIMULATIONS
        )

    if getattr(settings, 'SIMULER_CRISE_LOCALISEE', False):
        date_crise_ts = pd.Timestamp(settings.DATE_CRISE)
        date_pivot_ts = pd.Timestamp(settings.DATE_PIVOT_BACKTEST)

        # Vérification stricte unifiée : pas de choc déterministe dans le passé historique
        if date_crise_ts > date_pivot_ts:
            r_eq, r_bd = economics.shocks.injecter_crise_localisee(
                r_eq, r_bd, dates, settings.DATE_CRISE, settings.PARAMS_CRISE_DETAIL
            )
        else:
            print(f"ATTENTION : La date de crise ({settings.DATE_CRISE}) précède ou égale la date pivot ({settings.DATE_PIVOT_BACKTEST}). Le choc a été ignoré.")

    # =========================================================================
    # 2b. INITIALISATION GBI (historique + Nelson-Siegel VAR(1) forecast)
    # =========================================================================
    gpi = None
    gbi_tensor = None

    mode_comparaison = getattr(settings, 'MODE_COMPARAISON', False)
    strategies_run = settings.STRATEGIES_A_COMPARER if mode_comparaison else [settings.METHODE_DEFAUT]

    if "GBI" in strategies_run:
        # Courbe des taux historique pour le backtest
        yc = YieldCurveBuilder().load_from_csv(settings.CSV_YIELD_CURVE)
        gpi = GoalPriceIndex(
            yield_curve=yc,
            retirement_date=settings.DATE_RETRAITE_GBI
        )

        # Nelson-Siegel + VAR(1) pour le forecast uniquement
        nb_forecast = max(0, settings.NB_PERIODES_TOTAL - max(idx_split, 1))
        if nb_forecast > 0:
            gbi_tensor, gbi_factors, gbi_tau_grid = simulate_gbi_monte_carlo(
                nb_sims=settings.NB_SIMULATIONS,
                nb_months=nb_forecast,
                seed=getattr(settings, 'GBI_SEED', 42),
            )

        print(f"GBI : Courbe historique ({len(yc.dates)} obs) + "
              f"NS-VAR(1) forecast ({nb_forecast} mois, "
              f"{settings.NB_SIMULATIONS} sims x 360 maturites), "
              f"retraite {settings.DATE_RETRAITE_GBI}, "
              f"plancher {settings.FLOOR_PERCENT_GBI*100:.0f}%")

    # =========================================================================
    # 3. BOUCLE D'ÉVALUATION DES STRATÉGIES (ACCUMULATION)
    # =========================================================================
    resultats_comparaison = {}
    dernier_contexte = {}  # Stockage des données communes pour les graphiques post-boucle

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
                strategy = FalehStrategy(mu_e, sigma_e, mu_b, sigma_b, corr_eb)
                strategy.initialize_tree(dates)

        # Exécution du moteur
            mat_cap, courbe_investi, hist_apport, hist_dd, hist_salaire, inflation_factor = engine.run_simulation(
                strategy, r_eq, r_bd, dates, inflation=inflation_tensor
            )

        # =========================================================================
        # 4. POST-TRAITEMENT & ANALYTICS
        # =========================================================================
        capitaux_finaux = mat_cap[-1, :]
        total_investi = courbe_investi[-1]

        idx_sorted = np.argsort(capitaux_finaux)
        idx_p50 = idx_sorted[int(settings.NB_SIMULATIONS * 0.50)]

        #Metrics accumulation
        tri_median = analytics.metrics.calculer_tri_annualise(settings.CAPITAL_INITIAL, hist_apport, capitaux_finaux[idx_p50])
        kpis = analytics.metrics.calcul_kpi_complets(capitaux_finaux, total_investi, mat_cap)

        # Calcul du capital réel en P5 (prenant en compte l'inflation stochastique)
        idx_p5 = idx_sorted[int(settings.NB_SIMULATIONS * 0.05)]
        capital_p5_reel = capitaux_finaux[idx_p5] / inflation_factor[-1, idx_p5]
        gain_p5_reel = capital_p5_reel - total_investi

        # Sauvegarde des résultats
        resultats_comparaison[strat_actuelle] = mat_cap
        dernier_contexte = {
            "courbe_investi": courbe_investi, "hist_apport": hist_apport,
            "hist_salaire": hist_salaire, 
            "mat_cap": mat_cap,
            "inflation_factor": inflation_factor,
        }

        # =========================================================================
        # 4.5. REPORTING CONSOLE
        # =========================================================================
        if getattr(settings, 'PRINT_PERFORMANCE_GLOBALE', False) or getattr(settings, 'PRINT_METRIQUES_RISQUE', False):
            print("\n" + "="*80)
            print(f"ANALYSE QUANTITATIVE - STRATÉGIE : {strat_actuelle}")
            print("="*80)

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
            print("="*80 + "\n")

    # =========================================================================
    # 5. VISUALISATION ACCUMULATION (Exécution conditionnelle)
    # =========================================================================
    courbe_investi  = dernier_contexte["courbe_investi"]
    hist_salaire    = dernier_contexte["hist_salaire"]
    hist_apport     = dernier_contexte["hist_apport"]
    mat_cap         = dernier_contexte["mat_cap"]
    inflation_factor = dernier_contexte["inflation_factor"]

    # Graphiques d'accumulation (Comparatifs ou Isolés)
    if mode_comparaison:
        if getattr(settings, 'PLOT_COMPARAISON_CAPITAL', False):
            plotting.plot_comparaison_capital(dates, resultats_comparaison, reel=False)
        if getattr(settings, 'PLOT_COMPARAISON_CAPITAL_REEL', False):
            plotting.plot_comparaison_capital(dates, resultats_comparaison, reel=True, inflation_factor=inflation_factor)
    else:
        if getattr(settings, 'PLOT_CAPITAL', False):
            plotting.plot_capital(dates, mat_cap, courbe_investi, reel=False)
        if getattr(settings, 'PLOT_CAPITAL_REEL', False):
            plotting.plot_capital(dates, mat_cap, courbe_investi, reel=True, inflation_factor=inflation_factor)

    # Graphiques macro-économiques
    if getattr(settings, 'PLOT_SALAIRE', False):
        plotting.plot_salaire(dates, hist_salaire, reel=False)
    if getattr(settings, 'PLOT_SALAIRE_REEL', False):
        plotting.plot_salaire(dates, hist_salaire, reel=True, inflation_factor=inflation_factor)
    if getattr(settings, 'PLOT_APPORTS', False):
        plotting.plot_apports(dates, hist_apport, reel=False)
    if getattr(settings, 'PLOT_APPORTS_REEL', False):
        plotting.plot_apports(dates, hist_apport, reel=True, inflation_factor=inflation_factor)

    # Graphiques analytiques (crise)
    if getattr(settings, 'SIMULER_CRISE_LOCALISEE', False):
        if getattr(settings, 'PLOT_CRISE_RENDEMENTS', False):
            plotting.plot_zoom_crise_rendements(dates, mat_cap, settings.DATE_CRISE)
        if getattr(settings, 'PLOT_CRISE_CAPITAL_NOMINAL', False):
            plotting.plot_zoom_crise_capital(dates, mat_cap, settings.DATE_CRISE, reel=False)
        if getattr(settings, 'PLOT_CRISE_CAPITAL_REEL', False):
            plotting.plot_zoom_crise_capital(dates, mat_cap, settings.DATE_CRISE, reel=True, inflation_factor=inflation_factor)

    # =========================================================================
    # 6. SYNTHÈSE DES CAPITAUX À LA RETRAITE (Console)
    # =========================================================================
    if getattr(settings, 'PRINT_SYNTHESE_CAPITAL_RETRAITE', False):
        print("\n" + "="*80)
        print("SYNTHÈSE DES CAPITAUX FINAUX À LA RETRAITE (FIN D'ACCUMULATION)")
        print("="*80)

        for strat_nom, mat_c in resultats_comparaison.items():
            capitaux_finaux_strat = mat_c[-1, :]
            idx_sort = np.argsort(capitaux_finaux_strat)
            p5 = capitaux_finaux_strat[idx_sort[int(settings.NB_SIMULATIONS * 0.05)]]
            p50 = capitaux_finaux_strat[idx_sort[int(settings.NB_SIMULATIONS * 0.50)]]
            p95 = capitaux_finaux_strat[idx_sort[int(settings.NB_SIMULATIONS * 0.95)]]

            print(f"\n[ STRATÉGIE : {strat_nom} ]")
            print(f"  • Capital P5  (Pessimiste 5%)  : {p5:>15,.0f} EUR")
            print(f"  • Capital P50 (Médiane)        : {p50:>15,.0f} EUR")
            print(f"  • Capital P95 (Optimiste 95%)  : {p95:>15,.0f} EUR")

        print("="*80 + "\n")

    # =========================================================================
    # 7. PHASE DE DÉCUMULATION (Retraite)
    # =========================================================================
    if getattr(settings, 'SIMULER_DECUMULATION', False):
        from src.engine.decumulation_core import run_decumulation, print_decumulation_report

        nb_mois_decum = settings.NB_ANNEES_DECUMULATION * 12

        # ── 7a. Génération des rendements pour la phase retraite ─────────
        # Prolongation stochastique (pas de backtest en retraite, tout est forecast)
        print(f"\nDÉCUMULATION : Génération de {nb_mois_decum} mois de rendements...")

        # Récupérer le profil de décumulation (peut différer de l'accumulation)
        profil_decum = settings.PROFILS[settings.PROFIL_DECUMULATION]

        np.random.seed(settings.DECUMULATION_SEED)  # None = non-reproductible (prolongation)

        dt = 1.0 / 12.0
        mu_e_d, sigma_e_d = mu_e, sigma_e  # Mêmes paramètres de marché
        mu_b_d, sigma_b_d = mu_b, sigma_b
        s_e = sigma_e_d * np.sqrt(dt)
        s_b = sigma_b_d * np.sqrt(dt)
        cov = np.array([
            [s_e**2, corr_eb * s_e * s_b],
            [corr_eb * s_e * s_b, s_b**2]
        ])

        chocs = np.random.multivariate_normal(
            [0, 0], cov, size=(nb_mois_decum, settings.NB_SIMULATIONS)
        )
        r_eq_decum = (mu_e_d * dt - 0.5 * s_e**2) + chocs[:, :, 0]
        r_bd_decum = (mu_b_d * dt - 0.5 * s_b**2) + chocs[:, :, 1]

        # Inflation pour la phase retraite (prolongation Vasicek)
        inflation_decum = generer_inflation_vasicek(
            nb_periodes=nb_mois_decum,
            nb_sims=settings.NB_SIMULATIONS,
            kappa=getattr(settings, 'INFLATION_KAPPA', 0.15),
            theta=getattr(settings, 'INFLATION_THETA', 0.023),
            sigma=getattr(settings, 'INFLATION_SIGMA', 0.011),
            seed=settings.DECUMULATION_SEED,
        )

        # ── 7b. Construction de la stratégie de décumulation ─────────────
        strat_decum_name = settings.STRATEGIE_DECUMULATION

        if strat_decum_name == "TARGET_DATE":
            strategy_decum = strategies.TargetDateStrategy()
        elif strat_decum_name == "FIXED_MIX":
            strategy_decum = strategies.FixedMixStrategy(
                target_equity_pct=profil_decum["fixed_allocation"]
            )
        elif strat_decum_name == "FALEH":
            from src.strategies.faleh_strategy import FalehStrategy
            strategy_decum = FalehStrategy(mu_e, sigma_e, mu_b, sigma_b, corr_eb)
            dates_decum = pd.date_range(
                start=dates[-1] + pd.offsets.MonthEnd(1),
                periods=nb_mois_decum, freq='ME'
            )
            strategy_decum.initialize_tree(dates_decum)
        elif strat_decum_name == "GBI":
            # GBI en décumulation : fallback sur Fixed-Mix prudent
            # (le GPI n'a plus de sens sans date de retraite future)
            print(f"  Note : GBI non applicable en décumulation → fallback FIXED_MIX ({profil_decum['fixed_allocation']*100:.0f}%)")
            strategy_decum = strategies.FixedMixStrategy(
                target_equity_pct=profil_decum["fixed_allocation"]
            )
            strat_decum_name = f"FIXED_MIX (fallback GBI, {profil_decum['fixed_allocation']*100:.0f}%)"
        else:
            strategy_decum = strategies.FixedMixStrategy(
                target_equity_pct=profil_decum["fixed_allocation"]
            )

        # ── 7c. Capital de départ = capital final de la dernière stratégie d'accumulation ──
        # En mode comparaison, on prend la dernière stratégie exécutée
        capital_depart_decum = mat_cap[-1, :]

        # Dernier salaire pour le taux de remplacement
        dernier_salaire = hist_salaire[-1] if len(hist_salaire) > 0 else None

        print(f"DÉCUMULATION : Stratégie={strat_decum_name}, "
              f"Profil={settings.PROFIL_DECUMULATION}, "
              f"Retrait={settings.RETRAIT_MENSUEL_REEL}€/mois, "
              f"Horizon={settings.NB_ANNEES_DECUMULATION}ans, "
              f"Distribution surplus={settings.TAUX_DISTRIBUTION_SURPLUS*100:.0f}%")

        # ── 7d. Exécution du moteur de décumulation ──────────────────────
        result_decum = run_decumulation(
            strategy=strategy_decum,
            r_eq=r_eq_decum,
            r_bd=r_bd_decum,
            capital_initial_par_sim=capital_depart_decum,
            inflation=inflation_decum,
            dernier_salaire_mensuel=dernier_salaire,
        )

        # ── 7e. Reporting console ────────────────────────────────────────
        if getattr(settings, 'PRINT_DECUMULATION_METRICS', False):
            print_decumulation_report(
                result_decum["metrics"],
                strategy_name=f"{strat_decum_name} / Profil {settings.PROFIL_DECUMULATION}"
            )

        # ── 7f. Visualisation décumulation ───────────────────────────────
        if getattr(settings, 'PLOT_DECUMULATION_CAPITAL', False):
            plotting.plot_retraite_capital(
                result_decum["mat_capital"], reel=False,
                inflation_factor=result_decum["inflation_factor"]
            )
        if getattr(settings, 'PLOT_DECUMULATION_CAPITAL_REEL', False):
            plotting.plot_retraite_capital(
                result_decum["mat_capital"], reel=True,
                inflation_factor=result_decum["inflation_factor"]
            )

    return resultats_comparaison

if __name__ == "__main__":
    main()
