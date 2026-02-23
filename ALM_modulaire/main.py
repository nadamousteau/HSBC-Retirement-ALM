#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

# Importation des modules du projet
from config import settings, profiles
from data import loader
from src import economics, liabilities, strategies, engine, analytics
from src.analytics import plotting

def main():
    # =========================================================================
    # 1. INITIALISATION & CONFIGURATION
    # =========================================================================
    mu_e, sigma_e, mu_b, sigma_b, corr_eb = loader.load_market_parameters()

    # =========================================================================
    # 2. GÉNÉRATION DES SCÉNARIOS ÉCONOMIQUES (ESG) - EXÉCUTÉ UNE SEULE FOIS
    # =========================================================================
    dates = pd.date_range(start=settings.DATE_DEBUT_T0, periods=settings.NB_PERIODES_TOTAL, freq='ME')

    # Génération unifiée : Historique avant la date pivot, Stochastique après pour TOUTES les stratégies
    r_eq, r_bd, idx_split = economics.generators.generer_rendements_avec_backtest(
        mu_e, sigma_e, mu_b, sigma_b, corr_eb, dates, settings.DATE_PIVOT_BACKTEST, settings.NB_SIMULATIONS
    )

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
    # 3. BOUCLE D'ÉVALUATION DES STRATÉGIES
    # =========================================================================
    mode_comparaison = getattr(settings, 'MODE_COMPARAISON', False)
    strategies_run = settings.STRATEGIES_A_COMPARER if mode_comparaison else [settings.METHODE]

    resultats_comparaison = {}
    dernier_contexte = {} # Stockage des données communes pour les graphiques post-boucle

    for strat_actuelle in strategies_run:
        # Écrasement local pour garantir l'aiguillage dans les modules sous-jacents
        settings.METHODE = strat_actuelle
        
        if settings.METHODE == "TARGET_DATE":
            strategy = strategies.TargetDateStrategy()
        else: 
            strategy = strategies.FixedMixStrategy()

        # Exécution du moteur
        mat_cap, courbe_investi, hist_apport, hist_dd, hist_salaire = engine.run_simulation(
            strategy, r_eq, r_bd, dates
        )

        # =========================================================================
        # 4. POST-TRAITEMENT & ANALYTICS
        # =========================================================================
        capitaux_finaux = mat_cap[-1, :]
        total_investi = courbe_investi[-1]
        
        idx_sorted = np.argsort(capitaux_finaux)
        idx_p50 = idx_sorted[int(settings.NB_SIMULATIONS * 0.50)]
        
        tri_median = analytics.metrics.calculer_tri_annualise(settings.CAPITAL_INITIAL, hist_apport, capitaux_finaux[idx_p50])
        kpis = analytics.metrics.calcul_kpi_complets(capitaux_finaux, total_investi, mat_cap)
        
        coeff_inflation = 1 / ((1 + settings.TAUX_INFLATION) ** settings.NB_ANNEES_ACCUMULATION)
        capital_p5_reel = kpis['var_95'] * coeff_inflation
        gain_p5_reel = capital_p5_reel - total_investi
        
        dernier_salaire = hist_salaire[-1]
        taux_remp, mat_cap_retraite = liabilities.decumulation.simuler_decumulation(
            capitaux_finaux, dernier_salaire, settings.TAUX_LIVRET_A, settings.DUREE_RETRAITE
        )

        # Sauvegarde des résultats
        resultats_comparaison[strat_actuelle] = mat_cap
        dernier_contexte = {
            "courbe_investi": courbe_investi, "hist_apport": hist_apport,
            "hist_salaire": hist_salaire, "taux_remp": taux_remp,
            "mat_cap_retraite": mat_cap_retraite, "mat_cap": mat_cap
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
    # 5. VISUALISATION (Exécution conditionnelle)
    # =========================================================================
    # Extraction des variables communes pour éviter la duplication de code
    courbe_investi = dernier_contexte["courbe_investi"]
    hist_salaire = dernier_contexte["hist_salaire"]
    hist_apport = dernier_contexte["hist_apport"]
    mat_cap = dernier_contexte["mat_cap"]
    taux_remp = dernier_contexte["taux_remp"]
    mat_cap_retraite = dernier_contexte["mat_cap_retraite"]

    # Graphiques d'accumulation (Comparatifs ou Isolés)
    if mode_comparaison:
        if getattr(settings, 'PLOT_COMPARAISON_CAPITAL', False):
            plotting.plot_comparaison_capital(dates, resultats_comparaison, reel=False)
        if getattr(settings, 'PLOT_COMPARAISON_CAPITAL_REEL', False):
            plotting.plot_comparaison_capital(dates, resultats_comparaison, reel=True)
    else:
        if getattr(settings, 'PLOT_CAPITAL', False):
            plotting.plot_capital(dates, mat_cap, courbe_investi, reel=False)
        if getattr(settings, 'PLOT_CAPITAL_REEL', False):
            plotting.plot_capital(dates, mat_cap, courbe_investi, reel=True)

    # Graphiques macro-économiques (Indépendants de la stratégie)
    if getattr(settings, 'PLOT_SALAIRE', False):
        plotting.plot_salaire(dates, hist_salaire, reel=False)
    if getattr(settings, 'PLOT_SALAIRE_REEL', False):
        plotting.plot_salaire(dates, hist_salaire, reel=True)
    if getattr(settings, 'PLOT_APPORTS', False):
        plotting.plot_apports(dates, hist_apport, reel=False)
    if getattr(settings, 'PLOT_APPORTS_REEL', False):
        plotting.plot_apports(dates, hist_apport, reel=True)

    # Graphiques analytiques (Tracés sur la dernière stratégie exécutée)
    if getattr(settings, 'SIMULER_CRISE_LOCALISEE', False):
        if getattr(settings, 'PLOT_CRISE_RENDEMENTS', False):
            plotting.plot_zoom_crise_rendements(dates, mat_cap, settings.DATE_CRISE)
        if getattr(settings, 'PLOT_CRISE_CAPITAL_NOMINAL', False):
            plotting.plot_zoom_crise_capital(dates, mat_cap, settings.DATE_CRISE, reel=False)
        if getattr(settings, 'PLOT_CRISE_CAPITAL_REEL', False):
            plotting.plot_zoom_crise_capital(dates, mat_cap, settings.DATE_CRISE, reel=True)
        
    if getattr(settings, 'PLOT_RETRAITE_CAPITAL', False):
        plotting.plot_retraite_capital(mat_cap_retraite, reel=False)
    if getattr(settings, 'PLOT_RETRAITE_CAPITAL_REEL', False):
        plotting.plot_retraite_capital(mat_cap_retraite, reel=True)
        
    if getattr(settings, 'PLOT_TAUX_REMPLACEMENT', False):
        plotting.plot_taux_remplacement(taux_remp, reel=False)
    if getattr(settings, 'PLOT_TAUX_REMPLACEMENT_REEL', False):
        plotting.plot_taux_remplacement(taux_remp, reel=True)

    # =========================================================================
    # 6. SYNTHÈSE DES CAPITAUX À LA RETRAITE (Console)
    # =========================================================================
    if getattr(settings, 'PRINT_SYNTHESE_CAPITAL_RETRAITE', False):
        print("\n" + "="*80)
        print("SYNTHÈSE DES CAPITAUX FINAUX À LA RETRAITE (FIN D'ACCUMULATION)")
        print("="*80)
        
        # La boucle s'adapte automatiquement : 1 stratégie (mode normal) ou N stratégies (mode comparaison)
        for strat_nom, mat_c in resultats_comparaison.items():
            capitaux_finaux_strat = mat_c[-1, :]
            
            # Tri pour extraction des quantiles
            idx_sort = np.argsort(capitaux_finaux_strat)
            p5 = capitaux_finaux_strat[idx_sort[int(settings.NB_SIMULATIONS * 0.05)]]
            p50 = capitaux_finaux_strat[idx_sort[int(settings.NB_SIMULATIONS * 0.50)]]
            p95 = capitaux_finaux_strat[idx_sort[int(settings.NB_SIMULATIONS * 0.95)]]
            
            print(f"\n[ STRATÉGIE : {strat_nom} ]")
            print(f"  • Capital P5  (Pessimiste 5%)  : {p5:>15,.0f} EUR")
            print(f"  • Capital P50 (Médiane)        : {p50:>15,.0f} EUR")
            print(f"  • Capital P95 (Optimiste 95%)  : {p95:>15,.0f} EUR")
            
        print("="*80 + "\n")



    return resultats_comparaison

if __name__ == "__main__":
    main()