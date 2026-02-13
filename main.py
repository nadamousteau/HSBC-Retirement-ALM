#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import sys

# Importation des modules du projet
from config import settings, profiles
from data import loader
from src import economics, liabilities, strategies, engine, analytics
from src.analytics import plotting

def main():
    # =========================================================================
    # 1. INITIALISATION & CONFIGURATION
    # =========================================================================
    print("\n" + "="*80)
    print(f"🎯 SIMULATION ALM - MÉTHODE : {settings.METHODE}")
    print("="*80)
    print(f"Profil : {settings.PROFIL_CHOISI}")
    print(f"Horizon : {settings.NB_ANNEES_ACCUMULATION} ans (âge {settings.AGE_DEPART} → {settings.AGE_DEPART + settings.NB_ANNEES_ACCUMULATION})")
    print(f"Capital initial : {settings.CAPITAL_INITIAL:,.0f} €")
    print(f"Salaire initial : {settings.SALAIRE_INITIAL:,.0f} €/mois")
    print(f"Simulations : {settings.NB_SIMULATIONS}")
    
    # Chargement des paramètres de marché
    mu_e, sigma_e, mu_b, sigma_b, corr_eb = loader.load_market_parameters()
    
    print(f"\n📈 Paramètres marché :")
    print(f"  • Equity ({profiles.Equity[:30]}...)")
    print(f"    μ={mu_e*100:.2f}%, σ={sigma_e*100:.2f}%")
    print(f"  • Bond ({profiles.Bond[:30]}...)")
    print(f"    μ={mu_b*100:.2f}%, σ={sigma_b*100:.2f}%, ρ={corr_eb:.2f}")

    # Initialisation de la Stratégie (Factory Pattern simplifié)
    if settings.METHODE == "TARGET_DATE":
        strategy = strategies.TargetDateStrategy()
        
        alloc_init = profiles.allocation_initiale
        decr = profiles.decroissance_annuelle
        alloc_fin = max(0.05, alloc_init - decr * settings.NB_ANNEES_ACCUMULATION)
        
        print(f"\n🎯 Allocation Target Date :")
        print(f"  • Initiale : {alloc_init*100:.1f}% equity")
        print(f"  • Décroissance : {decr*100:.2f}%/an")
        print(f"  • Finale : {alloc_fin*100:.1f}% equity")
        print(f"  • Rééquilibrage : Annuel")
        print(f"  • Drawdown mesuré : {'AVANT apport (marché)' if settings.DRAWDOWN_AVANT_APPORT else 'APRÈS apport'}")
        
    else: # FIXED_MIX
        strategy = strategies.FixedMixStrategy()
        
        alloc_fixe = profiles.fixed_allocation
        print(f"\n🎯 Allocation Fixed Mix :")
        print(f"  • Constante : {alloc_fixe*100:.1f}% equity")
        print(f"  • Rééquilibrage : Implicite (via apports)")

    # =========================================================================
    # 2. GÉNÉRATION DES SCÉNARIOS ÉCONOMIQUES (ESG)
    # =========================================================================
    
    # Génération des dates
    dates = pd.date_range(start=settings.DATE_DEBUT_T0, periods=settings.NB_PERIODES_TOTAL, freq='ME')

    if settings.METHODE == "TARGET_DATE":
        print(f"\n🎲 Génération rendements : Stochastique pur (B&S)")
        r_eq, r_bd = economics.generators.generer_rendements_correles_base(
            mu_e, sigma_e, mu_b, sigma_b, corr_eb, settings.NB_PERIODES_TOTAL, settings.NB_SIMULATIONS
        )
        
        # Application crises (Jump-Diffusion de Merton)
        if settings.SIMULER_CRISE:
            print(f"💥 Ajout crises Jump-Diffusion (λ={settings.LAMBDA_CRISE*100:.1f}%/an)")
            r_eq, r_bd = economics.shocks.ajouter_chocs_merton(r_eq, r_bd, settings.NB_PERIODES_TOTAL, settings.NB_SIMULATIONS)
            
    else: # FIXED_MIX
        print(f"\n🎲 Génération rendements : Backtest/Forecast (pivot {settings.DATE_PIVOT_BACKTEST})")
        r_eq, r_bd, idx_split = economics.generators.generer_rendements_avec_backtest(
            mu_e, sigma_e, mu_b, sigma_b, corr_eb, dates, settings.DATE_PIVOT_BACKTEST, settings.NB_SIMULATIONS
        )
        print(f"  • Backtest : {idx_split} mois (historique commun)")
        print(f"  • Forecast : {settings.NB_PERIODES_TOTAL - idx_split} mois (stochastique)")
        
        # Application crise localisée
        if settings.SIMULER_CRISE and pd.Timestamp(settings.DATE_CRISE) > pd.Timestamp(settings.DATE_PIVOT_BACKTEST):
            print(f"💥 Injection crise localisée ({settings.DATE_CRISE})")
            print(f"   Drop equity : {settings.PARAMS_CRISE_DETAIL['drop_eq']*100:.1f}%")
            print(f"   Durée : {settings.PARAMS_CRISE_DETAIL['duree_mois']} mois")
            r_eq, r_bd = economics.shocks.injecter_crise_localisee(r_eq, r_bd, dates, settings.DATE_CRISE, settings.PARAMS_CRISE_DETAIL)

    # =========================================================================
    # 3. EXÉCUTION DU MOTEUR (ENGINE)
    # =========================================================================
    print(f"\n⚙️  Simulation en cours...")
    
    mat_cap, courbe_investi, hist_apport, hist_dd, hist_salaire = engine.run_simulation(
        strategy, r_eq, r_bd, dates
    )

    # =========================================================================
    # 4. POST-TRAITEMENT & ANALYTICS
    # =========================================================================
    
    # Analyse des résultats
    capitaux_finaux = mat_cap[-1, :]
    total_investi = courbe_investi[-1]
    
    # Indices percentiles
    idx_sorted = np.argsort(capitaux_finaux)
    idx_p5 = idx_sorted[int(settings.NB_SIMULATIONS * 0.05)]
    idx_p50 = idx_sorted[int(settings.NB_SIMULATIONS * 0.50)]
    idx_p95 = idx_sorted[int(settings.NB_SIMULATIONS * 0.95)]
    
    # TRI médian
    tri_median = analytics.metrics.calculer_tri_annualise(settings.CAPITAL_INITIAL, hist_apport, capitaux_finaux[idx_p50])
    
    # KPIs complets
    kpis = analytics.metrics.calcul_kpi_complets(capitaux_finaux, total_investi, mat_cap)
    
    # Correction inflation
    coeff_inflation = 1 / ((1 + settings.TAUX_INFLATION) ** settings.NB_ANNEES_ACCUMULATION)
    capital_p5_reel = kpis['var_95'] * coeff_inflation
    gain_p5_reel = capital_p5_reel - total_investi
    
    # Décumulation (Retraite)
    # Logique spécifique pour récupérer le dernier salaire
    if settings.METHODE == "FIXED_MIX":
        dernier_salaire = hist_salaire[-1]
    else:
        # Estimation pour Target Date (Approximation comme dans l'original)
        dernier_salaire = settings.SALAIRE_INITIAL * 1.5 
    
    taux_remp = liabilities.decumulation.simuler_decumulation(
        capitaux_finaux, dernier_salaire, settings.TAUX_LIVRET_A, settings.DUREE_RETRAITE
    )

    # =========================================================================
    # 5. AFFICHAGE RÉSULTATS (REPORTING)
    # =========================================================================
    
    print("\n" + "="*80)
    print(f"📊 RÉSULTATS - {settings.METHODE} - PROFIL {settings.PROFIL_CHOISI}")
    print("="*80)
    
    print(f"\n💰 FLUX & CAPITAL :")
    print(f"  • Capital initial           : {settings.CAPITAL_INITIAL:>15,.0f} €")
    print(f"  • Apports totaux            : {total_investi - settings.CAPITAL_INITIAL:>15,.0f} €")
    print(f"  • Total investi             : {total_investi:>15,.0f} €")
    print(f"  • Capital final P5          : {capitaux_finaux[idx_p5]:>15,.0f} €")
    print(f"  • Capital final P50         : {capitaux_finaux[idx_p50]:>15,.0f} €")
    print(f"  • Capital final P95         : {capitaux_finaux[idx_p95]:>15,.0f} €")
    print(f"  • TRI médian                : {tri_median:>15.2f} %/an")
    
    print(f"\n📉 RISQUE & DOWNSIDE :")
    print(f"  • Shortfall Risk            : {kpis['shortfall_prob']*100:>15.2f} %")
    print(f"  • VaR 95% (P5 nominal)      : {kpis['var_95']:>15,.0f} €")
    print(f"  • P&L en cas de crise       : {kpis['gain_p5']:>+16,.0f} €")
    print(f"  • Max Underwater            : {kpis['max_underwater']:>15.1f} années")
    print(f"  • Sortino Ratio             : {kpis['sortino']:>15.2f}")
    print(f"  • Dispersion (P95-P5)       : {kpis['dispersion']:>15,.0f} €")
    
    # Max drawdown médian
    max_dd_median = np.median([np.min(hist_dd[:, sim]) for sim in range(settings.NB_SIMULATIONS)])
    print(f"  • Max Drawdown médian       : {max_dd_median*100:>15.2f} %")
    
    print(f"\n💶 POUVOIR D'ACHAT (Inflation {settings.TAUX_INFLATION*100:.1f}%/an) :")
    print(f"  • Capital P5 réel           : {capital_p5_reel:>15,.0f} €")
    print(f"  • P&L réel (worst case)     : {gain_p5_reel:>+16,.0f} €")
    
    print(f"\n🏖️  RETRAITE (Livret A {settings.TAUX_LIVRET_A*100:.2f}%) :")
    print(f"  • Taux remplacement P5      : {taux_remp[0, idx_p5]*100:>15.1f} %")
    print(f"  • Taux remplacement P50     : {taux_remp[0, idx_p50]*100:>15.1f} %")
    print(f"  • Taux remplacement P95     : {taux_remp[0, idx_p95]*100:>15.1f} %")
    
    if gain_p5_reel < 0:
        print(f"\n⚠️  ALERTE : Destruction de richesse réelle en scénario adverse !")
        print(f"   Perte : {abs(gain_p5_reel):,.0f} € (pouvoir d'achat)")
    
    print("="*80)

    # =========================================================================
    # 6. VISUALISATION
    # =========================================================================
    plotting.afficher_resultats_graphiques(
        dates, mat_cap, capitaux_finaux, courbe_investi, taux_remp, tri_median
    )
    
    print("\n✅ Simulation terminée avec succès !\n")

if __name__ == "__main__":
    main()