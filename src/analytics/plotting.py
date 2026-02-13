import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import pandas as pd
import numpy as np
from config import settings

# Style global
plt.style.use('seaborn-v0_8-whitegrid')

def afficher_resultats_graphiques(dates, mat_cap, capitaux_finaux, courbe_investi, 
                                  taux_remp, tri_median):
    """
    Génère et affiche la planche de 4 graphiques (Distribution, Evolution, Retraite, Rendements).
    """
    
    # Indices percentiles
    nb_sims = len(capitaux_finaux)
    idx_sorted = np.argsort(capitaux_finaux)
    idx_p5 = idx_sorted[int(nb_sims * 0.05)]
    idx_p50 = idx_sorted[int(nb_sims * 0.50)]
    idx_p95 = idx_sorted[int(nb_sims * 0.95)]
    
    # Préparation des données temporelles
    dates_plot = [pd.Timestamp(settings.DATE_DEBUT_T0)] + list(dates)
    
    # Initialisation Figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"ANALYSE ALM - {settings.METHODE} - PROFIL {settings.PROFIL_CHOISI}", 
                 fontsize=16, fontweight='bold', y=0.96)
    
    # =========================================================================
    # GRAPHIQUE 1 : Distribution Capital Final
    # =========================================================================
    ax = axes[0, 0]
    ax.hist(capitaux_finaux, bins=40, alpha=0.75, edgecolor="black", color='steelblue')
    ax.axvline(capitaux_finaux[idx_p5], linestyle="--", linewidth=2, 
               label=f"P5  {capitaux_finaux[idx_p5]:,.0f} €", color="red")
    ax.axvline(capitaux_finaux[idx_p50], linestyle="--", linewidth=2, 
               label=f"P50 {capitaux_finaux[idx_p50]:,.0f} €", color="green")
    ax.axvline(capitaux_finaux[idx_p95], linestyle="--", linewidth=2, 
               label=f"P95 {capitaux_finaux[idx_p95]:,.0f} €", color="blue")
    ax.set_title("1. Distribution Capital Final")
    ax.set_xlabel("Capital (€)")
    ax.set_ylabel("Fréquence")
    ax.grid(True, alpha=0.25)
    ax.legend()
    
    # =========================================================================
    # GRAPHIQUE 2 : Fan Chart Capital (Evolution)
    # =========================================================================
    ax = axes[0, 1]
    ax.plot(dates_plot, courbe_investi, color='red', linestyle='--', 
            linewidth=2, label='Versements cumulés', alpha=0.7)
    ax.plot(dates_plot, mat_cap[:, idx_p95], color='#2ca02c', 
            linewidth=1.5, label='P95 (Optimiste)', alpha=0.8)
    ax.plot(dates_plot, mat_cap[:, idx_p50], color='black', 
            linewidth=2.5, label='Médiane (P50)')
    ax.plot(dates_plot, mat_cap[:, idx_p5], color='gray', 
            linewidth=1.5, label='P5 (Pessimiste)', alpha=0.8)
    ax.fill_between(dates_plot, mat_cap[:, idx_p5], mat_cap[:, idx_p95], 
                    color='gray', alpha=0.15)
    ax.set_title(f"2. Évolution Capital - TRI {tri_median:.2f}%")
    ax.set_ylabel("Capital (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(
        lambda x, p: format(int(x), ',').replace(',', ' ')
    ))
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.25)
    
    # =========================================================================
    # GRAPHIQUE 3 : Taux de Remplacement (Retraite)
    # =========================================================================
    ax = axes[1, 0]
    annees_retraite = np.arange(1, settings.DUREE_RETRAITE + 1)
    ax.plot(annees_retraite, taux_remp[:, idx_p95]*100, 
            color='#2ca02c', label='P95', alpha=0.7)
    ax.plot(annees_retraite, taux_remp[:, idx_p50]*100, 
            color='black', linewidth=2, label='P50')
    ax.plot(annees_retraite, taux_remp[:, idx_p5]*100, 
            color='#d62728', label='P5', alpha=0.7)
    ax.axhline(100, color='gray', linestyle=':', alpha=0.5, label='100%')
    ax.set_title("3. Taux de Remplacement (Retraite)")
    ax.set_xlabel("Année de retraite")
    ax.set_ylabel("% du dernier salaire")
    ax.legend()
    ax.grid(True, alpha=0.25)
    
    # =========================================================================
    # GRAPHIQUE 4 : Rendements Annuels
    # =========================================================================
    ax = axes[1, 1]
    df_perf = pd.DataFrame(mat_cap)
    # Calcul glissant annuel (toute les 12 périodes)
    perf_annuelle = df_perf.pct_change(12).iloc[12::12] * 100
    annees_simu = np.arange(1, len(perf_annuelle) + 1)
    
    ax.plot(annees_simu, perf_annuelle.iloc[:, idx_p95], 
            color="#4d2ca0", alpha=0.6, label='P95 (Scénario Haut)')
    ax.plot(annees_simu, perf_annuelle.iloc[:, idx_p50], 
            color='black', linewidth=2, label='Médiane')
    ax.plot(annees_simu, perf_annuelle.iloc[:, idx_p5], 
            color='#d62728', alpha=0.6, label='P5 (Scénario Bas)')
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    
    # Ligne pivot si Fixed Mix (Backtest vs Forecast)
    if settings.METHODE == "FIXED_MIX":
        idx_pivot_annee = int((pd.Timestamp(settings.DATE_PIVOT_BACKTEST) - pd.Timestamp(settings.DATE_DEBUT_T0)).days / 365)
        if 0 < idx_pivot_annee < len(annees_simu):
            ax.axvline(idx_pivot_annee, color='gray', linestyle='--', alpha=0.5)
            ax.text(idx_pivot_annee - 2, ax.get_ylim()[1] * 0.8, "BACKTEST", 
                    fontsize=8, color='gray', ha='right')
            ax.text(idx_pivot_annee + 1, ax.get_ylim()[1] * 0.8, "FORECAST", 
                    fontsize=8, color='gray', ha='left')
    
    ax.set_title("4. Rendements Annuels du Portefeuille")
    ax.set_xlabel("Année")
    ax.set_ylabel("Rendement (%)")
    ax.legend(loc='lower center', ncol=3)
    ax.grid(True, alpha=0.25)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()