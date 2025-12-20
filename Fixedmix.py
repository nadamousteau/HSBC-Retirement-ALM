#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMULATION ALM - DASHBOARD FINAL (CORRECTIF)
------------------------------------------------
- Réintégration de la MEDIANE dans les rendements
- Affichage Terminal restructuré (KPIs visibles)
- Clarification Shortfall & Inflation
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import optimize
import matplotlib.ticker as mtick

# Style graphique professionnel
plt.style.use('seaborn-v0_8-whitegrid')

''' =======================================================================
    1. CONFIGURATION
    ======================================================================= '''

# --- Paramètres Utilisateur ---
PROFIL_CHOISI = "EQUILIBRE"
NB_ANNEES_ACCUMULATION = 40
DATE_DEBUT_T0 = "2001-12-31" 
SALAIRE_INITIAL = 2000   
CAPITAL_INITIAL = 5000   

# --- Paramètres Retraite ---
DUREE_RETRAITE = 20     
TAUX_LIVRET_A = 0.017   

# --- Paramètres Simulation ---
NB_SIMULATIONS = 500
NB_PAS_PAR_AN = 12
NB_PERIODES_TOTAL = NB_ANNEES_ACCUMULATION * NB_PAS_PAR_AN

# --- Paramètres Éco ---
VITESSE_PROGRESSION = 0.10   
GAMMA_ELASTICITE = 1.5       
SEUIL_MATURITE = 0.935       
TAUX_APPORT_BASE = 0.10      
SALAIRE_MAX_CIBLE = SALAIRE_INITIAL * 2.5 
INFLATION_ANNUELLE = 0.02  

# --- Paramètres Crise ---
SIMULER_KRACH = True
DATE_CRISE = "2030-03-31"   
PARAMS_CRISE = {'drop_eq': 0.35, 'drop_bd': 0.05, 'duree_mois': 18, 'facteur_vol': 2.5}

# --- Profils ---
PROFILS = {
    "PRUDENT":   {"equity": "Global Equity USD Hedged", "bond": "US Government Bond USD Unhedged", "fixed_allocation": 0.20},
    "MODERE":    {"equity": "Global Equity USD Hedged", "bond": "US Inflation Linked Bond - USD Unhedged", "fixed_allocation": 0.40},
    "EQUILIBRE": {"equity": "Global Equity USD Hedged", "bond": "USD Corporate Bond - USD Unhedged", "fixed_allocation": 0.60},
    "DYNAMIQUE": {"equity": "US Equity USD Unhedged",   "bond": "USD Corporate Bond - USD Unhedged", "fixed_allocation": 0.80},
    "AGRESSIF":  {"equity": "Japan Equity - USD Unhedged","bond": "US High Yield Bond BB-B - USD Unhedged", "fixed_allocation": 0.95}
}

''' =======================================================================
    2. FONCTIONS CALCUL
    ======================================================================= '''

def get_params(asset_name, df_bs):
    row = df_bs[df_bs['Asset Name'] == asset_name]
    if row.empty: raise ValueError(f"Actif {asset_name} introuvable.")
    return row['Expected Return'].values[0], row['Volatility'].values[0], (row['Correlation'].values[0] if 'Correlation' in row.columns else 0.3)

def estimer_salaire_saturation(t_annees, S_init, S_max):
    return S_init + (S_max - S_init) * (1 - np.exp(-VITESSE_PROGRESSION * t_annees))

def precalculer_parametres_apport(S_init, S_max, duree_totale):
    ratio = S_max / S_init
    facteur = ratio ** GAMMA_ELASTICITE
    app_init, app_max = S_init * TAUX_APPORT_BASE, S_init * TAUX_APPORT_BASE * facteur
    s_cible = S_init + (S_max - S_init) * SEUIL_MATURITE
    if s_cible >= S_max: t_pic = duree_totale
    else: t_pic = -np.log(1 - min((s_cible - S_init) / (S_max - S_init), 0.9999)) / VITESSE_PROGRESSION
    return app_init, app_max, min(max(0, t_pic), duree_totale)

def calculer_apport_mensuel(t_annees, app_init, app_max, t_pic):
    if t_pic <= 0: return app_init 
    a = (app_init - app_max) / (t_pic**2)
    return max(a * (t_annees - t_pic)**2 + app_max, 0)

def calculer_tri_annualise(capital_init, liste_apports, val_finale, freq=12):
    flux = [-float(capital_init)]
    flux.extend([-float(m) for m in liste_apports]) 
    flux.append(float(val_finale))
    def npv(r):
        if r <= -1.0: return 1e10 
        valeurs = np.array(flux)
        puissances = np.arange(len(flux))
        return np.sum(valeurs / ((1 + r) ** puissances))
    try:
        tri_periodique = optimize.brentq(npv, -0.05, 0.10)
        return ((1 + tri_periodique) ** freq - 1) * 100
    except:
        return 0.0

def generer_rendements_simules(mu_e, sigma_e, mu_b, sigma_b, corr, dates, date_pivot, nb_sims):
    dt = 1.0/12.0
    nb_total_mois = len(dates)
    dates_pd = pd.to_datetime(dates)
    pivot_ts = pd.Timestamp(date_pivot)
    
    if pivot_ts < dates_pd[0]: idx_split = 0
    elif pivot_ts > dates_pd[-1]: idx_split = nb_total_mois
    else: idx_split = np.searchsorted(dates_pd, pivot_ts)

    s_e, s_b = sigma_e * np.sqrt(dt), sigma_b * np.sqrt(dt)
    cov = np.array([[s_e**2, corr*s_e*s_b], [corr*s_e*s_b, s_b**2]])

    # Partie Historique (Commune)
    if idx_split > 0:
        np.random.seed(42) 
        chocs_histo_unique = np.random.multivariate_normal([0, 0], cov, size=(idx_split))
        r_eq_h = (mu_e*dt - 0.5*s_e**2) + chocs_histo_unique[:, 0]
        r_bd_h = (mu_b*dt - 0.5*s_b**2) + chocs_histo_unique[:, 1]
        
        # On duplique le MÊME passé pour tout le monde
        r_eq_past = np.tile(r_eq_h.reshape(-1, 1), (1, nb_sims))
        r_bd_past = np.tile(r_bd_h.reshape(-1, 1), (1, nb_sims))
    else:
        r_eq_past = np.empty((0, nb_sims))
        r_bd_past = np.empty((0, nb_sims))

    # Partie Futur (Divergente)
    nb_mois_futur = nb_total_mois - idx_split
    if nb_mois_futur > 0:
        np.random.seed(None) 
        chocs_futur = np.random.multivariate_normal([0, 0], cov, size=(nb_mois_futur, nb_sims))
        r_eq_fut = (mu_e*dt - 0.5*s_e**2) + chocs_futur[:, :, 0]
        r_bd_fut = (mu_b*dt - 0.5*s_b**2) + chocs_futur[:, :, 1]
    else:
        r_eq_fut = np.empty((0, nb_sims))
        r_bd_fut = np.empty((0, nb_sims))

    return np.vstack([r_eq_past, r_eq_fut]), np.vstack([r_bd_past, r_bd_fut])

def injecter_crise_complexe(r_eq, r_bd, dates_list, date_depart, params):
    r_eq_m, r_bd_m = r_eq.copy(), r_bd.copy()
    dates_pd = pd.to_datetime(dates_list)
    idx = np.argmin(np.abs(dates_pd - pd.Timestamp(date_depart)))
    if abs((dates_pd[idx] - pd.Timestamp(date_depart)).days) > 40: return r_eq, r_bd 
    
    r_eq_m[idx, :] = np.log(1.0 - params.get('drop_eq', 0.3))
    r_bd_m[idx, :] = np.log(1.0 - params.get('drop_bd', 0.0))
    end = min(idx + params.get('duree_mois', 12), r_eq.shape[0])
    facteur = params.get('facteur_vol', 2.0)
    r_eq_m[idx+1:end, :] *= facteur
    r_bd_m[idx+1:end, :] *= facteur
    return r_eq_m, r_bd_m

def simuler_accumulation_complete(capital_init, r_eq, r_bd, alloc, dates):
    nb_steps, nb_sims = r_eq.shape
    mat_cap = np.zeros((nb_steps + 1, nb_sims))
    mat_cap[0, :] = capital_init
    courbe_investi = np.zeros(nb_steps + 1)
    courbe_investi[0] = capital_init
    hist_salaire = np.zeros(nb_steps)
    hist_apport = np.zeros(nb_steps)
    app_init, app_max, t_pic = precalculer_parametres_apport(SALAIRE_INITIAL, SALAIRE_MAX_CIBLE, NB_ANNEES_ACCUMULATION)
    
    for t in range(nb_steps):
        annee_f = t/12.0
        salaire_mensuel = estimer_salaire_saturation(annee_f, SALAIRE_INITIAL, SALAIRE_MAX_CIBLE)
        apport_mensuel = calculer_apport_mensuel(annee_f, app_init, app_max, t_pic)
        hist_salaire[t] = salaire_mensuel
        hist_apport[t] = apport_mensuel
        
        rendement = alloc * r_eq[t, :] + (1 - alloc) * r_bd[t, :]
        mat_cap[t+1, :] = mat_cap[t, :] * np.exp(rendement) + apport_mensuel
        courbe_investi[t+1] = courbe_investi[t] + apport_mensuel
        
    return mat_cap, courbe_investi, hist_salaire, hist_apport

def simuler_decumulation(capitaux_finaux, dernier_salaire, taux_livret, duree):
    nb_sims = len(capitaux_finaux)
    taux_histo = np.zeros((duree, nb_sims))
    cap_courant = capitaux_finaux.copy()
    for i in range(duree):
        restant = duree - i
        pension_mensuelle = cap_courant / (12 * restant)
        taux_histo[i, :] = pension_mensuelle / dernier_salaire
        cap_courant = (cap_courant - pension_mensuelle*12) * (1 + taux_livret)
        cap_courant = np.maximum(cap_courant, 0)
    return taux_histo

def analyse_drawdown(dates, courbe_cap):
    vals = np.array(courbe_cap)
    max_cum = np.maximum.accumulate(vals)
    with np.errstate(divide='ignore', invalid='ignore'):
        dds = (vals - max_cum) / max_cum
    dds = np.nan_to_num(dds)
    idx_worst = np.argmin(dds)
    return dds[idx_worst], dates[idx_worst], dds

def calcul_kpi_complets(capitaux_finaux, total_investi, mat_cap_historique):
    # Shortfall Probability
    nb_pertes = np.sum(capitaux_finaux < total_investi)
    proba_shortfall = nb_pertes / len(capitaux_finaux)
    
    # VaR 95 (Nominale)
    var_95 = np.percentile(capitaux_finaux, 5)
    
    # Sortino Ratio
    idx_med = np.argsort(capitaux_finaux)[len(capitaux_finaux)//2]
    trajectoire_med = mat_cap_historique[:, idx_med]
    rendements = np.diff(trajectoire_med) / trajectoire_med[:-1]
    rendements_neg = rendements[rendements < 0]
    downside = np.std(rendements_neg) * np.sqrt(12)
    sortino = (np.mean(rendements)*12) / downside if downside > 0 else 0

    # Max Underwater Duration (Années)
    plus_haut = np.maximum.accumulate(trajectoire_med)
    is_underwater = trajectoire_med < plus_haut
    duree_max, compteur = 0, 0
    for s in is_underwater:
        if s: compteur += 1
        else:
            duree_max = max(duree_max, compteur); compteur = 0
    max_underwater = max(duree_max, compteur) / 12.0
    
    # Dispersion P95 - P5
    p95 = np.percentile(capitaux_finaux, 95)
    p5 = np.percentile(capitaux_finaux, 5)
    
    return {
        "shortfall_prob": proba_shortfall,
        "var_95": var_95,
        "gain_p5": var_95 - total_investi,
        "sortino": sortino,
        "max_underwater": max_underwater,
        "dispersion": p95 - p5
    }

''' =======================================================================
    3. VISUALISATION & PRINT
    ======================================================================= '''

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(script_dir, "AssumptionForSimulation.xlsx")
    if not os.path.exists(excel_path): raise FileNotFoundError("Excel manquant.")
    
    df_BS = pd.read_excel(excel_path)
    mapping = {'US Government Bond': 'US Government Bond USD Unhedged', 'USD Corporate Bond': 'USD Corporate Bond - USD Unhedged', 
               'Global Equity': 'Global Equity USD Hedged', 'US Equity': 'US Equity USD Unhedged'}
    df_BS['Asset Name'] = df_BS['Asset Name'].replace(mapping)
    
    profil = PROFILS[PROFIL_CHOISI]
    mu_e, s_e, _ = get_params(profil["equity"], df_BS)
    mu_b, s_b, corr = get_params(profil["bond"], df_BS)
    
    # --- Calculs ---
    DATE_PIVOT = "2025-01-01" 
    dates = pd.date_range(start=DATE_DEBUT_T0, periods=NB_PERIODES_TOTAL, freq='ME')
    
    # 1. Génération Rendements
    r_eq, r_bd = generer_rendements_simules(mu_e, s_e, mu_b, s_b, corr, dates, DATE_PIVOT, NB_SIMULATIONS)
    
    # 2. Injection Crise (Optionnel)
    if SIMULER_KRACH and pd.Timestamp(DATE_CRISE) > pd.Timestamp(DATE_PIVOT):
        r_eq, r_bd = injecter_crise_complexe(r_eq, r_bd, dates, DATE_CRISE, PARAMS_CRISE)
    
    # 3. Accumulation
    mat_cap, courbe_investi, hist_sal, hist_app = simuler_accumulation_complete(
        CAPITAL_INITIAL, r_eq, r_bd, profil["fixed_allocation"], dates)
    
    # --- Indices ---
    idx_sorted = np.argsort(mat_cap[-1, :])
    idx_p5 = idx_sorted[int(NB_SIMULATIONS*0.05)]
    idx_med = idx_sorted[int(NB_SIMULATIONS*0.50)]
    idx_p95 = idx_sorted[int(NB_SIMULATIONS*0.95)]
    
    last_salary = hist_sal[-1]
    taux_remp = simuler_decumulation(mat_cap[-1, :], last_salary, TAUX_LIVRET_A, DUREE_RETRAITE)
    tri_median = calculer_tri_annualise(CAPITAL_INITIAL, hist_app, mat_cap[-1, idx_med])

    # =========================================================================
    # VISUALISATION GRAPHIQUE
    # =========================================================================
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"ANALYSE ALM - PROFIL {PROFIL_CHOISI}", fontsize=16, fontweight='bold', y=0.96)
    dates_plot = [pd.Timestamp(DATE_DEBUT_T0)] + list(dates)

    # AX1 : Flux
    ax1.plot(dates, hist_sal, color='#1f77b4', label='Salaire Net')
    ax1_bis = ax1.twinx()
    ax1_bis.plot(dates, hist_app, color='#2ca02c', linestyle='--', label='Épargne')
    ax1.set_title("1. Flux Financiers (Salaire & Épargne)", fontsize=11)
    ax1.legend(loc='upper left'); ax1_bis.legend(loc='upper right')
    
    # AX2 : Capital Nominal
    ax2.plot(dates_plot, courbe_investi, color='#d62728', label='Versements Cumulés (Cash)')
    ax2.plot(dates_plot, mat_cap[:, idx_p95], color='#2ca02c', linewidth=1, label='P95 (Optimiste)')
    ax2.plot(dates_plot, mat_cap[:, idx_med], color='black', linewidth=2, label='Médiane (P50)')
    ax2.plot(dates_plot, mat_cap[:, idx_p5], color='gray', linewidth=1, label='P5 (Pessimiste)')
    ax2.fill_between(dates_plot, mat_cap[:, idx_p5], mat_cap[:, idx_p95], color='gray', alpha=0.1)
    ax2.set_title(f"2. Projection Capital (Nominal) - TRI {tri_median:.2f}%", fontsize=11)
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    ax2.legend(loc='upper left')

    # AX3 : Remplacement
    ax3.plot(np.arange(1, 21), taux_remp[:, idx_p95]*100, label='P95')
    ax3.plot(np.arange(1, 21), taux_remp[:, idx_med]*100, color='black', label='P50')
    ax3.plot(np.arange(1, 21), taux_remp[:, idx_p5]*100, label='P5')
    ax3.set_title("3. Taux de Remplacement (%)", fontsize=11)
    ax3.legend()

    # AX4 : RENDEMENTS ANNUELS (CORRIGÉ)
    df_perf = pd.DataFrame(mat_cap)
    perf_annuelle = df_perf.pct_change(12).iloc[12::12] * 100
    annees_simu = np.arange(1, len(perf_annuelle) + 1)
    
    # Note : Backtest = même histoire pour tout le monde, donc courbes superposées au début
    ax4.plot(annees_simu, perf_annuelle.iloc[:, idx_p95], color="#4d2ca0", alpha=0.6, label='P95 (Scénario Haut)')
    ax4.plot(annees_simu, perf_annuelle.iloc[:, idx_med], color='black', linewidth=2, label='Médiane (Scénario Central)') # REAJOUT MEDIANE
    ax4.plot(annees_simu, perf_annuelle.iloc[:, idx_p5], color='#d62728', alpha=0.6, label='P5 (Scénario Bas)')
    
    ax4.axhline(0, color='black', linewidth=1)
    # Zone grisée pour distinguer passé (simulé) et futur
    idx_pivot_annee = int(pd.Timestamp(DATE_PIVOT).year - pd.Timestamp(DATE_DEBUT_T0).year)
    ax4.axvline(idx_pivot_annee, color='gray', linestyle='--', alpha=0.5)
    ax4.text(idx_pivot_annee - 2, 20, "BACKTEST SIMULÉ", fontsize=8, color='gray')
    ax4.text(idx_pivot_annee + 1, 20, "PROJECTIONS", fontsize=8, color='gray')
    
    ax4.set_title("4. Rendements Annuels (Backtest & Projection)", fontsize=11)
    ax4.set_ylabel("Rendement (%)")
    ax4.legend(loc='lower center', ncol=3)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # =========================================================================
    # TERMINAL REPORT (CLEAN & KPI ORIENTED)
    # =========================================================================
    
    total_verse = courbe_investi[-1]
    kpis = calcul_kpi_complets(mat_cap[-1, :], total_verse, mat_cap)
    
    # Correction Inflation
    coeff_inflation = 1 / ((1 + INFLATION_ANNUELLE) ** NB_ANNEES_ACCUMULATION)
    capital_p5_reel = kpis['var_95'] * coeff_inflation
    gain_p5_reel = capital_p5_reel - total_verse

    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║   RAPPORT ALM SYNTHÉTIQUE  |  PROFIL : {PROFIL_CHOISI:<23}   ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║ 1. FLUX & TRI                                                        ║")
    print(f"║    > Capital Total Investi (Cash) :   {total_verse:15,.0f} €          ║")
    print(f"║    > Capital Final Médian (P50)   :   {mat_cap[-1, idx_med]:15,.0f} €          ║")
    print(f"║    > TRI Médian                   :   {tri_median:14.2f} %           ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║ 2. RISQUE & DOWNSIDE (Les KPIs manquants)                            ║")
    print(f"║    > SHORTFALL RISK (Proba Perte) :   {kpis['shortfall_prob']*100:14.2f} %           ║")
    print(f"║    > Worst Case (P5 Nominal)      :   {kpis['var_95']:15,.0f} €          ║")
    print(f"║    > P&L en cas de Crise (P5)     :   {kpis['gain_p5']:+15,.0f} €          ║")
    print(f"║    > Max Underwater (Durée)       :   {kpis['max_underwater']:11.1f} années        ║")
    print(f"║    > Sortino Ratio                :   {kpis['sortino']:14.2f}             ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    print("║ 3. RÉALITÉ ÉCONOMIQUE (Inflation incluse)                            ║")
    print(f"║    > Capital P5 RÉEL (Net Infla)  :   {capital_p5_reel:15,.0f} €          ║")
    print(f"║    > P&L RÉEL (Pouvoir d'Achat)   :   {gain_p5_reel:+15,.0f} €          ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    if gain_p5_reel < 0:
        print(f"\n>>> ALERTE : Le modèle détruit {abs(gain_p5_reel):,.0f} € de richesse réelle en cas de scénario adverse (P5).")
