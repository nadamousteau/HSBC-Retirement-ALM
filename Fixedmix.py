#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import optimize
import matplotlib.ticker as mtick

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

# --- Paramètres Crise ---
SIMULER_KRACH = False
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
    """
    Génère un passé UNIQUE (identique pour toutes les simus) jusqu'à date_pivot,
    puis un futur DIVERGENT (Monte Carlo classique).
    """
    dt = 1.0/12.0
    nb_total_mois = len(dates)
    
    dates_pd = pd.to_datetime(dates)
    pivot_ts = pd.Timestamp(date_pivot)
    
    if pivot_ts < dates_pd[0]: idx_split = 0
    elif pivot_ts > dates_pd[-1]: idx_split = nb_total_mois
    else: idx_split = np.searchsorted(dates_pd, pivot_ts)

    s_e, s_b = sigma_e * np.sqrt(dt), sigma_b * np.sqrt(dt)
    cov = np.array([[s_e**2, corr*s_e*s_b], [corr*s_e*s_b, s_b**2]])

    # --- PARTIE 1 : HISTOIRE COMMUNE (0 à idx_split) ---
    if idx_split > 0:
        np.random.seed(42) 
        chocs_histo_unique = np.random.multivariate_normal([0, 0], cov, size=(idx_split))
        
        r_eq_h = (mu_e*dt - 0.5*s_e**2) + chocs_histo_unique[:, 0]
        r_bd_h = (mu_b*dt - 0.5*s_b**2) + chocs_histo_unique[:, 1]
        
        r_eq_past = np.tile(r_eq_h.reshape(-1, 1), (1, nb_sims))
        r_bd_past = np.tile(r_bd_h.reshape(-1, 1), (1, nb_sims))
    else:
        r_eq_past = np.empty((0, nb_sims))
        r_bd_past = np.empty((0, nb_sims))

    # --- PARTIE 2 : FUTUR INCERTAIN (idx_split à fin) ---
    nb_mois_futur = nb_total_mois - idx_split
    if nb_mois_futur > 0:
        np.random.seed(None) 
        chocs_futur = np.random.multivariate_normal([0, 0], cov, size=(nb_mois_futur, nb_sims))
        
        r_eq_fut = (mu_e*dt - 0.5*s_e**2) + chocs_futur[:, :, 0]
        r_bd_fut = (mu_b*dt - 0.5*s_b**2) + chocs_futur[:, :, 1]
    else:
        r_eq_fut = np.empty((0, nb_sims))
        r_bd_fut = np.empty((0, nb_sims))

    r_eq_final = np.vstack([r_eq_past, r_eq_fut])
    r_bd_final = np.vstack([r_bd_past, r_bd_fut])
    
    return r_eq_final, r_bd_final

def injecter_crise_complexe(r_eq, r_bd, dates_list, date_depart, params):
    r_eq_m, r_bd_m = r_eq.copy(), r_bd.copy()
    dates_pd = pd.to_datetime(dates_list)
    idx = np.argmin(np.abs(dates_pd - pd.Timestamp(date_depart)))
    
    if abs((dates_pd[idx] - pd.Timestamp(date_depart)).days) > 40:
        return r_eq, r_bd 
    
    print(f"⚡ KRACH INJECTÉ EN {dates_pd[idx].strftime('%Y-%m')}")
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

''' =======================================================================
    3. EXÉCUTION & VISUALISATION (DASHBOARD)
    ======================================================================= '''

if __name__ == "__main__":
    # --- Setup ---
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
    
    # --- Simulation ---
    DATE_PIVOT = "2025-01-01" 
    
    dates = pd.date_range(start=DATE_DEBUT_T0, periods=NB_PERIODES_TOTAL, freq='ME')
    
    # CORRECTION : s_e, s_b
    r_eq, r_bd = generer_rendements_simules(mu_e, s_e, mu_b, s_b, corr, dates, DATE_PIVOT, NB_SIMULATIONS)
    
    if SIMULER_KRACH:
        if pd.Timestamp(DATE_CRISE) > pd.Timestamp(DATE_PIVOT):
            r_eq, r_bd = injecter_crise_complexe(r_eq, r_bd, dates, DATE_CRISE, PARAMS_CRISE)
    
    mat_cap, courbe_investi, hist_sal, hist_app = simuler_accumulation_complete(
        CAPITAL_INITIAL, r_eq, r_bd, profil["fixed_allocation"], dates)
    
    # --- Sélection Scénarios ---
    idx_sorted = np.argsort(mat_cap[-1, :])
    idx_p5, idx_med, idx_p95 = idx_sorted[int(NB_SIMULATIONS*0.05)], idx_sorted[int(NB_SIMULATIONS*0.50)], idx_sorted[int(NB_SIMULATIONS*0.95)]
    
    # --- Décumulation & TRI ---
    last_salary = hist_sal[-1]
    taux_remp = simuler_decumulation(mat_cap[-1, :], last_salary, TAUX_LIVRET_A, DUREE_RETRAITE)
    tri_median = calculer_tri_annualise(CAPITAL_INITIAL, hist_app, mat_cap[-1, idx_med])

    # =========================================================================
    # VISUALISATION
    # =========================================================================
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"DASHBOARD ALM - PROFIL {PROFIL_CHOISI}", fontsize=16, fontweight='bold')
    dates_plot = [pd.Timestamp(DATE_DEBUT_T0)] + list(dates)

    # --- AX1 : Carrière (Salaire vs Apport) ---
    color1, color2 = 'tab:blue', 'tab:green'
    ax1.plot(dates, hist_sal, color=color1, linewidth=2, label='Salaire Net')
    ax1.set_ylabel('Salaire (€)', color=color1, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_title("1. Dynamique de Carrière & Épargne")
    ax1.grid(True, alpha=0.3)
    
    ax1_bis = ax1.twinx()
    ax1_bis.plot(dates, hist_app, color=color2, linewidth=2, linestyle='--', label='Épargne')
    ax1_bis.set_ylabel('Apport Mensuel (€)', color=color2, fontweight='bold')
    ax1_bis.tick_params(axis='y', labelcolor=color2)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_bis.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # --- AX2 : Accumulation Capital (LINEAIRE) ---
    ax2.plot(dates_plot, courbe_investi, color='#d62728', linewidth=2, linestyle='-', drawstyle='steps-post', label='Capital Investi (Cash)')
    ax2.plot(dates_plot, mat_cap[:, idx_p95], color='#2ca02c', linewidth=1.5, label='Marché (P95)')
    ax2.plot(dates_plot, mat_cap[:, idx_med], color='black', linewidth=2.5, label='Marché (Médian)')
    ax2.plot(dates_plot, mat_cap[:, idx_p5], color='gray', linewidth=1.5, label='Marché (P5)')
    
    ax2.fill_between(dates_plot, mat_cap[:, idx_p5], mat_cap[:, idx_p95], color='gray', alpha=0.1)
    
    # MODIFICATION ICI : Pas de log scale
    ax2.set_title(f"2. Accumulation (Linéaire) - TRI P50 : {tri_median:.2f}%")
    ax2.set_ylabel("Capital (€)") 
    # ax2.set_yscale('log') # <--- DESACTIVÉ
    
    ax2.legend(loc='upper left')
    ax2.grid(True, which="major", alpha=0.3) 
    ax2.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))

    # --- AX3 : Taux de Remplacement ---
    annees_ret = np.arange(1, DUREE_RETRAITE + 1)
    tr_p95 = taux_remp[:, idx_p95] * 100
    tr_med = taux_remp[:, idx_med] * 100
    tr_p5  = taux_remp[:, idx_p5] * 100
    
    ax3.plot(annees_ret, tr_p95, color='#2ca02c', marker='o', linewidth=2, label='Optimiste (P95)')
    ax3.plot(annees_ret, tr_med, color='black', marker='o', linewidth=2, label='Médian (P50)')
    ax3.plot(annees_ret, tr_p5, color='gray', marker='o', linewidth=2, label='Pessimiste (P5)')
    
    ax3.set_title("3. Taux de Remplacement (Phase Rente)")
    ax3.set_ylabel("% du dernier salaire")
    ax3.set_xlabel("Années après la retraite")
    ax3.set_xticks(np.arange(0, DUREE_RETRAITE + 2, 2))
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # --- AX4 : Rendements Annuels (3 Trajectoires) ---
    rendements_mensuels_portefeuille = profil["fixed_allocation"] * r_eq + (1 - profil["fixed_allocation"]) * r_bd
    
    nb_annees_reelles = NB_ANNEES_ACCUMULATION
    rendements_annuels_log = np.sum(rendements_mensuels_portefeuille.reshape(nb_annees_reelles, 12, NB_SIMULATIONS), axis=1)
    rendements_annuels_pct = (np.exp(rendements_annuels_log) - 1) * 100
    
    annees_simu = np.arange(1, nb_annees_reelles + 1)

    # CORRECTION ICI : Trajectoires réelles, pas de stats lissées
    traj_p5  = rendements_annuels_pct[:, idx_p5]
    traj_med = rendements_annuels_pct[:, idx_med]
    traj_p95 = rendements_annuels_pct[:, idx_p95]

    ax4.plot(annees_simu, traj_p95, color="#4d2ca0", linewidth=1.5, linestyle='-', alpha=0.8, label='Scénario Haut (P95)')
    ax4.plot(annees_simu, traj_p5, color='#d62728', linewidth=1.5, linestyle='-', alpha=0.8, label='Scénario Bas (P5)')
    ax4.plot(annees_simu, traj_med, color='black', linewidth=1.5, label='Scénario Médian (P50)')

    ax4.axhline(0, color='gray', linestyle='-', alpha=0.3, linewidth=1)

    ax4.set_title("4. Volatilité Réelle des Rendements (3 Trajectoires)")
    ax4.set_ylabel("Rendement Annuel (%)")
    ax4.set_xlabel("Année de simulation")
    
    # Légende en bas pour ne pas cacher les données
    ax4.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fancybox=True, shadow=False, ncol=3)
    ax4.grid(True, alpha=0.3)
    
    # Ajustement des marges
    plt.subplots_adjust(bottom=0.15, hspace=0.4)
    plt.show()

    # --- TABLEAU CONSOLE ---
    dd_med, date_dd, _ = analyse_drawdown(dates_plot, mat_cap[:, idx_med])
    max_cap_med = np.max(mat_cap[:, idx_med]) 

    data_res = {
        "Indicateur Clé": [
            "Total Capital Investi",
            "Pic Historique (P50)",
            "Capital Final (P50)", 
            "Plus-Value (P50)",
            "TRI Réel (P50)",
            "Pire Moment (Krach)",
            "Pire Drawdown",
            "Rente Initiale (P50)"
        ],
        "Valeur": [
            f"{courbe_investi[-1]:,.0f} €",
            f"{max_cap_med:,.0f} €",
            f"{mat_cap[-1, idx_med]:,.0f} €",
            f"{mat_cap[-1, idx_med] - courbe_investi[-1]:+,.0f} €",
            f"{tri_median:.2f} % / an",
            date_dd.strftime('%Y-%m'),
            f"{dd_med*100:.1f} %",
            f"{tr_med[0]:.1f} % salaire"
        ]
    }
    
    print("\n" + "="*40)
    print(f"VERDICT FINANCIER (Profil {PROFIL_CHOISI})")
    print("="*40)
    print(pd.DataFrame(data_res).to_string(index=False))
