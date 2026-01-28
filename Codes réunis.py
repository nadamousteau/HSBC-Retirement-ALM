#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SIMULATION ALM UNIFIÉE - TARGET DATE FUND vs FIXED MIX
========================================================
Permet de choisir entre deux stratégies d'allocation :
1. TARGET_DATE : Allocation décroissante avec l'âge (glide path)
2. FIXED_MIX : Allocation fixe constante sur toute la période

Auteur : Version unifiée
Date : 2026-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import optimize
import matplotlib.ticker as mtick

plt.style.use('seaborn-v0_8-whitegrid')

''' ═══════════════════════════════════════════════════════════════════════
    🎯 CHOIX DE LA MÉTHODE - PARAMÈTRE PRINCIPAL
    ═══════════════════════════════════════════════════════════════════════ '''

METHODE = "TARGET_DATE"  # Options: "TARGET_DATE" ou "FIXED_MIX"

''' ═══════════════════════════════════════════════════════════════════════
    📋 CONFIGURATION COMMUNE
    ═══════════════════════════════════════════════════════════════════════ '''

# --- Profil Investisseur ---
PROFIL_CHOISI = "EQUILIBRE"  # PRUDENT, MODERE, EQUILIBRE, DYNAMIQUE, AGRESSIF

# --- Paramètres Temporels ---
NB_ANNEES_ACCUMULATION = 23
AGE_DEPART = 40
DATE_DEBUT_T0 = "2010-12-31"
DATE_PIVOT_BACKTEST = "2025-01-01"  # Séparation historique/forecast

# --- Paramètres Financiers ---
CAPITAL_INITIAL = 50000
SALAIRE_INITIAL = 3000
TAUX_APPORT_BASE = 0.10

# --- Paramètres Économiques ---
TAUX_INFLATION = 0.02
TAUX_LIVRET_A = 0.017  # Pour phase de retraite

# --- Paramètres Retraite ---
DUREE_RETRAITE = 20

# --- Paramètres Simulation ---
NB_SIMULATIONS = 500
NB_PAS_PAR_AN = 12
NB_PERIODES_TOTAL = NB_ANNEES_ACCUMULATION * NB_PAS_PAR_AN

# --- Paramètres Apport (Modèle Quadratique / Exponentiel) ---
# Quadratique (Target Date)
RATIO_PIC_CARRIERE = 0.55
FACTEUR_CROISSANCE_MAX = 1.2

# Exponentiel (Fixed Mix)
VITESSE_PROGRESSION = 0.10
GAMMA_ELASTICITE = 1.5
SEUIL_MATURITE = 0.935
SALAIRE_MAX_CIBLE = SALAIRE_INITIAL * 2.5

# --- Paramètres Crise (Jump-Diffusion) ---
SIMULER_CRISE = True
DATE_CRISE = "2030-03-31"
LAMBDA_CRISE = 0.15  # Probabilité annuelle (Target Date)
SEVERITE_EQ_MOYENNE = -0.25
SEVERITE_EQ_SIGMA = 0.10
SEVERITE_BD_MOYENNE = -0.08
SEVERITE_BD_SIGMA = 0.03

# Paramètres crise détaillés (Fixed Mix)
PARAMS_CRISE_DETAIL = {
    'drop_eq': 0.35,
    'drop_bd': 0.05,
    'duree_mois': 18,
    'facteur_vol': 2.5
}

# --- Drawdown (Target Date uniquement) ---
DRAWDOWN_AVANT_APPORT = True  # True = marché, False = investisseur

''' ═══════════════════════════════════════════════════════════════════════
    🎨 DÉFINITION DES PROFILS
    ═══════════════════════════════════════════════════════════════════════ '''

PROFILS = {
    "PRUDENT": {
        "description": "Privilégie la sécurité",
        "equity": "Global Equity USD Hedged",
        "bond": "US Government Bond USD Unhedged",
        # Target Date
        "allocation_initiale": 0.90,
        "decroissance_annuelle": 0.03,
        # Fixed Mix
        "fixed_allocation": 0.20
    },
    "MODERE": {
        "description": "Équilibre sécurité et croissance",
        "equity": "Global Equity USD Hedged",
        "bond": "US Inflation Linked Bond - USD Unhedged",
        "allocation_initiale": 0.90,
        "decroissance_annuelle": 0.02,
        "fixed_allocation": 0.40
    },
    "EQUILIBRE": {
        "description": "Balance risque/rendement",
        "equity": "US Equity USD Unhedged",
        "bond": "USD Corporate Bond - USD Unhedged",
        "allocation_initiale": 0.95,
        "decroissance_annuelle": 0.01,
        "fixed_allocation": 0.60
    },
    "DYNAMIQUE": {
        "description": "Recherche de performance",
        "equity": "US Equity USD Unhedged",
        "bond": "USD Corporate Bond - USD Unhedged",
        "allocation_initiale": 1.00,
        "decroissance_annuelle": 0.008,
        "fixed_allocation": 0.80
    },
    "AGRESSIF": {
        "description": "Maximise le rendement",
        "equity": "US Equity USD Unhedged",
        "bond": "US High Yield Bond BB-B - USD Unhedged",
        "allocation_initiale": 1.00,
        "decroissance_annuelle": 0.005,
        "fixed_allocation": 0.95
    }
}

profil = PROFILS[PROFIL_CHOISI]
Equity = profil["equity"]
Bond = profil["bond"]

''' ═══════════════════════════════════════════════════════════════════════
    📊 CHARGEMENT DES DONNÉES
    ═══════════════════════════════════════════════════════════════════════ '''

def charger_parametres_actifs(script_dir):
    """Charge les paramètres μ, σ, ρ depuis Excel"""
    excel_path = os.path.join(script_dir, "AssumptionForSimulation.xlsx")
    
    if not os.path.exists(excel_path):
        print("⚠️  Fichier Excel manquant, utilisation paramètres par défaut")
        return 0.07, 0.15, 0.03, 0.05, 0.3
    
    try:
        df_BS = pd.read_excel(excel_path, sheet_name=0)
        
        # Mapping noms
        mapping = {
            'US Government Bond': 'US Government Bond USD Unhedged',
            'US Inflation Linked Bond': 'US Inflation Linked Bond - USD Unhedged',
            'USD Corporate Bond': 'USD Corporate Bond - USD Unhedged',
            'US High Yield Bond BB-B': 'US High Yield Bond BB-B - USD Unhedged',
            'Global Equity': 'Global Equity USD Hedged',
            'US Equity': 'US Equity USD Unhedged',
            'Japan Equity': 'Japan Equity - USD Unhedged',
            'Asia Pacific ex Japan Equity': 'Asia Pacific ex Japan Equity USD Hedged'
        }
        df_BS['Asset Name'] = df_BS['Asset Name'].replace(mapping)
        
        def get_params(asset_name):
            row = df_BS[df_BS['Asset Name'] == asset_name]
            if row.empty:
                print(f"⚠️  Actif {asset_name} non trouvé")
                return (0.05, 0.10, 0.3)
            mu = row['Expected Return'].values[0]
            sigma = row['Volatility'].values[0]
            corr = row['Correlation'].values[0] if 'Correlation' in row.columns else 0.3
            return (mu, sigma, corr)
        
        mu_e, sigma_e, _ = get_params(Equity)
        mu_b, sigma_b, corr_eb = get_params(Bond)
        
        return mu_e, sigma_e, mu_b, sigma_b, corr_eb
        
    except Exception as e:
        print(f"⚠️  Erreur lecture Excel: {e}")
        return 0.07, 0.15, 0.03, 0.05, 0.3

''' ═══════════════════════════════════════════════════════════════════════
    🧮 FONCTIONS COMMUNES - CALCUL DES APPORTS
    ═══════════════════════════════════════════════════════════════════════ '''

def calculer_apport_quadratique(t_annees, apport_init, duree_totale):
    """
    Modèle Target Date : courbe en cloche (quadratique)
    Pic à RATIO_PIC_CARRIERE de la durée totale
    """
    t_pic = duree_totale * RATIO_PIC_CARRIERE
    apport_max = apport_init * FACTEUR_CROISSANCE_MAX
    
    if t_pic > 0:
        a = (apport_init - apport_max) / (t_pic**2)
    else:
        return apport_init
    
    apport = a * (t_annees - t_pic)**2 + apport_max
    return max(apport, 0)

def estimer_salaire_saturation(t_annees, S_init, S_max):
    """Modèle Fixed Mix : salaire avec saturation exponentielle"""
    return S_init + (S_max - S_init) * (1 - np.exp(-VITESSE_PROGRESSION * t_annees))

def precalculer_parametres_apport_exponentiel(S_init, S_max, duree_totale):
    """Calcule les paramètres pour l'apport avec élasticité (Fixed Mix)"""
    ratio = S_max / S_init
    facteur = ratio ** GAMMA_ELASTICITE
    app_init = S_init * TAUX_APPORT_BASE
    app_max = app_init * facteur
    
    s_cible = S_init + (S_max - S_init) * SEUIL_MATURITE
    if s_cible >= S_max:
        t_pic = duree_totale
    else:
        t_pic = -np.log(1 - min((s_cible - S_init) / (S_max - S_init), 0.9999)) / VITESSE_PROGRESSION
    
    return app_init, app_max, min(max(0, t_pic), duree_totale)

def calculer_apport_exponentiel(t_annees, app_init, app_max, t_pic):
    """Calcule l'apport mensuel selon modèle exponentiel (Fixed Mix)"""
    if t_pic <= 0:
        return app_init
    a = (app_init - app_max) / (t_pic**2)
    return max(a * (t_annees - t_pic)**2 + app_max, 0)

''' ═══════════════════════════════════════════════════════════════════════
    🧮 FONCTIONS COMMUNES - ALLOCATION
    ═══════════════════════════════════════════════════════════════════════ '''

def calculer_allocation_target_date(age):
    """Allocation décroissante selon l'âge (Target Date)"""
    annees_ecoulees = age - AGE_DEPART
    pct_equity = profil['allocation_initiale'] - profil['decroissance_annuelle'] * annees_ecoulees
    pct_equity = max(0.05, min(1.0, pct_equity))
    return pct_equity, 1 - pct_equity

def calculer_allocation_fixed_mix():
    """Allocation fixe constante (Fixed Mix)"""
    pct_equity = profil['fixed_allocation']
    return pct_equity, 1 - pct_equity

''' ═══════════════════════════════════════════════════════════════════════
    📈 GÉNÉRATION DES RENDEMENTS
    ═══════════════════════════════════════════════════════════════════════ '''

def generer_rendements_correles_base(mu_e, sigma_e, mu_b, sigma_b, corr, nb_periodes, nb_sims):
    """
    Génère des rendements corrélés selon Black-Scholes
    Utilisé pour Target Date (tout stochastique)
    """
    r_e_m = mu_e / 12
    r_b_m = mu_b / 12
    sig_e_m = sigma_e / np.sqrt(12)
    sig_b_m = sigma_b / np.sqrt(12)
    
    cov = np.array([
        [sig_e_m**2, corr * sig_e_m * sig_b_m],
        [corr * sig_e_m * sig_b_m, sig_b_m**2]
    ])
    
    chocs = np.random.multivariate_normal([0, 0], cov, size=(nb_periodes, nb_sims))
    rend_eq = r_e_m - 0.5 * sig_e_m**2 + chocs[:, :, 0]
    rend_bd = r_b_m - 0.5 * sig_b_m**2 + chocs[:, :, 1]
    
    return rend_eq, rend_bd

def generer_rendements_avec_backtest(mu_e, sigma_e, mu_b, sigma_b, corr, dates, date_pivot, nb_sims):
    """
    Génère rendements avec séparation backtest/forecast (Fixed Mix)
    Backtest : même historique pour tous (seed fixe)
    Forecast : divergence stochastique
    """
    dt = 1.0 / 12.0
    nb_total_mois = len(dates)
    dates_pd = pd.to_datetime(dates)
    pivot_ts = pd.Timestamp(date_pivot)
    
    # Déterminer l'indice de split
    if pivot_ts < dates_pd[0]:
        idx_split = 0
    elif pivot_ts > dates_pd[-1]:
        idx_split = nb_total_mois
    else:
        idx_split = np.searchsorted(dates_pd, pivot_ts)
    
    s_e = sigma_e * np.sqrt(dt)
    s_b = sigma_b * np.sqrt(dt)
    cov = np.array([[s_e**2, corr*s_e*s_b], [corr*s_e*s_b, s_b**2]])
    
    # Partie Backtest (commune à toutes les simulations)
    if idx_split > 0:
        np.random.seed(42)  # Seed fixe pour reproductibilité
        chocs_histo = np.random.multivariate_normal([0, 0], cov, size=idx_split)
        r_eq_h = (mu_e*dt - 0.5*s_e**2) + chocs_histo[:, 0]
        r_bd_h = (mu_b*dt - 0.5*s_b**2) + chocs_histo[:, 1]
        
        # Duplication du même passé pour toutes les simulations
        r_eq_past = np.tile(r_eq_h.reshape(-1, 1), (1, nb_sims))
        r_bd_past = np.tile(r_bd_h.reshape(-1, 1), (1, nb_sims))
    else:
        r_eq_past = np.empty((0, nb_sims))
        r_bd_past = np.empty((0, nb_sims))
    
    # Partie Forecast (divergente)
    nb_mois_futur = nb_total_mois - idx_split
    if nb_mois_futur > 0:
        np.random.seed(None)  # Seed aléatoire pour divergence
        chocs_futur = np.random.multivariate_normal([0, 0], cov, size=(nb_mois_futur, nb_sims))
        r_eq_fut = (mu_e*dt - 0.5*s_e**2) + chocs_futur[:, :, 0]
        r_bd_fut = (mu_b*dt - 0.5*s_b**2) + chocs_futur[:, :, 1]
    else:
        r_eq_fut = np.empty((0, nb_sims))
        r_bd_fut = np.empty((0, nb_sims))
    
    return np.vstack([r_eq_past, r_eq_fut]), np.vstack([r_bd_past, r_bd_fut]), idx_split

''' ═══════════════════════════════════════════════════════════════════════
    💥 MODÈLES DE CRISE
    ═══════════════════════════════════════════════════════════════════════ '''

def ajouter_chocs_merton(rendements_eq, rendements_bd, nb_periodes, nb_sims):
    """
    Jump-Diffusion de Merton (Target Date)
    ⚠️ CORRECTION : Application probabiliste, pas systématique
    """
    lambda_mensuelle = LAMBDA_CRISE / 12
    
    for t in range(nb_periodes):
        # Tirage binomial : y a-t-il une crise ce mois ?
        crise = np.random.binomial(1, lambda_mensuelle, nb_sims)
        
        # Appliquer choc uniquement si crise=1
        choc_eq = crise * np.random.normal(SEVERITE_EQ_MOYENNE, SEVERITE_EQ_SIGMA, nb_sims)
        choc_bd = crise * np.random.normal(SEVERITE_BD_MOYENNE, SEVERITE_BD_SIGMA, nb_sims)
        
        rendements_eq[t] += choc_eq
        rendements_bd[t] += choc_bd
    
    return rendements_eq, rendements_bd

def injecter_crise_localisee(r_eq, r_bd, dates_list, date_depart, params):
    """
    Crise localisée à une date précise (Fixed Mix)
    Choc initial + volatilité accrue pendant N mois
    """
    r_eq_m = r_eq.copy()
    r_bd_m = r_bd.copy()
    
    dates_pd = pd.to_datetime(dates_list)
    idx = np.argmin(np.abs(dates_pd - pd.Timestamp(date_depart)))
    
    # Vérifier que la date est bien dans la période
    if abs((dates_pd[idx] - pd.Timestamp(date_depart)).days) > 40:
        return r_eq, r_bd
    
    # Choc initial (log-return pour cohérence)
    r_eq_m[idx, :] = np.log(1.0 - params.get('drop_eq', 0.3))
    r_bd_m[idx, :] = np.log(1.0 - params.get('drop_bd', 0.0))
    
    # Volatilité accrue pendant la période de récupération
    end = min(idx + params.get('duree_mois', 12), r_eq.shape[0])
    facteur = params.get('facteur_vol', 2.0)
    r_eq_m[idx+1:end, :] *= facteur
    r_bd_m[idx+1:end, :] *= facteur
    
    return r_eq_m, r_bd_m

''' ═══════════════════════════════════════════════════════════════════════
    🎯 SIMULATION PRINCIPALE
    ═══════════════════════════════════════════════════════════════════════ '''

def simuler_accumulation_target_date(capital_init, r_eq, r_bd, dates):
    """
    Simulation Target Date Fund avec :
    - Allocation décroissante
    - Rééquilibrage annuel
    - Apport quadratique
    - Drawdown avant/après apport
    """
    nb_steps, nb_sims = r_eq.shape
    
    # Initialisation
    eq_shares = np.zeros(nb_sims)
    bd_shares = np.zeros(nb_sims)
    eq_price = np.ones(nb_sims) * 100.0
    bd_price = np.ones(nb_sims) * 100.0
    
    age_actuel = AGE_DEPART
    pct_eq, pct_bd = calculer_allocation_target_date(age_actuel)
    
    # Allocation initiale
    eq_shares = (capital_init * pct_eq) / eq_price
    bd_shares = (capital_init * pct_bd) / bd_price
    
    apport_base_init = SALAIRE_INITIAL * TAUX_APPORT_BASE
    
    # Historiques
    historique_capital = np.zeros((nb_steps + 1, nb_sims))
    historique_capital[0, :] = capital_init
    historique_apport = np.zeros(nb_steps)
    historique_drawdown = np.zeros((nb_steps, nb_sims))
    courbe_investi = np.zeros(nb_steps + 1)
    courbe_investi[0] = capital_init
    
    total_apports = 0.0
    capital_max = np.ones(nb_sims) * capital_init
    
    for k in range(nb_steps):
        annee_courante = k // 12
        t_annees = k / 12.0
        age_actuel = AGE_DEPART + annee_courante
        
        # 1. Performance marché
        eq_price *= (1 + r_eq[k])
        bd_price *= (1 + r_bd[k])
        
        # 2. Capital avant apport (pour drawdown "marché")
        capital_avant = eq_shares * eq_price + bd_shares * bd_price
        
        # 3. Apport mensuel
        apport_mensuel = calculer_apport_quadratique(t_annees, apport_base_init, NB_ANNEES_ACCUMULATION)
        historique_apport[k] = apport_mensuel
        total_apports += apport_mensuel
        courbe_investi[k+1] = courbe_investi[k] + apport_mensuel
        
        pct_eq_cible, pct_bd_cible = calculer_allocation_target_date(age_actuel)
        
        # 4. Achat avec apport
        eq_buy = apport_mensuel * pct_eq_cible
        bd_buy = apport_mensuel * pct_bd_cible
        eq_shares += eq_buy / eq_price
        bd_shares += bd_buy / bd_price
        
        # 5. Rééquilibrage annuel (fin d'année)
        if (k % 12) == 11:
            total_val = eq_shares * eq_price + bd_shares * bd_price
            age_suivant = age_actuel + 1
            pct_eq_suivant, pct_bd_suivant = calculer_allocation_target_date(age_suivant)
            
            target_eq_val = total_val * pct_eq_suivant
            target_bd_val = total_val * pct_bd_suivant
            
            eq_val = eq_shares * eq_price
            bd_val = bd_shares * bd_price
            
            # Rééquilibrage par vente/achat (vectorisé pour gérer les arrays)
            # Cas 1: trop d'equity, vendre equity et acheter bonds
            surponderation_eq = (eq_val > target_eq_val) & (eq_shares > 0)
            vente_eq = np.where(surponderation_eq, (eq_val - target_eq_val) / eq_price, 0)
            achat_bd = np.where(surponderation_eq, (eq_val - target_eq_val) / bd_price, 0)
            
            # Cas 2: pas assez d'equity, vendre bonds et acheter equity
            sousponderation_eq = (eq_val < target_eq_val) & (bd_shares > 0)
            vente_bd_theorique = np.where(sousponderation_eq, (target_eq_val - eq_val) / bd_price, 0)
            vente_bd = np.minimum(vente_bd_theorique, bd_shares)
            achat_eq = np.where(sousponderation_eq, (vente_bd * bd_price) / eq_price, 0)
            
            # Application des transactions
            eq_shares = eq_shares - vente_eq + achat_eq
            bd_shares = bd_shares + achat_bd - vente_bd
        
        # 6. Capital après apport et rebalancing
        capital_apres = eq_shares * eq_price + bd_shares * bd_price
        historique_capital[k+1, :] = capital_apres
        
        # 7. Calcul drawdown
        if DRAWDOWN_AVANT_APPORT:
            capital_ref = capital_avant
        else:
            capital_ref = capital_apres
        
        capital_max = np.maximum(capital_max, capital_ref)
        dd = (capital_ref - capital_max) / capital_max
        dd = np.where(capital_max > 0, dd, 0.0)
        historique_drawdown[k, :] = dd
    
    return historique_capital, courbe_investi, historique_apport, historique_drawdown

def simuler_accumulation_fixed_mix(capital_init, r_eq, r_bd, dates):
    """
    Simulation Fixed Mix avec :
    - Allocation fixe
    - Apport avec saturation exponentielle
    - Pas de rééquilibrage explicite
    """
    nb_steps, nb_sims = r_eq.shape
    
    # Matrice de capital
    mat_cap = np.zeros((nb_steps + 1, nb_sims))
    mat_cap[0, :] = capital_init
    
    courbe_investi = np.zeros(nb_steps + 1)
    courbe_investi[0] = capital_init
    
    hist_salaire = np.zeros(nb_steps)
    hist_apport = np.zeros(nb_steps)
    
    # Précalcul paramètres apport
    app_init, app_max, t_pic = precalculer_parametres_apport_exponentiel(
        SALAIRE_INITIAL, SALAIRE_MAX_CIBLE, NB_ANNEES_ACCUMULATION
    )
    
    # Allocation fixe
    alloc_eq, alloc_bd = calculer_allocation_fixed_mix()
    
    for t in range(nb_steps):
        annee_f = t / 12.0
        
        # Salaire et apport
        salaire_mensuel = estimer_salaire_saturation(annee_f, SALAIRE_INITIAL, SALAIRE_MAX_CIBLE)
        apport_mensuel = calculer_apport_exponentiel(annee_f, app_init, app_max, t_pic)
        
        hist_salaire[t] = salaire_mensuel
        hist_apport[t] = apport_mensuel
        
        # Performance portefeuille (allocation fixe)
        rendement = alloc_eq * r_eq[t, :] + alloc_bd * r_bd[t, :]
        
        # Mise à jour capital
        mat_cap[t+1, :] = mat_cap[t, :] * np.exp(rendement) + apport_mensuel
        courbe_investi[t+1] = courbe_investi[t] + apport_mensuel
    
    # Drawdown (calculé a posteriori pour Fixed Mix)
    historique_drawdown = np.zeros((nb_steps, nb_sims))
    for sim in range(nb_sims):
        capital_max_sim = mat_cap[0, sim]
        for t in range(nb_steps):
            capital_max_sim = max(capital_max_sim, mat_cap[t+1, sim])
            dd = (mat_cap[t+1, sim] - capital_max_sim) / capital_max_sim if capital_max_sim > 0 else 0
            historique_drawdown[t, sim] = dd
    
    return mat_cap, courbe_investi, hist_apport, historique_drawdown, hist_salaire

''' ═══════════════════════════════════════════════════════════════════════
    📊 ANALYSE & KPIs
    ═══════════════════════════════════════════════════════════════════════ '''

def calculer_tri_annualise(capital_init, liste_apports, val_finale, freq=12):
    """Calcule le TRI (IRR) annualisé"""
    flux = [-float(capital_init)]
    flux.extend([-float(m) for m in liste_apports])
    flux.append(float(val_finale))
    
    def npv(r):
        if r <= -1.0:
            return 1e10
        valeurs = np.array(flux)
        puissances = np.arange(len(flux))
        return np.sum(valeurs / ((1 + r) ** puissances))
    
    try:
        tri_periodique = optimize.brentq(npv, -0.05, 0.10)
        return ((1 + tri_periodique) ** freq - 1) * 100
    except:
        return 0.0

def simuler_decumulation(capitaux_finaux, dernier_salaire, taux_livret, duree):
    """Simule la phase de retraite avec rente viagère"""
    nb_sims = len(capitaux_finaux)
    taux_remp = np.zeros((duree, nb_sims))
    cap_courant = capitaux_finaux.copy()
    
    for i in range(duree):
        restant = duree - i
        pension_mensuelle = cap_courant / (12 * restant)
        taux_remp[i, :] = pension_mensuelle / dernier_salaire
        cap_courant = (cap_courant - pension_mensuelle * 12) * (1 + taux_livret)
        cap_courant = np.maximum(cap_courant, 0)
    
    return taux_remp

def calcul_kpi_complets(capitaux_finaux, total_investi, mat_cap_historique):
    """Calcule tous les KPIs de risque"""
    
    # Shortfall Probability
    nb_pertes = np.sum(capitaux_finaux < total_investi)
    proba_shortfall = nb_pertes / len(capitaux_finaux)
    
    # VaR 95
    var_95 = np.percentile(capitaux_finaux, 5)
    
    # Sortino Ratio
    idx_med = np.argsort(capitaux_finaux)[len(capitaux_finaux)//2]
    trajectoire_med = mat_cap_historique[:, idx_med]
    rendements = np.diff(trajectoire_med) / trajectoire_med[:-1]
    rendements_neg = rendements[rendements < 0]
    downside = np.std(rendements_neg) * np.sqrt(12) if len(rendements_neg) > 0 else 1e-6
    sortino = (np.mean(rendements) * 12) / downside
    
    # Max Underwater Duration
    plus_haut = np.maximum.accumulate(trajectoire_med)
    is_underwater = trajectoire_med < plus_haut
    duree_max, compteur = 0, 0
    for s in is_underwater:
        if s:
            compteur += 1
        else:
            duree_max = max(duree_max, compteur)
            compteur = 0
    max_underwater = max(duree_max, compteur) / 12.0
    
    # Dispersion
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

''' ═══════════════════════════════════════════════════════════════════════
    🎬 EXÉCUTION PRINCIPALE
    ═══════════════════════════════════════════════════════════════════════ '''

def main():
    print("\n" + "="*80)
    print(f"🎯 SIMULATION ALM - MÉTHODE : {METHODE}")
    print("="*80)
    print(f"Profil : {PROFIL_CHOISI}")
    print(f"Horizon : {NB_ANNEES_ACCUMULATION} ans (âge {AGE_DEPART} → {AGE_DEPART + NB_ANNEES_ACCUMULATION})")
    print(f"Capital initial : {CAPITAL_INITIAL:,.0f} €")
    print(f"Salaire initial : {SALAIRE_INITIAL:,.0f} €/mois")
    print(f"Simulations : {NB_SIMULATIONS}")
    
    # Chargement paramètres
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mu_e, sigma_e, mu_b, sigma_b, corr_eb = charger_parametres_actifs(script_dir)
    
    print(f"\n📈 Paramètres marché :")
    print(f"  • Equity ({Equity[:30]}...)")
    print(f"    μ={mu_e*100:.2f}%, σ={sigma_e*100:.2f}%")
    print(f"  • Bond ({Bond[:30]}...)")
    print(f"    μ={mu_b*100:.2f}%, σ={sigma_b*100:.2f}%, ρ={corr_eb:.2f}")
    
    # Affichage allocation
    if METHODE == "TARGET_DATE":
        alloc_init = profil['allocation_initiale']
        decr = profil['decroissance_annuelle']
        alloc_fin = max(0.05, alloc_init - decr * NB_ANNEES_ACCUMULATION)
        print(f"\n🎯 Allocation Target Date :")
        print(f"  • Initiale : {alloc_init*100:.1f}% equity")
        print(f"  • Décroissance : {decr*100:.2f}%/an")
        print(f"  • Finale : {alloc_fin*100:.1f}% equity")
        print(f"  • Rééquilibrage : Annuel")
        print(f"  • Drawdown mesuré : {'AVANT apport (marché)' if DRAWDOWN_AVANT_APPORT else 'APRÈS apport'}")
    else:
        alloc_fixe = profil['fixed_allocation']
        print(f"\n🎯 Allocation Fixed Mix :")
        print(f"  • Constante : {alloc_fixe*100:.1f}% equity")
        print(f"  • Rééquilibrage : Implicite (via apports)")
    
    # Génération des dates
    dates = pd.date_range(start=DATE_DEBUT_T0, periods=NB_PERIODES_TOTAL, freq='ME')
    
    # Génération des rendements selon la méthode
    if METHODE == "TARGET_DATE":
        print(f"\n🎲 Génération rendements : Stochastique pur (B&S)")
        r_eq, r_bd = generer_rendements_correles_base(
            mu_e, sigma_e, mu_b, sigma_b, corr_eb, NB_PERIODES_TOTAL, NB_SIMULATIONS
        )
        
        # Application crises (Jump-Diffusion de Merton)
        if SIMULER_CRISE:
            print(f"💥 Ajout crises Jump-Diffusion (λ={LAMBDA_CRISE*100:.1f}%/an)")
            r_eq, r_bd = ajouter_chocs_merton(r_eq, r_bd, NB_PERIODES_TOTAL, NB_SIMULATIONS)
    
    else:  # FIXED_MIX
        print(f"\n🎲 Génération rendements : Backtest/Forecast (pivot {DATE_PIVOT_BACKTEST})")
        r_eq, r_bd, idx_split = generer_rendements_avec_backtest(
            mu_e, sigma_e, mu_b, sigma_b, corr_eb, dates, DATE_PIVOT_BACKTEST, NB_SIMULATIONS
        )
        print(f"  • Backtest : {idx_split} mois (historique commun)")
        print(f"  • Forecast : {NB_PERIODES_TOTAL - idx_split} mois (stochastique)")
        
        # Application crise localisée
        if SIMULER_CRISE and pd.Timestamp(DATE_CRISE) > pd.Timestamp(DATE_PIVOT_BACKTEST):
            print(f"💥 Injection crise localisée ({DATE_CRISE})")
            print(f"   Drop equity : {PARAMS_CRISE_DETAIL['drop_eq']*100:.1f}%")
            print(f"   Durée : {PARAMS_CRISE_DETAIL['duree_mois']} mois")
            r_eq, r_bd = injecter_crise_localisee(r_eq, r_bd, dates, DATE_CRISE, PARAMS_CRISE_DETAIL)
    
    # Simulation selon la méthode
    print(f"\n⚙️  Simulation en cours...")
    
    if METHODE == "TARGET_DATE":
        mat_cap, courbe_investi, hist_apport, hist_dd = simuler_accumulation_target_date(
            CAPITAL_INITIAL, r_eq, r_bd, dates
        )
        hist_salaire = None  # Pas calculé dans Target Date
    else:
        mat_cap, courbe_investi, hist_apport, hist_dd, hist_salaire = simuler_accumulation_fixed_mix(
            CAPITAL_INITIAL, r_eq, r_bd, dates
        )
    
    # Analyse des résultats
    capitaux_finaux = mat_cap[-1, :]
    total_investi = courbe_investi[-1]
    
    # Indices percentiles
    idx_sorted = np.argsort(capitaux_finaux)
    idx_p5 = idx_sorted[int(NB_SIMULATIONS * 0.05)]
    idx_p50 = idx_sorted[int(NB_SIMULATIONS * 0.50)]
    idx_p95 = idx_sorted[int(NB_SIMULATIONS * 0.95)]
    
    # TRI médian
    tri_median = calculer_tri_annualise(CAPITAL_INITIAL, hist_apport, capitaux_finaux[idx_p50])
    
    # KPIs complets
    kpis = calcul_kpi_complets(capitaux_finaux, total_investi, mat_cap)
    
    # Correction inflation
    coeff_inflation = 1 / ((1 + TAUX_INFLATION) ** NB_ANNEES_ACCUMULATION)
    capital_p5_reel = kpis['var_95'] * coeff_inflation
    gain_p5_reel = capital_p5_reel - total_investi
    
    # Décumulation
    if hist_salaire is not None:
        dernier_salaire = hist_salaire[-1]
    else:
        # Estimation pour Target Date
        dernier_salaire = SALAIRE_INITIAL * 1.5  # Approximation
    
    taux_remp = simuler_decumulation(capitaux_finaux, dernier_salaire, TAUX_LIVRET_A, DUREE_RETRAITE)
    
    # =========================================================================
    # AFFICHAGE RÉSULTATS
    # =========================================================================
    
    print("\n" + "="*80)
    print(f"📊 RÉSULTATS - {METHODE} - PROFIL {PROFIL_CHOISI}")
    print("="*80)
    
    print(f"\n💰 FLUX & CAPITAL :")
    print(f"  • Capital initial         : {CAPITAL_INITIAL:>15,.0f} €")
    print(f"  • Apports totaux          : {total_investi - CAPITAL_INITIAL:>15,.0f} €")
    print(f"  • Total investi           : {total_investi:>15,.0f} €")
    print(f"  • Capital final P5        : {capitaux_finaux[idx_p5]:>15,.0f} €")
    print(f"  • Capital final P50       : {capitaux_finaux[idx_p50]:>15,.0f} €")
    print(f"  • Capital final P95       : {capitaux_finaux[idx_p95]:>15,.0f} €")
    print(f"  • TRI médian              : {tri_median:>15.2f} %/an")
    
    print(f"\n📉 RISQUE & DOWNSIDE :")
    print(f"  • Shortfall Risk          : {kpis['shortfall_prob']*100:>15.2f} %")
    print(f"  • VaR 95% (P5 nominal)    : {kpis['var_95']:>15,.0f} €")
    print(f"  • P&L en cas de crise     : {kpis['gain_p5']:>+16,.0f} €")
    print(f"  • Max Underwater          : {kpis['max_underwater']:>15.1f} années")
    print(f"  • Sortino Ratio           : {kpis['sortino']:>15.2f}")
    print(f"  • Dispersion (P95-P5)     : {kpis['dispersion']:>15,.0f} €")
    
    # Max drawdown médian
    max_dd_median = np.median([np.min(hist_dd[:, sim]) for sim in range(NB_SIMULATIONS)])
    print(f"  • Max Drawdown médian     : {max_dd_median*100:>15.2f} %")
    
    print(f"\n💶 POUVOIR D'ACHAT (Inflation {TAUX_INFLATION*100:.1f}%/an) :")
    print(f"  • Capital P5 réel         : {capital_p5_reel:>15,.0f} €")
    print(f"  • P&L réel (worst case)   : {gain_p5_reel:>+16,.0f} €")
    
    print(f"\n🏖️  RETRAITE (Livret A {TAUX_LIVRET_A*100:.2f}%) :")
    print(f"  • Taux remplacement P5    : {taux_remp[0, idx_p5]*100:>15.1f} %")
    print(f"  • Taux remplacement P50   : {taux_remp[0, idx_p50]*100:>15.1f} %")
    print(f"  • Taux remplacement P95   : {taux_remp[0, idx_p95]*100:>15.1f} %")
    
    if gain_p5_reel < 0:
        print(f"\n⚠️  ALERTE : Destruction de richesse réelle en scénario adverse !")
        print(f"   Perte : {abs(gain_p5_reel):,.0f} € (pouvoir d'achat)")
    
    print("="*80)
    
    # =========================================================================
    # VISUALISATION
    # =========================================================================
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle(f"ANALYSE ALM - {METHODE} - PROFIL {PROFIL_CHOISI}", 
                 fontsize=16, fontweight='bold', y=0.96)
    
    dates_plot = [pd.Timestamp(DATE_DEBUT_T0)] + list(dates)
    
    # (1) Distribution capital final
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
    
    # (2) Fan chart capital
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
    
    # (3) Taux de remplacement
    ax = axes[1, 0]
    annees_retraite = np.arange(1, DUREE_RETRAITE + 1)
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
    
    # (4) Rendements annuels
    ax = axes[1, 1]
    df_perf = pd.DataFrame(mat_cap)
    perf_annuelle = df_perf.pct_change(12).iloc[12::12] * 100
    annees_simu = np.arange(1, len(perf_annuelle) + 1)
    
    ax.plot(annees_simu, perf_annuelle.iloc[:, idx_p95], 
            color="#4d2ca0", alpha=0.6, label='P95 (Scénario Haut)')
    ax.plot(annees_simu, perf_annuelle.iloc[:, idx_p50], 
            color='black', linewidth=2, label='Médiane')
    ax.plot(annees_simu, perf_annuelle.iloc[:, idx_p5], 
            color='#d62728', alpha=0.6, label='P5 (Scénario Bas)')
    ax.axhline(0, color='black', linewidth=0.8, alpha=0.5)
    
    # Ligne pivot si Fixed Mix
    if METHODE == "FIXED_MIX":
        idx_pivot_annee = int((pd.Timestamp(DATE_PIVOT_BACKTEST) - pd.Timestamp(DATE_DEBUT_T0)).days / 365)
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
    
    print("\n✅ Simulation terminée avec succès !\n")

if __name__ == "__main__":
    main()