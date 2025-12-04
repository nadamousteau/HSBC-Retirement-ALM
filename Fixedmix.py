#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation d'investissement - STRATÉGIE FIXED MIX (Allocation Constante)
- Backtesting sur données historiques puis forecasts Black & Scholes
- Allocation FIXE déterminée automatiquement par le PROFIL choisi
- Rééquilibrage MENSUEL
- MODELE D'APPORT : QUADRATIQUE (Réaliste selon Bruder et al. 2025)
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

''' ========== ACTIFS DISPONIBLES ========== '''

BONDS_SAFE = {
    "US Government Bond USD Unhedged": "Bond Gov",
    "US Inflation Linked Bond - USD Unhedged": "Bond Inflation"
}

BONDS_RISKY = {
    "US High Yield Bond BB-B - USD Unhedged": "High Yield",
    "USD Corporate Bond - USD Unhedged": "Corp Bond"
}

EQUITIES = {
    "Asia Pacific ex Japan Equity USD Hedged": "Asia Pacific",  
    "Global Equity USD Hedged": "Global Equity",
    "Japan Equity - USD Unhedged": "Japan Equity",
    "US Equity USD Unhedged": "US Equity"
}

''' ========== PROFILS INVESTISSEURS (ACTIFS + ALLOCATION FIXE) ========== '''

PROFILS = {
    "PRUDENT": {
        "description": "Sécuritaire (20% Actions / 80% Obligations)",
        "equity": "Global Equity USD Hedged", 
        "bond": "US Government Bond USD Unhedged", 
        "fixed_allocation": 0.20  # 20% Actions
    },
    "MODERE": {
        "description": "Défensif (40% Actions / 60% Obligations)",
        "equity": "Global Equity USD Hedged",
        "bond": "USD Corporate Bond - USD Unhedged", 
        "fixed_allocation": 0.40  # 40% Actions
    },
    "EQUILIBRE": {
        "description": "Le classique 60/40 (60% Actions / 40% Obligations)",
        "equity": "US Equity USD Unhedged", 
        "bond": "USD Corporate Bond - USD Unhedged",
        "fixed_allocation": 0.60  # 60% Actions
    },
    "DYNAMIQUE": {
        "description": "Offensif (80% Actions / 20% Obligations)",
        "equity": "US Equity USD Unhedged",
        "bond": "US High Yield Bond BB-B - USD Unhedged", 
        "fixed_allocation": 0.80  # 80% Actions
    },
    "AGRESSIF": {
        "description": "Maximisation (95% Actions / 5% Obligations)",
        "equity": "US Equity USD Unhedged", 
        "bond": "US High Yield Bond BB-B - USD Unhedged",
        "fixed_allocation": 0.95  # 95% Actions
    }
}

''' ========== PARAMÈTRES DE SIMULATION ========== '''


PROFIL_CHOISI = "EQUILIBRE"  # PRUDENT, MODERE, EQUILIBRE, DYNAMIQUE, AGRESSIF
# ----------------------------------

# Paramètres financiers
nb_annees = 35
t0 = "2001-12-31"  # Date de départ
age_depart = 30
capital_initial = 10000
salaire_initial = 3000
taux_apport = 0.10
taux_inflation = 0.02 

# Monte Carlo
nb_simulations = 500
nb_pas_par_an = 12
nb_periodes_total = nb_annees * nb_pas_par_an

# Sélection automatique des paramètres selon le profil
profil = PROFILS[PROFIL_CHOISI]
Equity = profil["equity"]
Bond = profil["bond"]
FIXED_ALLOCATION_EQUITY = profil["fixed_allocation"]

''' ========== CHARGEMENT DES DONNÉES ========== '''

script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "HistoricalAssetReturn.csv")

df_historical = pd.read_csv(file_path, header=None)
df_historical.columns = df_historical.iloc[1]  
df_historical = df_historical.rename(columns={"Asset Class - Name": "Date"})
df_historical = df_historical.drop(index=0) 
df_historical = df_historical.drop(index=1)
df_historical = df_historical.drop(index=2)  
df_historical["Date"] = pd.to_datetime(df_historical["Date"], errors="coerce")

for col in df_historical.columns:
    if col != "Date":
        df_historical[col] = df_historical[col].astype(float)

# Vérification actifs
if Equity not in df_historical.columns:
    raise ValueError(f"Actif {Equity} non trouvé dans HistoricalAssetReturn.csv")
if Bond not in df_historical.columns:
    raise ValueError(f"Actif {Bond} non trouvé dans HistoricalAssetReturn.csv")

''' ========== ANALYSE DE VOLATILITÉ HISTORIQUE ========== '''

print("=" * 80)
print("📊 ANALYSE DES ACTIFS DISPONIBLES")
print("=" * 80)

# Calcul volatilités historiques (annualisées)
print("\n🔵 ACTIONS (Equity) - Volatilité historique annualisée:")
for equity_name, short_name in EQUITIES.items():
    if equity_name in df_historical.columns:
        returns = df_historical[equity_name].dropna()
        vol_annuelle = returns.std() * np.sqrt(12) * 100
        rendement_moyen = returns.mean() * 12 * 100
        print(f"  • {short_name:20s}: Vol={vol_annuelle:5.2f}%  | Rdt moy={rendement_moyen:6.2f}%")

print("\n🟢 OBLIGATIONS SÛRES (Bonds Safe):")
for bond_name, short_name in BONDS_SAFE.items():
    if bond_name in df_historical.columns:
        returns = df_historical[bond_name].dropna()
        vol_annuelle = returns.std() * np.sqrt(12) * 100
        rendement_moyen = returns.mean() * 12 * 100
        print(f"  • {short_name:20s}: Vol={vol_annuelle:5.2f}%  | Rdt moy={rendement_moyen:6.2f}%")

print("\n🟡 OBLIGATIONS RISQUÉES (Bonds Risky):")
for bond_name, short_name in BONDS_RISKY.items():
    if bond_name in df_historical.columns:
        returns = df_historical[bond_name].dropna()
        vol_annuelle = returns.std() * np.sqrt(12) * 100
        rendement_moyen = returns.mean() * 12 * 100
        print(f"  • {short_name:20s}: Vol={vol_annuelle:5.2f}%  | Rdt moy={rendement_moyen:6.2f}%")

print("\n" + "=" * 80)
print(f"🎯 STRATÉGIE FIXED MIX ({FIXED_ALLOCATION_EQUITY*100:.0f}/{100-FIXED_ALLOCATION_EQUITY*100:.0f})")
print("=" * 80)
print(f"Profil sélectionné: {PROFIL_CHOISI}")
print(f"Description: {profil['description']}")
print(f"Equity: {EQUITIES.get(Equity, Equity)}")
print(f"Bond: {BONDS_SAFE.get(Bond, BONDS_RISKY.get(Bond, Bond))}")
print(f"Allocation CONSTANTE: {FIXED_ALLOCATION_EQUITY*100:.0f}% equity")
print("Rééquilibrage: MENSUEL")
print("Modèle d'apport: QUADRATIQUE (Réaliste)")

''' ========== PARAMÈTRES POUR FORECASTS ========== '''

excel_path = os.path.join(script_dir, "AssumptionForSimulation.xlsx")
df_BS = pd.read_excel(excel_path, sheet_name=0)

df_BS['Asset Name'] = df_BS['Asset Name'].replace({
    'Liquidity USD': 'Liquidity USD',
    'US Government Bond': 'US Government Bond USD Unhedged',
    'US Inflation Linked Bond': 'US Inflation Linked Bond - USD Unhedged',
    'US High Yield Bond BB-B': 'US High Yield Bond BB-B - USD Unhedged',
    'USD Corporate Bond': 'USD Corporate Bond - USD Unhedged',
    'Asia Pacific ex Japan Equity': 'Asia Pacific ex Japan Equity USD Hedged',
    'Global Equity': 'Global Equity USD Hedged',
    'Japan Equity': 'Japan Equity - USD Unhedged',
    'US Equity': 'US Equity USD Unhedged',
    'US Property': 'US Property'
})

def get_params(asset_name):
    row = df_BS[df_BS['Asset Name'] == asset_name]
    if row.empty:
        raise ValueError(f"Actif {asset_name} non trouvé")
    mu = row['Expected Return'].values[0]
    sigma = row['Volatility'].values[0]
    corr = row['Correlation'].values[0] if 'Correlation' in row.columns else 0.3
    return (mu, sigma, corr)

params_equity = get_params(Equity)
params_bond = get_params(Bond)

mu_e, sigma_e, _ = params_equity
mu_b, sigma_b, corr_eb = params_bond

print(f"\nParamètres de forecast (Black & Scholes):")
print(f"  • Equity: μ={mu_e*100:.2f}%, σ={sigma_e*100:.2f}%")
print(f"  • Bond: μ={mu_b*100:.2f}%, σ={sigma_b*100:.2f}%")
print(f"  • Corrélation: {corr_eb:.2f}")
print("=" * 80)

''' ========== ALLOCATION FIXED MIX ========== '''

def calculer_allocation(age):
    """
    Allocation FIXE dictée par le profil
    """
    return FIXED_ALLOCATION_EQUITY, 1 - FIXED_ALLOCATION_EQUITY

''' ========== MODÈLE D'APPORT QUADRATIQUE (CORRIGÉ & RÉALISTE) ========== '''
# ### MODIFICATION PAPIER ###
# Calibrage basé sur l'Appendix A.5 et la Figure 51 (Page 92)
# L'épargne monte, atteint un pic vers 50-55 ans, puis redescend.

def calculer_apport_quadratique(t_annees, apport_init, duree_totale):
    """
    Simule une courbe d'épargne en cloche asymétrique.
    Croissance jusqu'à un 'Age Pic', puis décroissance vers la retraite.
    
    t_annees : temps écoulé en années (0 à 35)
    apport_init : montant du premier versement (ex: 300€)
    duree_totale : durée de la simulation (ex: 35 ans)
    """
    
    # 1. Définition du Pic (Sommet de la carrière)
    # Selon le papier, le pic est souvent vers 50-55 ans.
    # Si on commence à 30 ans sur 35 ans, le pic est aux 2/3 du chemin.
    ratio_pic = 0.55  # Le pic arrive à 65% de la durée (env. 53 ans)
    t_pic = duree_totale * ratio_pic
    
    # 2. Hauteur du Pic
    # Facteur multiplicateur : On suppose qu'au sommet de sa carrière, 
    # on épargne 3x plus qu'au début (ex: 300€ -> 900€).
    facteur_croissance_max = 1.8
    apport_max = apport_init * facteur_croissance_max
    
    # 3. Calcul de la parabole (Forme Canonique : y = a(x-h)^2 + k)
    # On sait que c(0) = apport_init et c(t_pic) = apport_max
    # apport_init = a * (0 - t_pic)^2 + apport_max
    # a = (apport_init - apport_max) / (t_pic^2)
    
    if t_pic > 0:
        a = (apport_init - apport_max) / (t_pic**2)
    else:
        return apport_init

    apport = a * (t_annees - t_pic)**2 + apport_max
    
    # Sécurité : on ne peut pas avoir d'apport négatif (si la baisse est trop forte)
    return max(apport, 0)
''' ========== EXTRACTION RENDEMENTS HISTORIQUES ========== '''

def extraire_rendements_historiques(date_debut, nb_mois):
    start_index = df_historical.index[df_historical["Date"] == date_debut].tolist()
    if not start_index:
        raise ValueError(f"La date {date_debut} n'existe pas dans les données historiques")
    start_index = start_index[0]
    
    end_index = min(start_index + nb_mois, len(df_historical))
    df_periode = df_historical.iloc[start_index:end_index].reset_index(drop=True)
    
    rendements_eq = df_periode[Equity].values
    rendements_bd = df_periode[Bond].values
    dates = df_periode["Date"].values
    
    nb_mois_disponibles = len(rendements_eq)
    
    return rendements_eq, rendements_bd, dates, nb_mois_disponibles

''' ========== SIMULATION BLACK & SCHOLES ========== '''

def generer_rendements_correles(mu_e, sigma_e, mu_b, sigma_b, corr, nb_periodes, nb_sims):
    r_e_mensuel = mu_e / 12
    r_b_mensuel = mu_b / 12
    sigma_e_mensuel = sigma_e / np.sqrt(12)
    sigma_b_mensuel = sigma_b / np.sqrt(12)
    
    cov_matrix = np.array([
        [sigma_e_mensuel**2, corr * sigma_e_mensuel * sigma_b_mensuel],
        [corr * sigma_e_mensuel * sigma_b_mensuel, sigma_b_mensuel**2]
    ])
    
    chocs = np.random.multivariate_normal([0, 0], cov_matrix, size=(nb_periodes, nb_sims))
    
    rendements_equity = r_e_mensuel - 0.5 * sigma_e_mensuel**2 + chocs[:, :, 0]
    rendements_bonds = r_b_mensuel - 0.5 * sigma_b_mensuel**2 + chocs[:, :, 1]
    
    return rendements_equity, rendements_bonds

''' ========== MODE HYBRIDE ========== '''

rendements_eq_hist, rendements_bd_hist, dates_hist, nb_mois_hist = extraire_rendements_historiques(t0, nb_periodes_total)

if nb_mois_hist < nb_periodes_total:
    nb_mois_manquants = nb_periodes_total - nb_mois_hist
    print(f"\n📊 Mode HYBRIDE: {nb_mois_hist} mois historiques + {nb_mois_manquants} mois simulés (B&S)")
    
    rendements_eq_forecast, rendements_bd_forecast = generer_rendements_correles(
        mu_e, sigma_e, mu_b, sigma_b, corr_eb, nb_mois_manquants, nb_simulations
    )
    
    rendements_eq_scenarios = np.vstack([
        np.tile(rendements_eq_hist, (nb_simulations, 1)).T,
        rendements_eq_forecast
    ])
    rendements_bd_scenarios = np.vstack([
        np.tile(rendements_bd_hist, (nb_simulations, 1)).T,
        rendements_bd_forecast
    ])
else:
    print(f"\n📊 Période entièrement couverte par les données historiques ({nb_mois_hist} mois)")
    rendements_eq_scenarios = rendements_eq_hist.reshape(-1, 1)
    rendements_bd_scenarios = rendements_bd_hist.reshape(-1, 1)
    nb_simulations = 1

''' ========== BOUCLE DE SIMULATION (FIXED MIX REBALANCED MONTHLY) ========== '''

resultats_finaux = []
historiques_capital = []
# On stocke l'historique des apports pour le graphique
historique_apports_moyens = np.zeros(nb_periodes_total) 

for sim in range(nb_simulations):
    eq_shares = 0.0
    bd_shares = 0.0
    eq_price = 100.0
    bd_price = 100.0
    
    # Allocation initiale
    pct_eq, pct_bd = calculer_allocation(age_depart)
    
    eq_shares = (capital_initial * pct_eq) / eq_price
    bd_shares = (capital_initial * pct_bd) / bd_price
    
   
    # salaire_actuel = salaire_initial 
    historique_capital = [capital_initial]
    total_investi_sim = capital_initial
    
    for k in range(nb_periodes_total):
        mois = k + 1
        t_annees_ecoulees = k / 12.0
        annee_courante = k // 12
        age_actuel = age_depart + annee_courante
        
        # 1. Évolution des prix
        r_eq = rendements_eq_scenarios[k, sim]
        r_bd = rendements_bd_scenarios[k, sim]
        
        eq_price *= (1 + r_eq)
        bd_price *= (1 + r_bd)
        
        # 2. Apport mensuel : MODIFICATION QUADRATIQUE
       
        apport_base_init = salaire_initial * taux_apport
        apport_mensuel = calculer_apport_quadratique(t_annees_ecoulees, apport_base_init, nb_annees)
        
        # Ajustement optionnel pour l'inflation (pour garder la logique de pouvoir d'achat)
        apport_mensuel *= (1 + taux_inflation)**t_annees_ecoulees
        
        if sim == 0: 
            historique_apports_moyens[k] = apport_mensuel

        total_investi_sim += apport_mensuel
        
        # On achète selon la cible fixe
        pct_eq_cible, pct_bd_cible = calculer_allocation(age_actuel)
        
        eq_buy = apport_mensuel * pct_eq_cible
        bd_buy = apport_mensuel * pct_bd_cible
        
        eq_shares += eq_buy / eq_price
        bd_shares += bd_buy / bd_price
        
        # 4. Rééquilibrage MENSUEL (Target = Profil)
        total_val = eq_shares * eq_price + bd_shares * bd_price
        
        target_eq_val = total_val * pct_eq_cible
        target_bd_val = total_val * pct_bd_cible
        
        eq_val = eq_shares * eq_price
        bd_val = bd_shares * bd_price
        
        if eq_val > target_eq_val and eq_shares > 0:
            # On a trop d'actions, on vend equity -> bond
            vente_eq = (eq_val - target_eq_val) / eq_price
            eq_shares -= vente_eq
            bd_shares += (eq_val - target_eq_val) / bd_price
        elif eq_val < target_eq_val and bd_shares > 0:
            # On n'a pas assez d'actions, on vend bond -> equity
            vente_bd = (target_eq_val - eq_val) / bd_price
            vente_bd = min(vente_bd, bd_shares)
            bd_shares -= vente_bd
            eq_shares += (vente_bd * bd_price) / eq_price
        
        # Enregistrement
        capital_actuel = eq_shares * eq_price + bd_shares * bd_price
        historique_capital.append(capital_actuel)
    
    capital_final = eq_shares * eq_price + bd_shares * bd_price
    
    resultats_finaux.append({
        'capital_final': capital_final,
        'total_investi': total_investi_sim
    })
    
    historiques_capital.append(historique_capital)

''' ========== ANALYSE DES RÉSULTATS ========== '''

capitaux_finaux = [r['capital_final'] for r in resultats_finaux]
total_investi_moyen = np.mean([r['total_investi'] for r in resultats_finaux])

capital_median = np.median(capitaux_finaux)
capital_moyen = np.mean(capitaux_finaux)
capital_p10 = np.percentile(capitaux_finaux, 10)
capital_p90 = np.percentile(capitaux_finaux, 90)

rendement_median = ((capital_median / total_investi_moyen) - 1) * 100
rendement_moyen = ((capital_moyen / total_investi_moyen) - 1) * 100

print("\n" + "=" * 80)
print(f"📊 RÉSULTATS - PROFIL {PROFIL_CHOISI} (Fixed Mix)")
print("=" * 80)
print(f"Simulations: {nb_simulations} | Horizon: {nb_annees} ans")
print(f"Allocation: {FIXED_ALLOCATION_EQUITY*100:.0f}% Equity / {100-FIXED_ALLOCATION_EQUITY*100:.0f}% Bonds")
print(f"Modèle d'apport: Quadratique (Pic à {nb_annees*0.75:.0f} ans)")
print(f"\n💰 Investissement:")
print(f"  • Capital initial: {capital_initial:>15,.2f} €")
print(f"  • Apports totaux: {total_investi_moyen - capital_initial:>15,.2f} €")
print(f"  • Total investi: {total_investi_moyen:>15,.2f} €")
print(f"\n🎯 Capital final:")
print(f"  • Médian: {capital_median:>15,.2f} € (rdt: {rendement_median:>6.2f}%)")
print(f"  • Moyen: {capital_moyen:>15,.2f} € (rdt: {rendement_moyen:>6.2f}%)")
if nb_simulations > 1:
    print(f"  • P10 (pessimiste): {capital_p10:>15,.2f} €")
    print(f"  • P90 (optimiste): {capital_p90:>15,.2f} €")
    print(f"  • Écart P90-P10: {capital_p90 - capital_p10:>15,.2f} € ({(capital_p90/capital_p10-1)*100:.1f}%)")
print(f"\n✨ Gain médian: {capital_median - total_investi_moyen:>15,.2f} €")
print("=" * 80)

''' ========== VISUALISATIONS ========== '''

if nb_simulations == 1:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    mois = range(len(historiques_capital[0]))
    # Calcul manuel du cumul pour le graph single simulation
    cumul_investi = [capital_initial]
    curr_invest = capital_initial
    for m in range(len(historique_apports_moyens)):
        curr_invest += historique_apports_moyens[m]
        cumul_investi.append(curr_invest)

    axes[0, 0].plot(mois, historiques_capital[0], color='#2E86AB', linewidth=2, label='Capital valorisé')
    axes[0, 0].plot(mois, cumul_investi, color='#A23B72', linewidth=2, linestyle='--', label='Capital investi')
    axes[0, 0].set_title(f'Evolution du capital - {PROFIL_CHOISI}', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Mois', fontsize=11)
    axes[0, 0].set_ylabel('Capital (€)', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    ages = range(age_depart, age_depart + nb_annees + 1)
    allocations_eq = [calculer_allocation(age)[0] * 100 for age in ages]
    allocations_bd = [calculer_allocation(age)[1] * 100 for age in ages]
    
    
    annees_graph = np.array(range(nb_periodes_total)) / 12 + age_depart
    axes[0, 1].plot(annees_graph, historique_apports_moyens, linewidth=2.5, color='#D4AF37', label='Apport mensuel')
    axes[0, 1].set_title(f'Profil des Apports (Modèle Quadratique)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Âge', fontsize=11)
    axes[0, 1].set_ylabel('Apport Mensuel (€)', fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(rendements_eq_scenarios[:, 0] * 100, alpha=0.7, linewidth=1, label='Equity', color='#2E86AB')
    axes[1, 0].plot(rendements_bd_scenarios[:, 0] * 100, alpha=0.7, linewidth=1, label='Bonds', color='#A23B72')
    axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Rendements mensuels historiques', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Mois', fontsize=11)
    axes[1, 0].set_ylabel('Rendement (%)', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].axis('off')
    stats_text = f"""
    PROFIL {PROFIL_CHOISI}
    {profil['description']}
    
    Capital final: {capital_median:,.0f} €
    Total investi: {total_investi_moyen:,.0f} €
    Gain: {capital_median - total_investi_moyen:,.0f} €
    Rendement: {rendement_median:.2f}%
    
    Actifs:
    • {EQUITIES.get(Equity, Equity)[:20]}
    • {BONDS_SAFE.get(Bond, BONDS_RISKY.get(Bond, Bond))[:20]}
    """
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', 
                    family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

else:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    axes[0, 0].hist(capitaux_finaux, bins=50, color='#2E86AB', alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(capital_median, color='red', linestyle='--', linewidth=2, label=f'Médiane: {capital_median:,.0f} €')
    axes[0, 0].axvline(capital_moyen, color='green', linestyle='--', linewidth=2, label=f'Moyenne: {capital_moyen:,.0f} €')
    axes[0, 0].set_title(f'Distribution - Profil {PROFIL_CHOISI}', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Capital (€)', fontsize=11)
    axes[0, 0].set_ylabel('Fréquence', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    percentiles = [10, 25, 50, 75, 90]
    evolutions_percentiles = {p: [] for p in percentiles}
    
    for k in range(len(historiques_capital[0])):
        valeurs_k = [hist[k] for hist in historiques_capital]
        for p in percentiles:
            evolutions_percentiles[p].append(np.percentile(valeurs_k, p))
    
    mois = range(len(historiques_capital[0]))
    axes[0, 1].fill_between(mois, evolutions_percentiles[10], evolutions_percentiles[90], 
                             alpha=0.2, color='#2E86AB', label='P10-P90')
    axes[0, 1].fill_between(mois, evolutions_percentiles[25], evolutions_percentiles[75], 
                             alpha=0.3, color='#2E86AB', label='P25-P75')
    axes[0, 1].plot(mois, evolutions_percentiles[50], color='#A23B72', linewidth=2.5, label='Médiane')
    axes[0, 1].set_title('Evolution du capital (percentiles)', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Mois', fontsize=11)
    axes[0, 1].set_ylabel('Capital (€)', fontsize=11)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Remplacement du graph d'allocation par le graph des apports (plus pertinent ici)
    annees_graph = np.array(range(nb_periodes_total)) / 12 + age_depart
    axes[1, 0].plot(annees_graph, historique_apports_moyens, linewidth=2.5, color='#D4AF37', label='Apport mensuel')
    axes[1, 0].set_title(f'Profil des Apports (Modèle Quadratique)', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Âge', fontsize=11)
    axes[1, 0].set_ylabel('Apport Mensuel (€)', fontsize=11)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    rendements_mensuels_eq = np.mean(rendements_eq_scenarios, axis=1) * 100
    rendements_mensuels_bd = np.mean(rendements_bd_scenarios, axis=1) * 100
    
    axes[1, 1].plot(rendements_mensuels_eq, alpha=0.7, linewidth=1, label='Equity', color='#2E86AB')
    axes[1, 1].plot(rendements_mensuels_bd, alpha=0.7, linewidth=1, label='Bonds', color='#A23B72')
    axes[1, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Rendements mensuels moyens', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Mois', fontsize=11)
    axes[1, 1].set_ylabel('Rendement (%)', fontsize=11)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
