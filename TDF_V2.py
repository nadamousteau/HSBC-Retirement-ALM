#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulation d'investissement avec profils de risque personnalisés
- Backtesting sur données historiques puis forecasts Black & Scholes
- Profils investisseurs : Prudent, Modéré, Équilibré, Dynamique, Agressif
- Allocation adaptative selon l'âge avec décroissance annuelle
- Enveloppe PER (pas de taxation)
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
    "Asia Pacific ex Japan Equity USD Hedged": "Asia Pacific",  # Commence 31/07/2014
    "Global Equity USD Hedged": "Global Equity",
    "Japan Equity - USD Unhedged": "Japan Equity",
    "US Equity USD Unhedged": "US Equity"
}

''' ========== PROFILS INVESTISSEURS ========== '''

PROFILS = {
    "PRUDENT": {
        "description": "Privilégie la sécurité, volatilité minimale",
        "equity": "Global Equity USD Hedged",  # Plus diversifié
        "bond": "US Government Bond USD Unhedged",  # Bond sûr
        "allocation_initiale": 0.30,  # 30% equity au départ
        "decroissance_annuelle": 0.005  # -0.5% par an
    },
    "MODERE": {
        "description": "Équilibre sécurité et croissance modérée",
        "equity": "Global Equity USD Hedged",
        "bond": "USD Corporate Bond - USD Unhedged",  # Bond risqué mais modéré
        "allocation_initiale": 0.50,
        "decroissance_annuelle": 0.008  # -0.8% par an
    },
    "EQUILIBRE": {
        "description": "Balance risque/rendement, approche classique",
        "equity": "US Equity USD Unhedged",  # Plus volatile
        "bond": "USD Corporate Bond - USD Unhedged",
        "allocation_initiale": 0.70,
        "decroissance_annuelle": 0.010  # -1.0% par an
    },
    "DYNAMIQUE": {
        "description": "Recherche de performance, accepte volatilité",
        "equity": "US Equity USD Unhedged",
        "bond": "US High Yield Bond BB-B - USD Unhedged",  # Bond risqué
        "allocation_initiale": 0.85,
        "decroissance_annuelle": 0.012  # -1.2% par an
    },
    "AGRESSIF": {
        "description": "Maximise le rendement, très haute volatilité",
        "equity": "US Equity USD Unhedged",  # Le plus volatile historiquement
        "bond": "US High Yield Bond BB-B - USD Unhedged",
        "allocation_initiale": 0.95,
        "decroissance_annuelle": 0.015  # -1.5% par an
    }
}

''' ========== PARAMÈTRES DE SIMULATION ========== '''

# CHOISIR LE PROFIL ICI
PROFIL_CHOISI = "EQUILIBRE"  # PRUDENT, MODERE, EQUILIBRE, DYNAMIQUE, AGRESSIF

# Paramètres financiers
nb_annees = 35
t0 = "2001-12-31"  # Date de départ
age_depart = 30
capital_initial = 10000
salaire_initial = 3000
taux_apport = 0.10
croissance_salaire_annuelle = 0.02
taux_inflation = 0.02

# Monte Carlo
nb_simulations = 500
nb_pas_par_an = 12
nb_periodes_total = nb_annees * nb_pas_par_an

# Sélection des actifs selon le profil
profil = PROFILS[PROFIL_CHOISI]
Equity = profil["equity"]
Bond = profil["bond"]

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
print(f"🎯 PROFIL SÉLECTIONNÉ: {PROFIL_CHOISI}")
print("=" * 80)
print(f"Description: {profil['description']}")
print(f"Equity choisi: {EQUITIES.get(Equity, Equity)}")
print(f"Bond choisi: {BONDS_SAFE.get(Bond, BONDS_RISKY.get(Bond, Bond))}")
print(f"Allocation initiale (âge {age_depart}): {profil['allocation_initiale']*100:.1f}% equity")
print(f"Décroissance annuelle: {profil['decroissance_annuelle']*100:.2f}% par an")

# Calcul allocation finale
age_final = age_depart + nb_annees
allocation_finale = max(0.05, profil['allocation_initiale'] - profil['decroissance_annuelle'] * nb_annees)
print(f"Allocation finale (âge {age_final}): {allocation_finale*100:.1f}% equity")

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

''' ========== PROFIL DE RISQUE ADAPTATIF ========== '''

def calculer_allocation(age):
    """
    Allocation basée sur l'âge avec décroissance annuelle paramétrable
    pct_equity = allocation_initiale - decroissance_annuelle × (age - age_depart)
    Avec un plancher minimum de 5% equity
    """
    annees_ecoulees = age - age_depart
    pct_equity = profil['allocation_initiale'] - profil['decroissance_annuelle'] * annees_ecoulees
    
    # Plancher minimum de 5% equity
    pct_equity = max(0.05, pct_equity)
    # Plafond maximum de 100%
    pct_equity = min(1.0, pct_equity)
    
    return pct_equity, 1 - pct_equity

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

''' ========== BOUCLE DE SIMULATION ========== '''

resultats_finaux = []
historiques_capital = []

for sim in range(nb_simulations):
    eq_shares = 0.0
    bd_shares = 0.0
    eq_price = 100.0
    bd_price = 100.0
    
    age_actuel = age_depart
    pct_eq, pct_bd = calculer_allocation(age_actuel)
    
    eq_shares = (capital_initial * pct_eq) / eq_price
    bd_shares = (capital_initial * pct_bd) / bd_price
    
    salaire_actuel = salaire_initial
    historique_capital = [capital_initial]
    
    for k in range(nb_periodes_total):
        mois = k + 1
        annee_courante = k // 12
        age_actuel = age_depart + annee_courante
        
        r_eq = rendements_eq_scenarios[k, sim]
        r_bd = rendements_bd_scenarios[k, sim]
        
        eq_price *= (1 + r_eq)
        bd_price *= (1 + r_bd)
        
        if mois % 12 == 0 and mois > 0:
            salaire_actuel *= (1 + croissance_salaire_annuelle + taux_inflation)
        
        apport_mensuel = salaire_actuel * taux_apport
        
        pct_eq_cible, pct_bd_cible = calculer_allocation(age_actuel)
        
        eq_buy = apport_mensuel * pct_eq_cible
        bd_buy = apport_mensuel * pct_bd_cible
        
        eq_shares += eq_buy / eq_price
        bd_shares += bd_buy / bd_price
        
        # Rééquilibrage annuel
        if (k % 12) == 11:
            total_val = eq_shares * eq_price + bd_shares * bd_price
            age_suivant = age_actuel + 1
            pct_eq_suivant, pct_bd_suivant = calculer_allocation(age_suivant)
            
            target_eq_val = total_val * pct_eq_suivant
            target_bd_val = total_val * pct_bd_suivant
            
            eq_val = eq_shares * eq_price
            bd_val = bd_shares * bd_price
            
            if eq_val > target_eq_val and eq_shares > 0:
                vente_eq = (eq_val - target_eq_val) / eq_price
                eq_shares -= vente_eq
                bd_shares += (eq_val - target_eq_val) / bd_price
            elif eq_val < target_eq_val and bd_shares > 0:
                vente_bd = (target_eq_val - eq_val) / bd_price
                vente_bd = min(vente_bd, bd_shares)
                bd_shares -= vente_bd
                eq_shares += (vente_bd * bd_price) / eq_price
        
        capital_actuel = eq_shares * eq_price + bd_shares * bd_price
        historique_capital.append(capital_actuel)
    
    capital_final = eq_shares * eq_price + bd_shares * bd_price
    total_investi = capital_initial + sum([
        salaire_initial * (1 + croissance_salaire_annuelle + taux_inflation)**(k//12) * taux_apport
        for k in range(nb_periodes_total)
    ])
    
    resultats_finaux.append({
        'capital_final': capital_final,
        'total_investi': total_investi
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
print(f"📊 RÉSULTATS - PROFIL {PROFIL_CHOISI}")
print("=" * 80)
print(f"Simulations: {nb_simulations} | Horizon: {nb_annees} ans | Âge: {age_depart}→{age_depart + nb_annees}")
print(f"Décroissance: {profil['decroissance_annuelle']*100:.2f}% par an")
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
    capital_investi_cumul = [capital_initial + salaire_initial * (1 + croissance_salaire_annuelle + taux_inflation)**(k//12) * taux_apport * k for k in mois]
    
    axes[0, 0].plot(mois, historiques_capital[0], color='#2E86AB', linewidth=2, label='Capital valorisé')
    axes[0, 0].plot(mois, capital_investi_cumul, color='#A23B72', linewidth=2, linestyle='--', label='Capital investi')
    axes[0, 0].set_title(f'Evolution du capital - Profil {PROFIL_CHOISI}', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Mois', fontsize=11)
    axes[0, 0].set_ylabel('Capital (€)', fontsize=11)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    ages = range(age_depart, age_depart + nb_annees + 1)
    allocations_eq = [calculer_allocation(age)[0] * 100 for age in ages]
    allocations_bd = [calculer_allocation(age)[1] * 100 for age in ages]
    
    axes[0, 1].plot(ages, allocations_eq, linewidth=2.5, color='#2E86AB', label='Equity', marker='o')
    axes[0, 1].plot(ages, allocations_bd, linewidth=2.5, color='#A23B72', label='Bonds', marker='s')
    axes[0, 1].set_title(f'Allocation - Profil {PROFIL_CHOISI}', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Âge', fontsize=11)
    axes[0, 1].set_ylabel('Allocation (%)', fontsize=11)
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
    
    Décroissance: {profil['decroissance_annuelle']*100:.2f}%/an
    
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
    axes[0, 0].set_title(f'Distribution capital final - Profil {PROFIL_CHOISI}', fontsize=14, fontweight='bold')
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
    
    ages = range(age_depart, age_depart + nb_annees + 1)
    allocations_eq = [calculer_allocation(age)[0] * 100 for age in ages]
    allocations_bd = [calculer_allocation(age)[1] * 100 for age in ages]
    
    axes[1, 0].plot(ages, allocations_eq, linewidth=2.5, color='#2E86AB', label='Equity', marker='o')
    axes[1, 0].plot(ages, allocations_bd, linewidth=2.5, color='#A23B72', label='Bonds', marker='s')
    axes[1, 0].set_title(f'Allocation - Profil {PROFIL_CHOISI}', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Âge', fontsize=11)
    axes[1, 0].set_ylabel('Allocation (%)', fontsize=11)
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