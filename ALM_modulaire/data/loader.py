import pandas as pd
import os
import numpy as np
from config import settings

def load_market_parameters():
    """
    Charge les paramètres de marché (μ, σ, ρ) depuis le fichier Excel.
    Extraction dynamique basée sur le profil d'investissement actif.
    """
    excel_path = settings.XLSX_ASSUMPTIONS
    
    if not os.path.exists(excel_path):
        print("ATTENTION : Fichier Excel manquant. Chargement des paramètres par défaut.")
        return 0.07, 0.15, 0.03, 0.05, 0.3
    
    try:
        df_BS = pd.read_excel(excel_path, sheet_name=0)
        
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
                print(f"ATTENTION : Actif {asset_name} introuvable. Application de valeurs par défaut.")
                return 0.05, 0.10, 0.3
            
            mu = row['Expected Return'].values[0]
            sigma = row['Volatility'].values[0]
            corr = row['Correlation'].values[0] if 'Correlation' in row.columns else 0.3
            return mu, sigma, corr
        
        # Extraction dynamique via le fichier settings centralisé
        nom_profil = settings.PROFIL_CHOISI
        actif_equity = settings.PROFILS[nom_profil]["equity"]
        actif_bond = settings.PROFILS[nom_profil]["bond"]
        
        mu_e, sigma_e, _ = get_params(actif_equity)
        mu_b, sigma_b, corr_eb = get_params(actif_bond)
        
        return mu_e, sigma_e, mu_b, sigma_b, corr_eb
        
    except Exception as e:
        print(f"ERREUR CRITIQUE : Échec de la lecture Excel ({e}).")
        return 0.07, 0.15, 0.03, 0.05, 0.3

def load_macro_history(dates_backtest):
    """
    Synchronise les données macroéconomiques annuelles sur l'index mensuel du backtest.
    """
    try:
        # Chargement des fichiers simplifiés
        df_inf = pd.read_csv(settings.CSV_HISTORICAL_INFLATION, sep=';')
        df_livret = pd.read_csv(settings.CSV_HISTORICAL_LIVRETA, sep=';')
        
        # Création du squelette mensuel basé sur les dates de la simulation
        df_sim = pd.DataFrame({'Date': pd.to_datetime(dates_backtest)})
        df_sim['Année'] = df_sim['Date'].dt.year
        
        # Jointure sur l'année pour propager le taux annuel sur chaque mois de l'année concernée
        df_sim = df_sim.merge(df_inf, on='Année', how='left')
        df_sim = df_sim.merge(df_livret, on='Année', how='left')
        
        # Comblement par propagation pour les années manquantes ou incomplètes
        df_sim = df_sim.ffill().bfill().fillna(0.0)
        
        # Conversion des pourcentages en décimaux (ex: 2.16 -> 0.0216)
        # On utilise les noms de colonnes exacts de vos nouveaux fichiers
        inf_values = df_sim['Indice des prix à la consommation'].values / 100.0
        livret_values = df_sim['Taux annuel du livret A (%)'].values / 100.0
        
        return inf_values, livret_values
        
    except Exception as e:
        print(f"ERREUR CHARGEMENT MACRO : {e}. Utilisation des cibles de settings.")
        size = len(dates_backtest)
        return np.full(size, settings.INFLATION_TARGET), np.full(size, 0.03)