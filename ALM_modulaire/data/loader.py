"""
MODULE DE CHARGEMENT DES PARAMÈTRES DE MARCHÉ
==============================================
Charge les paramètres μ, σ, ρ depuis le fichier Excel AssumptionForSimulation.xlsx.
"""

import pandas as pd
import os
import numpy as np
from config import settings, profiles
from pathlib import Path

# =============================================================================
# PARAMÈTRES PAR DÉFAUT
# =============================================================================

DEFAULT_MARKET_PARAMS = {
    'mu_e': 0.07,       # Rendement equity par défaut (7%)
    'sigma_e': 0.15,    # Volatilité equity par défaut (15%)
    'mu_b': 0.03,       # Rendement bonds par défaut (3%)
    'sigma_b': 0.05,    # Volatilité bonds par défaut (5%)
    'corr_eb': 0.30     # Corrélation equity-bonds par défaut (0.3)
}

DEFAULT_ASSET_PARAMS = (0.05, 0.10, 0.30)  # (mu, sigma, corr) 

# =============================================================================
# MAPPING DES NOMS D'ACTIFS
# =============================================================================

ASSET_NAME_MAPPING = {
    'US Government Bond': 'US Government Bond USD Unhedged',
    'US Inflation Linked Bond': 'US Inflation Linked Bond - USD Unhedged',
    'USD Corporate Bond': 'USD Corporate Bond - USD Unhedged',
    'US High Yield Bond BB-B': 'US High Yield Bond BB-B - USD Unhedged',
    'Global Equity': 'Global Equity USD Hedged',
    'US Equity': 'US Equity USD Unhedged',
    'Japan Equity': 'Japan Equity - USD Unhedged',
    'Asia Pacific ex Japan Equity': 'Asia Pacific ex Japan Equity USD Hedged'
}


def load_market_parameters():
    """
    Charge les paramètres de marché depuis le fichier Excel.
    
    Étapes :
    1. Vérification de l'existence du fichier
    2. Chargement et normalisation des noms d'actifs
    3. Extraction des paramètres pour les actifs du profil actif
    
    Returns:
        tuple: (mu_e, sigma_e, mu_b, sigma_b, corr_eb)
               Rendement annuel, volatilité annuelle et corrélation
    """

    excel_path = settings.XLSX_ASSUMPTIONS

    # Vérification de l'existence du fichier
    if not os.path.exists(excel_path):
        print("Fichier Excel manquant, utilisation des paramètres par défaut")
        return tuple(DEFAULT_MARKET_PARAMS.values())  

    try:
        # Chargement du fichier Excel
        df_BS = pd.read_excel(excel_path, sheet_name=0)

        # Normalisation des noms d'actifs
        df_BS['Asset Name'] = df_BS['Asset Name'].replace(ASSET_NAME_MAPPING)

        def get_params(asset_name):
            """
            Extrait les paramètres (μ, σ, ρ) pour un actif donné.
            
            Args:
                asset_name: Nom de l'actif (après normalisation)
            
            Returns:
                tuple: (mu, sigma, corr)
            """
            row = df_BS[df_BS['Asset Name'] == asset_name]

            if row.empty:
                print(f"Actif '{asset_name}' non trouvé dans le fichier Excel")
                return DEFAULT_ASSET_PARAMS 

            mu = row['Expected Return'].values[0]
            sigma = row['Volatility'].values[0]

            # Gestion de la colonne Correlation
            if 'Correlation' in df_BS.columns:  
                corr = row['Correlation'].values[0]
            else:
                corr = DEFAULT_ASSET_PARAMS[2]  

            return (mu, sigma, corr)

        # Extraction des paramètres pour les actifs du profil
        mu_e, sigma_e, _ = get_params(profiles.Equity)
        mu_b, sigma_b, corr_eb = get_params(profiles.Bond)

        return mu_e, sigma_e, mu_b, sigma_b, corr_eb

    except FileNotFoundError:  
        print(f"Fichier Excel introuvable : {excel_path}")
        return tuple(DEFAULT_MARKET_PARAMS.values())
    
    except pd.errors.ExcelFileError as e:  
        print(f"Erreur de lecture Excel : {e}")
        return tuple(DEFAULT_MARKET_PARAMS.values())
    
    except KeyError as e:  
        print(f"Colonne manquante dans le fichier Excel : {e}")
        return tuple(DEFAULT_MARKET_PARAMS.values())
    
    except Exception as e:  
        print(f"Erreur inattendue : {e}")
        return tuple(DEFAULT_MARKET_PARAMS.values())
    


def charger_rendements_historiques(asset_equity, asset_bond, date_pivot):
    """
    Charge les rendements historiques depuis le fichier CSV jusqu'à la date pivot.
    
    Args:
        asset_equity: Nom de l'actif equity (ex: "US Equity USD Unhedged")
        asset_bond: Nom de l'actif bond (ex: "USD Corporate Bond - USD Unhedged")
        date_pivot: Date pivot pour filtrer les données historiques
    
    Returns:
        tuple: (rendements_equity, rendements_bond, nombre_periodes)
               Rendements mensuels jusqu'à la date pivot
    """
    # Chemin du fichier CSV
    base_dir = Path(__file__).resolve().parent.parent.parent
    csv_path = base_dir / "data" / "inputs" / "HistoricalAssetReturn.csv"
    
    if not os.path.exists(csv_path):
        return None, None, 0
    
    # Chargement du CSV
    # Le CSV a 3 lignes de header:
    #   - ligne 0: Asset Class - Type
    #   - ligne 1: Asset Class - Name (noms des colonnes)
    #   - ligne 2: Asset Data - Date
    # On skippe lignes 0 et 2, utilise ligne 1 comme header, première colonne comme index (dates)
    df = pd.read_csv(csv_path, skiprows=[0, 2], header=0, index_col=0)
    
    # Convertir l'index en datetime
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    
    # Conversion de la date pivot
    pivot_ts = pd.Timestamp(date_pivot)
    
    # Filtrage jusqu'à la date pivot
    df_filtered = df[df.index <= pivot_ts].copy()
    
    if df_filtered.empty:
        return None, None, 0
    
    # Extraction des colonnes pour equity et bond
    try:
        r_equity = df_filtered[asset_equity].dropna().values
        r_bond = df_filtered[asset_bond].dropna().values
        
        # Prendre la longueur minimale pour avoir les mêmes dates
        min_len = min(len(r_equity), len(r_bond))
        return r_equity[-min_len:], r_bond[-min_len:], min_len
    except KeyError as e:
        print(f"Colonne non trouvée dans le CSV: {e}")
        print(f"Colonnes disponibles: {list(df_filtered.columns)}")
        return None, None, 0

def generer_rendements_correles_base(mu_e, sigma_e, mu_b, sigma_b, corr, nb_periodes, nb_sims):
    """
    Génère des rendements corrélés selon Black-Scholes.
    Utilisé pour Target Date (tout stochastique).
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