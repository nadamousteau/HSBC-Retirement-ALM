"""
MODULE DE CHARGEMENT DES PARAMÈTRES DE MARCHÉ
==============================================
Charge les paramètres μ, σ, ρ depuis le fichier Excel AssumptionForSimulation.xlsx.
"""

import pandas as pd
import os
import numpy as np
from config import settings, profiles

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