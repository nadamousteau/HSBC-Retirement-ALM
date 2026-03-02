import pandas as pd
import os
import numpy as np
from config import settings, profiles

def load_market_parameters():
    """
    Charge les paramètres μ, σ, ρ depuis le fichier Excel AssumptionForSimulation.xlsx.
    
    Logique reprise de 'charger_parametres_actifs' du script original :
    1. Vérification de l'existence du fichier.
    2. Chargement et application du mapping de noms.
    3. Extraction des données pour l'Equity et le Bond du profil actif.
    
    Returns:
        tuple: (mu_e, sigma_e, mu_b, sigma_b, corr_eb)
    """
    
    # Utilisation du chemin défini dans settings.py
    excel_path = settings.XLSX_ASSUMPTIONS
    
    if not os.path.exists(excel_path):
        print("⚠️  Fichier Excel manquant, utilisation paramètres par défaut")
        return 0.07, 0.15, 0.03, 0.05, 0.3
    
    try:
        df_BS = pd.read_excel(excel_path, sheet_name=0)
        
        # Mapping noms (Strictement identique à l'original)
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
            """Extrait les paramètres pour un actif donné."""
            row = df_BS[df_BS['Asset Name'] == asset_name]
            
            if row.empty:
                print(f"⚠️  Actif {asset_name} non trouvé")
                return (0.05, 0.10, 0.3)
            
            mu = row['Expected Return'].values[0]
            sigma = row['Volatility'].values[0]
            
            # Gestion de la colonne Correlation
            corr = row['Correlation'].values[0] if 'Correlation' in row.columns else 0.3
            return (mu, sigma, corr)
        
        # Récupération des actifs définis dans le profil (config/profiles.py)
        mu_e, sigma_e, _ = get_params(profiles.Equity)
        mu_b, sigma_b, corr_eb = get_params(profiles.Bond)
        
        return mu_e, sigma_e, mu_b, sigma_b, corr_eb
        
    except Exception as e:
        print(f"⚠️  Erreur lecture Excel: {e}")
        # Valeurs par défaut en cas de crash
        return 0.07, 0.15, 0.03, 0.05, 0.3


def load_gbi_nominal_forecast_monthly_rates(nb_months):
    """
    Charge les taux forecast GBI depuis la feuille Excel Nominal Forecast.

    Mapping utilisé (ordre de préférence) :
      - Forecast Short Rate (10Y)
      - Forecast Short Rate (3M)

    Args:
        nb_months: longueur de la série mensuelle requise

    Returns:
        np.ndarray shape (nb_months,) avec taux annualisés (décimal),
        ou None si la feuille/structure est indisponible.
    """
    excel_path = settings.XLSX_ASSUMPTIONS
    if not os.path.exists(excel_path):
        return None

    try:
        xls = pd.ExcelFile(excel_path)
        if "Nominal Forecast" not in set(xls.sheet_names):
            return None

        df = pd.read_excel(excel_path, sheet_name="Nominal Forecast")
        if df.empty:
            return None

        horizon_col = "Year Fraction Horizon"
        rate_col = None
        for c in ["Forecast Short Rate (10Y)", "Forecast Short Rate (3M)"]:
            if c in df.columns:
                rate_col = c
                break

        if horizon_col not in df.columns or rate_col is None:
            return None

        work = df[[horizon_col, rate_col]].copy()
        work[horizon_col] = pd.to_numeric(work[horizon_col], errors="coerce")
        work[rate_col] = pd.to_numeric(work[rate_col], errors="coerce")
        work = work.dropna().sort_values(horizon_col)

        if work.empty:
            return None

        horizons_years = work[horizon_col].to_numpy(dtype=float)
        rates_annual = work[rate_col].to_numpy(dtype=float)

        months_grid_years = np.arange(nb_months, dtype=float) / 12.0
        return np.interp(months_grid_years, horizons_years, rates_annual)

    except Exception as e:
        print(f"⚠️  Erreur lecture Nominal Forecast Excel: {e}")
        return None