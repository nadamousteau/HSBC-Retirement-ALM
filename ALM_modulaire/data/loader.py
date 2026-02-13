import pandas as pd
import os
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