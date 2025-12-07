
# Fichiers
YIELD_FILE = 'data/yield-curve-rates-1990-2024.csv'
ASSET_FILE = 'data/HistoricalAssetReturn.csv'

# Période de Simulation (Backtest)
SIMULATION_START = '2000-01-01'
SIMULATION_END   = '2023-12-31'

# Paramètres du Client
INITIAL_WEALTH   = 50000         # Capital de départ
RETIREMENT_DATE  = '2023-12-31'  # Date cible de la retraite
FLOOR_PERCENT    = 0.80          # Protection 80%


CLIENT_AGE_START = 40            # Âge du client au début de la simulation
CONTRIB_START    = 5000          # Épargne annuelle au début 
CONTRIB_PEAK     = 15000         # Épargne annuelle MAXIMALE (au sommet de la carrière)
AGE_PEAK         = 40           # Âge où l'épargne est maximale
INFLATION_SALAIRE = 0.01        # Inflation du salaire 
SAVINGS_RATE     = 0.10          # Taux d'épargne (pourcentage du salaire épargné chaque année)
INITIAL_SALARY   = 3000         # Salaire initial annuel



PROFILS_FIXE_MIXED = {
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



PROFILS_TDF = {
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
