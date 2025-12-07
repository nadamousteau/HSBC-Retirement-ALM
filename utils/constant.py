
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