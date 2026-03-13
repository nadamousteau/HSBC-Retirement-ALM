from pathlib import Path

# =============================================================================
# 0. STRATÉGIE GLOBALE ET BENCHMARKING
# =============================================================================
MODE_COMPARAISON = True
STRATEGIES_A_COMPARER = ["GBI", "FALEH", "FIXED_MIX", "TARGET_DATE"]  
METHODE_DEFAUT = "GBI"

# =============================================================================
# 1. PATHS & DATA CONFIGURATION
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUTS_DIR = DATA_DIR / "inputs"

XLSX_ASSUMPTIONS = INPUTS_DIR / "AssumptionForSimulation.xlsx"
CSV_HISTORICAL_RETURNS = INPUTS_DIR / "HistoricalAssetReturn.csv"  
CSV_YIELD_CURVE = INPUTS_DIR / "yield-curve-rates-1990-2024.csv"

# =============================================================================
# 2. PROFIL
# =============================================================================
PROFIL_CHOISI = "EQUILIBRE"  # PRUDENT, MODERE, EQUILIBRE, DYNAMIQUE, AGRESSIF

# =============================================================================
# 3. PARAMÈTRES TEMPORELS
# =============================================================================
NB_ANNEES_ACCUMULATION = 40
AGE_DEPART = 20
DATE_DEBUT_T0 = "2001-12-31"
DATE_PIVOT_BACKTEST = "2025-08-31"

# =============================================================================
# 4. PARAMÈTRES FINANCIERS
# =============================================================================
CAPITAL_INITIAL = 5000
SALAIRE_INITIAL = 2000
TAUX_APPORT_BASE = 0.10

# =============================================================================
# 5. PARAMÈTRES ÉCONOMIQUES
# =============================================================================
TAUX_INFLATION = 0.02  # Utilisé uniquement si INFLATION_STOCHASTIQUE = False

# =============================================================================
# 5b. PARAMÈTRES INFLATION VASICEK (Stochastique)
# =============================================================================
INFLATION_STOCHASTIQUE = True  # Activer le modèle Vasicek pour l'inflation
INFLATION_KAPPA = 0.15         # Vitesse de retour à la moyenne (demi-vie ~4.6 ans)
INFLATION_THETA = 0.023        # Cible d'inflation long terme (2.3%)
INFLATION_SIGMA = 0.011        # Volatilité de l'inflation (~1.1% annualisée)
INFLATION_SEED = 42            # Graine aléatoire pour reproductibilité

# =============================================================================
# 6. PARAMÈTRES GBI (Goal-Based Investing / CPPI dynamique)
# =============================================================================

DATE_RETRAITE_GBI = "2041-12-31"  # DATE_DEBUT_T0 + NB_ANNEES_ACCUMULATION
FLOOR_PERCENT_GBI = 0.80  # Plancher de sécurité (80% de la richesse cible)
GBI_SEED = 42  # Graine aléatoire pour NS-VAR(1)

# =============================================================================
# 11. PARAMÈTRES SIMULATION
# =============================================================================

NB_SIMULATIONS = 1000
NB_PAS_PAR_AN = 12
NB_PERIODES_TOTAL = NB_ANNEES_ACCUMULATION * NB_PAS_PAR_AN

# =============================================================================
# 12. PARAMÈTRES APPORT (Contributions dynamiques)
# =============================================================================

VITESSE_PROGRESSION = 0.10
GAMMA_ELASTICITE = 1.5
SEUIL_MATURITE = 0.935
SALAIRE_MAX_CIBLE = SALAIRE_INITIAL * 2.5

# =============================================================================
# 13. PARAMÈTRES DES CRISES (ESG)
# =============================================================================

# Modèle de Jump-Diffusion (Merton)
SIMULER_CRISE_MERTON = False
LAMBDA_CRISE = 0.05  # Fréquence (1 crise tous les 20 ans)
SEVERITE_EQ_MOYENNE = -0.20
SEVERITE_EQ_SIGMA = 0.10
SEVERITE_BD_MOYENNE = -0.02
SEVERITE_BD_SIGMA = 0.05

# Choc Déterministe (Crise Localisée)
SIMULER_CRISE_LOCALISEE = False
DATE_CRISE = "2030-03-01"
PARAMS_CRISE_DETAIL = {
    'drop_eq': 0.35,
    'drop_bd': 0.05,
    'duree_mois': 12,
    'facteur_vol': 2.5
}

# =============================================================================
# 14. PARAMÈTRES FALEH (Optimisation Stochastique)
# =============================================================================

FALEH_NB_TREE_STAGES = 10
FALEH_PENALTY_RUINE = 50
FALEH_TARGET_WEALTH = None  # Calculé automatiquement si None
FALEH_SEED = 42  # Graine aléatoire pour les scénarios FALEH
FALEH_MAX_BRANCHES = 5  # Nombre maximal de branches par nœud dans l'arbre

# =============================================================================
# 15. PARAMÈTRES DRAWDOWN
# =============================================================================

DRAWDOWN_AVANT_APPORT = True

# =============================================================================
# 16. PROFILS D'INVESTISSEMENT
# =============================================================================
PROFILS = {
    "PRUDENT": {
        "description": "Privilégie la sécurité",
        "equity": "Global Equity USD Hedged",
        "bond": "US Government Bond USD Unhedged",
        "risk_aversion": 10.0,
        "allocation_initiale": 0.90,
        "decroissance_annuelle": 0.03,
        "fixed_allocation": 0.20
    },
    "MODERE": {
        "description": "Équilibre sécurité et croissance",
        "equity": "Global Equity USD Hedged",
        "bond": "US Inflation Linked Bond - USD Unhedged",
        "risk_aversion": 7.0,
        "allocation_initiale": 0.90,
        "decroissance_annuelle": 0.02,
        "fixed_allocation": 0.40
    },
    "EQUILIBRE": {
        "description": "Balance risque/rendement",
        "equity": "US Equity USD Unhedged",
        "bond": "USD Corporate Bond - USD Unhedged",
        "risk_aversion": 4.0,
        "allocation_initiale": 0.95,
        "decroissance_annuelle": 0.01,
        "fixed_allocation": 0.60
    },
    "DYNAMIQUE": {
        "description": "Recherche de performance",
        "equity": "US Equity USD Unhedged",
        "bond": "USD Corporate Bond - USD Unhedged",
        "risk_aversion": 2.5,
        "allocation_initiale": 1.00,
        "decroissance_annuelle": 0.008,
        "fixed_allocation": 0.80
    },
    "AGRESSIF": {
        "description": "Maximise le rendement",
        "equity": "US Equity USD Unhedged",
        "bond": "US High Yield Bond BB-B - USD Unhedged",
        "risk_aversion": 1.5,
        "allocation_initiale": 1.00,
        "decroissance_annuelle": 0.005,
        "fixed_allocation": 0.95
    }
}

# =============================================================================
# 17. VISUALISATION
# =============================================================================

# Accumulation
PLOT_CAPITAL = False
PLOT_SALAIRE = False
PLOT_APPORTS = False
PLOT_CAPITAL_REEL = False
PLOT_SALAIRE_REEL = False
PLOT_APPORTS_REEL = False

# Crises
PLOT_CRISE_RENDEMENTS = False
PLOT_CRISE_CAPITAL_NOMINAL = False
PLOT_CRISE_CAPITAL_REEL = False

# KPI
PRINT_PERFORMANCE_GLOBALE = True
PRINT_METRIQUES_RISQUE = True

# Comparaison
PLOT_COMPARAISON_CAPITAL = False
PLOT_COMPARAISON_CAPITAL_REEL = False
PRINT_SYNTHESE_CAPITAL_RETRAITE = True