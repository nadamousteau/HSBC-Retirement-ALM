import os
from pathlib import Path

# =============================================================================
# 1. PATHS & DATA CONFIGURATION
# =============================================================================

# Définition de la racine du projet
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INPUTS_DIR = DATA_DIR / "inputs"

# Fichiers de données
XLSX_ASSUMPTIONS = INPUTS_DIR / "AssumptionForSimulation.xlsx"
CSV_HISTORICAL_RETURNS = INPUTS_DIR / "HistoricalAssetReturn.csv"

# =============================================================================
# 2. MÉTHODE ET PROFIL
# =============================================================================

METHODE = "TARGET_DATE"  # Options: "TARGET_DATE" ou "FIXED_MIX"
PROFIL_CHOISI = "EQUILIBRE"  # PRUDENT, MODERE, EQUILIBRE, DYNAMIQUE, AGRESSIF

# =============================================================================
# 3. PARAMÈTRES TEMPORELS
# =============================================================================

NB_ANNEES_ACCUMULATION = 23
AGE_DEPART = 40
DATE_DEBUT_T0 = "2010-12-31"
DATE_PIVOT_BACKTEST = "2025-01-01"

# =============================================================================
# 4. PARAMÈTRES FINANCIERS
# =============================================================================

CAPITAL_INITIAL = 50000
SALAIRE_INITIAL = 3000
TAUX_APPORT_BASE = 0.10

# =============================================================================
# 5. PARAMÈTRES ÉCONOMIQUES
# =============================================================================

TAUX_INFLATION = 0.02
TAUX_LIVRET_A = 0.017

# =============================================================================
# 6. PARAMÈTRES RETRAITE
# =============================================================================

DUREE_RETRAITE = 20

# =============================================================================
# 7. PARAMÈTRES SIMULATION
# =============================================================================

NB_SIMULATIONS = 500
NB_PAS_PAR_AN = 12
NB_PERIODES_TOTAL = NB_ANNEES_ACCUMULATION * NB_PAS_PAR_AN

# =============================================================================
# 8. PARAMÈTRES APPORT (TARGET DATE & FIXED MIX)
# =============================================================================

# Quadratique (Target Date)
RATIO_PIC_CARRIERE = 0.55
FACTEUR_CROISSANCE_MAX = 1.2

# Exponentiel (Fixed Mix)
VITESSE_PROGRESSION = 0.10
GAMMA_ELASTICITE = 1.5
SEUIL_MATURITE = 0.935
SALAIRE_MAX_CIBLE = SALAIRE_INITIAL * 2.5

# =============================================================================
# 9. PARAMÈTRES CRISE
# =============================================================================

SIMULER_CRISE = True
DATE_CRISE = "2030-03-31"

# Target Date (Jump-Diffusion)
LAMBDA_CRISE = 0.15
SEVERITE_EQ_MOYENNE = -0.25
SEVERITE_EQ_SIGMA = 0.10
SEVERITE_BD_MOYENNE = -0.08
SEVERITE_BD_SIGMA = 0.03

# Fixed Mix (Crise détaillée)
PARAMS_CRISE_DETAIL = {
    'drop_eq': 0.35,
    'drop_bd': 0.05,
    'duree_mois': 18,
    'facteur_vol': 2.5
}

# =============================================================================
# 10. PARAMÈTRES DRAWDOWN
# =============================================================================

DRAWDOWN_AVANT_APPORT = True

# =============================================================================
# 11. PROFILS D'INVESTISSEMENT
# =============================================================================

PROFILS = {
    "PRUDENT": {
        "description": "Privilégie la sécurité",
        "equity": "Global Equity USD Hedged",
        "bond": "US Government Bond USD Unhedged",
        # Target Date
        "allocation_initiale": 0.90,
        "decroissance_annuelle": 0.03,
        # Fixed Mix
        "fixed_allocation": 0.20
    },
    "MODERE": {
        "description": "Équilibre sécurité et croissance",
        "equity": "Global Equity USD Hedged",
        "bond": "US Inflation Linked Bond - USD Unhedged",
        "allocation_initiale": 0.90,
        "decroissance_annuelle": 0.02,
        "fixed_allocation": 0.40
    },
    "EQUILIBRE": {
        "description": "Balance risque/rendement",
        "equity": "US Equity USD Unhedged",
        "bond": "USD Corporate Bond - USD Unhedged",
        "allocation_initiale": 0.95,
        "decroissance_annuelle": 0.01,
        "fixed_allocation": 0.60
    },
    "DYNAMIQUE": {
        "description": "Recherche de performance",
        "equity": "US Equity USD Unhedged",
        "bond": "USD Corporate Bond - USD Unhedged",
        "allocation_initiale": 1.00,
        "decroissance_annuelle": 0.008,
        "fixed_allocation": 0.80
    },
    "AGRESSIF": {
        "description": "Maximise le rendement",
        "equity": "US Equity USD Unhedged",
        "bond": "US High Yield Bond BB-B - USD Unhedged",
        "allocation_initiale": 1.00,
        "decroissance_annuelle": 0.005,
        "fixed_allocation": 0.95
    }
}