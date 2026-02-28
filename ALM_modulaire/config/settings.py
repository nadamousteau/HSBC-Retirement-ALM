import os
from pathlib import Path

# =============================================================================
# 0. STRATÉGIE GLOBALE ET BENCHMARKING
# =============================================================================
MODE_COMPARAISON = False            # Si True, exécute et compare les stratégies listées
STRATEGIES_A_COMPARER = ["TARGET_DATE", "FIXED_MIX"] 
METHODE = "TARGET_DATE"              # Utilisée si MODE_COMPARAISON = False


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


PROFIL_CHOISI = "EQUILIBRE"  # PRUDENT, MODERE, EQUILIBRE, DYNAMIQUE, AGRESSIF
PROFIL_DECUMULATION_ACTIF = "EQUILIBRE"  # Peut être changé par "PRESERVATION", "TRANSMISSION", "FLEXIBLE", "EQUILIBRE"

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

TAUX_INFLATION = 0.02
TAUX_LIVRET_A = 0.02 #La BdF fait en sorte que taux_livret_A >= taux_inflation

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


# Exponentiel 
VITESSE_PROGRESSION = 0.10
GAMMA_ELASTICITE = 1.5
SEUIL_MATURITE = 0.935
SALAIRE_MAX_CIBLE = SALAIRE_INITIAL * 2.5

## =============================================================================
# PARAMÈTRES DES CRISES (ESG)
# =============================================================================

# 1. Modèle de Jump-Diffusion (Merton)
SIMULER_CRISE_MERTON = False
LAMBDA_CRISE = 0.05               # Fréquence (ex: 0.05 = 1 crise tous les 20 ans en moyenne)
SEVERITE_EQ_MOYENNE = -0.20       # Choc moyen sur les actions (log-rendement)
SEVERITE_EQ_SIGMA = 0.10          # Volatilité du choc actions
SEVERITE_BD_MOYENNE = -0.02       # Choc moyen sur les obligations
SEVERITE_BD_SIGMA = 0.05          # Volatilité du choc obligations

# 2. Choc Déterministe (Crise Localisée)
SIMULER_CRISE_LOCALISEE = False
DATE_CRISE = "2030-03-01"         # Date d'injection de la crise
PARAMS_CRISE_DETAIL = {
    'drop_eq': 0.35,              # Choc instantané actions (-35%)
    'drop_bd': 0.05,              # Choc instantané obligations (-5%)
    'duree_mois': 12,             # Période de stress (persistance)
    'facteur_vol': 2.5            # Multiplicateur de volatilité post-choc
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





# =============================================================================
# 12. PROFILS DE DÉCUMULATION (ALM)
# =============================================================================

PROFILS_DECUMULATION = {
    "PRESERVATION": {
        "description": "Minimise strictement le risque de ruine et la perte en capital.",
        "type_strategie": "MONTANT_FIXE",
        "gamma_aversion": 5.0,  # Forte aversion au risque (Utilité CRRA)
        "taux_remplacement_cible": 0.70,  # Exigence élevée de maintien du niveau de vie
        "poids_kpi": {
            "prob_ruine": 0.50,
            "expected_shortfall": 0.30,
            "equivalent_certain": 0.10,
            "transmission": 0.10
        },
        # Paramètres d'exécution des stratégies de retrait
        "taux_retrait_initial": 0.03,  # Règle des 3% (Ultra-conservateur)
        "gk_seuil_hausse": 0.20,       # Paramètres Guyton-Klinger
        "gk_seuil_baisse": 0.20,
        "gk_ajustement": 0.10
    },
    "EQUILIBRE": {
        "description": "Compromis rationnel entre maintien du pouvoir d'achat et risque de longévité.",
        "type_strategie": "MONTANT_FIXE",
        "gamma_aversion": 3.0,
        "taux_remplacement_cible": 0.60,  # Objectif standard académique
        "poids_kpi": {
            "prob_ruine": 0.30,
            "expected_shortfall": 0.20,
            "equivalent_certain": 0.30,
            "transmission": 0.20
        },
        "taux_retrait_initial": 0.04,  # Standard académique (Règle de Bengen)
        "gk_seuil_hausse": 0.20,
        "gk_seuil_baisse": 0.20,
        "gk_ajustement": 0.10
    },
    "TRANSMISSION": {
        "description": "Maximise le capital résiduel au décès, tolère une volatilité des rentes.",
        "type_strategie": "POURCENTAGE_FIXE",  # Le retrait baisse avec les marchés, préservant le capital
        "gamma_aversion": 2.0,
        "taux_remplacement_cible": 0.50,  # Acceptation d'un niveau de vie réduit pour léguer
        "poids_kpi": {
            "prob_ruine": 0.10,
            "expected_shortfall": 0.10,
            "equivalent_certain": 0.20,
            "transmission": 0.60       # Priorité absolue au legs
        },
        "taux_retrait_initial": 0.035,
        "gk_seuil_hausse": 0.15,
        "gk_seuil_baisse": 0.25,
        "gk_ajustement": 0.10
    },
    "FLEXIBLE": {
        "description": "Maximise la consommation initiale via des règles de retrait dynamiques.",
        "type_strategie": "GUYTON_KLINGER",  # Ajustement dynamique aux chocs de marché
        "gamma_aversion": 1.5,         # Faible aversion, recherche de consommation
        "taux_remplacement_cible": 0.65,  # Vise une forte consommation initiale
        "poids_kpi": {
            "prob_ruine": 0.15,
            "expected_shortfall": 0.15,
            "equivalent_certain": 0.60,  # Priorité à la maximisation de l'utilité courante
            "transmission": 0.10
        },
        "taux_retrait_initial": 0.05,  # Retrait agressif
        "gk_seuil_hausse": 0.15,
        "gk_seuil_baisse": 0.15,
        "gk_ajustement": 0.15
    }
}



# =============================================================================
# 13. VISUALISATION
# =============================================================================

PLOT_CAPITAL = False
PLOT_SALAIRE = False
PLOT_APPORTS = False

PLOT_CAPITAL_REEL = False
PLOT_SALAIRE_REEL = False
PLOT_APPORTS_REEL = False

# Visualisation ciblée : Impact de la Crise (1 an avant -> 5 ans après) seulement pour crise localisée
PLOT_CRISE_RENDEMENTS = False
PLOT_CRISE_CAPITAL_NOMINAL = False
PLOT_CRISE_CAPITAL_REEL = False

#Décumulation

PLOT_RETRAITE_CAPITAL = False
PLOT_TAUX_REMPLACEMENT = False

PLOT_RETRAITE_CAPITAL_REEL = False
PLOT_TAUX_REMPLACEMENT_REEL = False

#KPI

PRINT_PERFORMANCE_GLOBALE = False   # Affiche le TRI médian et la dispersion
PRINT_METRIQUES_RISQUE = False     # Affiche Shortfall, VaR, Max Drawdown, Sortino, Max Underwater

#Comparaison 

PLOT_COMPARAISON_CAPITAL = False
PLOT_COMPARAISON_CAPITAL_REEL = False
PRINT_SYNTHESE_CAPITAL_RETRAITE = False  # Affiche le comparatif final des capitaux à l'âge de retraite

#KPI décumulation

PRINT_KPI_DECUMULATION=False


#Etude évolution capital fin de l'accumulation

PRINT_KPI_END_CARRIERE=False

#Objectif final

TROUVER_LA_MEILLEURE_STRAT = False #posbbile seulement si mode comparaison activé