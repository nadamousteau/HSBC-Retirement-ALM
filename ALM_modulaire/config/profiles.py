"""
MODULE DE GESTION DU PROFIL ACTIF
==================================
Récupère le profil sélectionné dans settings.py et expose les variables
nécessaires (Equity, Bond, allocations) pour le reste de l'application.
"""

from config import settings

# =============================================================================
# EXTRACTION DU PROFIL ACTIF
# =============================================================================

active_profile = settings.PROFILS[settings.PROFIL_CHOISI]

Equity = active_profile["equity"]
Bond = active_profile["bond"]

# =============================================================================
# PARAMÈTRES DE STRATÉGIE
# =============================================================================

# Aversion au risque (défaut : 4.0 pour profil EQUILIBRE)
gamma = active_profile.get("risk_aversion", 4.0)

# Paramètres Target Date
allocation_initiale = active_profile.get("allocation_initiale", 0.95)      
decroissance_annuelle = active_profile.get("decroissance_annuelle", 0.01)  

# Paramètre Fixed Mix
fixed_allocation = active_profile.get("fixed_allocation", 0.60)            