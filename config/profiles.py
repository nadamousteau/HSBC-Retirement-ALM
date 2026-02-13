from config import settings

"""
MODULE DE GESTION DU PROFIL ACTIF
=================================
Ce module récupère le profil sélectionné dans settings.py et expose
les variables nécessaires (Equity, Bond) pour le reste de l'application.
"""

# Récupération du dictionnaire correspondant au profil choisi
# Logique originale : profil = PROFILS[PROFIL_CHOISI]
active_profile = settings.PROFILS[settings.PROFIL_CHOISI]

# Extraction des noms d'actifs pour le mapping Excel
# Logique originale : Equity = profil["equity"]
Equity = active_profile["equity"]

# Logique originale : Bond = profil["bond"]
Bond = active_profile["bond"]

# Paramètres d'allocation (accessibles directement via active_profile, 
# mais on peut définir des alias si besoin pour la clarté dans les stratégies)
allocation_initiale = active_profile.get("allocation_initiale")
decroissance_annuelle = active_profile.get("decroissance_annuelle")
fixed_allocation = active_profile.get("fixed_allocation")