from .base_strategy import BaseStrategy
from config import settings, profiles

class TargetDateStrategy(BaseStrategy):
    """
    Stratégie Target Date : 
    - Allocation en actions décroissante selon l'âge.
    - Rééquilibrage annuel (mois 11 de chaque année).
    """

    def get_allocation(self, t_index, current_age):
        # Logique reprise de 'calculer_allocation_target_date'
        annees_ecoulees = current_age - settings.AGE_DEPART
        
        pct_equity = profiles.allocation_initiale - profiles.decroissance_annuelle * annees_ecoulees
        
        # Bornage entre 5% et 100%
        pct_equity = max(0.05, min(1.0, pct_equity))
        
        return pct_equity, 1 - pct_equity

    def should_rebalance(self, t_index):
        # Logique originale : if (k % 12) == 11
        # On rééquilibre uniquement le 12ème mois de chaque année
        return (t_index % 12) == 11