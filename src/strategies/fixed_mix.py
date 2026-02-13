from .base_strategy import BaseStrategy
from config import settings, profiles

class FixedMixStrategy(BaseStrategy):
    """
    Stratégie Fixed Mix :
    - Allocation constante définie dans le profil.
    - Rééquilibrage constant (à chaque pas) pour maintenir le mix.
    """

    def get_allocation(self, t_index, current_age):
        # Logique reprise de 'calculer_allocation_fixed_mix'
        pct_equity = profiles.fixed_allocation
        return pct_equity, 1 - pct_equity

    def should_rebalance(self, t_index):
        # Pour maintenir un Fixed Mix strict (ex: 60/40 constant), 
        # il faut rééquilibrer dès que les marchés bougent.
        return True