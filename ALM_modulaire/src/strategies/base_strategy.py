from abc import ABC, abstractmethod

class BaseStrategy(ABC):
    """
    Classe abstraite définissant l'interface d'une stratégie d'allocation.
    """

    @abstractmethod
    def get_allocation(self, t_index, current_age):
        """
        Retourne la répartition cible (Equity, Bond) pour un pas de temps donné.
        
        Args:
            t_index (int): Index du mois en cours (0 à T).
            current_age (float): Âge actuel du participant.
            
        Returns:
            tuple: (pct_equity, pct_bond)
        """
        pass

    @abstractmethod
    def should_rebalance(self, t_index):
        """
        Indique si un rééquilibrage du portefeuille doit être effectué à ce pas de temps.
        
        Returns:
            bool: True si on doit rééquilibrer, False sinon.
        """
        pass