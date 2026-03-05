from .base_strategy import BaseStrategy

class FixedMixStrategy(BaseStrategy):
    """
    Stratégie d'allocation à pondération constante (Fixed Mix).
    L'allocation cible est injectée à l'instanciation, garantissant 
    l'indépendance de la classe vis-à-vis de la configuration globale.
    """

    def __init__(self, target_equity_pct: float):
        super().__init__()
        
        # Validation stricte des entrées (Point 3)
        if not isinstance(target_equity_pct, (int, float)):
            raise TypeError("Le pourcentage d'allocation doit être un numérique.")
        if not (0.0 <= target_equity_pct <= 1.0):
            raise ValueError(
                f"L'allocation cible (target_equity_pct) doit être comprise dans [0, 1]. "
                f"Valeur reçue : {target_equity_pct}"
            )
            
        # Encapsulation de l'attribut (Point 1)
        self._target_equity_pct = float(target_equity_pct)

    def get_allocation(self, t_index: int, current_age: float) -> tuple[float, float]:
        """
        Retourne l'allocation constante : (poids_risqué, poids_sans_risque).
        """
        return self._target_equity_pct, 1.0 - self._target_equity_pct

    def should_rebalance(self, t_index: int) -> bool:
        """
        Force un rééquilibrage à chaque pas de temps.
        """
        return True
