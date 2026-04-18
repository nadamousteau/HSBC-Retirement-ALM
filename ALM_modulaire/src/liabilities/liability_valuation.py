"""
LIABILITY VALUATION — Couverture actif/passif retraite
=======================================================
Vague 2 / Tâche C : utilitaires de valorisation du passif `RetirementLiability`
en regard d'un capital actif (typiquement le capital final d'accumulation).

Une seule fonction publique pour l'instant : `funded_ratio` — le ratio de
couverture instantané au moment de la retraite. Conçu pour être étendu
ultérieurement (PV du passif vu de t < T_retraite, immunisation par
duration, etc.) sans casser l'API actuelle.
"""

from __future__ import annotations

import numpy as np

from .retirement_objective import RetirementLiability


def funded_ratio(capital, liability: RetirementLiability):
    """
    Ratio de couverture du passif retraite par un capital donné.

        funded_ratio = capital / liability.required_capital_at_retirement()

    Convention :
        - funded_ratio = 1.0 → capital exactement égal à la valeur présente
          du passif (couverture cible).
        - < 1.0 → sous-capitalisation (le client devra réduire ses retraits
                  ou faire face à un risque de ruine).
        - > 1.0 → sur-capitalisation (marge de sécurité).

    Args:
        capital   : float ou ndarray — capital actif évalué (typiquement
                    `mat_capital[-1, :]` en sortie d'accumulation).
        liability : RetirementLiability — passif client.

    Returns:
        float ou ndarray (mêmes dimensions que `capital`) — ratio sans unité.
    """
    pv = liability.required_capital_at_retirement()
    if pv <= 0.0:
        raise ValueError(
            f"funded_ratio : passif de valeur présente {pv} <= 0, indéfini"
        )
    capital_arr = np.asarray(capital, dtype=np.float64)
    return capital_arr / pv
