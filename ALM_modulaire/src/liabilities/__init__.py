"""
LIABILITIES — Modélisation du passif retraite
==============================================
Vague 2 / Tâche C : refonte complète. Le passif retraite est désormais
représenté par un objet `RetirementLiability` (dataclass immuable) qui
porte explicitement :
    - le revenu cible mensuel (en € réels ou nominaux),
    - le mode d'horizon (FIXED ou ACTUARIAL via la table de mortalité),
    - le taux d'actualisation et l'inflation anticipée du plancher.

`build_liability_from_settings(settings)` est le point d'entrée principal :
il construit l'objet à partir des LIABILITY_* du fichier de configuration.
`liability_valuation.funded_ratio()` calcule la couverture instantanée
d'un capital donné par rapport à la valeur présente du passif.

Modules :
    retirement_objective : `RetirementLiability` + builder + utilitaires.
    liability_valuation  : `funded_ratio()` — couverture actif/passif.
    mortality            : Table TH 00-02 (FR) + facteur de rente viagère.
    goal_price_index     : GPI utilisé par GBI (passif spécifique CPPI).
"""

from . import retirement_objective
from . import liability_valuation
from . import mortality
from . import goal_price_index

from .retirement_objective import RetirementLiability, build_liability_from_settings
from .liability_valuation import funded_ratio
