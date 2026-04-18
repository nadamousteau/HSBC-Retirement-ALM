"""
ASSETS — Modélisation du capital humain et de la politique de contribution
===========================================================================
Issu de la Vague 2 / Tâche C : ce package isole les composants "actif" du
profil ALM client (capital humain stochastique, politique d'apport mensuel)
de l'objectif de retraite (modélisé désormais comme un passif dans
`src.liabilities.retirement_objective`).

Modules :
    human_capital       : `HumanCapitalCurve` — diffusion stochastique
                          hétérogène des salaires avec élasticité dynamique
                          à l'inflation.
    contribution_policy : Fonctions standalone (`precalculer_parametres_apport_exponentiel`,
                          `calculer_apport_exponentiel`, `estimer_salaire_saturation`)
                          consommées par les moteurs `engine.core` et
                          `engine.gbi_core`.

Avant la Vague 2, ces composants vivaient dans `src.liabilities.contributions`
— vestige d'une assimilation incorrecte entre apport (flux d'actif) et
retrait (flux de passif). La séparation actif/passif est désormais nette.
"""

from . import human_capital
from . import contribution_policy
