"""
POLITIQUE DE CONTRIBUTION — Apport mensuel paramétrique
========================================================
Issu de la Vague 2 / Tâche C : déplacé verbatim depuis
`src.liabilities.contributions` (où ces fonctions vivaient par erreur — un
apport mensuel est un flux d'actif, pas un passif retraite).

Trois fonctions standalone consommées par `engine.core` et `engine.gbi_core` :

    precalculer_parametres_apport_exponentiel(s_init, s_max, duree_totale)
        → (app_init, app_max, t_pic) — paramètres de la parabole d'apport
        centrée sur le pic de carrière. La cible s_cible = s_init + 93.5 %
        de l'amplitude (s_max - s_init) marque le moment où la vitesse de
        progression (≈ 10 %/an) sature.

    calculer_apport_exponentiel(t, app_init, app_max, t_pic)
        → apport mensuel à l'instant t (années) : parabole inversée
        f(t) = a (t - t_pic)² + app_max avec a < 0.

    estimer_salaire_saturation(t, s_init, s_max)
        → courbe logistique exponentielle inversée modélisant la saturation
        salariale ; utilisée pour la cohérence du reporting.
"""

import numpy as np


def precalculer_parametres_apport_exponentiel(s_init, s_max, duree_totale):
    from config import settings
    ratio = s_max / s_init
    facteur = ratio ** 1.5
    app_init = s_init * settings.TAUX_APPORT_BASE
    app_max = app_init * facteur
    s_cible = s_init + (s_max - s_init) * 0.935
    if s_cible >= s_max:
        t_pic = duree_totale
    else:
        val_log = 1 - min((s_cible - s_init) / max(1.0, (s_max - s_init)), 0.9999)
        t_pic = -np.log(val_log) / 0.10
    return app_init, app_max, np.clip(t_pic, 0, duree_totale)


def calculer_apport_exponentiel(t, app_init, app_max, t_pic):
    if t_pic <= 0:
        return app_init
    a = (app_init - app_max) / (t_pic ** 2)
    return max(0.0, a * (t - t_pic) ** 2 + app_max)


def estimer_salaire_saturation(t, s_init, s_max):
    return s_init + (s_max - s_init) * (1 - np.exp(-0.10 * t))
