import numpy as np
from config import settings


class HumanCapitalCurve:
    """
    Modèle de contribution quadratique (en cloche, hump-shaped) basé sur l'âge.

    Formule : C(age) = a * (age - age_pic)^2 + contrib_pic
        → Parabole avec le MAXIMUM à age_pic, qui décline symétriquement.
        → Garantit C(age) >= 0.

    Paramètres (depuis settings.GBI_*) :
        age_depart    : âge du client au début de la simulation
        contrib_debut : contribution annuelle à age_depart (€/an)
        age_pic       : âge où la contribution est maximale
        contrib_pic   : contribution annuelle maximale (€/an)

    Usage :
        hc = HumanCapitalCurve(20, 2400, 45, 8400)
        apport_mensuel = hc.get_contribution_mensuelle(current_age)
    """

    def __init__(self, age_depart, contrib_debut, age_pic, contrib_pic):
        self.age_pic = age_pic
        self.contrib_pic = contrib_pic

        if age_depart != age_pic:
            self.a = (contrib_debut - contrib_pic) / ((age_depart - age_pic) ** 2)
        else:
            self.a = 0.0

    def get_contribution_annuelle(self, current_age):
        """Retourne la contribution annuelle pour un âge donné (float, >= 0)."""
        amount = self.a * (current_age - self.age_pic) ** 2 + self.contrib_pic
        return max(0.0, amount)

    def get_contribution_mensuelle(self, current_age):
        """Retourne la contribution mensuelle pour un âge donné (float, >= 0)."""
        return self.get_contribution_annuelle(current_age) / 12.0

def calculer_apport_quadratique(t_annees, apport_init, duree_totale):
    """
    Modèle Target Date : courbe en cloche (quadratique)
    Pic à RATIO_PIC_CARRIERE de la durée totale.
    """
    t_pic = duree_totale * settings.RATIO_PIC_CARRIERE
    apport_max = apport_init * settings.FACTEUR_CROISSANCE_MAX
    
    if t_pic > 0:
        a = (apport_init - apport_max) / (t_pic**2)
    else:
        return apport_init
    
    apport = a * (t_annees - t_pic)**2 + apport_max
    return max(apport, 0)

def estimer_salaire_saturation(t_annees, S_init, S_max):
    """
    Modèle Fixed Mix : salaire avec saturation exponentielle.
    Utilise VITESSE_PROGRESSION défini dans les settings.
    """
    return S_init + (S_max - S_init) * (1 - np.exp(-settings.VITESSE_PROGRESSION * t_annees))

def precalculer_parametres_apport_exponentiel(S_init, S_max, duree_totale):
    """
    Calcule les paramètres pour l'apport avec élasticité (Fixed Mix).
    Utilise GAMMA_ELASTICITE, TAUX_APPORT_BASE, SEUIL_MATURITE.
    """
    ratio = S_max / S_init
    facteur = ratio ** settings.GAMMA_ELASTICITE
    
    # Note: TAUX_APPORT_BASE est dans settings
    app_init = S_init * settings.TAUX_APPORT_BASE
    app_max = app_init * facteur
    
    s_cible = S_init + (S_max - S_init) * settings.SEUIL_MATURITE
    
    if s_cible >= S_max:
        t_pic = duree_totale
    else:
        # Évite le log de nombre négatif ou nul
        val_log = 1 - min((s_cible - S_init) / (S_max - S_init), 0.9999)
        t_pic = -np.log(val_log) / settings.VITESSE_PROGRESSION
    
    return app_init, app_max, min(max(0, t_pic), duree_totale)

def calculer_apport_exponentiel(t_annees, app_init, app_max, t_pic):
    """
    Calcule l'apport mensuel selon modèle exponentiel (Fixed Mix).
    """
    if t_pic <= 0:
        return app_init
    
    a = (app_init - app_max) / (t_pic**2)
    return max(a * (t_annees - t_pic)**2 + app_max, 0)