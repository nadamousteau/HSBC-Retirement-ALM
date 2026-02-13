import numpy as np
import pandas as pd
from config import settings

def ajouter_chocs_merton(rendements_eq, rendements_bd, nb_periodes, nb_sims):
    """
    Jump-Diffusion de Merton (Target Date)
    Application probabiliste d'un saut (crise) sur les rendements.
    Utilise les paramètres définis dans settings (LAMBDA_CRISE, etc.).
    """
    lambda_mensuelle = settings.LAMBDA_CRISE / 12
    
    for t in range(nb_periodes):
        # Tirage binomial : y a-t-il une crise ce mois ?
        crise = np.random.binomial(1, lambda_mensuelle, nb_sims)
        
        # Appliquer choc uniquement si crise=1
        choc_eq = crise * np.random.normal(settings.SEVERITE_EQ_MOYENNE, settings.SEVERITE_EQ_SIGMA, nb_sims)
        choc_bd = crise * np.random.normal(settings.SEVERITE_BD_MOYENNE, settings.SEVERITE_BD_SIGMA, nb_sims)
        
        rendements_eq[t] += choc_eq
        rendements_bd[t] += choc_bd
    
    return rendements_eq, rendements_bd

def injecter_crise_localisee(r_eq, r_bd, dates_list, date_depart, params):
    """
    Crise localisée à une date précise (Fixed Mix).
    Choc initial + volatilité accrue pendant N mois.
    """
    r_eq_m = r_eq.copy()
    r_bd_m = r_bd.copy()
    
    dates_pd = pd.to_datetime(dates_list)
    idx = np.argmin(np.abs(dates_pd - pd.Timestamp(date_depart)))
    
    # Vérifier que la date est bien dans la période (marge de 40 jours)
    if abs((dates_pd[idx] - pd.Timestamp(date_depart)).days) > 40:
        return r_eq, r_bd
    
    # Choc initial (log-return pour cohérence)
    r_eq_m[idx, :] = np.log(1.0 - params.get('drop_eq', 0.3))
    r_bd_m[idx, :] = np.log(1.0 - params.get('drop_bd', 0.0))
    
    # Volatilité accrue pendant la période de récupération
    end = min(idx + params.get('duree_mois', 12), r_eq.shape[0])
    facteur = params.get('facteur_vol', 2.0)
    
    r_eq_m[idx+1:end, :] *= facteur
    r_bd_m[idx+1:end, :] *= facteur
    
    return r_eq_m, r_bd_m