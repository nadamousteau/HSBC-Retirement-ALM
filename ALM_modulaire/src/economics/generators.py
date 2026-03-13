import numpy as np
import pandas as pd
from data.loader import charger_rendements_historiques

def generer_rendements_correles_base(mu_e, sigma_e, mu_b, sigma_b, corr, nb_periodes, nb_sims):
    """
    Génère des rendements corrélés selon Black-Scholes.
    Utilisé pour Target Date (tout stochastique).
    """
    r_e_m = mu_e / 12
    r_b_m = mu_b / 12
    sig_e_m = sigma_e / np.sqrt(12)
    sig_b_m = sigma_b / np.sqrt(12)
    
    cov = np.array([
        [sig_e_m**2, corr * sig_e_m * sig_b_m],
        [corr * sig_e_m * sig_b_m, sig_b_m**2]
    ])
    
    chocs = np.random.multivariate_normal([0, 0], cov, size=(nb_periodes, nb_sims))
    rend_eq = r_e_m - 0.5 * sig_e_m**2 + chocs[:, :, 0]
    rend_bd = r_b_m - 0.5 * sig_b_m**2 + chocs[:, :, 1]
    
    return rend_eq, rend_bd

def generer_rendements_avec_backetest(mu_e, sigma_e, mu_b, sigma_b, corr, dates, date_pivot, nb_sims, asset_equity=None, asset_bond=None):
    """
    Génère rendements avec séparation backtest/forecast (Fixed Mix).
    Backtest : utilise les données historiques réelles jusqu'à la date pivot.
    Forecast : divergence stochastique après la date pivot.
    
    Args:
        mu_e, sigma_e: Paramètres annuels pour equity
        mu_b, sigma_b: Paramètres annuels pour bonds
        corr: Corrélation equity-bonds
        dates: Liste des dates (mensuelles)
        date_pivot: Date de transition entre backtest et forecast
        nb_sims: Nombre de simulations
        asset_equity: Nom de l'actif equity pour charger les données historiques
        asset_bond: Nom de l'actif bond pour charger les données historiques
    
    Returns:
        tuple: (rendements_equity, rendements_bonds, idx_split)
    """
    dt = 1.0 / 12.0
    nb_total_mois = len(dates)
    dates_pd = pd.to_datetime(dates)
    pivot_ts = pd.Timestamp(date_pivot)
    
    # Déterminer l'indice de split
    if pivot_ts < dates_pd[0]:
        idx_split = 0
    elif pivot_ts > dates_pd[-1]:
        idx_split = nb_total_mois
    else:
        idx_split = np.searchsorted(dates_pd, pivot_ts)
    
    s_e = sigma_e * np.sqrt(dt)
    s_b = sigma_b * np.sqrt(dt)
    cov = np.array([[s_e**2, corr*s_e*s_b], [corr*s_e*s_b, s_b**2]])
    
    # Partie Backtest : utiliser les données historiques si fournies
    if idx_split > 0:
        if asset_equity is not None and asset_bond is not None:
            # Charger les données historiques
            r_eq_h, r_bd_h, nb_periodes_loaded = charger_rendements_historiques(
                asset_equity, asset_bond, date_pivot
            )
            
            if r_eq_h is not None and len(r_eq_h) > 0:
                # Utiliser les données historiques réelles (même pour toutes les simulations)
                nb_histo = min(len(r_eq_h), idx_split)
                r_eq_past = np.tile(r_eq_h[-nb_histo:].reshape(-1, 1), (1, nb_sims))
                r_bd_past = np.tile(r_bd_h[-nb_histo:].reshape(-1, 1), (1, nb_sims))
                
                # Si les données historiques sont plus courtes que idx_split, générer le reste
                if nb_histo < idx_split:
                    remaining = idx_split - nb_histo
                    np.random.seed(42)
                    chocs_remaining = np.random.multivariate_normal([0, 0], cov, size=remaining)
                    r_eq_remaining = (mu_e*dt - 0.5*s_e**2) + chocs_remaining[:, 0:1]
                    r_bd_remaining = (mu_b*dt - 0.5*s_b**2) + chocs_remaining[:, 1:2]
                    r_eq_past = np.vstack([r_eq_remaining.repeat(nb_sims, axis=1), r_eq_past])
                    r_bd_past = np.vstack([r_bd_remaining.repeat(nb_sims, axis=1), r_bd_past])
            else:
                # Si le chargement échoue, utiliser la seed fixe
                np.random.seed(42)
                chocs_histo = np.random.multivariate_normal([0, 0], cov, size=idx_split)
                r_eq_h = (mu_e*dt - 0.5*s_e**2) + chocs_histo[:, 0]
                r_bd_h = (mu_b*dt - 0.5*s_b**2) + chocs_histo[:, 1]
                r_eq_past = np.tile(r_eq_h.reshape(-1, 1), (1, nb_sims))
                r_bd_past = np.tile(r_bd_h.reshape(-1, 1), (1, nb_sims))
        else:
            # Seed fixe pour reproductibilité (comportement par défaut)
            np.random.seed(42)
            chocs_histo = np.random.multivariate_normal([0, 0], cov, size=idx_split)
            r_eq_h = (mu_e*dt - 0.5*s_e**2) + chocs_histo[:, 0]
            r_bd_h = (mu_b*dt - 0.5*s_b**2) + chocs_histo[:, 1]
            r_eq_past = np.tile(r_eq_h.reshape(-1, 1), (1, nb_sims))
            r_bd_past = np.tile(r_bd_h.reshape(-1, 1), (1, nb_sims))
    else:
        r_eq_past = np.empty((0, nb_sims))
        r_bd_past = np.empty((0, nb_sims))
    
    # Partie Forecast (divergente)
    nb_mois_futur = nb_total_mois - idx_split
    if nb_mois_futur > 0:
        np.random.seed(None)  # Seed aléatoire pour divergence
        chocs_futur = np.random.multivariate_normal([0, 0], cov, size=(nb_mois_futur, nb_sims))
        r_eq_fut = (mu_e*dt - 0.5*s_e**2) + chocs_futur[:, :, 0]
        r_bd_fut = (mu_b*dt - 0.5*s_b**2) + chocs_futur[:, :, 1]
    else:
        r_eq_fut = np.empty((0, nb_sims))
        r_bd_fut = np.empty((0, nb_sims))
    
    return np.vstack([r_eq_past, r_eq_fut]), np.vstack([r_bd_past, r_bd_fut]), idx_split