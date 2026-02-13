import numpy as np
import pandas as pd

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

def generer_rendements_avec_backtest(mu_e, sigma_e, mu_b, sigma_b, corr, dates, date_pivot, nb_sims):
    """
    Génère rendements avec séparation backtest/forecast (Fixed Mix).
    Backtest : même historique pour tous (seed fixe = 42).
    Forecast : divergence stochastique (seed None).
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
    
    # Partie Backtest (commune à toutes les simulations)
    if idx_split > 0:
        np.random.seed(42)  # Seed fixe pour reproductibilité
        chocs_histo = np.random.multivariate_normal([0, 0], cov, size=idx_split)
        r_eq_h = (mu_e*dt - 0.5*s_e**2) + chocs_histo[:, 0]
        r_bd_h = (mu_b*dt - 0.5*s_b**2) + chocs_histo[:, 1]
        
        # Duplication du même passé pour toutes les simulations
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