import numpy as np
import pandas as pd

def generer_rendements_actions(mu_e, sigma_e, dates, date_pivot, nb_sims):
    dt = 1.0 / 12.0
    nb_total_mois = len(dates)
    dates_pd = pd.to_datetime(dates)
    pivot_ts = pd.Timestamp(date_pivot)
    
    if pivot_ts < dates_pd[0]:
        idx_split = 0
    elif pivot_ts > dates_pd[-1]:
        idx_split = nb_total_mois
    else:
        idx_split = np.searchsorted(dates_pd, pivot_ts)
    
    s_e = sigma_e * np.sqrt(dt)
    
    # Partie Backtest
    if idx_split > 0:
        np.random.seed(42)
        chocs_histo = np.random.normal(0, s_e, size=idx_split)
        r_eq_h = (mu_e*dt - 0.5*s_e**2) + chocs_histo
        r_eq_past = np.tile(r_eq_h.reshape(-1, 1), (1, nb_sims))
    else:
        r_eq_past = np.empty((0, nb_sims))
    
    # Partie Forecast
    nb_mois_futur = nb_total_mois - idx_split
    if nb_mois_futur > 0:
        np.random.seed(None)
        chocs_futur = np.random.normal(0, s_e, size=(nb_mois_futur, nb_sims))
        r_eq_fut = (mu_e*dt - 0.5*s_e**2) + chocs_futur
    else:
        r_eq_fut = np.empty((0, nb_sims))
    r_eq=np.vstack([r_eq_past, r_eq_fut])
    return r_eq, idx_split