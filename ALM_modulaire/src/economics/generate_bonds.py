import numpy as np
import pandas as pd
from src.economics.yield_curve import YieldCurveBuilder

def generer_rendements_bond(Bond, dates, date_pivot, nb_sims, yc_builder, maturity, duration, sigma_bd):
    """
    Génère les rendements d'un type d'obligation spécifique.
    Utilise YieldCurveBuilder pour l'historique et simule le futur.
    """
    dt = 1.0 / 12.0
    nb_total_mois = len(dates)
    dates_pd = pd.to_datetime(dates)
    pivot_ts = pd.Timestamp(date_pivot)
    
    # 1. Dictionnaire des spreads 
    spreads_dict = {
        "US Inflation Linked Bond - USD Unhedged": 0.0101,  # 1.01%
        "US High Yield Bond BB-B - USD Unhedged": 0.0079,   # 0.79%
        "USD Corporate Bond - USD Unhedged": 0.0127         # 1.27%
    }
    
    spread = spreads_dict.get(Bond, 0.0)

    # 2. Déterminer l'indice de séparation Passé (Backtest) / Futur (Forecast)
    if pivot_ts < dates_pd[0]:
        idx_split = 0
    elif pivot_ts > dates_pd[-1]:
        idx_split = nb_total_mois
    else:
        idx_split = np.searchsorted(dates_pd, pivot_ts)

    # ---------------------------------------------------------
    # PARTIE BACKTEST (Historique lu depuis yield_curve.py)
    # ---------------------------------------------------------
    last_y_hist = 0.0
    if idx_split > 0:
        dates_backtest = dates_pd[:idx_split]
        y_hist = np.zeros(idx_split)
        
        for i, d in enumerate(dates_backtest):
            # Appel de TA fonction pour récupérer le ZC historique
            zc = yc_builder.get_zero_rate(d, maturity)
            # Le yield est le ZC + le spread fixe
            y_hist[i] = zc + spread
            
        # Calcul de la performance (Return) historique
        y_hist_prev = np.roll(y_hist, shift=1)
        y_hist_prev[0] = y_hist[0] 
        
        # Formule du rendement = Portage - Effet Taux
        r_bd_h = y_hist_prev * dt - duration * (y_hist - y_hist_prev)
        
        # Duplication du passé pour toutes les simulations
        r_bd_past = np.tile(r_bd_h.reshape(-1, 1), (1, nb_sims))
        last_y_hist = y_hist[-1]
    else:
        r_bd_past = np.empty((0, nb_sims))
        # Initialisation si pas de backtest
        zc_init = yc_builder.get_zero_rate(pivot_ts, maturity)
        last_y_hist = zc_init + spread

    # ---------------------------------------------------------
    # PARTIE FORECAST (Simulation du futur)
    # ---------------------------------------------------------
    nb_mois_futur = nb_total_mois - idx_split
    if nb_mois_futur > 0:
        # Volatilité du ZC convertie en décimale mensuelle
        s_zc = (sigma_bd / 10000.0) * np.sqrt(dt)
        
        np.random.seed(None)
        # Chocs aléatoires normaux pour les variations du taux ZC futur
        delta_zc = np.random.normal(0, s_zc, size=(nb_mois_futur, nb_sims))
        
        # Le spread étant fixe ici, la variation du Yield est égale à la variation du ZC
        delta_y = delta_zc 
        
        # Reconstruction des trajectoires de Yield
        y_futur = np.zeros((nb_mois_futur, nb_sims))
        y_futur[0, :] = last_y_hist + delta_y[0, :]
        for t in range(1, nb_mois_futur):
            y_futur[t, :] = y_futur[t-1, :] + delta_y[t, :]
            
        # Calcul des performances (Returns) futures
        y_futur_prev = np.vstack([np.full((1, nb_sims), last_y_hist), y_futur[:-1, :]])
        r_bd_fut = y_futur_prev * dt - duration * delta_y
    else:
        r_bd_fut = np.empty((0, nb_sims))

    r_bd= np.vstack([r_bd_past, r_bd_fut])

    return r_bd, idx_split