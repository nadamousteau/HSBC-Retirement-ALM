import numpy as np
import pandas as pd
from src.economics.nelson_siegel_var import simulate_gbi_monte_carlo, get_calibration

def generer_rendements_bond(Bond, dates, date_pivot, nb_sims, maturity, duration, sigma_bd,
                             gbi_tensor=None, seed=None):
    """
    Génère les rendements d'un type d'obligation spécifique.
    Utilise le modèle Nelson-Siegel + VAR(1) pour les taux ZC (passé et futur).

    Args:
        Bond        : str   — nom de l'obligation (clé du dictionnaire de spreads)
        dates       : array-like — dates mensuelles de la simulation
        date_pivot  : str ou Timestamp — date séparant backtest et forecast
        nb_sims     : int   — nombre de scénarios Monte Carlo
        maturity    : float — maturité cible en années (ex : 10.0)
        duration    : float — duration modifiée de l'obligation
        sigma_bd    : float — volatilité additionnelle en bps (non utilisée si gbi_tensor fourni)
        gbi_tensor  : ndarray (nb_sims, nb_months, 360) ou None
                      Si None, une simulation est lancée en interne (usage autonome).
        seed        : int ou None — graine aléatoire si gbi_tensor=None

    Returns:
        r_bd        : ndarray (nb_total_mois, nb_sims) — rendements mensuels
        idx_split   : int — indice de séparation backtest / forecast
    """
    dt = 1.0 / 12.0
    dates_pd = pd.to_datetime(dates)
    pivot_ts = pd.Timestamp(date_pivot)
    nb_total_mois = len(dates_pd)

    # ------------------------------------------------------------------
    # 1. Spreads par type d'obligation
    # ------------------------------------------------------------------
    spreads_dict = {
        "US Inflation Linked Bond - USD Unhedged": 0.0101,
        "US High Yield Bond BB-B - USD Unhedged":  0.0079,
        "USD Corporate Bond - USD Unhedged":        0.0127,
    }
    spread = spreads_dict.get(Bond, 0.0)

    # ------------------------------------------------------------------
    # 2. Indice de séparation backtest / forecast
    # ------------------------------------------------------------------
    if pivot_ts < dates_pd[0]:
        idx_split = 0
    elif pivot_ts > dates_pd[-1]:
        idx_split = nb_total_mois
    else:
        idx_split = int(np.searchsorted(dates_pd, pivot_ts))

    # ------------------------------------------------------------------
    # 3. Simulation GBI si tenseur non fourni
    #    (utile pour appel autonome ou tests unitaires)
    # ------------------------------------------------------------------
    if gbi_tensor is None:
        calibration = get_calibration()
        gbi_tensor, _, _ = simulate_gbi_monte_carlo(
            nb_sims=nb_sims,
            nb_months=nb_total_mois,
            calibration=calibration,
            seed=seed,
        )
        # gbi_tensor : (nb_sims, nb_total_mois, 360)

    # ------------------------------------------------------------------
    # 4. Extraction du taux ZC à la maturité cible depuis le tenseur GBI
    #    Index dans la grille mensuelle (0-indexed) :
    #      tau_idx = round(maturity * 12) - 1
    # ------------------------------------------------------------------
    tau_idx = int(round(maturity * 12)) - 1
    tau_idx = max(0, min(tau_idx, gbi_tensor.shape[2] - 1))

    # zc_matrix : (nb_total_mois, nb_sims)
    # gbi_tensor[sim, t, tau] -> on transpose : (T, N)
    zc_matrix = gbi_tensor[:, :, tau_idx].T      # (nb_total_mois, nb_sims)

    # Yield = ZC + spread (spread constant)
    y_matrix = zc_matrix + spread                 # (nb_total_mois, nb_sims)

    # ------------------------------------------------------------------
    # 5. Calcul des rendements : carry - effet taux
    #    r(t) = y(t-1) * dt - duration * (y(t) - y(t-1))
    # ------------------------------------------------------------------
    y_prev = np.vstack([y_matrix[0:1, :], y_matrix[:-1, :]])  # y décalé d'un pas
    delta_y = y_matrix - y_prev                               # variation de yield

    r_bd = y_prev * dt - duration * delta_y                   # (nb_total_mois, nb_sims)

    # ------------------------------------------------------------------
    # 6. Cohérence backtest : les scénarios passés sont identiques
    #    Le tenseur GBI est déterministe sur [0, idx_split) si le seed
    #    est fixé, mais on peut aussi forcer la colonne unique dupliquée
    #    pour reproduire le comportement d'origine.
    # ------------------------------------------------------------------
    if idx_split > 0:
        # Moyenne des scénarios sur la partie historique -> signal unique
        r_hist_mean = r_bd[:idx_split, :].mean(axis=1, keepdims=True)  # (idx_split, 1)
        r_bd[:idx_split, :] = np.tile(r_hist_mean, (1, nb_sims))

    return r_bd, idx_split
'''
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
'''