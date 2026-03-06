"""
Moteur de simulation GBI (Goal-Based Investing / CPPI dynamique).

Architecture distincte du moteur standard (core.py) car la strategie GBI
requiert l'acces au patrimoine courant et au GoalPriceIndex pour calculer
l'allocation a chaque pas de temps.

Architecture de courbes GBI :
  - Les courbes de taux GBI sont simulees via Nelson-Siegel + VAR(1)
    (module src.economics.gbi_nelson_siegel).
  - Le tenseur (N, T, 360) de taux annualises alimente le calcul du GPI.
  - Phase backtest  : betas GPI calcules via la courbe des taux historique (deterministe)
  - Phase forecast  : betas GPI propages via les taux NS-VAR simules (stochastiques)

Contributions & salaire : identiques au moteur standard (calculer_apport_exponentiel).
Glide path TDF          : identique a TargetDateStrategy (profiles.allocation_initiale).
"""

import numpy as np
import pandas as pd
from config import settings, profiles
from src.liabilities.contributions import (
    precalculer_parametres_apport_exponentiel,
    calculer_apport_exponentiel as _calculer_apport,
    estimer_salaire_saturation as _estimer_salaire,
)


def _compute_beta_matrix(gpi, dates, idx_split, gbi_curves, tau, nb_sims):
    """
    Construit la matrice de GPI (betas) de forme (nb_periods, nb_sims).

    - Periode backtest [0, idx_split[  : meme beta pour toutes les simulations
                                        (courbe des taux historique reelle via YieldCurveBuilder)
    - Periode forecast [idx_split, end[ : beta calcule a partir des courbes NS-VAR simulees
                                          (stochastique, different par simulation)

    Args:
        gpi        : GoalPriceIndex (deja initialise avec yield_curve)
        dates      : DatetimeIndex de longueur nb_periods
        idx_split  : indice de separation backtest/forecast
        gbi_curves : np.ndarray (N, T_forecast, 360) - courbes NS-VAR simulees
                     pour la periode forecast uniquement
        tau        : np.ndarray (360,) - grille de maturites en annees
        nb_sims    : int - nombre de simulations
    """
    nb_periods = len(dates)
    beta_matrix = np.zeros((nb_periods, nb_sims))

    # -- Backtest : betas deterministes (courbe historique reelle) ---------
    if idx_split > 0:
        beta_hist = gpi.compute_beta_series(dates[:idx_split])
        beta_matrix[:idx_split, :] = beta_hist.reshape(-1, 1)
    else:
        beta_matrix[0, :] = gpi.calculate(dates[0])

    # -- Forecast : betas a partir des courbes NS-VAR simulees -------------
    forecast_start = max(idx_split, 1)

    if gbi_curves is not None:
        ret_date = gpi.ret_date
        dec_years = gpi.dec_years

        for t in range(forecast_start, nb_periods):
            t_forecast = t - forecast_start
            date = dates[t]

            # Annees restantes jusqu'a la retraite
            t_ret = (ret_date - date).days / 365.25

            if t_ret < 0:
                rem = max(0, dec_years + t_ret)
            else:
                rem = dec_years

            if rem <= 0:
                beta_matrix[t, :] = 1.0
                continue

            n_steps = int(np.ceil(rem))
            beta_vals = np.zeros(nb_sims)

            for k in range(n_steps):
                tau_k = max(0.0, t_ret) + k
                if tau_k < 1e-6:
                    tau_k = 1.0 / 12.0

                # Trouver l'indice de maturite le plus proche dans la grille
                idx_tau = np.argmin(np.abs(tau - tau_k))

                # Taux pour toutes les simulations a cette maturite : (N,)
                r_k = gbi_curves[:, t_forecast, idx_tau]

                beta_vals += np.exp(-r_k * tau_k)

            beta_matrix[t, :] = np.maximum(beta_vals, 1.0)
    else:
        # Fallback : dernier beta connu replique
        for t in range(forecast_start, nb_periods):
            beta_matrix[t, :] = beta_matrix[forecast_start - 1, :]

    beta_matrix = np.maximum(beta_matrix, 1e-6)
    return beta_matrix


def run_simulation_gbi(gpi, r_eq, r_bd, dates, idx_split, gbi_curves=None, tau=None):
    """
    Execute la simulation Monte Carlo GBI (CPPI avec plancher lie au GPI).

    Les courbes GBI sont fournies sous forme d'un tenseur (N, T_forecast, 360)
    issu de la simulation Nelson-Siegel + VAR(1).

    Args:
        gpi        : GoalPriceIndex (deja initialise avec yield_curve)
        r_eq       : ndarray (nb_periods, nb_sims) - rendements actions
        r_bd       : ndarray (nb_periods, nb_sims) - rendements obligations
        dates      : DatetimeIndex de longueur nb_periods
        idx_split  : indice de separation backtest/forecast
        gbi_curves : ndarray (N, T_forecast, 360) - courbes GBI simulees (NS-VAR)
        tau        : ndarray (360,) - grille de maturites en annees

    Returns:
        mat_capital   : ndarray (nb_periods+1, nb_sims)
        courbe_investi: ndarray (nb_periods+1,)
        hist_apport   : ndarray (nb_periods,)
        hist_drawdown : ndarray (nb_periods, nb_sims)
        hist_salaire  : ndarray (nb_periods,)
        hist_alloc_psp: ndarray (nb_periods, nb_sims) - allocation PSP par sim
    """
    nb_periods, nb_sims = r_eq.shape
    ret_date = gpi.ret_date

    floor_pct    = settings.FLOOR_PERCENT_GBI
    age_depart   = settings.AGE_DEPART
    capital_init = settings.CAPITAL_INITIAL

    # -- Pre-calcul de la matrice de betas ---------------------------------
    beta_matrix = _compute_beta_matrix(gpi, dates, idx_split, gbi_curves, tau, nb_sims)

    # -- Pre-calcul des parametres de contribution -------------------------
    app_init, app_max, t_pic = precalculer_parametres_apport_exponentiel(
        settings.SALAIRE_INITIAL, settings.SALAIRE_MAX_CIBLE, settings.NB_ANNEES_ACCUMULATION
    )

    # -- Initialisation ----------------------------------------------------
    mat_capital    = np.zeros((nb_periods + 1, nb_sims))
    mat_capital[0] = capital_init
    hist_alloc_psp = np.zeros((nb_periods, nb_sims))

    courbe_investi = np.zeros(nb_periods + 1)
    courbe_investi[0] = capital_init

    hist_apport   = np.zeros(nb_periods)
    hist_drawdown = np.zeros((nb_periods, nb_sims))
    hist_salaire  = np.zeros(nb_periods)

    capital_max = np.full(nb_sims, capital_init)

    # Suivi du patrimoine de debut d'annee pour le plancher dynamique
    W_annee_debut    = np.full(nb_sims, float(capital_init))
    beta_annee_debut = beta_matrix[0, :].copy()

    # -- Boucle temporelle -------------------------------------------------
    for k in range(nb_periods):
        date     = dates[k]
        t_annees = k / 12.0
        age      = age_depart + t_annees

        W = mat_capital[k].copy()  # (nb_sims,)

        # 1. Contribution mensuelle
        apport_mensuel = _calculer_apport(
            t_annees, app_init, app_max, t_pic
        ) if k > 0 else 0.0

        salaire = _estimer_salaire(
            t_annees, settings.SALAIRE_INITIAL, settings.SALAIRE_MAX_CIBLE
        )

        W += apport_mensuel
        hist_apport[k]      = apport_mensuel
        hist_salaire[k]     = salaire
        courbe_investi[k+1] = courbe_investi[k] + apport_mensuel

        # 2. Reinitialisation annuelle du plancher (chaque janvier)
        if k > 0:
            prev_date = dates[k - 1]
            if date.month == 1 and prev_date.month == 12:
                W_annee_debut    = W.copy()
                beta_annee_debut = beta_matrix[k, :].copy()

        # 3. GPI courant et plancher dynamique
        beta_t    = beta_matrix[k, :]
        beta_safe = np.where(beta_annee_debut > 1e-9, beta_annee_debut, 1.0)
        floor     = floor_pct * (W_annee_debut / beta_safe) * beta_t
        floor     = np.maximum(floor, 0.0)

        # 4. Multiplicateur GBI  (glide path TDF)
        annees_ecoulees = age - age_depart
        alloc_tdf = profiles.allocation_initiale - profiles.decroissance_annuelle * annees_ecoulees
        alloc_tdf = max(0.05, min(1.0, alloc_tdf))

        m       = alloc_tdf / (1.0 - floor_pct + 1e-9)
        cushion = np.maximum(W - floor, 0.0)
        w_psp   = np.where(W > 1e-9, np.minimum(1.0, m * cushion / W), 0.0)

        hist_alloc_psp[k, :] = w_psp

        # 5. Rendement du portefeuille
        r_port = w_psp * r_eq[k] + (1.0 - w_psp) * r_bd[k]
        W *= (1.0 + r_port)
        W  = np.maximum(W, 0.0)

        mat_capital[k+1] = W

        # 6. Drawdown
        capital_max      = np.maximum(capital_max, W)
        dd               = np.where(capital_max > 1e-9, (W - capital_max) / capital_max, 0.0)
        hist_drawdown[k] = dd

    return mat_capital, courbe_investi, hist_apport, hist_drawdown, hist_salaire, hist_alloc_psp
