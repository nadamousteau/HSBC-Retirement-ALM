"""
Moteur de simulation GBI (Goal-Based Investing / CPPI dynamique).

Architecture distincte du moteur standard (core.py) car la stratégie GBI
requiert l'accès au patrimoine courant et au GoalPriceIndex pour calculer
l'allocation à chaque pas de temps.

Approche hybride backtest/forecast :
  - Phase backtest  : betas calculés via la courbe des taux réelle (déterministe)
  - Phase forecast  : betas propagés par sim via le rendement obligataire
                      (proxy LDI : r_bond ≈ d(beta)/beta)

Contributions & salaire : identiques au moteur standard (calculer_apport_exponentiel).
Glide path TDF          : identique à TargetDateStrategy (profiles.allocation_initiale).
"""

import numpy as np
import pandas as pd
from config import settings, profiles
from src.liabilities import contributions


def _compute_beta_matrix(gpi, dates, idx_split, r_bd, nb_sims):
    """
    Construit la matrice de GPI (betas) de forme (nb_periods, nb_sims).

    - Période backtest [0, idx_split[   : même beta pour toutes les simulations
                                         (courbe des taux historique réelle)
    - Période forecast [idx_split, end[ : beta diverge par sim via r_bd comme proxy
                                         beta(t+1) = beta(t) * (1 + r_bd(t))
    """
    nb_periods = len(dates)
    beta_matrix = np.zeros((nb_periods, nb_sims))

    # ── Backtest : betas déterministes ──────────────────────────────────────
    if idx_split > 0:
        beta_hist = gpi.compute_beta_series(dates[:idx_split])
        beta_matrix[:idx_split, :] = beta_hist.reshape(-1, 1)
    else:
        # Aucun historique : initialiser avec la première date forecast
        beta_matrix[0, :] = gpi.calculate(dates[0])

    # ── Forecast : propagation par simulation ───────────────────────────────
    forecast_start = max(idx_split, 1)
    for t in range(forecast_start, nb_periods):
        beta_matrix[t, :] = beta_matrix[t - 1, :] * (1.0 + r_bd[t - 1, :])

    beta_matrix = np.maximum(beta_matrix, 1e-6)
    return beta_matrix


def run_simulation_gbi(gpi, r_eq, r_bd, dates, idx_split):
    """
    Exécute la simulation Monte Carlo GBI (CPPI avec plancher lié au GPI).

    Contributions et salaire : identiques au moteur standard (calculer_apport_exponentiel).
    Glide path TDF            : identique à TargetDateStrategy via profiles.

    Args:
        gpi        : GoalPriceIndex (déjà initialisé avec yield_curve)
        r_eq       : ndarray (nb_periods, nb_sims) - rendements actions
        r_bd       : ndarray (nb_periods, nb_sims) - rendements obligations
        dates      : DatetimeIndex de longueur nb_periods
        idx_split  : indice de séparation backtest/forecast

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

    # ── Pré-calcul de la matrice de betas ──────────────────────────────────
    beta_matrix = _compute_beta_matrix(gpi, dates, idx_split, r_bd, nb_sims)

    # ── Pré-calcul des paramètres de contribution (identique à core.py) ────
    app_init, app_max, t_pic = contributions.precalculer_parametres_apport_exponentiel(
        settings.SALAIRE_INITIAL, settings.SALAIRE_MAX_CIBLE, settings.NB_ANNEES_ACCUMULATION
    )

    # ── Initialisation ─────────────────────────────────────────────────────
    mat_capital    = np.zeros((nb_periods + 1, nb_sims))
    mat_capital[0] = capital_init
    hist_alloc_psp = np.zeros((nb_periods, nb_sims))

    courbe_investi = np.zeros(nb_periods + 1)
    courbe_investi[0] = capital_init

    hist_apport   = np.zeros(nb_periods)
    hist_drawdown = np.zeros((nb_periods, nb_sims))
    hist_salaire  = np.zeros(nb_periods)

    capital_max = np.full(nb_sims, capital_init)

    # Suivi du patrimoine de début d'année pour le plancher dynamique
    W_annee_debut    = np.full(nb_sims, float(capital_init))
    beta_annee_debut = beta_matrix[0, :].copy()

    # ── Boucle temporelle ──────────────────────────────────────────────────
    for k in range(nb_periods):
        date     = dates[k]
        t_annees = k / 12.0
        age      = age_depart + t_annees

        W = mat_capital[k].copy()  # (nb_sims,)

        # ── 1. Contribution mensuelle — identique à core.py ───────────────
        apport_mensuel = contributions.calculer_apport_exponentiel(
            t_annees, app_init, app_max, t_pic
        ) if k > 0 else 0.0

        salaire = contributions.estimer_salaire_saturation(
            t_annees, settings.SALAIRE_INITIAL, settings.SALAIRE_MAX_CIBLE
        )

        W += apport_mensuel
        hist_apport[k]      = apport_mensuel
        hist_salaire[k]     = salaire
        courbe_investi[k+1] = courbe_investi[k] + apport_mensuel

        # ── 2. Réinitialisation annuelle du plancher (chaque janvier) ─────
        if k > 0:
            prev_date = dates[k - 1]
            if date.month == 1 and prev_date.month == 12:
                W_annee_debut    = W.copy()
                beta_annee_debut = beta_matrix[k, :].copy()

        # ── 3. GPI courant et plancher dynamique ──────────────────────────
        beta_t    = beta_matrix[k, :]
        beta_safe = np.where(beta_annee_debut > 1e-9, beta_annee_debut, 1.0)
        floor     = floor_pct * (W_annee_debut / beta_safe) * beta_t
        floor     = np.maximum(floor, 0.0)

        # ── 4. Multiplicateur GBI — glide path TDF identique à TargetDateStrategy ──
        # Formule miroir de target_date.py :
        #   pct_equity = allocation_initiale - decroissance_annuelle * annees_ecoulees
        annees_ecoulees = age - age_depart
        alloc_tdf = profiles.allocation_initiale - profiles.decroissance_annuelle * annees_ecoulees
        alloc_tdf = max(0.05, min(1.0, alloc_tdf))

        m       = alloc_tdf / (1.0 - floor_pct + 1e-9)
        cushion = np.maximum(W - floor, 0.0)
        w_psp   = np.where(W > 1e-9, np.minimum(1.0, m * cushion / W), 0.0)

        hist_alloc_psp[k, :] = w_psp

        # ── 5. Rendement du portefeuille ──────────────────────────────────
        r_port = w_psp * r_eq[k] + (1.0 - w_psp) * r_bd[k]
        W *= (1.0 + r_port)
        W  = np.maximum(W, 0.0)

        mat_capital[k+1] = W

        # ── 6. Drawdown ───────────────────────────────────────────────────
        capital_max      = np.maximum(capital_max, W)
        dd               = np.where(capital_max > 1e-9, (W - capital_max) / capital_max, 0.0)
        hist_drawdown[k] = dd

    return mat_capital, courbe_investi, hist_apport, hist_drawdown, hist_salaire, hist_alloc_psp

