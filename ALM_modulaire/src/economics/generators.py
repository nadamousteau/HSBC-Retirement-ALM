import numpy as np
import pandas as pd
from data.loader import charger_rendements_historiques
from src.economics.inflation_vasicek import VasicekInflation


def generer_rendements_correles_base(mu_e, sigma_e, mu_b, sigma_b, corr,
                                     nb_periodes, nb_sims, rng=None):
    """
    Génère des rendements corrélés selon Black-Scholes.
    Utilisé pour Target Date (tout stochastique).
    """
    if rng is None:
        rng = np.random.default_rng()

    r_e_m = mu_e / 12
    r_b_m = mu_b / 12
    sig_e_m = sigma_e / np.sqrt(12)
    sig_b_m = sigma_b / np.sqrt(12)

    cov = np.array([
        [sig_e_m**2, corr * sig_e_m * sig_b_m],
        [corr * sig_e_m * sig_b_m, sig_b_m**2]
    ])

    chocs = rng.multivariate_normal([0, 0], cov, size=(nb_periodes, nb_sims))
    rend_eq = r_e_m - 0.5 * sig_e_m**2 + chocs[:, :, 0]
    rend_bd = r_b_m - 0.5 * sig_b_m**2 + chocs[:, :, 1]

    return rend_eq, rend_bd


def _compute_idx_split(dates, date_pivot):
    """Indice du premier mois forecast (date >= date_pivot)."""
    nb_total_mois = len(dates)
    dates_pd = pd.to_datetime(dates)
    pivot_ts = pd.Timestamp(date_pivot)

    if pivot_ts < dates_pd[0]:
        return 0
    if pivot_ts > dates_pd[-1]:
        return nb_total_mois
    return int(np.searchsorted(dates_pd, pivot_ts))


def generer_rendements_backtest(mu_e, sigma_e, mu_b, sigma_b, corr,
                                dates, date_pivot, nb_sims,
                                asset_equity=None, asset_bond=None, rng=None):
    """
    Génère uniquement la portion BACKTEST des rendements (avant `date_pivot`).

    Comportement :
        1. Calcule `idx_split` (premier mois forecast) à partir de `dates`/`date_pivot`.
        2. Si idx_split == 0 (pivot avant la première date) : retourne deux matrices vides.
        3. Sinon : tente de charger les historiques pour `asset_equity`/`asset_bond`.
           - Succès : tile la trajectoire historique (même pour toutes les sims).
                     Si l'historique est plus court que `idx_split`, le préfixe manquant
                     est rempli par tirage Black-Scholes COMMUN (un seul tirage répété
                     pour toutes les sims, cohérent avec la philosophie déterministe
                     du backtest).
           - Échec / asset non fourni : tirage Black-Scholes commun à toutes les sims
             sur tout `idx_split` (fallback).

    Le RNG n'est consommé que dans la branche fallback (historique manquant) — les
    runs sans fallback sont parfaitement reproductibles puisque la seule source de
    variabilité backtest est la table CSV.

    Returns:
        tuple : (r_eq_past, r_bd_past, idx_split)
                r_eq_past, r_bd_past de shape (idx_split, nb_sims) — log-rendements
                                       mensuels.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = 1.0 / 12.0
    idx_split = _compute_idx_split(dates, date_pivot)

    if idx_split == 0:
        return (
            np.empty((0, nb_sims)),
            np.empty((0, nb_sims)),
            idx_split,
        )

    s_e = sigma_e * np.sqrt(dt)
    s_b = sigma_b * np.sqrt(dt)
    cov = np.array([[s_e**2, corr*s_e*s_b], [corr*s_e*s_b, s_b**2]])

    if asset_equity is not None and asset_bond is not None:
        r_eq_h, r_bd_h, _ = charger_rendements_historiques(
            asset_equity, asset_bond, date_pivot
        )

        if r_eq_h is not None and len(r_eq_h) > 0:
            nb_histo = min(len(r_eq_h), idx_split)
            r_eq_past = np.tile(r_eq_h[-nb_histo:].reshape(-1, 1), (1, nb_sims))
            r_bd_past = np.tile(r_bd_h[-nb_histo:].reshape(-1, 1), (1, nb_sims))

            # Si l'historique est plus court que idx_split, préfixer par tirage BS
            # commun à toutes les sims (path déterministe partagé).
            if nb_histo < idx_split:
                remaining = idx_split - nb_histo
                chocs_remaining = rng.multivariate_normal([0, 0], cov, size=remaining)
                r_eq_remaining = (mu_e*dt - 0.5*s_e**2) + chocs_remaining[:, 0:1]
                r_bd_remaining = (mu_b*dt - 0.5*s_b**2) + chocs_remaining[:, 1:2]
                r_eq_past = np.vstack([
                    r_eq_remaining.repeat(nb_sims, axis=1), r_eq_past
                ])
                r_bd_past = np.vstack([
                    r_bd_remaining.repeat(nb_sims, axis=1), r_bd_past
                ])
            return r_eq_past, r_bd_past, idx_split

    # Fallback : pas d'historique disponible → tirage BS commun à toutes les sims.
    chocs_histo = rng.multivariate_normal([0, 0], cov, size=idx_split)
    r_eq_h = (mu_e*dt - 0.5*s_e**2) + chocs_histo[:, 0]
    r_bd_h = (mu_b*dt - 0.5*s_b**2) + chocs_histo[:, 1]
    r_eq_past = np.tile(r_eq_h.reshape(-1, 1), (1, nb_sims))
    r_bd_past = np.tile(r_bd_h.reshape(-1, 1), (1, nb_sims))
    return r_eq_past, r_bd_past, idx_split


def generer_rendements_forecast(mu_e, sigma_e, mu_b, sigma_b, corr,
                                nb_mois, nb_sims, rng=None):
    """
    Génère uniquement la portion FORECAST des rendements (après `date_pivot`),
    par tirage Black-Scholes corrélé EQ/BD divergent entre simulations.

    Cette fonction NE génère PAS d'inflation. Elle est utile pour les modules
    qui n'ont pas besoin de corrélation avec l'inflation (ex. : sous-modules
    de stratégie qui pré-tirent leurs propres scénarios). Le pipeline principal
    utilise `generer_scenarios_marche_correles` à la place, qui produit
    conjointement EQ/BD/inflation avec la corrélation bonds-inflation cible.

    Args:
        mu_e, sigma_e, mu_b, sigma_b : paramètres ANNUELS (drift, vol).
        corr                         : corrélation EQ-BD.
        nb_mois                      : nombre de mois à générer.
        nb_sims                      : nombre de simulations.
        rng                          : np.random.Generator (requis pour
                                       reproductibilité).

    Returns:
        tuple : (r_eq, r_bd) de shape (nb_mois, nb_sims) — log-rendements
                mensuels divergents entre sims.
    """
    if rng is None:
        rng = np.random.default_rng()

    dt = 1.0 / 12.0
    if nb_mois <= 0:
        return (
            np.empty((0, nb_sims)),
            np.empty((0, nb_sims)),
        )

    s_e = sigma_e * np.sqrt(dt)
    s_b = sigma_b * np.sqrt(dt)
    cov = np.array([[s_e**2, corr*s_e*s_b], [corr*s_e*s_b, s_b**2]])

    chocs = rng.multivariate_normal([0, 0], cov, size=(nb_mois, nb_sims))
    r_eq = (mu_e*dt - 0.5*s_e**2) + chocs[:, :, 0]
    r_bd = (mu_b*dt - 0.5*s_b**2) + chocs[:, :, 1]
    return r_eq, r_bd


def generer_scenarios_marche_correles(
    mu_e, sigma_e, mu_b, sigma_b, corr_eb,
    vasicek_kappa, vasicek_theta, vasicek_sigma,
    corr_bond_infl,
    nb_mois, nb_sims,
    rng,
    inflation_init=None,
    corr_eq_infl=0.0,
):
    """
    Génère conjointement (r_eq, r_bd, inflation) sur `nb_mois` périodes
    mensuelles, via une matrice de covariance 3×3 cohérente.

    Schéma de discrétisation :
        - EQ et BD : Black-Scholes mensuel (log-rendements)
              r_eq[t] = mu_e*dt - 0.5*s_e^2 + eps_e[t]
              r_bd[t] = mu_b*dt - 0.5*s_b^2 + eps_b[t]
              avec s_e = sigma_e * sqrt(dt) et s_b = sigma_b * sqrt(dt).

        - Inflation : Vasicek discrétisé Euler-Maruyama (taux mensuel)
              I[0]   = inflation_init  (ou theta_m si None)
              I[t]   = I[t-1] + kappa_m * dt * (theta_m - I[t-1]) + eps_i[t-1]
              avec kappa_m, theta_m, sigma_m = annualize_to_monthly(...)
              et s_i = sigma_m * sqrt(dt) (volatilité du pas mensuel).

        - Couplage : (eps_e, eps_b, eps_i) ~ N(0, Sigma) avec
              Sigma = [[ s_e^2,            rho_eb s_e s_b,    rho_ei s_e s_i ],
                       [ rho_eb s_e s_b,   s_b^2,             rho_bi s_b s_i ],
                       [ rho_ei s_e s_i,   rho_bi s_b s_i,    s_i^2          ]]
          où rho_ei (corr equity-inflation) = `corr_eq_infl` (défaut 0)
             et rho_bi = `corr_bond_infl` (typiquement -0.30, cf. settings).
          Cholesky : L = chol(Sigma) ; eps = z @ L.T avec z ~ N(0, I_3).

    Cohérence temporelle : r_bd[t] et l'innovation d'inflation pendant le
    mois t (qui produit I[t+1]) sont tirés à partir du même triplet eps[t,:],
    donc partagent la même structure de corrélation. La dernière innovation
    d'inflation (eps[-1, :, 2]) n'est pas utilisée pour la transition (puisqu'il
    n'y a pas de I[nb_mois]) : c'est un artefact de bord acceptable, sans
    impact statistique pour des horizons longs (300+ mois).

    Args:
        mu_e, sigma_e        : drift et vol ANNUELS equity.
        mu_b, sigma_b        : drift et vol ANNUELS bonds.
        corr_eb              : corrélation EQ-BD (cf. loader).
        vasicek_kappa        : Vasicek κ ANNUEL (vitesse de retour).
        vasicek_theta        : Vasicek θ ANNUEL (cible long terme).
        vasicek_sigma        : Vasicek σ ANNUEL (vol).
        corr_bond_infl       : corrélation bonds ↔ inflation
                               (typiquement -0.30, cf. settings.INFLATION_BONDS_CORRELATION).
        nb_mois              : nombre de mois à générer.
        nb_sims              : nombre de simulations.
        rng                  : np.random.Generator (requis).
        inflation_init       : valeur initiale d'inflation. None → theta_m.
                               Scalaire OU ndarray de shape (nb_sims,) pour
                               assurer la continuité avec une phase backtest.
        corr_eq_infl         : corrélation equity ↔ inflation (défaut 0,
                               hypothèse d'orthogonalité actions/inflation).

    Returns:
        tuple (r_eq, r_bd, inflation) chacun de shape (nb_mois, nb_sims).
    """
    if nb_mois <= 0:
        return (
            np.empty((0, nb_sims)),
            np.empty((0, nb_sims)),
            np.empty((0, nb_sims)),
        )

    dt = 1.0 / 12.0

    # Conversion annuel → mensuel pour Vasicek (mêmes formules que VasicekInflation).
    kappa_m, theta_m, sigma_m_infl = VasicekInflation.annualize_to_monthly(
        vasicek_kappa, vasicek_theta, vasicek_sigma
    )

    # Volatilités du pas mensuel (innovations brownien × sqrt(dt)).
    s_e = sigma_e * np.sqrt(dt)
    s_b = sigma_b * np.sqrt(dt)
    s_i = sigma_m_infl * np.sqrt(dt)

    # Matrice de covariance 3x3 (symétrique).
    cov = np.array([
        [s_e**2,                    corr_eb * s_e * s_b,        corr_eq_infl * s_e * s_i],
        [corr_eb * s_e * s_b,       s_b**2,                     corr_bond_infl * s_b * s_i],
        [corr_eq_infl * s_e * s_i,  corr_bond_infl * s_b * s_i, s_i**2],
    ])

    # Cholesky : L lower triangular, L @ L.T = cov.
    # Si la matrice n'est pas PSD (cas pathologique : corr trop fortes), lève
    # explicitement np.linalg.LinAlgError — on ne corrige pas silencieusement.
    L = np.linalg.cholesky(cov)

    z = rng.standard_normal(size=(nb_mois, nb_sims, 3))
    eps = z @ L.T  # shape (nb_mois, nb_sims, 3)

    # Rendements EQ/BD : drift + innovation corrélée.
    r_eq = (mu_e * dt - 0.5 * s_e**2) + eps[:, :, 0]
    r_bd = (mu_b * dt - 0.5 * s_b**2) + eps[:, :, 1]

    # Inflation Vasicek (Euler-Maruyama) : init + transitions corrélées avec eps_b.
    inflation = np.empty((nb_mois, nb_sims))
    if inflation_init is None:
        init = theta_m
        inflation[0, :] = init
    elif np.isscalar(inflation_init):
        inflation[0, :] = inflation_init
    else:
        init_arr = np.asarray(inflation_init)
        if init_arr.shape != (nb_sims,):
            raise ValueError(
                f"inflation_init array doit être de shape ({nb_sims},), "
                f"reçu {init_arr.shape}"
            )
        inflation[0, :] = init_arr

    drift_factor = kappa_m * dt
    for t in range(1, nb_mois):
        # eps[t-1, :, 2] : innovation d'inflation pendant le mois (t-1) → t,
        # corrélée avec eps[t-1, :, 1] = innovation des bonds pendant le même mois.
        inflation[t, :] = (
            inflation[t-1, :]
            + drift_factor * (theta_m - inflation[t-1, :])
            + eps[t-1, :, 2]
        )

    return r_eq, r_bd, inflation


def generer_inflation_vasicek(nb_periodes, nb_sims, kappa=None, theta=None, sigma=None, rng=None):
    # IMPORTANT : On pré-convertit manuellement les paramètres annuels → mensuels
    # pour contrôler inflation_init (= theta_m, en fréquence mensuelle).
    # Si on laissait frequency="auto", simulate() convertirait kappa/theta/sigma
    # mais PAS inflation_init, créant un saut artificiel au premier pas.
    # frequency="monthly" empêche simulate() de reconvertir une seconde fois.

    """
    Génère des trajectoires d'inflation stochastique via le modèle Vasicek.

    IMPORTANT : Le modèle Vasicek génère des taux périodiques. Les paramètres par défaut
    sont calibrés en **fréquence annuelle**, mais sont automatiquement convertis en
    fréquence mensuelle pour la simulation.

    Formule utilisée pour conversion annuel → mensuel:
    - kappa_monthly = kappa_annual (la vitesse de retour ne change pas)
    - theta_monthly = theta_annual / 12
    - sigma_monthly = sigma_annual / sqrt(12)

    Args:
        nb_periodes : Nombre de périodes (mois)
        nb_sims : Nombre de simulations
        kappa : Vitesse de retour à la moyenne ANNUELLE (si None, calibration par défaut)
        theta : Inflation cible long terme ANNUELLE (si None, calibration par défaut)
        sigma : Volatilité ANNUALISÉE (si None, calibration par défaut)
        rng : np.random.Generator ou None

    Returns:
        np.array : (nb_periodes, nb_sims) - taux d'inflation mensuels
    """
    # Calibration par défaut si nécessaire (PARAMÈTRES ANNUELS)
    if kappa is None or theta is None or sigma is None:
        calib = VasicekInflation.calibration_default()
        kappa = kappa if kappa is not None else calib['kappa']
        theta = theta if theta is not None else calib['theta']
        sigma = sigma if sigma is not None else calib['sigma']

    # Convertir les paramètres annuels en mensuels
    kappa_m, theta_m, sigma_m = VasicekInflation.annualize_to_monthly(kappa, theta, sigma)

    # Créer l'instance Vasicek avec paramètres mensuels
    vasicek = VasicekInflation(
        kappa=kappa_m,
        theta=theta_m,
        sigma=sigma_m,
        inflation_init=theta_m  # Condition initiale en fréquence mensuelle
    )

    # Simuler avec fréquence=monthly pour éviter une double conversion
    return vasicek.simulate(
        nb_periods=nb_periodes,
        nb_scenarios=nb_sims,
        dt=1/12,
        rng=rng,
        frequency="monthly"  # Les paramètres sont déjà en fréquence mensuelle
    )
