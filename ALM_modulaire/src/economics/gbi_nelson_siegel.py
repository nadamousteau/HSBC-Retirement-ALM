"""
Simulation Monte Carlo de la courbe GBI via Nelson-Siegel + VAR(1).

Architecture :
    1. Courbe instantanee :  r(tau) = beta0 + beta1 * exp(-lambda*tau)
                                     + beta2 * tau * exp(-lambda*tau)

    2. Dynamique factorielle :  beta_{t+1} = A @ beta_t + b + Sigma @ eps_t
       avec eps_t ~ N(0, I_3).

    3. Monte Carlo :  N scenarios, T mois, grille de 360 maturites (1M .. 30Y).
       Sortie : tenseur (N, T, 360) de taux annualises.

Calibration :
    - Si un historique de parametres NS est fourni, A, b et Sigma sont estimes
      par regression lineaire (moindres carres ordinaires).
    - Sinon, des valeurs plausibles calibrees sur la litterature (Diebold & Li 2006)
      sont utilisees par defaut.

Reference :
    Diebold, F.X. & Li, C. (2006). "Forecasting the Term Structure of Government
    Bond Yields". Journal of Econometrics, 130(2), 337-364.
"""

import numpy as np
from numpy.linalg import cholesky


# ─────────────────────────────────────────────────────────────────────────────
# 1. Modele Nelson-Siegel (evaluation de la courbe)
# ─────────────────────────────────────────────────────────────────────────────

def nelson_siegel(tau, beta0, beta1, beta2, lam):
    """
    Evalue la courbe de taux Nelson-Siegel pour un vecteur de maturites.

    Args:
        tau   : np.ndarray (M,) - maturites en annees (> 0)
        beta0 : float - facteur de niveau (long terme)
        beta1 : float - facteur de pente (court terme)
        beta2 : float - facteur de courbure (moyen terme)
        lam   : float - parametre de decroissance (> 0)

    Returns:
        np.ndarray (M,) - taux annualises pour chaque maturite
    """
    exp_lt = np.exp(-lam * tau)
    return beta0 + beta1 * exp_lt + beta2 * tau * exp_lt


def nelson_siegel_matrix(tau, betas, lam):
    """
    Evalue Nelson-Siegel pour une matrice de facteurs.

    Args:
        tau   : np.ndarray (M,)        - grille de maturites
        betas : np.ndarray (K, 3)      - K jeux de facteurs (beta0, beta1, beta2)
        lam   : float                  - parametre de decroissance

    Returns:
        np.ndarray (K, M) - taux pour chaque jeu de facteurs et maturite
    """
    exp_lt = np.exp(-lam * tau)                    # (M,)
    tau_exp = tau * exp_lt                          # (M,)
    # betas[:, 0] -> (K,1), exp_lt -> (1,M)  =>  broadcast (K, M)
    return (betas[:, 0:1]
            + betas[:, 1:2] * exp_lt[np.newaxis, :]
            + betas[:, 2:3] * tau_exp[np.newaxis, :])


# ─────────────────────────────────────────────────────────────────────────────
# 2. Calibration VAR(1) sur un historique de facteurs NS
# ─────────────────────────────────────────────────────────────────────────────

def calibrate_var1(beta_history):
    """
    Estime les parametres VAR(1) a partir d'un historique mensuel de facteurs NS.

        beta_{t+1} = A @ beta_t + b + Sigma @ eps_t

    Args:
        beta_history : np.ndarray (T, 3) - historique mensuel de (beta0, beta1, beta2)

    Returns:
        A     : np.ndarray (3, 3) - matrice de transition
        b     : np.ndarray (3,)   - vecteur d'intercept
        Sigma : np.ndarray (3, 3) - matrice de Cholesky de la covariance des residus
    """
    Y = beta_history[1:]     # (T-1, 3)
    X = beta_history[:-1]    # (T-1, 3)

    T_obs = X.shape[0]

    # Ajout d'une colonne de 1 pour l'intercept : X_aug = [X, 1]
    X_aug = np.column_stack([X, np.ones(T_obs)])   # (T-1, 4)

    # OLS : [A | b]' = (X_aug' X_aug)^{-1} X_aug' Y
    coeffs = np.linalg.lstsq(X_aug, Y, rcond=None)[0]  # (4, 3)

    A = coeffs[:3, :].T       # (3, 3)
    b = coeffs[3, :]           # (3,)

    # Residus et matrice de covariance
    residuals = Y - X_aug @ coeffs   # (T-1, 3)
    cov_resid = np.cov(residuals, rowvar=False)

    # Regularisation pour garantir la semi-definie positivite
    cov_resid += np.eye(3) * 1e-10
    Sigma = cholesky(cov_resid)

    return A, b, Sigma


# ─────────────────────────────────────────────────────────────────────────────
# 3. Parametres par defaut (Diebold-Li 2006, adaptes)
# ─────────────────────────────────────────────────────────────────────────────

def default_var1_params():
    """
    Retourne des parametres VAR(1) plausibles pour la dynamique des facteurs
    Nelson-Siegel, calibres sur la litterature (Diebold & Li 2006).

    Niveau initial :
        beta0 ~ 4.5% (taux long terme)
        beta1 ~ -2.0% (pente : court - long)
        beta2 ~ 1.0%  (courbure)

    La matrice A est quasi-diagonale avec forte persistence (~0.99/mois).

    Returns:
        A, b, Sigma, beta0_init, lam
    """
    # Matrice de transition : forte persistance, faible couplage
    A = np.array([
        [0.995,  0.002,  0.000],
        [0.000,  0.990, -0.001],
        [0.000,  0.005,  0.985]
    ])

    # Moyenne inconditionnelle : mu = (I - A)^{-1} b
    # On vise mu = [0.045, -0.020, 0.010]
    mu = np.array([0.045, -0.020, 0.010])
    b = (np.eye(3) - A) @ mu

    # Volatilite mensuelle des facteurs (ecart-type ~ 15-30 bps/mois)
    sigma_diag = np.array([0.0015, 0.0025, 0.0030])
    corr = np.array([
        [1.00, -0.30,  0.10],
        [-0.30, 1.00, -0.20],
        [0.10, -0.20,  1.00]
    ])
    cov = np.outer(sigma_diag, sigma_diag) * corr
    cov += np.eye(3) * 1e-10
    Sigma = cholesky(cov)

    # Facteurs initiaux
    beta0_init = mu.copy()

    # Lambda (Diebold-Li : lambda optimal ~ 0.0609 pour maximiser la courbure a ~30 mois)
    lam = 0.0609

    return A, b, Sigma, beta0_init, lam


# ─────────────────────────────────────────────────────────────────────────────
# 4. Simulation Monte Carlo
# ─────────────────────────────────────────────────────────────────────────────

def simulate_gbi_curves(nb_sims, nb_months, A=None, b=None, Sigma=None,
                        beta0_init=None, lam=None, seed=None):
    """
    Simule N trajectoires de courbes GBI via Nelson-Siegel + VAR(1).

    A chaque pas mensuel t :
        1. Tire eps_t ~ N(0, I_3)
        2. Propage beta_{t+1} = A @ beta_t + b + Sigma @ eps_t
        3. Reconstruit la courbe r(tau) sur 360 maturites (1M .. 30Y)

    Args:
        nb_sims    : int   - nombre de scenarios Monte Carlo (N)
        nb_months  : int   - nombre de pas mensuels (T)
        A          : (3,3) - matrice de transition VAR(1)
        b          : (3,)  - intercept VAR(1)
        Sigma      : (3,3) - Cholesky de la covariance des innovations
        beta0_init : (3,)  - facteurs NS initiaux (beta0, beta1, beta2)
        lam        : float - parametre de decroissance NS
        seed       : int   - graine aleatoire (reproductibilite)

    Returns:
        curves : np.ndarray (N, T, 360) - taux annualises sur la grille de maturites
        betas  : np.ndarray (N, T, 3)   - trajectoires des facteurs NS
        tau    : np.ndarray (360,)       - grille de maturites en annees
    """
    # Parametres par defaut si non fournis
    if A is None or b is None or Sigma is None or beta0_init is None or lam is None:
        A_def, b_def, Sigma_def, beta0_def, lam_def = default_var1_params()
        A = A if A is not None else A_def
        b = b if b is not None else b_def
        Sigma = Sigma if Sigma is not None else Sigma_def
        beta0_init = beta0_init if beta0_init is not None else beta0_def
        lam = lam if lam is not None else lam_def

    rng = np.random.default_rng(seed)

    # Grille de maturites : 1 mois a 30 ans (360 points)
    tau = np.arange(1, 361) / 12.0   # (360,) en annees

    # Allocation des tenseurs
    betas = np.zeros((nb_sims, nb_months, 3))
    betas[:, 0, :] = beta0_init[np.newaxis, :]

    # Tirage de toutes les innovations en une fois : (N, T-1, 3)
    eps = rng.standard_normal((nb_sims, nb_months - 1, 3))

    # Propagation VAR(1)
    for t in range(nb_months - 1):
        # betas[:, t, :] est (N, 3)
        # A @ beta_t : on transpose pour multiplier par A (3,3)
        betas[:, t + 1, :] = (betas[:, t, :] @ A.T
                              + b[np.newaxis, :]
                              + eps[:, t, :] @ Sigma.T)

    # Reconstruction des courbes pour tous les scenarios et dates
    # betas reshaped (N*T, 3), calcul vectorise, reshape en (N, T, 360)
    betas_flat = betas.reshape(-1, 3)
    curves_flat = nelson_siegel_matrix(tau, betas_flat, lam)
    curves = curves_flat.reshape(nb_sims, nb_months, 360)

    return curves, betas, tau


# ─────────────────────────────────────────────────────────────────────────────
# 5. Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def extract_rate_at_maturity(curves, tau, target_maturity_years):
    """
    Extrait les taux a une maturite donnee depuis le tenseur de courbes.

    Args:
        curves              : np.ndarray (N, T, 360)
        tau                 : np.ndarray (360,)
        target_maturity_years : float - maturite cible en annees

    Returns:
        np.ndarray (N, T) - taux extraits
    """
    idx = np.argmin(np.abs(tau - target_maturity_years))
    return curves[:, :, idx]


def curves_to_discount_factors(curves, tau):
    """
    Convertit les taux annualises en facteurs d'actualisation.

    Args:
        curves : np.ndarray (N, T, 360) - taux annualises
        tau    : np.ndarray (360,)       - maturites en annees

    Returns:
        np.ndarray (N, T, 360) - facteurs d'actualisation exp(-r * tau)
    """
    return np.exp(-curves * tau[np.newaxis, np.newaxis, :])
