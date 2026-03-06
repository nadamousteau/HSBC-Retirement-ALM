"""
Simulation Monte Carlo des courbes GBI via Nelson-Siegel + VAR(1).

Architecture :
    1. Modèle Nelson-Siegel pour la courbe des taux :
       r(tau) = beta0 + beta1 * exp(-lambda * tau) + beta2 * tau * exp(-lambda * tau)

    2. Dynamique des facteurs via processus VAR(1) :
       beta_{t+1} = A @ beta_t + b + Sigma @ epsilon_t
       avec epsilon_t ~ N(0, I_3)

    3. Calibration par défaut sur des valeurs plausibles inspirées de la littérature
       (Diebold & Li, 2006 ; Christensen, Diebold & Rudebusch, 2011).

    4. Sortie : tenseur NumPy (N, T, 360) de taux annualisés pour les maturités
       mensuelles de 1 mois à 30 ans.

Référence :
    Diebold, F.X. & Li, C. (2006). Forecasting the term structure of government
    bond yields. Journal of Econometrics, 130(2), 337-364.
"""

import numpy as np
from config import settings


# ---------------------------------------------------------------------------
# 1. Modèle Nelson-Siegel
# ---------------------------------------------------------------------------

def nelson_siegel(tau, beta0, beta1, beta2, lam):
    """
    Calcule le taux zéro-coupon Nelson-Siegel pour un vecteur de maturités.

    Args:
        tau   : ndarray (M,) — maturités en années (> 0)
        beta0 : float — niveau (long terme)
        beta1 : float — pente (court terme)
        beta2 : float — courbure (moyen terme)
        lam   : float — paramètre de décroissance (> 0)

    Returns:
        ndarray (M,) — taux annualisés pour chaque maturité
    """
    x = lam * tau
    exp_x = np.exp(-x)
    return beta0 + beta1 * exp_x + beta2 * tau * exp_x


def nelson_siegel_vectorized(tau, betas, lam):
    """
    Version vectorisée : calcule les courbes NS pour un lot de facteurs.

    Args:
        tau   : ndarray (M,) — grille de maturités en années
        betas : ndarray (..., 3) — facteurs [beta0, beta1, beta2]
        lam   : float — paramètre de décroissance

    Returns:
        ndarray (..., M) — taux annualisés
    """
    x = lam * tau                           # (M,)
    exp_x = np.exp(-x)                      # (M,)
    b0 = betas[..., 0:1]                    # (..., 1)
    b1 = betas[..., 1:2]                    # (..., 1)
    b2 = betas[..., 2:3]                    # (..., 1)
    return b0 + b1 * exp_x + b2 * tau * exp_x  # (..., M)


# ---------------------------------------------------------------------------
# 2. Calibration par défaut (Diebold-Li plausible)
# ---------------------------------------------------------------------------

def default_calibration():
    """
    Retourne les paramètres de calibration par défaut du modèle VAR(1)
    pour les facteurs Nelson-Siegel du GBI.

    Les valeurs sont inspirées de la littérature empirique sur les taux
    souverains US (Diebold & Li, 2006) et représentent un environnement
    de taux modérés avec retour à la moyenne.

    Returns:
        dict avec les clés :
            beta0_init : float          — valeur initiale du niveau
            beta1_init : float          — valeur initiale de la pente
            beta2_init : float          — valeur initiale de la courbure
            lam        : float          — paramètre de décroissance NS
            A          : ndarray (3, 3) — matrice de transition VAR(1)
            b          : ndarray (3,)   — vecteur d'intercept VAR(1)
            Sigma      : ndarray (3, 3) — matrice de diffusion (Cholesky)
    """
    # Paramètre de décroissance NS (contrôle la maturité du pic de courbure)
    # lam = 0.0609 dans Diebold-Li (2006) correspond à un pic de courbure
    # à tau* ≈ 1/lam ≈ 16.4 ans (pour la composante tau*exp(-lam*tau)).
    # Valeur standard de la littérature pour les taux souverains.
    lam = 0.0609

    # Conditions initiales des facteurs (typiques Diebold-Li, données US)
    # r(tau) = beta0 + beta1 * exp(-lam*tau) + beta2 * lam*tau * exp(-lam*tau)
    # beta0 : niveau asymptotique (taux très long terme)
    # beta1 : facteur de pente (négatif => courbe ascendante, car exp(-lam*tau)
    #          est maximal en tau=0 et décroît vers 0)
    # beta2 : facteur de courbure (positif => bosse dans la courbe)
    beta0_init = 0.045   # niveau long terme ≈ 4.5%
    beta1_init = -0.020  # taux court ≈ beta0+beta1 ≈ 2.5%
    beta2_init = 0.002   # courbure positive modérée

    # Moyennes de long terme (cibles de retour à la moyenne)
    mu_beta0 = 0.045
    mu_beta1 = -0.020
    mu_beta2 = 0.002

    # Matrice de transition VAR(1) — mensuelle
    # Forte persistance (typique : diagonale entre 0.95 et 0.999)
    # beta0 (niveau) : très persistant
    # beta1 (pente)  : persistant
    # beta2 (courbure) : moins persistant
    A = np.array([
        [0.998,  0.000,  0.000],   # beta0 quasi random walk
        [0.000,  0.990,  0.000],   # beta1 retour à la moyenne modéré
        [0.000,  0.000,  0.980],   # beta2 retour à la moyenne plus rapide
    ])

    # Vecteur d'intercept : b = (I - A) @ mu (retour à la moyenne)
    mu = np.array([mu_beta0, mu_beta1, mu_beta2])
    b = (np.eye(3) - A) @ mu

    # Matrice de volatilité (Cholesky de la covariance des innovations)
    # Volatilités mensuelles typiques pour les facteurs NS
    sigma_beta0 = 0.0010   # ≈ 10 bps/mois pour le niveau
    sigma_beta1 = 0.0025   # ≈ 25 bps/mois pour la pente
    sigma_beta2 = 0.0015   # ≈ 15 bps/mois pour la courbure (proportionnel à beta2)

    # Corrélations entre innovations
    rho_01 = -0.30   # niveau-pente : corrélation négative modérée
    rho_02 = 0.10    # niveau-courbure : faible corrélation positive
    rho_12 = 0.20    # pente-courbure : corrélation positive modérée

    # Construction de la matrice de corrélation
    C = np.array([
        [1.0,    rho_01, rho_02],
        [rho_01, 1.0,    rho_12],
        [rho_02, rho_12, 1.0   ],
    ])

    # Matrice de covariance
    D = np.diag([sigma_beta0, sigma_beta1, sigma_beta2])
    Cov = D @ C @ D

    # Décomposition de Cholesky pour la simulation
    Sigma = np.linalg.cholesky(Cov)

    return {
        'beta0_init': beta0_init,
        'beta1_init': beta1_init,
        'beta2_init': beta2_init,
        'lam': lam,
        'A': A,
        'b': b,
        'Sigma': Sigma,
    }


# ---------------------------------------------------------------------------
# 3. Simulation VAR(1) des facteurs Nelson-Siegel
# ---------------------------------------------------------------------------

def simulate_ns_factors(nb_sims, nb_months, calibration=None, rng=None):
    """
    Simule les trajectoires des facteurs Nelson-Siegel via un processus VAR(1).

    Modèle :
        beta_{t+1} = A @ beta_t + b + Sigma @ epsilon_t
        avec epsilon_t ~ N(0, I_3)

    Args:
        nb_sims     : int — nombre de scénarios Monte Carlo (N)
        nb_months   : int — horizon de simulation en mois (T)
        calibration : dict ou None — paramètres du modèle (cf. default_calibration)
        rng         : np.random.Generator ou None — générateur aléatoire

    Returns:
        ndarray (N, T, 3) — trajectoires des facteurs [beta0, beta1, beta2]
    """
    if calibration is None:
        calibration = default_calibration()

    if rng is None:
        rng = np.random.default_rng()

    A = calibration['A']          # (3, 3)
    b_vec = calibration['b']      # (3,)
    Sigma = calibration['Sigma']  # (3, 3)

    beta_init = np.array([
        calibration['beta0_init'],
        calibration['beta1_init'],
        calibration['beta2_init'],
    ])  # (3,)

    # Pré-allocation
    factors = np.zeros((nb_sims, nb_months, 3))

    # Condition initiale identique pour tous les scénarios
    factors[:, 0, :] = beta_init

    # Génération des innovations : (N, T-1, 3)
    eps = rng.standard_normal((nb_sims, nb_months - 1, 3))

    # Simulation pas-à-pas
    for t in range(nb_months - 1):
        # beta_{t+1} = A @ beta_t + b + Sigma @ eps_t
        # Vectorisé sur les N scénarios
        beta_t = factors[:, t, :]                       # (N, 3)
        innovation = eps[:, t, :] @ Sigma.T             # (N, 3)
        factors[:, t + 1, :] = beta_t @ A.T + b_vec + innovation

    return factors


# ---------------------------------------------------------------------------
# 4. Reconstruction des courbes GBI
# ---------------------------------------------------------------------------

def build_gbi_curves(factors, lam=None, nb_maturities=360):
    """
    Reconstruit les courbes de taux GBI à partir des facteurs Nelson-Siegel.

    Grille de maturités : 1 mois à 30 ans (360 points mensuels).

    Args:
        factors       : ndarray (N, T, 3) — facteurs NS simulés
        lam           : float ou None — paramètre de décroissance NS
        nb_maturities : int — nombre de points de la grille (défaut 360)

    Returns:
        ndarray (N, T, 360) — taux annualisés pour chaque (scénario, date, maturité)
    """
    if lam is None:
        lam = default_calibration()['lam']

    # Grille de maturités en années : 1/12, 2/12, ..., 360/12 = 30 ans
    tau = np.arange(1, nb_maturities + 1, dtype=np.float64) / 12.0  # (360,)

    # Calcul vectorisé des courbes NS
    rates = nelson_siegel_vectorized(tau, factors, lam)  # (N, T, 360)

    return rates


# ---------------------------------------------------------------------------
# 5. Fonction principale : simulation Monte Carlo GBI complète
# ---------------------------------------------------------------------------

def simulate_gbi_monte_carlo(nb_sims=None, nb_months=None,
                             calibration=None, seed=None):
    """
    Point d'entrée principal : simule N scénarios de courbes GBI sur T mois
    via le modèle Nelson-Siegel + VAR(1).

    Pipeline :
        1. Calibration des paramètres (défaut ou fournis)
        2. Simulation des facteurs NS par VAR(1)
        3. Reconstruction des courbes GBI sur 360 maturités mensuelles

    Args:
        nb_sims     : int ou None — nombre de scénarios (défaut: settings.NB_SIMULATIONS)
        nb_months   : int ou None — horizon en mois (défaut: settings.NB_PERIODES_TOTAL)
        calibration : dict ou None — paramètres du modèle
        seed        : int ou None — graine pour la reproductibilité

    Returns:
        gbi_tensor  : ndarray (N, T, 360) — taux GBI annualisés
        factors     : ndarray (N, T, 3)   — facteurs NS sous-jacents
        tau_grid    : ndarray (360,)      — maturités en années
    """
    if nb_sims is None:
        nb_sims = settings.NB_SIMULATIONS
    if nb_months is None:
        nb_months = settings.NB_PERIODES_TOTAL

    if calibration is None:
        calibration = get_calibration()

    rng = np.random.default_rng(seed)

    # Étape 1 : simulation des facteurs
    factors = simulate_ns_factors(nb_sims, nb_months,
                                  calibration=calibration, rng=rng)

    # Étape 2 : reconstruction des courbes
    lam = calibration['lam']
    gbi_tensor = build_gbi_curves(factors, lam=lam, nb_maturities=360)

    # Grille de maturités (pour référence)
    tau_grid = np.arange(1, 361, dtype=np.float64) / 12.0

    return gbi_tensor, factors, tau_grid


# ---------------------------------------------------------------------------
# 6. Interface avec le pipeline ALM : extraction du taux proxy pour beta
# ---------------------------------------------------------------------------

def extract_beta_proxy_rates(gbi_tensor, target_maturity_years=10.0):
    """
    Extrait un taux proxy (par ex. 10 ans) pour piloter le GPI/beta
    dans le moteur GBI-CPPI.

    Args:
        gbi_tensor            : ndarray (N, T, 360) — tenseur GBI complet
        target_maturity_years : float — maturité cible en années

    Returns:
        ndarray (T, N) — taux annualisés pour la maturité cible,
                         transposé au format (nb_periods, nb_sims) pour gbi_core.py
    """
    # Index dans la grille : maturité en mois - 1 (0-indexed)
    mat_idx = int(round(target_maturity_years * 12)) - 1
    mat_idx = max(0, min(mat_idx, gbi_tensor.shape[2] - 1))

    # gbi_tensor[n, t, mat_idx] -> (N, T) -> transpose -> (T, N)
    return gbi_tensor[:, :, mat_idx].T


def compute_beta_from_ns(gbi_tensor, retirement_date_offset_months,
                         dec_years=20):
    """
    Calcule la matrice de GPI (betas) directement depuis le tenseur GBI
    Nelson-Siegel, sans passer par YieldCurveBuilder.

    beta(t) = SUM_{k=0}^{dec_years} exp(-r(t, tau_k) * tau_k)
    où tau_k = max(0, t_ret(t)) + k

    Args:
        gbi_tensor                  : ndarray (N, T, 360)
        retirement_date_offset_months : ndarray (T,) — nombre de mois
                                        jusqu'à la retraite à chaque date t
        dec_years                   : int — durée de décumulation

    Returns:
        ndarray (T, N) — matrice de betas au format (nb_periods, nb_sims)
    """
    nb_sims, nb_months, nb_mat = gbi_tensor.shape
    tau_grid = np.arange(1, nb_mat + 1, dtype=np.float64) / 12.0  # (360,)

    beta_matrix = np.zeros((nb_months, nb_sims))

    for t in range(nb_months):
        # Années restantes jusqu'à la retraite
        t_ret = retirement_date_offset_months[t] / 12.0

        if t_ret < 0:
            rem = max(0.0, dec_years + t_ret)
        else:
            rem = float(dec_years)

        if rem <= 0:
            beta_matrix[t, :] = 1.0
            continue

        n_steps = int(np.ceil(rem))
        beta_sum = np.zeros(nb_sims)

        for k in range(n_steps):
            tau_k = max(0.0, t_ret) + k  # maturité en années

            if tau_k < 1.0 / 12.0:
                # Maturité trop courte, utiliser le taux à 1 mois
                tau_idx = 0
            else:
                tau_idx = int(round(tau_k * 12)) - 1
                tau_idx = max(0, min(tau_idx, nb_mat - 1))

            # r(t, tau_k) pour chaque scénario : gbi_tensor[:, t, tau_idx]
            r_k = gbi_tensor[:, t, tau_idx]  # (N,)
            beta_sum += np.exp(-r_k * tau_k)

        beta_matrix[t, :] = np.maximum(beta_sum, 1.0)

    return beta_matrix


# ---------------------------------------------------------------------------
# 7. Accès à la calibration (centralisé)
# ---------------------------------------------------------------------------

def get_calibration():
    """
    Retourne la calibration du modèle NS-VAR(1).

    Utilise les paramètres de settings.py s'ils existent,
    sinon retombe sur la calibration par défaut.

    Returns:
        dict — paramètres de calibration
    """
    custom = getattr(settings, 'GBI_NS_VAR_CALIBRATION', None)
    if custom is not None:
        return custom
    return default_calibration()
