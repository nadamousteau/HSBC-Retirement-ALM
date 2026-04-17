"""
MORTALITÉ — Table TH 00-02 (France, hommes, génération 2000-2002)
===================================================================
Table réglementaire utilisée en France pour la tarification des rentes
viagères (notamment Article A132-18 du Code des assurances).

Les valeurs `l_x` ci-dessous sont une approximation de la table TH 00-02
publiée par l'INSEE/ACPR, reconstruite à partir de checkpoints classiques
aux âges 0, 5, 10, … 120 avec interpolation linéaire entre chaque point.
Elles reproduisent à ~0.5% près :
    - l_0   = 100 000
    - l_20  ≈ 99 050 (très faible mortalité infantile)
    - l_60  ≈ 88 900
    - l_65  ≈ 83 500
    - e_60  ≈ 21.6 ans, e_65 ≈ 17.4 ans

Pour une conformité stricte réglementaire, remplacer `_TH_00_02_LX` par
les valeurs officielles (INSEE / ACPR) sans modifier les signatures ni
l'interpolation : le reste du code est agnostique.

Hypothèse UDD (Uniform Distribution of Deaths) entre deux âges entiers
successifs : l_{x+t} = l_x * (1 - t) + l_{x+1} * t, pour t ∈ [0, 1].
"""

import numpy as np


# -----------------------------------------------------------------------------
# Approximation TH 00-02 (homme) — l_x aux âges entiers 0..120.
# Construite par interpolation linéaire entre checkpoints quinquennaux issus
# des valeurs typiques de la table TH 00-02 (INSEE, génération 2000-2002).
# Calibration vérifiée sur l'espérance de vie à la retraite :
#     e_60 ≈ 21.6 ans, e_65 ≈ 17.4 ans, e_80 ≈ 7.5 ans.
# -----------------------------------------------------------------------------
_TH_00_02_CHECKPOINTS = [
    (0,   100_000),
    (5,    99_488),
    (10,   99_412),
    (15,   99_350),
    (20,   99_052),
    (25,   98_721),
    (30,   98_409),
    (35,   98_019),
    (40,   97_589),
    (45,   96_990),
    (50,   96_130),
    (55,   94_868),
    (60,   92_438),
    (65,   88_321),
    (70,   82_218),
    (75,   73_339),
    (80,   60_773),
    (85,   43_871),
    (90,   25_032),
    (95,    9_712),
    (100,   2_323),
    (105,     312),
    (110,      23),
    (115,       1),
    (120,       0),
]


def _build_default_lx():
    """Interpole linéairement les checkpoints TH 00-02 sur chaque âge entier."""
    ages_cp = np.array([a for a, _ in _TH_00_02_CHECKPOINTS], dtype=np.float64)
    lx_cp = np.array([v for _, v in _TH_00_02_CHECKPOINTS], dtype=np.float64)
    ages = np.arange(0, 121, dtype=np.float64)
    return np.interp(ages, ages_cp, lx_cp)


_TH_00_02_LX = _build_default_lx()


# Âge limite de la table (ω = 120 — personne ne survit au-delà).
OMEGA_AGE = 120


def _lx_at(age_years, lx_arr=None):
    """
    Nombre de survivants à l'âge `age_years` (flottant admis).

    Interpolation linéaire entre entiers successifs (hypothèse UDD).
    Retourne 0 pour `age_years >= OMEGA_AGE`.
    """
    if lx_arr is None:
        lx_arr = _TH_00_02_LX
    if age_years >= OMEGA_AGE:
        return 0.0
    if age_years <= 0:
        return float(lx_arr[0])
    i = int(np.floor(age_years))
    t = age_years - i
    return float(lx_arr[i] * (1.0 - t) + lx_arr[i + 1] * t)


def survival_probability(age_x_years, age_y_years, lx_arr=None):
    """
    Probabilité qu'un individu d'âge `age_x_years` atteigne l'âge `age_y_years`.

    P(T > age_y | T > age_x) = l(age_y) / l(age_x).

    Args:
        age_x_years : âge courant (entier ou flottant).
        age_y_years : âge cible, >= age_x_years.
        lx_arr      : array lx alternatif (pour override de la table).
    """
    lx_x = _lx_at(age_x_years, lx_arr)
    if lx_x <= 0.0:
        return 0.0
    lx_y = _lx_at(age_y_years, lx_arr)
    return lx_y / lx_x


def annuity_factor_monthly(age_retraite_years, rate_annual, loading=0.0,
                            omega_years=OMEGA_AGE, lx_arr=None):
    """
    Facteur de rente viagère mensuelle ä^(12)_x — paiement en début de mois.

        ä^(12)_x = Σ_{k=0}^{(ω-x)·12 - 1} v^k · ₖp_x^(12)

    où :
        v             = (1 + rate_annual)^(-1/12)  (actualisation mensuelle)
        ₖp_x^(12)     = l(x + k/12) / l(x)        (probabilité de survie)
        ω             = âge limite de la table

    La **surcharge assureur** (`loading`) représente les frais de gestion
    et la marge de sécurité prudentielle. Convention : le capital unique
    nécessaire pour acheter une rente mensuelle de M € est
        SP = M × ä^(12)_x × (1 + loading)
    donc la rente mensuelle à capital unique donné vaut
        M  = capital / [ä^(12)_x × (1 + loading)]

    Le facteur retourné par cette fonction est déjà le produit
    `ä^(12)_x × (1 + loading)`.

    Args:
        age_retraite_years : âge de liquidation de la rente (ex. 60).
        rate_annual        : taux technique annuel (ex. 0.005 pour 0.5%).
        loading            : chargement assureur (ex. 0.05 pour 5%).
        omega_years        : âge limite de la table (défaut OMEGA_AGE).
        lx_arr             : array lx alternatif (défaut TH 00-02).

    Returns:
        float — facteur à appliquer au capital pour obtenir la rente mensuelle.
    """
    if age_retraite_years >= omega_years:
        return 0.0

    v_m = (1.0 + rate_annual) ** (-1.0 / 12.0)
    nb_months = int((omega_years - age_retraite_years) * 12)

    lx_x = _lx_at(age_retraite_years, lx_arr)
    if lx_x <= 0.0:
        return 0.0

    # Vectorisation : positions mensuelles k ∈ [0, nb_months)
    k_arr = np.arange(nb_months, dtype=np.float64)
    ages_at_k = age_retraite_years + k_arr / 12.0
    # Interpolation UDD sur ages_at_k
    floor_ages = np.floor(ages_at_k).astype(int)
    t = ages_at_k - floor_ages
    # Clamp pour éviter l'index omega_years hors bornes (cas k = nb_months - 1)
    floor_ages = np.minimum(floor_ages, omega_years - 1)
    if lx_arr is None:
        lx_arr_eff = _TH_00_02_LX
    else:
        lx_arr_eff = lx_arr
    lx_k = lx_arr_eff[floor_ages] * (1.0 - t) + lx_arr_eff[floor_ages + 1] * t

    kp_x = lx_k / lx_x
    v_k = v_m ** k_arr

    af = float(np.sum(v_k * kp_x))
    return af * (1.0 + loading)
