"""
Tests Vague 2 — Tâche B
========================
Vérifie que `generer_scenarios_marche_correles` produit :
    1. Une corrélation empirique (bonds, inflation) ≈ rho_bi cible.
    2. Des distributions marginales EQ/BD/inflation cohérentes avec
       les générateurs indépendants existants (mêmes drifts/volatilités).

Tests numériques tolérants : large nombre de simulations (10 000)
sur horizon long (480 mois) → bornes ±0.02 sur la corrélation,
±10 % sur les volatilités annualisées des marges.

Lancement :
    pytest tests/test_inflation_bonds_corr.py -v

Ces tests sont indépendants du pipeline (pas d'import settings.py)
et utilisent une seed locale fixée pour reproductibilité.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Permet de lancer pytest depuis ALM_modulaire/ ou la racine du repo.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.economics.generators import (
    generer_scenarios_marche_correles,
    generer_inflation_vasicek,
    generer_rendements_forecast,
)


# Paramètres de marché US calibrés (cohérents avec settings.py).
MU_E, SIGMA_E = 0.07, 0.16
MU_B, SIGMA_B = 0.03, 0.06
CORR_EB = 0.0
KAPPA, THETA, SIGMA_INFL = 0.15, 0.023, 0.011


def test_correlation_empirique_forte():
    """
    Avec rho_bi = -0.30, la corrélation empirique entre les innovations
    bonds et inflation doit converger vers -0.30 ± 0.02 sur 10 000 sims
    × 480 mois (4.8 M de paires).
    """
    rng = np.random.default_rng(123)
    nb_mois, nb_sims = 480, 10_000
    rho_bi = -0.30

    r_eq, r_bd, inflation = generer_scenarios_marche_correles(
        MU_E, SIGMA_E, MU_B, SIGMA_B, CORR_EB,
        KAPPA, THETA, SIGMA_INFL,
        rho_bi,
        nb_mois, nb_sims,
        rng=rng,
    )

    # On compare l'innovation des bonds (r_bd - drift_bd) à l'innovation
    # d'inflation (inflation[t+1] - inflation[t] - drift_vasicek).
    dt = 1.0 / 12.0
    s_b = SIGMA_B * np.sqrt(dt)
    drift_bd = MU_B * dt - 0.5 * s_b**2
    eps_b = r_bd - drift_bd  # (nb_mois, nb_sims)

    # Reconstitue les innovations Vasicek à partir des transitions de inflation.
    kappa_m = KAPPA
    theta_m = THETA / 12.0
    drift_factor = kappa_m * dt
    # eps_i[t-1] = inflation[t] - inflation[t-1] - drift_factor*(theta_m - inflation[t-1])
    eps_i = np.empty((nb_mois - 1, nb_sims))
    for t in range(1, nb_mois):
        eps_i[t-1, :] = (
            inflation[t, :]
            - inflation[t-1, :]
            - drift_factor * (theta_m - inflation[t-1, :])
        )

    # Aligne : eps_b[t-1] doit être corrélé avec eps_i[t-1] (même triplet
    # eps[t-1, :] dans le générateur Cholesky).
    eps_b_aligned = eps_b[:nb_mois - 1, :].ravel()
    eps_i_flat = eps_i.ravel()

    corr_emp = np.corrcoef(eps_b_aligned, eps_i_flat)[0, 1]
    assert abs(corr_emp - rho_bi) < 0.02, (
        f"Corrélation empirique bonds-inflation = {corr_emp:.4f}, "
        f"attendue = {rho_bi:.4f} ± 0.02"
    )


def test_distributions_marginales():
    """
    Les marginales EQ/BD/inflation du générateur joint doivent avoir des
    moments (moyenne, vol annualisée) cohérents avec un générateur
    indépendant (à 10 % près sur la vol).
    """
    rng_joint = np.random.default_rng(456)
    rng_indep_eb = np.random.default_rng(789)
    rng_indep_infl = np.random.default_rng(101)

    nb_mois, nb_sims = 480, 5_000
    rho_bi = -0.30

    r_eq_j, r_bd_j, infl_j = generer_scenarios_marche_correles(
        MU_E, SIGMA_E, MU_B, SIGMA_B, CORR_EB,
        KAPPA, THETA, SIGMA_INFL,
        rho_bi,
        nb_mois, nb_sims,
        rng=rng_joint,
    )

    # Référence indépendante : forecast EQ/BD seul + Vasicek seul.
    r_eq_i, r_bd_i = generer_rendements_forecast(
        MU_E, SIGMA_E, MU_B, SIGMA_B, CORR_EB,
        nb_mois, nb_sims, rng=rng_indep_eb,
    )
    infl_i = generer_inflation_vasicek(
        nb_periodes=nb_mois, nb_sims=nb_sims,
        kappa=KAPPA, theta=THETA, sigma=SIGMA_INFL,
        rng=rng_indep_infl,
    )

    # Volatilité annualisée des log-rendements EQ.
    vol_eq_joint = np.std(r_eq_j) * np.sqrt(12)
    vol_eq_indep = np.std(r_eq_i) * np.sqrt(12)
    assert abs(vol_eq_joint - vol_eq_indep) / vol_eq_indep < 0.10, (
        f"Vol EQ jointe = {vol_eq_joint:.4f}, indep = {vol_eq_indep:.4f}"
    )

    # Volatilité annualisée bonds.
    vol_bd_joint = np.std(r_bd_j) * np.sqrt(12)
    vol_bd_indep = np.std(r_bd_i) * np.sqrt(12)
    assert abs(vol_bd_joint - vol_bd_indep) / vol_bd_indep < 0.10, (
        f"Vol BD jointe = {vol_bd_joint:.4f}, indep = {vol_bd_indep:.4f}"
    )

    # Niveau moyen d'inflation (mensuel).
    mean_infl_joint = np.mean(infl_j)
    mean_infl_indep = np.mean(infl_i)
    # Théorique : theta_m ≈ 0.023/12 ≈ 0.00192. Tolérance large car
    # process Vasicek convergent.
    assert abs(mean_infl_joint - mean_infl_indep) / abs(mean_infl_indep) < 0.20, (
        f"Mean infl jointe = {mean_infl_joint:.6f}, indep = {mean_infl_indep:.6f}"
    )


def test_correlation_nulle_par_defaut():
    """
    Si on appelle generer_scenarios_marche_correles avec rho_bi=0,
    on doit retomber sur une corrélation empirique ≈ 0 (sanity check
    de la matrice de covariance).
    """
    rng = np.random.default_rng(999)
    nb_mois, nb_sims = 240, 5_000

    _, r_bd, inflation = generer_scenarios_marche_correles(
        MU_E, SIGMA_E, MU_B, SIGMA_B, CORR_EB,
        KAPPA, THETA, SIGMA_INFL,
        corr_bond_infl=0.0,
        nb_mois=nb_mois, nb_sims=nb_sims,
        rng=rng,
    )

    dt = 1.0 / 12.0
    s_b = SIGMA_B * np.sqrt(dt)
    drift_bd = MU_B * dt - 0.5 * s_b**2
    eps_b = r_bd - drift_bd

    kappa_m, theta_m = KAPPA, THETA / 12.0
    drift_factor = kappa_m * dt
    eps_i = np.empty((nb_mois - 1, nb_sims))
    for t in range(1, nb_mois):
        eps_i[t-1, :] = (
            inflation[t, :]
            - inflation[t-1, :]
            - drift_factor * (theta_m - inflation[t-1, :])
        )

    corr_emp = np.corrcoef(
        eps_b[:nb_mois - 1, :].ravel(),
        eps_i.ravel()
    )[0, 1]

    assert abs(corr_emp) < 0.02, (
        f"Avec rho_bi=0, corrélation empirique = {corr_emp:.4f} (devrait être ≈ 0)"
    )


def test_continuite_inflation_init_per_sim():
    """
    L'argument `inflation_init` accepte un vecteur per-sim et la chaîne
    Vasicek doit démarrer exactement à ces valeurs pour assurer la
    continuité backtest → forecast.
    """
    rng = np.random.default_rng(77)
    nb_mois, nb_sims = 60, 100
    inflation_init = np.linspace(0.001, 0.005, nb_sims)  # valeurs per-sim distinctes

    _, _, inflation = generer_scenarios_marche_correles(
        MU_E, SIGMA_E, MU_B, SIGMA_B, CORR_EB,
        KAPPA, THETA, SIGMA_INFL,
        corr_bond_infl=-0.30,
        nb_mois=nb_mois, nb_sims=nb_sims,
        rng=rng,
        inflation_init=inflation_init,
    )

    np.testing.assert_allclose(inflation[0, :], inflation_init, rtol=1e-12)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
