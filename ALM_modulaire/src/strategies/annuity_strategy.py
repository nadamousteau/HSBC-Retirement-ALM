"""
STRATÉGIE DE DÉCUMULATION — RENTE VIAGÈRE (ANNUITY)
====================================================
Conversion du capital d'accumulation en rente viagère mensuelle fixe
(paiement en début de mois, à vie).

Le capital est intégralement versé à l'assureur au moment du départ à
la retraite, en échange d'une mensualité nominale constante. Il n'y a
ni exposition aux marchés, ni risque de ruine (le risque de longévité
est transféré à l'assureur). En contrepartie, le capital n'est plus
disponible : le patrimoine résiduel à tout horizon est zéro.

Le calcul du flux de rente s'appuie sur la table de mortalité
`src.liabilities.mortality` (TH 00-02) et le facteur d'annuité
mensuelle `annuity_factor_monthly`.

Convention :
    SP = M × ä^(12)_x × (1 + loading)
    ⇒ M  = SP / [ä^(12)_x × (1 + loading)]
où SP est le capital initial (single premium), M la mensualité.

La sortie est strictement compatible avec le dict produit par
`run_decumulation()` afin de ne pas perturber le reporting aval.
"""

import numpy as np

from src.liabilities.mortality import annuity_factor_monthly


def run_annuity_decumulation(capital_initial_par_sim, inflation,
                              age_retraite_years, rate_annual, loading,
                              nb_mois, dernier_salaire_mensuel=None):
    """
    Simule la phase de retraite sous contrat de rente viagère.

    Args:
        capital_initial_par_sim : ndarray (nb_sims,) — capital au moment de
                                  la conversion en rente.
        inflation               : ndarray (nb_mois, nb_sims) — taux mensuels
                                  (utilisé pour le facteur d'indexation et
                                  les métriques réelles).
        age_retraite_years      : float — âge au moment de la conversion.
        rate_annual             : float — taux technique annuel.
        loading                 : float — chargement assureur.
        nb_mois                 : int — horizon de simulation (pour le
                                  reporting : la rente est en réalité à vie).
        dernier_salaire_mensuel : float ou None — pour le taux de remplacement.

    Returns:
        dict identique à `run_decumulation()` — mat_capital, mat_retrait_*,
        inflation_factor, mois_ruine, metrics.
    """
    capital_initial_par_sim = np.asarray(capital_initial_par_sim, dtype=np.float64)
    nb_sims = len(capital_initial_par_sim)

    af = annuity_factor_monthly(
        age_retraite_years=age_retraite_years,
        rate_annual=rate_annual,
        loading=loading,
    )
    if af <= 0.0:
        raise ValueError(
            f"Facteur d'annuité mensuel nul (âge={age_retraite_years}, "
            f"rate={rate_annual}) — rente non achetable."
        )

    mensualite_nominale = capital_initial_par_sim / af  # (nb_sims,)

    # ── Flux de paiement : constant nominal sur l'horizon ────────────────
    mat_retrait_total = np.tile(mensualite_nominale, (nb_mois, 1))
    mat_retrait_plancher = mat_retrait_total.copy()
    mat_retrait_variable = np.zeros_like(mat_retrait_total)
    mat_richesse_plancher = np.zeros_like(mat_retrait_total)

    # ── Capital détenu par le retraité : 0 après versement à l'assureur ──
    # mat_capital[0] = capital_initial (avant achat de la rente), puis 0.
    mat_capital = np.zeros((nb_mois + 1, nb_sims))
    mat_capital[0, :] = capital_initial_par_sim

    # ── Facteur d'inflation cumulatif (mêmes conventions que run_decumulation) ──
    inflation_factor = np.ones((nb_mois + 1, nb_sims))
    for k in range(nb_mois):
        inflation_factor[k + 1, :] = inflation_factor[k, :] * (1.0 + inflation[k, :])

    hist_alloc_eq = np.zeros((nb_mois, nb_sims))
    mois_ruine = np.zeros(nb_sims, dtype=int)  # La rente ne tombe jamais en ruine.

    metrics = _compute_annuity_metrics(
        mat_capital, mat_retrait_total, mat_retrait_plancher,
        inflation_factor, mensualite_nominale, nb_mois,
        dernier_salaire_mensuel, nb_sims,
        af=af, loading=loading, rate_annual=rate_annual,
        age_retraite_years=age_retraite_years,
    )

    return {
        "mat_capital": mat_capital,
        "mat_retrait_plancher": mat_retrait_plancher,
        "mat_retrait_variable": mat_retrait_variable,
        "mat_retrait_total": mat_retrait_total,
        "mat_richesse_plancher": mat_richesse_plancher,
        "hist_alloc_eq": hist_alloc_eq,
        "inflation_factor": inflation_factor,
        "mois_ruine": mois_ruine,
        "metrics": metrics,
    }


def _compute_annuity_metrics(mat_capital, mat_retrait_total, mat_retrait_plancher,
                              inflation_factor, mensualite_nominale, nb_mois,
                              dernier_salaire_mensuel, nb_sims,
                              af, loading, rate_annual, age_retraite_years):
    """
    Métriques adaptées à la rente : pas de ruine, patrimoine résiduel nul,
    revenu réel décroissant (rente nominale fixe, inflation positive).
    """
    horizon_annees = nb_mois // 12

    capital_final = mat_capital[-1, :]
    capital_final_reel = capital_final / np.maximum(inflation_factor[-1, :], 1e-9)

    revenu_total_nominal = np.sum(mat_retrait_total, axis=0)
    revenu_total_reel = np.sum(
        mat_retrait_total / np.maximum(inflation_factor[:-1, :], 1e-9), axis=0
    )

    retrait_moyen_nominal = np.mean(mat_retrait_total, axis=0)
    retrait_moyen_reel = np.mean(
        mat_retrait_total / np.maximum(inflation_factor[:-1, :], 1e-9), axis=0
    )

    if dernier_salaire_mensuel is not None and dernier_salaire_mensuel > 0:
        taux_remplacement_p50 = float(np.median(retrait_moyen_nominal / dernier_salaire_mensuel))
        taux_remplacement_reel_p50 = float(np.median(retrait_moyen_reel / dernier_salaire_mensuel))
    else:
        taux_remplacement_p50 = None
        taux_remplacement_reel_p50 = None

    return {
        "proba_ruine": 0.0,
        "nb_ruine": 0,
        "duree_couverture_p5": float(horizon_annees),
        "duree_couverture_p50": float(horizon_annees),
        "duree_couverture_p95": float(horizon_annees),
        "capital_residuel_p5": float(np.percentile(capital_final, 5)),
        "capital_residuel_p50": float(np.percentile(capital_final, 50)),
        "capital_residuel_p95": float(np.percentile(capital_final, 95)),
        "capital_residuel_reel_p50": float(np.percentile(capital_final_reel, 50)),
        "revenu_total_nominal_p50": float(np.median(revenu_total_nominal)),
        "revenu_total_reel_p50": float(np.median(revenu_total_reel)),
        "retrait_mensuel_nominal_p50": float(np.median(retrait_moyen_nominal)),
        "retrait_mensuel_reel_p50": float(np.median(retrait_moyen_reel)),
        "taux_remplacement_p50": taux_remplacement_p50,
        "taux_remplacement_reel_p50": taux_remplacement_reel_p50,
        "part_variable_p50": 0.0,
        "retrait_plancher_cible": float(np.median(mensualite_nominale)),
        # Métriques spécifiques rente
        "annuity_factor": float(af),
        "annuity_loading": float(loading),
        "annuity_rate_annual": float(rate_annual),
        "annuity_age_retraite": float(age_retraite_years),
        "mensualite_nominale_p50": float(np.median(mensualite_nominale)),
    }
