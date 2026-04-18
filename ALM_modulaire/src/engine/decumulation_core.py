"""
MOTEUR DE DÉCUMULATION — Retrait Plancher + Variable (CPPI inversé)
====================================================================

Emplacement : src/engine/decumulation_core.py
Ajouter dans src/engine/__init__.py :
    from .decumulation_core import run_decumulation, print_decumulation_report

Architecture symétrique à core.py :
    - Paramètres lus depuis config.settings (pas d'arguments ad hoc)
    - Stratégie d'investissement réutilisée (même interface que l'accumulation)
    - Inflation stochastique propagée depuis le main

Stratégie de retrait Plancher + Variable :
    1. Plancher = retrait minimum indexé à l'inflation (besoins essentiels)
    2. Richesse plancher = VA des retraits plancher futurs (annuité certaine)
    3. Surplus = max(Capital - Richesse plancher, 0)
    4. Variable = taux_distribution × surplus / 12
    5. Si capital < plancher → retrait = min(capital, plancher), ruine imminente
"""

import numpy as np
from config import settings


# =============================================================================
# MOTEUR PRINCIPAL
# =============================================================================

def run_decumulation(strategy, r_eq, r_bd, capital_initial_par_sim,
                     inflation, liability, dernier_salaire_mensuel=None):
    """
    Exécute la simulation Monte Carlo de décumulation.

    Le passif client `liability` (RetirementLiability) porte désormais le
    revenu mensuel cible et le taux d'actualisation du plancher. Les
    paramètres restants (horizon, distribution surplus) sont toujours
    lus depuis settings.py.

    Vague 2 / Tâche C — l'horizon de simulation reste piloté par
    `settings.NB_ANNEES_DECUMULATION` ; `validate_settings()` vérifie
    qu'il couvre `liability.expected_duration_years() + marge`.

    Args:
        strategy                : Objet stratégie (get_allocation, should_rebalance)
                                  Construit dans main.py selon STRATEGIE_DECUMULATION
        r_eq                    : ndarray (nb_steps, nb_sims) — rendements actions
        r_bd                    : ndarray (nb_steps, nb_sims) — rendements obligations
        capital_initial_par_sim : ndarray (nb_sims,) — capital de départ par scénario
                                  (= capital final de l'accumulation, varie par sim)
        inflation               : ndarray (nb_steps, nb_sims) — taux d'inflation mensuels
        liability               : RetirementLiability — passif client.
                                  Fournit `target_income_monthly`,
                                  `income_mode`, `discount_rate`.
        dernier_salaire_mensuel : float ou None — pour le taux de remplacement

    Returns:
        dict avec matrices de simulation et métriques
    """
    # ── Lecture des paramètres depuis settings + liability ────────────────
    horizon_annees = settings.NB_ANNEES_DECUMULATION
    taux_distribution = settings.TAUX_DISTRIBUTION_SURPLUS

    # Retrait plancher : monetisé au moment de la liquidation (NOMINAL).
    # En mode REAL, on capitalise par (1+inflation_expected)^accum ; en
    # mode NOMINAL, on prend directement la cible saisie.
    retrait_mensuel_plancher = liability.monthly_income_nominal_at_retirement()
    taux_actu_plancher = liability.discount_rate

    nb_steps = horizon_annees * 12
    nb_sims = len(capital_initial_par_sim)

    assert r_eq.shape[0] >= nb_steps, (
        f"Rendements trop courts : {r_eq.shape[0]} mois fournis, {nb_steps} nécessaires"
    )

    # Taux mensuel d'actualisation pour la richesse plancher
    r_plancher_m = (1 + taux_actu_plancher) ** (1 / 12) - 1

    # =========================================================================
    # INITIALISATION
    # =========================================================================
    mat_capital = np.zeros((nb_steps + 1, nb_sims))
    mat_capital[0, :] = capital_initial_par_sim

    mat_retrait_plancher = np.zeros((nb_steps, nb_sims))
    mat_retrait_variable = np.zeros((nb_steps, nb_sims))
    mat_retrait_total = np.zeros((nb_steps, nb_sims))
    mat_richesse_plancher = np.zeros((nb_steps, nb_sims))
    hist_alloc_eq = np.zeros((nb_steps, nb_sims))

    # Facteur cumulatif d'inflation (indexation des retraits)
    inflation_factor = np.ones((nb_steps + 1, nb_sims))

    # Mois de ruine par scénario (0 = jamais)
    mois_ruine = np.zeros(nb_sims, dtype=int)

    # =========================================================================
    # BOUCLE TEMPORELLE
    # =========================================================================
    for k in range(nb_steps):
        t_annees = k / 12.0
        age_actuel = settings.AGE_DEPART + settings.NB_ANNEES_ACCUMULATION + t_annees
        mois_restants = nb_steps - k

        W = mat_capital[k, :].copy()

        # ── Facteur d'inflation cumulatif ────────────────────────────────
        inflation_factor[k + 1, :] = inflation_factor[k, :] * (1 + inflation[k, :])

        # ── 1. Retrait plancher indexé ───────────────────────────────────
        retrait_plancher = retrait_mensuel_plancher * inflation_factor[k, :]
        retrait_plancher = np.minimum(retrait_plancher, np.maximum(W, 0.0))

        # ── 2. Richesse plancher (VA des retraits futurs) ────────────────
        if r_plancher_m > 1e-10:
            annuite = (1 - (1 + r_plancher_m) ** (-mois_restants)) / r_plancher_m
        else:
            annuite = float(mois_restants)
        richesse_plancher = retrait_plancher * annuite

        # ── 3. Surplus et retrait variable ───────────────────────────────
        W_apres_plancher = W - retrait_plancher
        surplus = np.maximum(W_apres_plancher - richesse_plancher, 0.0)
        retrait_variable = taux_distribution * surplus / 12.0
        retrait_variable = np.minimum(retrait_variable, np.maximum(W_apres_plancher, 0.0))

        # ── 4. Retrait total et capital post-retrait ─────────────────────
        retrait_total = retrait_plancher + retrait_variable
        W_apres_retrait = np.maximum(W - retrait_total, 0.0)

        # ── Sauvegarde ───────────────────────────────────────────────────
        mat_retrait_plancher[k, :] = retrait_plancher
        mat_retrait_variable[k, :] = retrait_variable
        mat_retrait_total[k, :] = retrait_total
        mat_richesse_plancher[k, :] = richesse_plancher

        # ── 5. Investissement du capital restant ─────────────────────────
        pct_eq, pct_bd = strategy.get_allocation(k, int(age_actuel))
        hist_alloc_eq[k, :] = pct_eq

        r_port = pct_eq * r_eq[k, :] + pct_bd * r_bd[k, :]
        W_fin = W_apres_retrait * (1 + r_port)
        W_fin = np.maximum(W_fin, 0.0)

        mat_capital[k + 1, :] = W_fin

        # ── 6. Détection de la ruine ─────────────────────────────────────
        newly_ruined = (W_fin <= 0.0) & (mois_ruine == 0)
        mois_ruine[newly_ruined] = k + 1

    # =========================================================================
    # MÉTRIQUES
    # =========================================================================
    metrics = _compute_metrics(
        mat_capital, mat_retrait_total, mat_retrait_plancher,
        mat_retrait_variable, inflation_factor, mois_ruine,
        horizon_annees, retrait_mensuel_plancher, dernier_salaire_mensuel, nb_sims,
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


# =============================================================================
# MÉTRIQUES
# =============================================================================

def _compute_metrics(mat_capital, mat_retrait_total, mat_retrait_plancher,
                     mat_retrait_variable, inflation_factor, mois_ruine,
                     horizon_annees, retrait_mensuel_plancher,
                     dernier_salaire_mensuel, nb_sims):
    """Calcule toutes les métriques de décumulation."""
    nb_steps = horizon_annees * 12

    # ── Ruine ────────────────────────────────────────────────────────────
    nb_ruine = int(np.sum(mois_ruine > 0))
    proba_ruine = nb_ruine / nb_sims

    # ── Durée de couverture ──────────────────────────────────────────────
    duree_couverture = np.where(mois_ruine > 0, mois_ruine / 12.0, float(horizon_annees))

    # ── Patrimoine résiduel ──────────────────────────────────────────────
    capital_final = mat_capital[-1, :]
    capital_final_reel = capital_final / np.maximum(inflation_factor[-1, :], 1e-9)

    # ── Revenu total distribué ───────────────────────────────────────────
    revenu_total_nominal = np.sum(mat_retrait_total, axis=0)
    revenu_total_reel = np.sum(
        mat_retrait_total / np.maximum(inflation_factor[:-1, :], 1e-9), axis=0
    )

    # ── Retrait mensuel moyen (avant ruine) ──────────────────────────────
    retrait_moyen_nominal = np.zeros(nb_sims)
    retrait_moyen_reel = np.zeros(nb_sims)
    for s in range(nb_sims):
        dur = mois_ruine[s] if mois_ruine[s] > 0 else nb_steps
        if dur > 0:
            retrait_moyen_nominal[s] = np.mean(mat_retrait_total[:dur, s])
            retrait_moyen_reel[s] = np.mean(
                mat_retrait_total[:dur, s] / np.maximum(inflation_factor[:dur, s], 1e-9)
            )

    # ── Taux de remplacement ─────────────────────────────────────────────
    if dernier_salaire_mensuel is not None and dernier_salaire_mensuel > 0:
        taux_remplacement_p50 = np.median(retrait_moyen_nominal / dernier_salaire_mensuel)
        taux_remplacement_reel_p50 = np.median(retrait_moyen_reel / dernier_salaire_mensuel)
    else:
        taux_remplacement_p50 = None
        taux_remplacement_reel_p50 = None

    # ── Part variable ────────────────────────────────────────────────────
    tot_p = np.sum(mat_retrait_plancher, axis=0)
    tot_v = np.sum(mat_retrait_variable, axis=0)
    part_variable_p50 = np.median(tot_v / np.maximum(tot_p + tot_v, 1e-9))

    return {
        "proba_ruine": proba_ruine,
        "nb_ruine": nb_ruine,
        "duree_couverture_p5": np.percentile(duree_couverture, 5),
        "duree_couverture_p50": np.percentile(duree_couverture, 50),
        "duree_couverture_p95": np.percentile(duree_couverture, 95),
        "capital_residuel_p5": np.percentile(capital_final, 5),
        "capital_residuel_p50": np.percentile(capital_final, 50),
        "capital_residuel_p95": np.percentile(capital_final, 95),
        "capital_residuel_reel_p50": np.percentile(capital_final_reel, 50),
        "revenu_total_nominal_p50": np.median(revenu_total_nominal),
        "revenu_total_reel_p50": np.median(revenu_total_reel),
        "retrait_mensuel_nominal_p50": np.median(retrait_moyen_nominal),
        "retrait_mensuel_reel_p50": np.median(retrait_moyen_reel),
        "taux_remplacement_p50": taux_remplacement_p50,
        "taux_remplacement_reel_p50": taux_remplacement_reel_p50,
        "part_variable_p50": part_variable_p50,
        "retrait_plancher_cible": retrait_mensuel_plancher,
    }


# =============================================================================
# REPORTING CONSOLE
# =============================================================================

def print_decumulation_report(metrics, strategy_name=""):
    """Affiche un rapport synthétique — appelé depuis main.py si PRINT_DECUMULATION_METRICS."""
    print("\n" + "=" * 80)
    print(f"RÉSULTATS DÉCUMULATION — {strategy_name}")
    print("=" * 80)

    print(f"\n[ PARAMÈTRES CLIENT ]")
    print(f"  • Retrait plancher cible     : {metrics['retrait_plancher_cible']:>10,.0f} €/mois (réel)")
    print(f"  • Taux distribution surplus  : {settings.TAUX_DISTRIBUTION_SURPLUS * 100:>10.0f} %")
    print(f"  • Horizon retraite           : {settings.NB_ANNEES_DECUMULATION:>10d} ans")

    print(f"\n[ RISQUE DE RUINE ]")
    print(f"  • Probabilité de ruine       : {metrics['proba_ruine'] * 100:>10.1f} %")
    print(f"  • Simulations en ruine       : {metrics['nb_ruine']:>10d}")

    print(f"\n[ DURÉE DE COUVERTURE ]")
    print(f"  • P5  (Pessimiste)           : {metrics['duree_couverture_p5']:>10.1f} ans")
    print(f"  • P50 (Médiane)              : {metrics['duree_couverture_p50']:>10.1f} ans")
    print(f"  • P95 (Optimiste)            : {metrics['duree_couverture_p95']:>10.1f} ans")

    print(f"\n[ REVENU DISTRIBUÉ ]")
    print(f"  • Retrait mensuel moyen (nom): {metrics['retrait_mensuel_nominal_p50']:>10,.0f} €")
    print(f"  • Retrait mensuel moyen (réel): {metrics['retrait_mensuel_reel_p50']:>10,.0f} €")
    print(f"  • Revenu total médian (nom)  : {metrics['revenu_total_nominal_p50']:>10,.0f} €")
    print(f"  • Part variable du revenu    : {metrics['part_variable_p50'] * 100:>10.1f} %")

    if metrics['taux_remplacement_p50'] is not None:
        print(f"\n[ TAUX DE REMPLACEMENT ]")
        print(f"  • Nominal (médian)           : {metrics['taux_remplacement_p50'] * 100:>10.1f} %")
        print(f"  • Réel (médian)              : {metrics['taux_remplacement_reel_p50'] * 100:>10.1f} %")

    print(f"\n[ PATRIMOINE RÉSIDUEL ]")
    print(f"  • P5  (Pessimiste)           : {metrics['capital_residuel_p5']:>10,.0f} €")
    print(f"  • P50 (Médiane)              : {metrics['capital_residuel_p50']:>10,.0f} €")
    print(f"  • P95 (Optimiste)            : {metrics['capital_residuel_p95']:>10,.0f} €")
    print(f"  • P50 Réel (€ constants)     : {metrics['capital_residuel_reel_p50']:>10,.0f} €")

    print("=" * 80 + "\n")
