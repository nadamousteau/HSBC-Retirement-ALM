"""
PIPELINE — REPORTING POST-ACCUMULATION
========================================
Centralise les plots et impressions synthèse déclenchés après la boucle
d'accumulation (sections 5 et 6 de l'ancien main.py).

Aucune logique de simulation : cette fonction ne fait que lire des contextes
déjà calculés et les restituer à l'utilisateur selon les flags dans
settings.py.

Sélection du contexte solo (Tâche 4) : les plots solo et les graphiques
macro-économiques s'appuient sur la stratégie `settings.METHODE_DEFAUT`,
fixée explicitement par l'utilisateur. Si cette stratégie n'a pas été
exécutée (absence de `METHODE_DEFAUT` dans `STRATEGIES_A_COMPARER`), on
retombe sur la première stratégie du dict plutôt que sur la dernière,
pour ne pas dépendre de l'ordre d'itération.
"""

import numpy as np

from config import settings
from src.analytics import plotting


def _select_solo_context(contextes_par_strat):
    """
    Retourne (strat_name, contexte) de la stratégie choisie pour les plots solo.

    Règle : `settings.METHODE_DEFAUT` si présente dans le dict, sinon première
    stratégie du dict (ordre d'insertion — déterministe).
    """
    methode_defaut = getattr(settings, 'METHODE_DEFAUT', None)
    if methode_defaut is not None and methode_defaut in contextes_par_strat:
        return methode_defaut, contextes_par_strat[methode_defaut]
    strat_name = next(iter(contextes_par_strat))
    return strat_name, contextes_par_strat[strat_name]


def run_reporting(contextes_par_strat, dates):
    """
    Exécute le reporting post-accumulation.

    Args:
        contextes_par_strat : dict {strat_name: contexte_strat} produit par
                              run_accumulation().
        dates               : DatetimeIndex de l'accumulation.
    """
    if not contextes_par_strat:
        return

    mode_comparaison = getattr(settings, 'MODE_COMPARAISON', False)

    solo_strat_name, solo_ctx = _select_solo_context(contextes_par_strat)

    courbe_investi = solo_ctx["courbe_investi"]
    hist_salaire = solo_ctx["hist_salaire"]
    hist_apport = solo_ctx["hist_apport"]
    mat_cap = solo_ctx["mat_cap"]
    inflation_factor = solo_ctx["inflation_factor"]

    # Matrice de capitaux par stratégie pour les plots comparatifs
    resultats_comparaison = {
        strat: ctx["mat_cap"] for strat, ctx in contextes_par_strat.items()
    }

    # =========================================================================
    # 5. VISUALISATION ACCUMULATION (Exécution conditionnelle)
    # =========================================================================
    if mode_comparaison:
        if getattr(settings, 'PLOT_COMPARAISON_CAPITAL', False):
            plotting.plot_comparaison_capital(dates, resultats_comparaison, reel=False)
        if getattr(settings, 'PLOT_COMPARAISON_CAPITAL_REEL', False):
            plotting.plot_comparaison_capital(
                dates, resultats_comparaison, reel=True, inflation_factor=inflation_factor
            )
    else:
        if getattr(settings, 'PLOT_CAPITAL', False):
            plotting.plot_capital(dates, mat_cap, courbe_investi, reel=False)
        if getattr(settings, 'PLOT_CAPITAL_REEL', False):
            plotting.plot_capital(
                dates, mat_cap, courbe_investi, reel=True, inflation_factor=inflation_factor
            )

    # Graphiques macro-économiques
    if getattr(settings, 'PLOT_SALAIRE', False):
        plotting.plot_salaire(dates, hist_salaire, reel=False)
    if getattr(settings, 'PLOT_SALAIRE_REEL', False):
        plotting.plot_salaire(dates, hist_salaire, reel=True, inflation_factor=inflation_factor)
    if getattr(settings, 'PLOT_APPORTS', False):
        plotting.plot_apports(dates, hist_apport, reel=False)
    if getattr(settings, 'PLOT_APPORTS_REEL', False):
        plotting.plot_apports(dates, hist_apport, reel=True, inflation_factor=inflation_factor)

    # Graphiques analytiques (crise)
    if getattr(settings, 'SIMULER_CRISE_LOCALISEE', False):
        if getattr(settings, 'PLOT_CRISE_RENDEMENTS', False):
            plotting.plot_zoom_crise_rendements(dates, mat_cap, settings.DATE_CRISE)
        if getattr(settings, 'PLOT_CRISE_CAPITAL_NOMINAL', False):
            plotting.plot_zoom_crise_capital(dates, mat_cap, settings.DATE_CRISE, reel=False)
        if getattr(settings, 'PLOT_CRISE_CAPITAL_REEL', False):
            plotting.plot_zoom_crise_capital(
                dates, mat_cap, settings.DATE_CRISE, reel=True, inflation_factor=inflation_factor
            )

    # =========================================================================
    # 6. SYNTHÈSE DES CAPITAUX À LA RETRAITE (Console)
    # =========================================================================
    if getattr(settings, 'PRINT_SYNTHESE_CAPITAL_RETRAITE', False):
        print("\n" + "=" * 80)
        print("SYNTHÈSE DES CAPITAUX FINAUX À LA RETRAITE (FIN D'ACCUMULATION)")
        print("=" * 80)

        for strat_nom, mat_c in resultats_comparaison.items():
            capitaux_finaux_strat = mat_c[-1, :]
            idx_sort = np.argsort(capitaux_finaux_strat)
            p5 = capitaux_finaux_strat[idx_sort[int(settings.NB_SIMULATIONS * 0.05)]]
            p50 = capitaux_finaux_strat[idx_sort[int(settings.NB_SIMULATIONS * 0.50)]]
            p95 = capitaux_finaux_strat[idx_sort[int(settings.NB_SIMULATIONS * 0.95)]]

            print(f"\n[ STRATÉGIE : {strat_nom} ]")
            print(f"  • Capital P5  (Pessimiste 5%)  : {p5:>15,.0f} EUR")
            print(f"  • Capital P50 (Médiane)        : {p50:>15,.0f} EUR")
            print(f"  • Capital P95 (Optimiste 95%)  : {p95:>15,.0f} EUR")

        print("=" * 80 + "\n")
