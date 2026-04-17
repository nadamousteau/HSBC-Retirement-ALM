import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from config import settings

plt.style.use('seaborn-v0_8-whitegrid')


# =============================================================================
# DÉFLATION
# =============================================================================

def generer_facteur_actualisation(nb_periodes, inflation_factor=None):
    """
    Calcule le vecteur de déflation.
    
    Si inflation_factor stochastique est fourni, utilise sa médiane.
    Sinon, fallback sur TAUX_INFLATION constant (rétrocompatibilité).
    
    Args:
        nb_periodes      : int — longueur du vecteur à produire
        inflation_factor : np.array (nb_periodes, nb_sims), optionnel
                           Facteur cumulatif stochastique issu du moteur.
    Returns:
        np.array (nb_periodes,) — vecteur de déflation (à multiplier au nominal)
    """
    if inflation_factor is not None:
        median_factor = np.median(inflation_factor, axis=1)
        # Éviter division par zéro au pas 0 (facteur = 1.0)
        median_factor = np.maximum(median_factor, 1e-9)
        return 1.0 / median_factor
    else:
        annees = np.arange(nb_periodes) / 12.0
        return (1 + settings.TAUX_INFLATION) ** (-annees)


def _representative_sim_index(mat):
    """
    Retourne l'indice de la simulation la plus proche de la trajectoire
    percentile-P50 au sens de la distance L2 (sim-médoïde).

    Contexte ALM : un fan chart percentile-par-pas lisse toute la volatilité
    (les percentiles sont calculés indépendamment à chaque date, ce qui donne
    une courbe anormalement régulière, inexistante dans la réalité). Pour
    afficher UN chemin unique représentatif des dynamiques de marché
    (crises, rebonds, séquences de rendements), on sélectionne la sim réelle
    dont la trajectoire s'écarte le moins, en L2, de la trajectoire médiane
    percentile. C'est le scénario "représentatif" de la littérature ALM
    (cf. scénarios CTE / LDI / Solvency II).

    Args:
        mat : ndarray (nb_pas, nb_sims) — matrice des trajectoires simulées.
    Returns:
        int — indice de la colonne la plus proche de la médiane.
    """
    traj_median = np.percentile(mat, 50, axis=1)
    dist = np.sum((mat - traj_median[:, np.newaxis]) ** 2, axis=0)
    return int(np.argmin(dist))


# =============================================================================
# GRAPHIQUES — PHASE D'ACCUMULATION
# =============================================================================

def plot_capital(dates, mat_cap, courbe_investi, reel=False, inflation_factor=None):
    """Génère le graphique de l'évolution du capital, en nominal ou réel."""
    fig, ax = plt.subplots(figsize=(10, 6))
    dates_plot = [pd.Timestamp(settings.DATE_DEBUT_T0)] + list(dates)
    
    nb_periodes = mat_cap.shape[0]
    facteur = generer_facteur_actualisation(nb_periodes, inflation_factor=inflation_factor)
    
    mat_cap_plot = mat_cap * facteur[:, np.newaxis] if reel else mat_cap
    courbe_investi_plot = courbe_investi * facteur if reel else courbe_investi

    # Enveloppe P5-P95 pour l'incertitude (fan chart).
    # Trajectoire centrale : scénario représentatif (sim-médoïde L2), qui
    # préserve la volatilité réelle du marché contrairement à une courbe
    # de percentiles lissés.
    traj_p5 = np.percentile(mat_cap_plot, 5, axis=1)
    traj_p95 = np.percentile(mat_cap_plot, 95, axis=1)
    idx_rep = _representative_sim_index(mat_cap_plot)
    traj_rep = mat_cap_plot[:, idx_rep]

    ax.plot(dates_plot, courbe_investi_plot, color='red', linestyle='--', linewidth=2, label='Versements cumulés')
    ax.plot(dates_plot, traj_p95, color='#2ca02c', linewidth=1.2, alpha=0.7, label='P95 (Optimiste)')
    ax.plot(dates_plot, traj_rep, color='black', linewidth=2.5, label='Scénario représentatif')
    ax.plot(dates_plot, traj_p5, color='gray', linewidth=1.2, alpha=0.7, label='P5 (Pessimiste)')
    ax.fill_between(dates_plot, traj_p5, traj_p95, color='gray', alpha=0.15)
    
    titre = f"Évolution du Capital Accumulé - {settings.METHODE}"
    titre += " (Corrigé de l'inflation)" if reel else " (Nominal)"
    
    ax.set_title(titre)
    ax.set_ylabel("Capital Constant (€)" if reel else "Capital (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_salaire(dates, hist_salaire, reel=False, inflation_factor=None):
    """Génère le graphique de l'évolution du salaire, en nominal ou réel."""
    fig, ax = plt.subplots(figsize=(10, 6))
    nb_periodes = len(hist_salaire)
    facteur = generer_facteur_actualisation(nb_periodes, inflation_factor=inflation_factor)
    
    salaire_plot = hist_salaire * facteur if reel else hist_salaire
    
    ax.plot(dates, salaire_plot, color='navy', linewidth=2, label='Salaire mensuel')
    
    titre = "Évolution du Salaire" + (" (Corrigé de l'inflation)" if reel else " (Nominal)")
    ax.set_title(titre)
    ax.set_ylabel("Salaire Constant (€)" if reel else "Salaire (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


def plot_apports(dates, hist_apport, reel=False, inflation_factor=None):
    """Génère le graphique de l'évolution des apports, en nominal ou réel."""
    fig, ax = plt.subplots(figsize=(10, 6))
    nb_periodes = len(hist_apport)
    facteur = generer_facteur_actualisation(nb_periodes, inflation_factor=inflation_factor)
    
    apport_plot = hist_apport * facteur if reel else hist_apport
    
    ax.plot(dates, apport_plot, color='darkred', linewidth=2, label='Apport mensuel')
    
    titre = "Évolution des Apports Mensuels" + (" (Corrigés de l'inflation)" if reel else " (Nominaux)")
    ax.set_title(titre)
    ax.set_ylabel("Apport Constant (€)" if reel else "Apport (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()


# =============================================================================
# GRAPHIQUES — ZOOM CRISE
# =============================================================================

def _get_window_indices(dates_plot, date_crise):
    """Calcule les bornes de la fenêtre [T-12, T+60] autour de la crise."""
    date_c = pd.Timestamp(date_crise)
    idx_crise = np.argmin(np.abs(dates_plot - date_c))
    idx_start = max(0, idx_crise - 12)
    idx_end = min(len(dates_plot), idx_crise + 61)
    return idx_start, idx_end, idx_crise


def plot_zoom_crise_capital(dates, mat_cap, date_crise, reel=False, inflation_factor=None):
    """Génère le zoom sur l'évolution du capital (Nominal ou Réel) autour du choc."""
    dates_plot = pd.DatetimeIndex([pd.Timestamp(settings.DATE_DEBUT_T0)] + list(dates))
    idx_start, idx_end, idx_crise = _get_window_indices(dates_plot, date_crise)

    window_dates = dates_plot[idx_start:idx_end]
    window_cap = mat_cap[idx_start:idx_end, :]

    if reel:
        facteur = generer_facteur_actualisation(mat_cap.shape[0], inflation_factor=inflation_factor)
        window_facteur = facteur[idx_start:idx_end]
        window_cap = window_cap * window_facteur[:, np.newaxis]

    # Enveloppe P5-P95 + scénario représentatif (sim-médoïde L2) pour
    # conserver les vraies dynamiques de choc/rebond à l'écran.
    traj_p5 = np.percentile(window_cap, 5, axis=1)
    traj_p95 = np.percentile(window_cap, 95, axis=1)
    idx_rep = _representative_sim_index(window_cap)
    traj_rep = window_cap[:, idx_rep]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(window_dates, traj_p95, color='#2ca02c', linewidth=1.2, alpha=0.7, label='P95')
    ax.plot(window_dates, traj_rep, color='black', linewidth=2.5, label='Scénario représentatif')
    ax.plot(window_dates, traj_p5, color='gray', linewidth=1.2, alpha=0.7, label='P5')
    ax.fill_between(window_dates, traj_p5, traj_p95, color='gray', alpha=0.15)

    ax.axvline(dates_plot[idx_crise], color='red', linestyle='--', linewidth=1.5, label='Choc de marché')

    titre = "Impact Crise : Capital Accumulé" + (" (Réel)" if reel else " (Nominal)")
    ax.set_title(titre)
    ax.set_ylabel("Capital (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


def plot_zoom_crise_rendements(dates, mat_cap, date_crise):
    """Génère la performance cumulée (Base 100 = 1 an avant crise) du portefeuille."""
    dates_plot = pd.DatetimeIndex([pd.Timestamp(settings.DATE_DEBUT_T0)] + list(dates))
    idx_start, idx_end, idx_crise = _get_window_indices(dates_plot, date_crise)

    window_dates = dates_plot[idx_start:idx_end]
    window_cap = mat_cap[idx_start:idx_end, :]

    perf_cumul = (window_cap / window_cap[0, :]) * 100

    # Enveloppe P5-P95 + scénario représentatif (sim-médoïde L2).
    traj_p5 = np.percentile(perf_cumul, 5, axis=1)
    traj_p95 = np.percentile(perf_cumul, 95, axis=1)
    idx_rep = _representative_sim_index(perf_cumul)
    traj_rep = perf_cumul[:, idx_rep]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(window_dates, traj_p95, color='#2ca02c', linewidth=1.2, alpha=0.7, label='P95')
    ax.plot(window_dates, traj_rep, color='black', linewidth=2.5, label='Scénario représentatif')
    ax.plot(window_dates, traj_p5, color='gray', linewidth=1.2, alpha=0.7, label='P5')
    ax.fill_between(window_dates, traj_p5, traj_p95, color='gray', alpha=0.15)

    ax.axvline(dates_plot[idx_crise], color='red', linestyle='--', linewidth=1.5, label='Choc de marché')
    ax.axhline(100, color='black', linestyle=':', alpha=0.8)

    ax.set_title("Impact Crise : Performance cumulée du portefeuille (Base 100)")
    ax.set_ylabel("Indice (Base 100 = 1 an avant choc)")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


# =============================================================================
# GRAPHIQUES — PHASE DE RETRAITE (DÉCUMULATION)
# =============================================================================

def plot_retraite_capital(mat_cap_retraite, reel=False, inflation_factor=None):
    """Génère le graphique de l'évolution du capital pendant la retraite (Nominal ou Réel)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    nb_pas = mat_cap_retraite.shape[0] - 1
    annees = np.arange(nb_pas + 1) / 12.0  # Conversion mois → années
    duree = annees[-1]
    
    if reel:
        if inflation_factor is not None:
            # Déflation stochastique : médiane du facteur cumulatif
            # inflation_factor couvre la phase retraite uniquement ici
            median_factor = np.median(inflation_factor, axis=1)
            median_factor = np.maximum(median_factor, 1e-9)
            facteur = 1.0 / median_factor
        else:
            # Fallback déterministe depuis T0
            facteur = (1 + settings.TAUX_INFLATION) ** -(settings.NB_ANNEES_ACCUMULATION + annees)
        mat_cap_plot = mat_cap_retraite * facteur[:, np.newaxis]
    else:
        mat_cap_plot = mat_cap_retraite
    
    # Enveloppe P5-P95 + scénario représentatif (sim-médoïde L2).
    traj_p5 = np.percentile(mat_cap_plot, 5, axis=1)
    traj_p95 = np.percentile(mat_cap_plot, 95, axis=1)
    idx_rep = _representative_sim_index(mat_cap_plot)
    traj_rep = mat_cap_plot[:, idx_rep]

    ax.plot(annees, traj_p95, color='#2ca02c', linewidth=1.2, alpha=0.7, label='P95')
    ax.plot(annees, traj_rep, color='black', linewidth=2.5, label='Scénario représentatif')
    ax.plot(annees, traj_p5, color='gray', linewidth=1.2, alpha=0.7, label='P5')
    ax.fill_between(annees, traj_p5, traj_p95, color='gray', alpha=0.15)

    titre = f"Évolution du Capital en Retraite ({duree} ans)"
    titre += " (Réel - € constants T0)" if reel else " (Nominal)"
    
    ax.set_title(titre)
    ax.set_xlabel("Années de retraite")
    ax.set_ylabel("Capital Restant Constant (€)" if reel else "Capital Restant (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    ax.set_xlim(0, duree)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.show()


def plot_taux_remplacement(taux_remp, reel=False, inflation_factor=None):
    """Génère le graphique de l'évolution du taux de remplacement (Nominal ou Réel)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    duree = taux_remp.shape[0]
    annees = np.arange(1, duree + 1)
    
    if reel:
        if inflation_factor is not None:
            median_factor = np.median(inflation_factor, axis=1)
            median_factor = np.maximum(median_factor, 1e-9)
            facteur = 1.0 / median_factor
        else:
            facteur = (1 + settings.TAUX_INFLATION) ** -annees
        taux_plot = taux_remp * facteur[:, np.newaxis]
    else:
        taux_plot = taux_remp
        
    # Enveloppe P5-P95 + scénario représentatif (sim-médoïde L2).
    traj_p5 = np.percentile(taux_plot, 5, axis=1)
    traj_p95 = np.percentile(taux_plot, 95, axis=1)
    idx_rep = _representative_sim_index(taux_plot)
    traj_rep = taux_plot[:, idx_rep]

    ax.plot(annees, traj_p95 * 100, color='#2ca02c', linewidth=1.2, alpha=0.7, label='P95')
    ax.plot(annees, traj_rep * 100, color='black', linewidth=2.5, label='Scénario représentatif')
    ax.plot(annees, traj_p5 * 100, color='gray', linewidth=1.2, alpha=0.7, label='P5')
    
    ax.axhline(100, color='red', linestyle='--', linewidth=1, alpha=0.5, label='100% (Maintien pouvoir d\'achat)')
    
    titre = "Évolution du Taux de Remplacement en Retraite"
    titre += " (Réel)" if reel else " (Nominal)"
    
    ax.set_title(titre)
    ax.set_xlabel("Année de retraite")
    ax.set_ylabel("Taux de remplacement (%)")
    ax.set_xlim(1, duree)
    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.show()


# =============================================================================
# GRAPHIQUES — COMPARAISON MULTI-STRATÉGIES
# =============================================================================

def plot_comparaison_capital(dates, resultats_dict, reel=False, inflation_factor=None):
    """
    Superpose l'évolution du capital pour plusieurs stratégies sur le même graphique.
    
    Args:
        resultats_dict   : dict { 'NOM_STRATEGIE': matrice_capital }
        inflation_factor : np.array (nb_periodes+1, nb_sims), optionnel
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    dates_plot = [pd.Timestamp(settings.DATE_DEBUT_T0)] + list(dates)
    
    couleurs = {"TARGET_DATE": "navy", "FIXED_MIX": "darkred", "GBI": "green", "FALEH": "purple"}
    
    for strat, mat_cap in resultats_dict.items():
        if reel:
            facteur = generer_facteur_actualisation(mat_cap.shape[0], inflation_factor=inflation_factor)
            mat_cap_plot = mat_cap * facteur[:, np.newaxis]
        else:
            mat_cap_plot = mat_cap

        # Enveloppe P5-P95 pour l'incertitude, trajectoire centrale =
        # scénario représentatif (sim-médoïde L2). On évite ainsi à la fois :
        #   - le biais d'endpoint d'un tri sur valeur finale (les sims P50
        #     se terminent mécaniquement au même fractile = convergence
        #     artificielle des médianes),
        #   - et le lissage excessif d'une courbe percentile-par-pas qui
        #     efface toute la volatilité du marché.
        traj_p5 = np.percentile(mat_cap_plot, 5, axis=1)
        traj_p95 = np.percentile(mat_cap_plot, 95, axis=1)
        idx_rep = _representative_sim_index(mat_cap_plot)
        traj_rep = mat_cap_plot[:, idx_rep]

        c = couleurs.get(strat, "black")

        ax.plot(dates_plot, traj_rep, color=c, linewidth=2.5, label=f'{strat} (Scénario représentatif)')
        ax.plot(dates_plot, traj_p5, color=c, linewidth=1.2, linestyle='--', alpha=0.7, label=f'{strat} (P5)')
        ax.fill_between(dates_plot, traj_p5, traj_p95, color=c, alpha=0.1)
        
    titre = "Comparaison Stratégique : Évolution du Capital Accumulé"
    titre += " (Réel - € constants)" if reel else " (Nominal)"
    
    ax.set_title(titre)
    ax.set_ylabel("Capital Constant (€)" if reel else "Capital (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()