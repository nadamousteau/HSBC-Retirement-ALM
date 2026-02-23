import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from config import settings

plt.style.use('seaborn-v0_8-whitegrid')

def generer_facteur_actualisation(nb_periodes):
    """
    Calcule le vecteur de déflation basé sur l'inflation annuelle paramétrée.
    nb_periodes: int (nombre de mois à actualiser)
    """
    annees = np.arange(nb_periodes) / 12.0
    return (1 + settings.TAUX_INFLATION) ** (-annees)

def plot_capital(dates, mat_cap, courbe_investi, reel=False):
    """Génère le graphique de l'évolution du capital, en nominal ou réel."""
    fig, ax = plt.subplots(figsize=(10, 6))
    dates_plot = [pd.Timestamp(settings.DATE_DEBUT_T0)] + list(dates)
    
    nb_periodes = mat_cap.shape[0]
    facteur = generer_facteur_actualisation(nb_periodes)
    
    # Application du vecteur d'actualisation si le mode réel est activé
    mat_cap_plot = mat_cap * facteur[:, np.newaxis] if reel else mat_cap
    courbe_investi_plot = courbe_investi * facteur if reel else courbe_investi
    
    capitaux_finaux = mat_cap_plot[-1, :]
    idx_sorted = np.argsort(capitaux_finaux)
    idx_p5 = idx_sorted[int(len(capitaux_finaux) * 0.05)]
    idx_p50 = idx_sorted[int(len(capitaux_finaux) * 0.50)]
    idx_p95 = idx_sorted[int(len(capitaux_finaux) * 0.95)]

    ax.plot(dates_plot, courbe_investi_plot, color='red', linestyle='--', linewidth=2, label='Versements cumulés')
    ax.plot(dates_plot, mat_cap_plot[:, idx_p95], color='#2ca02c', linewidth=1.5, label='P95 (Optimiste)')
    ax.plot(dates_plot, mat_cap_plot[:, idx_p50], color='black', linewidth=2.5, label='Médiane (P50)')
    ax.plot(dates_plot, mat_cap_plot[:, idx_p5], color='gray', linewidth=1.5, label='P5 (Pessimiste)')
    ax.fill_between(dates_plot, mat_cap_plot[:, idx_p5], mat_cap_plot[:, idx_p95], color='gray', alpha=0.15)
    
    titre = f"Évolution du Capital Accumulé - {settings.METHODE}"
    titre += " (Corrigé de l'inflation)" if reel else " (Nominal)"
    
    ax.set_title(titre)
    ax.set_ylabel("Capital Constant (€)" if reel else "Capital (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_salaire(dates, hist_salaire, reel=False):
    """Génère le graphique de l'évolution du salaire, en nominal ou réel."""
    fig, ax = plt.subplots(figsize=(10, 6))
    nb_periodes = len(hist_salaire)
    facteur = generer_facteur_actualisation(nb_periodes)
    
    salaire_plot = hist_salaire * facteur if reel else hist_salaire
    
    ax.plot(dates, salaire_plot, color='navy', linewidth=2, label='Salaire mensuel')
    
    titre = "Évolution du Salaire" + (" (Corrigé de l'inflation)" if reel else " (Nominal)")
    ax.set_title(titre)
    ax.set_ylabel("Salaire Constant (€)" if reel else "Salaire (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_apports(dates, hist_apport, reel=False):
    """Génère le graphique de l'évolution des apports, en nominal ou réel."""
    fig, ax = plt.subplots(figsize=(10, 6))
    nb_periodes = len(hist_apport)
    facteur = generer_facteur_actualisation(nb_periodes)
    
    apport_plot = hist_apport * facteur if reel else hist_apport
    
    ax.plot(dates, apport_plot, color='darkred', linewidth=2, label='Apport mensuel')
    
    titre = "Évolution des Apports Mensuels" + (" (Corrigés de l'inflation)" if reel else " (Nominaux)")
    ax.set_title(titre)
    ax.set_ylabel("Apport Constant (€)" if reel else "Apport (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

def _get_window_indices(dates_plot, date_crise):
    """Calcule les bornes de la fenêtre [T-12, T+60] autour de la crise."""
    date_c = pd.Timestamp(date_crise)
    idx_crise = np.argmin(np.abs(dates_plot - date_c))
    idx_start = max(0, idx_crise - 12)
    idx_end = min(len(dates_plot), idx_crise + 61) # +60 mois inclus
    return idx_start, idx_end, idx_crise

def plot_zoom_crise_capital(dates, mat_cap, date_crise, reel=False):
    """Génère le zoom sur l'évolution du capital (Nominal ou Réel) autour du choc."""
    dates_plot = pd.DatetimeIndex([pd.Timestamp(settings.DATE_DEBUT_T0)] + list(dates))
    idx_start, idx_end, idx_crise = _get_window_indices(dates_plot, date_crise)

    window_dates = dates_plot[idx_start:idx_end]
    window_cap = mat_cap[idx_start:idx_end, :]

    if reel:
        facteur = generer_facteur_actualisation(mat_cap.shape[0])
        window_facteur = facteur[idx_start:idx_end]
        window_cap = window_cap * window_facteur[:, np.newaxis]

    # Conservation des trajectoires fractiles globales
    capitaux_finaux = mat_cap[-1, :]
    idx_sorted = np.argsort(capitaux_finaux)
    idx_p5 = idx_sorted[int(len(capitaux_finaux) * 0.05)]
    idx_p50 = idx_sorted[int(len(capitaux_finaux) * 0.50)]
    idx_p95 = idx_sorted[int(len(capitaux_finaux) * 0.95)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(window_dates, window_cap[:, idx_p95], color='#2ca02c', linewidth=1.5, label='P95')
    ax.plot(window_dates, window_cap[:, idx_p50], color='black', linewidth=2.5, label='Médiane (P50)')
    ax.plot(window_dates, window_cap[:, idx_p5], color='gray', linewidth=1.5, label='P5')
    ax.fill_between(window_dates, window_cap[:, idx_p5], window_cap[:, idx_p95], color='gray', alpha=0.15)

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

    # Rebasage de la trajectoire : 100 * (Valeur(t) / Valeur(t_start))
    perf_cumul = (window_cap / window_cap[0, :]) * 100

    capitaux_finaux = mat_cap[-1, :]
    idx_sorted = np.argsort(capitaux_finaux)
    idx_p5 = idx_sorted[int(len(capitaux_finaux) * 0.05)]
    idx_p50 = idx_sorted[int(len(capitaux_finaux) * 0.50)]
    idx_p95 = idx_sorted[int(len(capitaux_finaux) * 0.95)]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(window_dates, perf_cumul[:, idx_p95], color='#2ca02c', linewidth=1.5, label='P95')
    ax.plot(window_dates, perf_cumul[:, idx_p50], color='black', linewidth=2.5, label='Médiane (P50)')
    ax.plot(window_dates, perf_cumul[:, idx_p5], color='gray', linewidth=1.5, label='P5')
    ax.fill_between(window_dates, perf_cumul[:, idx_p5], perf_cumul[:, idx_p95], color='gray', alpha=0.15)

    ax.axvline(dates_plot[idx_crise], color='red', linestyle='--', linewidth=1.5, label='Choc de marché')
    ax.axhline(100, color='black', linestyle=':', alpha=0.8)

    ax.set_title("Impact Crise : Performance cumulée du portefeuille (Base 100)")
    ax.set_ylabel("Indice (Base 100 = 1 an avant choc)")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()

def plot_retraite_capital(mat_cap_retraite, reel=False):
    """Génère le graphique de l'évolution du capital pendant la retraite (Nominal ou Réel)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    duree = mat_cap_retraite.shape[0] - 1
    annees = np.arange(duree + 1)
    
    # Actualisation en euros constants T0
    if reel:
        facteur = (1 + settings.TAUX_INFLATION) ** -(settings.NB_ANNEES_ACCUMULATION + annees)
        mat_cap_plot = mat_cap_retraite * facteur[:, np.newaxis]
    else:
        mat_cap_plot = mat_cap_retraite
    
    # Identification des quantiles sur le capital initial réel ou nominal
    capitaux_initiaux = mat_cap_plot[0, :]
    idx_sorted = np.argsort(capitaux_initiaux)
    idx_p5 = idx_sorted[int(len(capitaux_initiaux) * 0.05)]
    idx_p50 = idx_sorted[int(len(capitaux_initiaux) * 0.50)]
    idx_p95 = idx_sorted[int(len(capitaux_initiaux) * 0.95)]
    
    ax.plot(annees, mat_cap_plot[:, idx_p95], color='#2ca02c', linewidth=1.5, label='P95')
    ax.plot(annees, mat_cap_plot[:, idx_p50], color='black', linewidth=2.5, label='Médiane (P50)')
    ax.plot(annees, mat_cap_plot[:, idx_p5], color='gray', linewidth=1.5, label='P5')
    ax.fill_between(annees, mat_cap_plot[:, idx_p5], mat_cap_plot[:, idx_p95], color='gray', alpha=0.15)
    
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

def plot_taux_remplacement(taux_remp, reel=False):
    """Génère le graphique de l'évolution du taux de remplacement (Nominal ou Réel)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    duree = taux_remp.shape[0]
    annees = np.arange(1, duree + 1)
    
    # Actualisation relative : seule la dérive post-départ retraite impacte le ratio
    if reel:
        facteur = (1 + settings.TAUX_INFLATION) ** -annees
        taux_plot = taux_remp * facteur[:, np.newaxis]
    else:
        taux_plot = taux_remp
        
    taux_initiaux = taux_plot[0, :]
    idx_sorted = np.argsort(taux_initiaux)
    idx_p5 = idx_sorted[int(len(taux_initiaux) * 0.05)]
    idx_p50 = idx_sorted[int(len(taux_initiaux) * 0.50)]
    idx_p95 = idx_sorted[int(len(taux_initiaux) * 0.95)]
    
    ax.plot(annees, taux_plot[:, idx_p95] * 100, color='#2ca02c', linewidth=1.5, label='P95')
    ax.plot(annees, taux_plot[:, idx_p50] * 100, color='black', linewidth=2.5, label='Médiane (P50)')
    ax.plot(annees, taux_plot[:, idx_p5] * 100, color='gray', linewidth=1.5, label='P5')
    
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

def plot_comparaison_capital(dates, resultats_dict, reel=False):
    """
    Superpose l'évolution du capital pour plusieurs stratégies sur le même graphique.
    resultats_dict: dictionnaire { 'NOM_STRATEGIE': matrice_capital }
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    dates_plot = [pd.Timestamp(settings.DATE_DEBUT_T0)] + list(dates)
    
    # Palette de couleurs stricte pour différencier les stratégies
    couleurs = {"TARGET_DATE": "navy", "FIXED_MIX": "darkred"}
    
    for strat, mat_cap in resultats_dict.items():
        if reel:
            facteur = generer_facteur_actualisation(mat_cap.shape[0])
            mat_cap_plot = mat_cap * facteur[:, np.newaxis]
        else:
            mat_cap_plot = mat_cap
            
        capitaux_finaux = mat_cap_plot[-1, :]
        idx_sorted = np.argsort(capitaux_finaux)
        idx_p5 = idx_sorted[int(len(capitaux_finaux) * 0.05)]
        idx_p50 = idx_sorted[int(len(capitaux_finaux) * 0.50)]
        idx_p95 = idx_sorted[int(len(capitaux_finaux) * 0.95)]
        
        c = couleurs.get(strat, "black") # Noir par défaut si stratégie inconnue
        
        # Tracé de la médiane et du P5 pour chaque stratégie
        ax.plot(dates_plot, mat_cap_plot[:, idx_p50], color=c, linewidth=2.5, label=f'{strat} (Médiane)')
        ax.plot(dates_plot, mat_cap_plot[:, idx_p5], color=c, linewidth=1.5, linestyle='--', label=f'{strat} (P5)')
        ax.fill_between(dates_plot, mat_cap_plot[:, idx_p5], mat_cap_plot[:, idx_p95], color=c, alpha=0.1)
        
    titre = "Comparaison Stratégique : Évolution du Capital Accumulé"
    titre += " (Réel - € constants)" if reel else " (Nominal)"
    
    ax.set_title(titre)
    ax.set_ylabel("Capital Constant (€)" if reel else "Capital (€)")
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',').replace(',', ' ')))
    ax.legend(loc='upper left')
    plt.tight_layout()
    plt.show()