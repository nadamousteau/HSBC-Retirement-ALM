import numpy as np
from scipy import optimize

def calculer_tri_annualise(capital_init, liste_apports, val_finale, freq=12):
    """
    Calcule le TRI (IRR) annualisé du portefeuille.
    Utilise scipy.optimize pour résoudre l'équation de la VAN.
    """
    flux = [-float(capital_init)]
    # Ajout des flux négatifs (investissements)
    flux.extend([-float(m) for m in liste_apports])
    # Ajout du flux positif final (valeur de sortie)
    flux.append(float(val_finale))
    
    def npv(r):
        # Protection contre taux aberrant
        if r <= -1.0:
            return 1e10
        valeurs = np.array(flux)
        puissances = np.arange(len(flux))
        return np.sum(valeurs / ((1 + r) ** puissances))
    
    try:
        # Recherche du zéro entre -5% et +10% mensuel
        tri_periodique = optimize.brentq(npv, -0.05, 0.10)
        return ((1 + tri_periodique) ** freq - 1) * 100
    except:
        return 0.0

def calcul_kpi_complets(capitaux_finaux, total_investi, mat_cap_historique):
    """
    Calcule l'ensemble des indicateurs de risque et de performance.
    
    Returns:
        dict: Dictionnaire contenant Shortfall, VaR, Sortino, Underwater, Dispersion.
    """
    
    # Shortfall Probability (Probabilité de finir sous le montant investi)
    nb_pertes = np.sum(capitaux_finaux < total_investi)
    proba_shortfall = nb_pertes / len(capitaux_finaux)
    
    # VaR 95 (Valeur à risque au seuil de 5%)
    var_95 = np.percentile(capitaux_finaux, 5)
    
    # Sortino Ratio
    # On prend la trajectoire médiane pour calculer le ratio représentatif
    idx_med = np.argsort(capitaux_finaux)[len(capitaux_finaux)//2]
    trajectoire_med = mat_cap_historique[:, idx_med]
    
    rendements = np.diff(trajectoire_med) / trajectoire_med[:-1]
    rendements_neg = rendements[rendements < 0]
    
    downside = np.std(rendements_neg) * np.sqrt(12) if len(rendements_neg) > 0 else 1e-6
    sortino = (np.mean(rendements) * 12) / downside
    
    # Max Underwater Duration (Durée maximale sous le précédent plus haut)
    plus_haut = np.maximum.accumulate(trajectoire_med)
    is_underwater = trajectoire_med < plus_haut
    
    duree_max, compteur = 0, 0
    for s in is_underwater:
        if s:
            compteur += 1
        else:
            duree_max = max(duree_max, compteur)
            compteur = 0
    max_underwater = max(duree_max, compteur) / 12.0
    
    # Dispersion (Écart entre scénario optimiste et pessimiste)
    p95 = np.percentile(capitaux_finaux, 95)
    p5 = np.percentile(capitaux_finaux, 5)
    
    return {
        "shortfall_prob": proba_shortfall,
        "var_95": var_95,
        "gain_p5": var_95 - total_investi,
        "sortino": sortino,
        "max_underwater": max_underwater,
        "dispersion": p95 - p5
    }