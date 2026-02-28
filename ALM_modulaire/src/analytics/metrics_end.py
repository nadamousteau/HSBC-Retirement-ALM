import numpy as np

def calculer_kpis_fin_accumulation(mat_cap, dates):
    """
    Calcule les extrema de capital et les différentiels sur la fenêtre stricte des 5 
    dernières années (60 mois) précédant la date de départ à la retraite.
    
    Args:
        mat_cap (np.ndarray): Matrice d'accumulation (nb_steps+1 x nb_sims).
        dates_acc (pd.DatetimeIndex): Vecteur des dates de la simulation.
        
    Returns:
        dict: Dictionnaire contenant les vecteurs des métriques pour chaque trajectoire.
    """
    # Extraction de la fenêtre de 5 ans (60 périodes + le point final)
    fenetre_5a = mat_cap[-61:, :]
    
    # Capital exact au moment du départ à la retraite (dernier point de la matrice)
    cap_retraite = mat_cap[-1, :]
    
    # Évaluation des extrema sur la fenêtre
    max_5ans = np.max(fenetre_5a, axis=0)
    min_5ans = np.min(fenetre_5a, axis=0)
    
    # Extraction vectorisée des dates d'occurrence du maximum
   # Extraction vectorisée des dates d'occurrence du maximum
    idx_max_relatif = np.argmax(fenetre_5a, axis=0)
    idx_max_global = (mat_cap.shape[0] - 61) + idx_max_relatif
    
    # CORRECTION DE L'ALIGNEMENT TEMPOREL :
    # mat_cap est de taille N+1 (indices 0 à N)
    # dates_acc est de taille N (indices 0 à N-1)
    # On décale l'indice de -1 pour correspondre au vecteur dates, 
    # avec np.clip pour sécuriser mathématiquement les bornes.
    idx_dates = np.clip(idx_max_global - 1, 0, len(dates) - 1)
    dates_max = dates[idx_dates]

    # Calculs des différentiels demandés
    benefice_vs_pire = cap_retraite - min_5ans
    manque_a_gagner = max_5ans - cap_retraite

    return {
        "max_5ans": max_5ans,
        "dates_max": dates_max,
        "min_5ans": min_5ans,
        "benefice_vs_pire": benefice_vs_pire,
        "manque_a_gagner": manque_a_gagner
    }

def synthetiser_kpis_fin_accumulation(kpis_end):
    """
    Agrége les distributions par leurs médianes pour l'affichage des résultats globaux.
    """
    return {
        "mediane_max_5ans": np.median(kpis_end["max_5ans"]),
        "mediane_benefice": np.median(kpis_end["benefice_vs_pire"]),
        "mediane_manque": np.median(kpis_end["manque_a_gagner"]),
        "date_max": kpis_end["dates_max"][-1]
    }