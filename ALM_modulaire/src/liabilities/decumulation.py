import numpy as np

def simuler_decumulation(capitaux_finaux, dernier_salaire, taux_livret, duree):
    """
    Simule la phase de retraite avec une pension géométrique (croissance au rythme du taux sans risque).
    """
    nb_sims = len(capitaux_finaux)
    taux_remp = np.zeros((duree, nb_sims))
    mat_cap_retraite = np.zeros((duree + 1, nb_sims))
    
    cap_courant = capitaux_finaux.copy()
    mat_cap_retraite[0, :] = cap_courant
    
    for i in range(duree):
        restant = duree - i
        
        # Calcul de la pension mensuelle (variable chaque année)
        pension_mensuelle = cap_courant / (12 * restant)
        
        # Enregistrement du taux de remplacement de l'année i
        taux_remp[i, :] = pension_mensuelle / dernier_salaire
        
        # Capitalisation du solde
        cap_courant = (cap_courant - pension_mensuelle * 12) * (1 + taux_livret)
        cap_courant = np.maximum(cap_courant, 0)
        
        mat_cap_retraite[i+1, :] = cap_courant
        
    return taux_remp, mat_cap_retraite