import numpy as np

def simuler_decumulation(capitaux_finaux, dernier_salaire, taux_livret, duree):
    """
    Simule la phase de retraite avec rente viagère.
    
    Args:
        capitaux_finaux (np.array): Capital accumulé en fin de phase d'épargne pour chaque simulation.
        dernier_salaire (float): Dernier salaire simulé ou estimé.
        taux_livret (float): Taux de rendement sans risque pendant la retraite.
        duree (int): Durée de la retraite en années.
        
    Returns:
        np.array: Matrice des taux de remplacement (années x simulations).
    """
    nb_sims = len(capitaux_finaux)
    taux_remp = np.zeros((duree, nb_sims))
    cap_courant = capitaux_finaux.copy()
    
    for i in range(duree):
        restant = duree - i
        
        # Calcul de la pension mensuelle (rente viagère simple sur capital restant)
        pension_mensuelle = cap_courant / (12 * restant)
        
        # Taux de remplacement
        taux_remp[i, :] = pension_mensuelle / dernier_salaire
        
        # Mise à jour du capital (consommation + intérêts)
        cap_courant = (cap_courant - pension_mensuelle * 12) * (1 + taux_livret)
        cap_courant = np.maximum(cap_courant, 0)
    
    return taux_remp