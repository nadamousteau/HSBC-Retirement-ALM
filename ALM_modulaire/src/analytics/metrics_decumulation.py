import numpy as np

def calculer_probabilite_ruine(mat_cap_retraite):
    """
    Calcule la probabilité que le capital tombe à zéro (ou en dessous) 
    durant la phase de décumulation.
    
    Args:
        mat_cap_retraite (np.ndarray): Matrice des capitaux durant la retraite (Temps x Simulations).
        
    Returns:
        float: Probabilité de ruine (entre 0.0 et 1.0).
    """
    # Évaluation vectorisée : True si le capital passe sous 0 à n'importe quel moment de la trajectoire
    ruines_par_simulation = np.any(mat_cap_retraite <= 1e-4, axis=0)
    return np.mean(ruines_par_simulation)

def calculer_expected_shortfall_passif(capitaux_finaux, capital_cible, alpha=0.05):
    """
    Calcule l'Expected Shortfall (CVaR) relatif au déficit de capital par rapport à l'objectif.
    
    Args:
        capitaux_finaux (np.ndarray): Vecteur des capitaux à l'instant de la retraite.
        capital_cible (float): Montant requis pour financer le passif (ex: rente cible).
        alpha (float): Seuil de probabilité pour les pires scénarios (défaut 5%).
        
    Returns:
        float: Espérance du manque à gagner dans les pires (alpha * 100)% des cas.
    """
    # Calcul du déficit L (Loss). L = 0 si l'objectif est dépassé.
    deficits = np.maximum(0.0, capital_cible - capitaux_finaux)
    
    # La VaR du déficit correspond au quantile (1 - alpha) de la distribution des déficits
    var_deficit = np.percentile(deficits, 100 * (1 - alpha))
    
    # Extraction des pires scénarios (déficits stricts supérieurs à la VaR)
    pires_deficits = deficits[deficits > var_deficit]
    
    if len(pires_deficits) == 0:
        return 0.0
    return np.mean(pires_deficits)

def calculer_distribution_taux_remplacement(taux_remplacement_initiaux):
    """
    Extrait les quantiles conditionnels du pouvoir d'achat à la retraite.
    
    Args:
        taux_remplacement_initiaux (np.ndarray): Vecteur des taux de remplacement 
                                                 (Rente année 1 / Dernier salaire).
                                                 
    Returns:
        dict: Quantiles 5%, 50% (médiane) et 95%.
    """
    return {
        "p5_pessimiste": np.percentile(taux_remplacement_initiaux, 5),
        "p50_median": np.percentile(taux_remplacement_initiaux, 50),
        "p95_optimiste": np.percentile(taux_remplacement_initiaux, 95)
    }

def calculer_equivalent_certain(capitaux_finaux, gamma=3.0):
    """
    Calcule l'Équivalent Certain (Certainty Equivalent Wealth) basé sur une 
    fonction d'utilité CRRA (Constant Relative Risk Aversion).
    
    Args:
        capitaux_finaux (np.ndarray): Vecteur des capitaux à l'instant de la retraite.
        gamma (float): Coefficient d'aversion au risque de l'investisseur.
                       gamma = 1 -> Utilité logarithmique.
                       gamma > 1 -> Aversion forte.
                       
    Returns:
        float: Le montant garanti qui procure la même utilité que la stratégie risquée.
    """
    # Troncature pour éviter les erreurs mathématiques (log ou puissance de nombres négatifs/nuls)
    w_safe = np.maximum(capitaux_finaux, 1e-2)
    
    if abs(gamma - 1.0) < 1e-5:
        # Utilité logarithmique pour gamma = 1
        utilite_esperee = np.mean(np.log(w_safe))
        cew = np.exp(utilite_esperee)
    else:
        # Utilité puissance pour gamma != 1
        utilite_esperee = np.mean((w_safe ** (1 - gamma)) / (1 - gamma))
        cew = (utilite_esperee * (1 - gamma)) ** (1 / (1 - gamma))
        
    return cew

def executer_analyse_alm(capitaux_finaux, mat_cap_retraite, dernier_salaire_vecteur, capital_cible, gamma=3.0):
    """
    Orchestre le calcul des 4 KPIs stochastiques.
    """
    # Hypothèse : le taux de remplacement est évalué sur la première année de retraite
    rente_an_1 = mat_cap_retraite[0, :] - mat_cap_retraite[1, :]
    taux_remp_initiaux = rente_an_1 / np.maximum(dernier_salaire_vecteur, 1.0)
    
    kpis = {
        "probabilite_ruine": calculer_probabilite_ruine(mat_cap_retraite),
        "expected_shortfall_5pct": calculer_expected_shortfall_passif(capitaux_finaux, capital_cible, alpha=0.05),
        "taux_remplacement": calculer_distribution_taux_remplacement(taux_remp_initiaux),
        "equivalent_certain": calculer_equivalent_certain(capitaux_finaux, gamma)
    }
    
    return kpis