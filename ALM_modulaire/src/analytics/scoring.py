import numpy as np

def calculer_scores_strategies(dictionnaire_kpis_bruts, profil_config):
    """
    Normalise les KPIs de plusieurs stratégies et calcule le score pondéré
    selon le profil de décumulation via normalisation Min-Max.
    
    Args:
        dictionnaire_kpis_bruts (dict): {nom_strat: {kpi: valeur, ...}}
        profil_config (dict): Dictionnaire contenant les poids ('poids_kpi').
        
    Returns:
        tuple: (Dictionnaire des scores, Nom de la meilleure stratégie)
    """
    strategies = list(dictionnaire_kpis_bruts.keys())
    poids = profil_config["poids_kpi"]
    
    # 1. Extraction vectorielle des métriques
    ruines = np.array([dictionnaire_kpis_bruts[s]['prob_ruine'] for s in strategies])
    shortfalls = np.array([dictionnaire_kpis_bruts[s]['expected_shortfall'] for s in strategies])
    cews = np.array([dictionnaire_kpis_bruts[s]['equivalent_certain'] for s in strategies])
    transmissions = np.array([dictionnaire_kpis_bruts[s]['transmission'] for s in strategies])
    
    # 2. Fonctions de normalisation Min-Max
    def normaliser_a_minimiser(vecteur):
        """Pour les risques (Ruine, Shortfall) : le plus petit a le score de 1"""
        ptp = np.ptp(vecteur)
        return (np.max(vecteur) - vecteur) / ptp if ptp > 1e-8 else np.ones_like(vecteur)
        
    def normaliser_a_maximiser(vecteur):
        """Pour les gains (CEW, Transmission) : le plus grand a le score de 1"""
        ptp = np.ptp(vecteur)
        return (vecteur - np.min(vecteur)) / ptp if ptp > 1e-8 else np.ones_like(vecteur)

    # 3. Projection sur [0, 1]
    norm_ruines = normaliser_a_minimiser(ruines)
    norm_shortfalls = normaliser_a_minimiser(shortfalls)
    norm_cews = normaliser_a_maximiser(cews)
    norm_transmissions = normaliser_a_maximiser(transmissions)
    
    # 4. Calcul des scores par produit scalaire
    scores = {}
    for i, strat in enumerate(strategies):
        score_strat = (
            norm_ruines[i] * poids["prob_ruine"] +
            norm_shortfalls[i] * poids["expected_shortfall"] +
            norm_cews[i] * poids["equivalent_certain"] +
            norm_transmissions[i] * poids["transmission"]
        )
        scores[strat] = score_strat
        
    meilleure_strategie = max(scores, key=scores.get)
    
    return scores, meilleure_strategie