import numpy as np

def generer_survie_stochastique(nb_sims, horizon_max=40):
    """
    Modélise le risque de longévité via une distribution stochastique.
    Utilise une approximation normale pour l'espérance de vie résiduelle à 65 ans.
    (Dans un environnement de production, ceci est remplacé par l'inversion 
    de la fonction de survie d'une table réglementaire type TGH05/TGF05).
    """
    # Espérance résiduelle ~ 21 ans, écart-type ~ 7 ans
    annees_a_vivre = np.random.normal(loc=21.0, scale=7.0, size=nb_sims)
    # Contrainte mathématique : décès entre l'année 1 et l'horizon maximum
    durees_vie = np.clip(np.round(annees_a_vivre), 1, horizon_max).astype(int)
    return durees_vie

def actualiser_capital_intra_annuel(capital, retrait_annuel, rendement_annuel):
    """
    Calcule la valeur du capital après une année de rendements et de prélèvements fractionnés.
    Utilise le taux équivalent mensuel pour refléter l'amortissement continu.
    """
    # Taux équivalent mensuel : (1+r_a)^(1/12) - 1
    rm = (1.0 + rendement_annuel)**(1.0/12.0) - 1.0
    
    # Protection algorithmique contre la division par zéro si rm est nul
    rm = np.where(np.abs(rm) < 1e-6, 1e-6, rm)
    
    # Valeur acquise de la rente (versements en début de mois)
    valeur_retraits = (retrait_annuel / 12.0) * (((1.0 + rm)**12.0 - 1.0) / rm) * (1.0 + rm)
    
    nouveau_capital = capital * (1.0 + rendement_annuel) - valeur_retraits
    return np.maximum(nouveau_capital, 0.0)

def simuler_decumulation(capitaux_finaux, dernier_salaire_vecteur, rendements_matrice, 
                         inflation_matrice, profil_config, 
                         horizon_max=40):
    """
    Moteur de décumulation ALM. Exécute la stratégie de retrait stochastique.
    
    Args:
        capitaux_finaux (np.ndarray): Vecteur (N,) des capitaux à la date de retraite.
        dernier_salaire_vecteur (np.ndarray): Vecteur (N,) des salaires terminaux.
        rendements_matrice (np.ndarray ou float): Rendements du portefeuille en retraite (T x N).
        inflation_matrice (np.ndarray ou float): Inflation annuelle (T x N).
        profil_config (dict): Paramètres de décumulation (issus de settings.PROFILS_DECUMULATION).
        type_strategie (str): "MONTANT_FIXE", "POURCENTAGE_FIXE", ou "GUYTON_KLINGER".
        horizon_max (int): Durée maximale de modélisation (ex: 40 ans, jusqu'à 105 ans).
        
    Returns:
        tuple: (matrice_capitaux, matrice_rentes, vecteur_durees_vie)
    """
    nb_sims = len(capitaux_finaux)
    type_strategie = profil_config.get("type_strategie", "MONTANT_FIXE")
    
    # Formatage des matrices de marché (Broadcasting si scalaires fournis)
    if np.isscalar(rendements_matrice):
        rendements_matrice = np.full((horizon_max, nb_sims), rendements_matrice)
    if np.isscalar(inflation_matrice):
        inflation_matrice = np.full((horizon_max, nb_sims), inflation_matrice)

    # Initialisation des tenseurs de stockage
    mat_cap_retraite = np.zeros((horizon_max + 1, nb_sims))
    mat_rentes = np.zeros((horizon_max, nb_sims))
    
    # État initial
    mat_cap_retraite[0, :] = capitaux_finaux.copy()
    cap_courant = capitaux_finaux.copy()
    
    # Paramètres de la stratégie
    taux_initial = profil_config.get("taux_retrait_initial", 0.04)
    rente_courante = cap_courant * taux_initial

    # Génération stochastique de la mortalité (Risque de longévité)
    durees_vie = generer_survie_stochastique(nb_sims, horizon_max)

    for t in range(horizon_max):
        # 1. Mise à jour de la rente cible selon la stratégie
        if type_strategie == "POURCENTAGE_FIXE":
            rente_courante = cap_courant * taux_initial
            
        elif type_strategie == "MONTANT_FIXE":
            if t > 0:
                rente_courante = rente_courante * (1.0 + inflation_matrice[t-1, :])
                
        elif type_strategie == "GUYTON_KLINGER":
            if t > 0:
                rente_cible = rente_courante * (1.0 + inflation_matrice[t-1, :])
                
                # Évaluation du taux de retrait instantané
                taux_instantané = np.divide(rente_cible, cap_courant, out=np.zeros_like(rente_cible), where=(cap_courant > 1e-4))
                
                # Règle de Préservation du Capital : Si le retrait dépasse la limite de tolérance à la hausse
                limite_haute = taux_initial * (1.0 + profil_config["gk_seuil_baisse"])
                condition_baisse = taux_instantané > limite_haute
                
                # Règle de Prospérité : Si le portefeuille a trop crû par rapport aux retraits
                limite_basse = taux_initial * (1.0 - profil_config["gk_seuil_hausse"])
                condition_hausse = (taux_instantané < limite_basse) & (cap_courant > 0)
                
                # Application des ajustements
                ajustement = profil_config["gk_ajustement"]
                rente_cible = np.where(condition_baisse, rente_cible * (1.0 - ajustement), rente_cible)
                rente_cible = np.where(condition_hausse, rente_cible * (1.0 + ajustement), rente_cible)
                
                rente_courante = rente_cible

        # 2. Contrainte de solvabilité (On ne peut pas retirer plus que le capital disponible)
        rente_effective = np.minimum(rente_courante, cap_courant)
        mat_rentes[t, :] = rente_effective
        
        # 3. Évolution stochastique du capital (Actualisation exacte intra-annuelle)
        cap_courant = actualiser_capital_intra_annuel(cap_courant, rente_effective, rendements_matrice[t, :])
        
        # Le capital tombe définitivement à zéro pour les individus décédés
        # (Hypothèse de liquidation de la succession à l'année du décès)
        individus_vivants = (durees_vie > t)
        cap_courant = np.where(individus_vivants, cap_courant, 0.0)
        
        mat_cap_retraite[t+1, :] = cap_courant

    return mat_cap_retraite, mat_rentes, durees_vie