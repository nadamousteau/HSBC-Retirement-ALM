
import numpy as np
from config import settings

class HumanCapitalCurve:
    """
    Modélisation ALM de la trajectoire du capital humain et des contributions retraite.
    Intègre une diffusion stochastique hétérogène des salaires avec élasticité dynamique à l'inflation.
    """

    def __init__(
        self, 
        age_depart: int, 
        salaire_depart: float, 
        taux_apport_depart: float, 
        mode_investissement: str, 
        age_retraite: int,
        matrice_inflation: np.ndarray,
        n_simulations: int 
    ):
        # Variables d'état de l'individu
        self.age_depart = age_depart
        self.salaire_depart = salaire_depart
        self.contrib_depart = taux_apport_depart 
        self.mode_investissement = mode_investissement
        self.age_retraite = age_retraite
        self.matrice_inflation = matrice_inflation
        self.n_simulations = n_simulations
        
        # Paramètres structurels du modèle stochastique salarial
        self.lambda_mu = 0.03                 # Espérance du potentiel de croissance
        self.lambda_sigma = 0.02              # Écart-type du potentiel de croissance
        self.alpha_amortissement = 0.35       # Facteur d'amortissement de fin de carrière
        self.volatilite_idiosyncratique = 0.015 # Bruit résiduel
        self.beta_max = 0.6                   # Élasticité max à l'inflation
        self.beta_min = 0.0                   # Élasticité min à l'inflation
        self.salaire_seuil = 3000.0           # Pivot de désindexation
        self.age_reference = 23               # Âge d'entrée théorique sur le marché

    def matrice_salaire(self) -> np.ndarray:
        """
        Exécute la simulation de Monte-Carlo et retourne une matrice (3, n_mois).
        Ligne 0 : Médiane (P50)
        Ligne 1 : P5 (Défavorable)
        Ligne 2 : P95 (Favorable)
        """
        n_annees = self.age_retraite - self.age_depart
        n_mois = n_annees * 12
        
        # Validation de la dimension d'entrée (mensuelle)
        if len(self.matrice_inflation) != n_mois:
            raise ValueError(f"La matrice d'inflation doit contenir {n_mois} points mensuels.")

        # Extraction de l'inflation annuelle (un point tous les 12 mois)
        inflation_annuelle = self.matrice_inflation[0::12]

        # Initialisation
        ages_annuels = np.arange(self.age_depart, self.age_retraite)
        salaires_simules = np.zeros((self.n_simulations, n_annees))
        salaires_simules[:, 0] = self.salaire_depart

        # Hétérogénéité du capital humain
        vecteur_lambda = np.maximum(0.0, np.random.normal(
            self.lambda_mu, self.lambda_sigma, self.n_simulations
        ))

        # Boucle de diffusion annuelle
        for k in range(1, n_annees):
            age_precedent = ages_annuels[k-1]
            salaires_precedents = salaires_simules[:, k-1]
            inflation_precedente = inflation_annuelle[k-1]
            
            annees_experience = max(0, age_precedent - self.age_reference)
            
            # Dérive et élasticité
            derive_carriere = vecteur_lambda * np.exp(-self.alpha_amortissement * annees_experience)
            facteur_desindexation = (self.salaire_seuil / salaires_precedents)
            beta_dynamique = np.clip(self.beta_max * facteur_desindexation, self.beta_min, self.beta_max)
            
            # Choc stochastique
            Z = np.random.standard_normal(self.n_simulations)
            choc_aleatoire = self.volatilite_idiosyncratique * Z
            
            # Transition avec plancher de rigidité
            taux_revalorisation = np.maximum(-0.02, (beta_dynamique * inflation_precedente) + derive_carriere + choc_aleatoire)
            salaires_simules[:, k] = salaires_precedents * (1.0 + taux_revalorisation)

        # Calcul des centiles transversaux
        p50_annuel = np.percentile(salaires_simules, 50, axis=0)
        p5_annuel = np.percentile(salaires_simules, 5, axis=0)
        p95_annuel = np.percentile(salaires_simules, 95, axis=0)

        # Construction de la matrice finale par empilement vertical et répétition mensuelle
        matrice_salaire = np.vstack([
            np.repeat(p50_annuel, 12),
            np.repeat(p5_annuel, 12),
            np.repeat(p95_annuel, 12)
        ])

        return matrice_salaire
    
    def precalculer_parametres_apport_exponentiel(self, s_init: float, s_max: float, duree_totale: float):
        """
        Calcule les paramètres de la courbe quadratique d'apport.
        """
        ratio = s_max / s_init
        facteur = ratio ** 1.5
        app_init = s_init * self.contrib_depart
        app_max = app_init * facteur
        
        # Calcul du temps de pic (t_pic) via la dynamique de progression
        s_cible = s_init + (s_max - s_init) * 0.935
        
        if s_cible >= s_max:
            t_pic = duree_totale
        else:
            val_log = 1 - min((s_cible - s_init) / max(1.0, (s_max - s_init)), 0.9999)
            t_pic = -np.log(val_log) / 0.10
            
        return app_init, app_max, np.clip(t_pic, 0, duree_totale)

    def matrice_apport(self, matrice_salaires: np.ndarray) -> np.ndarray:
        """
        Génère la matrice des apports mensuels (3, n_mois). Les lignes correspondes aux memes scenarios que les salaires
        Basé sur une fonction quadratique centrée sur le pic de carrière.
        """
        n_scenarios, n_mois = matrice_salaires.shape
        duree_ans = n_mois / 12
        matrice_apports = np.zeros_like(matrice_salaires)
        
        t_mois = np.arange(n_mois)
        t_annees = t_mois / 12

        for i in range(n_scenarios):
            s_traj = matrice_salaires[i]
            app_init, app_max, t_pic = self.precalculer_parametres_apport_exponentiel(
                s_traj[0], s_traj[-1], duree_ans
            )
            
            if t_pic <= 0:
                matrice_apports[i, :] = app_init
                continue
                
            # Coefficient de la parabole : a = (app_init - app_max) / (t_pic^2)
            # f(t) = a * (t - t_pic)^2 + app_max
            a = (app_init - app_max) / (t_pic**2)
            apports = a * (t_annees - t_pic)**2 + app_max
            
            matrice_apports[i, :] = np.maximum(apports, 0.0)

        return matrice_apports


# =============================================================================
# Fonctions utilitaires de niveau module (utilisées par engine/core.py et engine/gbi_core.py)
# =============================================================================

def precalculer_parametres_apport_exponentiel(s_init: float, s_max: float, duree_totale: float):
    """
    Calcule les paramètres (app_init, app_max, t_pic) de la courbe quadratique d'apport.

    Args:
        s_init      : Salaire initial.
        s_max       : Salaire maximal cible.
        duree_totale: Durée totale d'accumulation en années.

    Returns:
        app_init (float), app_max (float), t_pic (float)
    """
    ratio = s_max / s_init
    facteur = ratio ** settings.GAMMA_ELASTICITE
    app_init = s_init * settings.TAUX_APPORT_BASE
    app_max = app_init * facteur

    s_cible = s_init + (s_max - s_init) * settings.SEUIL_MATURITE

    if s_cible >= s_max:
        t_pic = duree_totale
    else:
        val_log = 1 - min((s_cible - s_init) / max(1.0, (s_max - s_init)), 0.9999)
        t_pic = -np.log(val_log) / settings.VITESSE_PROGRESSION

    return app_init, app_max, np.clip(t_pic, 0, duree_totale)


def calculer_apport_exponentiel(t_annees: float, app_init: float, app_max: float, t_pic: float) -> float:
    """
    Calcule l'apport mensuel à un instant t (en années) selon la courbe quadratique.

    Args:
        t_annees: Temps écoulé depuis le début de l'accumulation (en années).
        app_init: Apport initial.
        app_max : Apport maximal.
        t_pic   : Temps du pic d'apport (en années).

    Returns:
        Apport mensuel (float >= 0).
    """
    if t_pic <= 0:
        return app_init
    a = (app_init - app_max) / (t_pic ** 2)
    return max(0.0, a * (t_annees - t_pic) ** 2 + app_max)


def estimer_salaire_saturation(t_annees: float, s_init: float, s_max: float) -> float:
    """
    Estime le salaire à un instant t par saturation exponentielle.

    Args:
        t_annees: Temps écoulé depuis le début de l'accumulation (en années).
        s_init  : Salaire initial.
        s_max   : Salaire maximal cible.

    Returns:
        Salaire estimé (float).
    """
    return s_max - (s_max - s_init) * np.exp(-settings.VITESSE_PROGRESSION * t_annees)
