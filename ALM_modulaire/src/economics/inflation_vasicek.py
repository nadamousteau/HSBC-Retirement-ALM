"""
MODULE INFLATION VASICEK
==========================
Génère des trajectoires d'inflation stochastique via le modèle Vasicek.

Processus Vasicek pour l'inflation :
    dI(t) = κ(θ - I(t))dt + σ dW(t)
    
Où :
    I(t)    : taux d'inflation instantané (annualisé)
    κ       : vitesse de retour à la moyenne (mean reversion speed)
    θ       : inflation cible de long terme (annualisée)
    σ       : volatilité (annualisée)
    dW(t)   : mouvement brownien standard
"""

import numpy as np
import warnings

class VasicekInflation:
    """
    Modèle Vasicek pour simuler des chemins d'inflation stochastique.
    
    Utilisation typique :
        inflation_sim = VasicekInflation(kappa=0.15, theta=0.02, sigma=0.01)
        inflation_paths = inflation_sim.simulate(nb_periods=300, nb_scenarios=1000)
    """
    
    def __init__(self, kappa=0.15, theta=0.02, sigma=0.01, inflation_init=None):
        """
        Args:
            kappa          : Vitesse de retour à la moyenne (typique : 0.05 à 0.30)
            theta          : Inflation cible de long terme (typique : 0.02 = 2%)
            sigma          : Volatilité de l'inflation (typique : 0.005 à 0.015)
            inflation_init : Inflation initiale (défaut : theta)
        """
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.inflation_init = inflation_init if inflation_init is not None else theta
        
    def simulate(self, nb_periods, nb_scenarios, dt=1/12, rng=None, frequency="auto"):
        """
        Simule des trajectoires d'inflation selon Vasicek (discrétisation d'Euler).

        Args:
            nb_periods   : Nombre de périodes (mois)
            nb_scenarios : Nombre de trajectoires parallèles
            dt           : Pas de temps en années (1/12 pour mensuel)
            rng          : np.random.Generator ou None (entropie OS si None)
            frequency    : "auto" (défaut) ou "annual" ou "monthly"
                           NOTE : Ce paramètre est conservé pour compatibilité mais 
                           ignoré pour le calcul stochastique afin d'éviter les erreurs 
                           d'échelle sur sigma et theta.

        Returns:
            np.array : (nb_periods, nb_scenarios) - taux d'inflation annualisés simulés

        IMPORTANT : La simulation conserve les paramètres annuels. L'ajustement 
                    temporel est géré par dt et sqrt(dt) dans le schéma d'Euler.
        """
        if rng is None:
            rng = np.random.default_rng()

        # Correction : On utilise directement les attributs de classe (annuels)
        # pour garantir la cohérence de l'EDS.
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma

        # La variable d'état 'inflation' représente le taux annualisé à l'instant t
        inflation = np.zeros((nb_periods, nb_scenarios))
        inflation[0, :] = self.inflation_init

        # Pré-calcul des facteurs (optimisation numérique)
        drift_factor = kappa * dt
        diffusion_factor = sigma * np.sqrt(dt)

        for t in range(1, nb_periods):
            # Discrétisation d'Euler-Maruyama : dI = kappa(theta - I)dt + sigma*sqrt(dt)*Z
            shock = rng.standard_normal(nb_scenarios)
            inflation[t, :] = (
                inflation[t-1, :]
                + drift_factor * (theta - inflation[t-1, :])
                + diffusion_factor * shock
            )

        return inflation
    
    def to_cumulative_factor(self, inflation_annualized, dt=1/12):
        """
        Convertit des taux d'inflation annualisés en facteurs cumulatifs.
    
        Utile pour déflation a posteriori : capital_reel = capital_nominal / factor(T)
    
        Args:
            inflation_annualized : np.array (nb_periods, nb_scenarios) - taux annuels
            dt                   : Pas de temps (défaut 1/12 pour passer en mensuel)
        Returns:
            np.array (nb_periods, nb_scenarios) — facteur cumulatif mois par mois
        """
        return np.cumprod(1 + inflation_annualized * dt, axis=0)
    
    def to_indexation_factor(self, inflation_annualized, dt=1/12):
        """
        Convertit des taux d'inflation annualisés en facteurs d'indexation des revenus.
        
        Utile pour indexation des salaires/apports : salaire(t) = salaire_init * factor(t)
        
        Args:
            inflation_annualized : np.array (nb_periods, nb_scenarios) - taux annuels
            dt                   : Pas de temps
            
        Returns:
            np.array : (nb_periods, nb_scenarios) - facteurs cumulatifs d'indexation
        """
        return np.cumprod(1 + inflation_annualized * dt, axis=0)
    
    @staticmethod
    def calibration_default():
        """
        Retourne les paramètres Vasicek calibrés par défaut sur données historiques US.
        
        Basés sur l'inflation PCE (1990-2024) :
        - Moyenne : ~2.3%
        - Écart-type : ~1.1%
        - Autocorrélation (1 an) : ~0.75
        
        IMPORTANT : Les paramètres retournés ici sont ANNUALISÉS.
        
        Returns:
            dict : {'kappa': float, 'theta': float, 'sigma': float} (ANNUELS)
        """
        return {
            'kappa': 0.15,      # Vitesse de retour à la moyenne modérée (demi-vie ~4.6 ans)
            'theta': 0.023,     # Cible d'inflation long terme (2.3% ANNUEL)
            'sigma': 0.011      # Volatilité (~1.1% ANNUALISÉE)
        }
    
    @staticmethod
    def annualize_to_monthly(kappa_annual, theta_annual, sigma_annual):
        """
        Convertit les paramètres Vasicek d'une fréquence annuelle à mensuelle.
        
        /!\ ATTENTION : Cette fonction est maintenue pour compatibilité externe. 
        Elle ne doit pas être utilisée pour modifier les paramètres avant simulate().
        
        Args:
            kappa_annual : float - vitesse annuelle
            theta_annual : float - cible annuelle
            sigma_annual : float - volatilité annuelle
        
        Returns:
            tuple : (kappa_monthly, theta_monthly, sigma_monthly)
        """
        warnings.warn(
            "annualize_to_monthly ne doit être utilisée que pour l'affichage ou des "
            "calculs statiques, pas pour la simulation stochastique.", 
            DeprecationWarning
        )
        
        kappa_monthly = kappa_annual
        theta_monthly = theta_annual / 12.0
        sigma_monthly = sigma_annual / np.sqrt(12)
        
        return kappa_monthly, theta_monthly, sigma_monthly
