"""
MODULE INFLATION VASICEK
==========================
Génère des trajectoires d'inflation stochastique via le modèle Vasicek.

Processus Vasicek pour l'inflation :
    dI(t) = κ(θ - I(t))dt + σ dW(t)
    
Où :
    I(t)    : taux d'inflation instantané
    κ       : vitesse de retour à la moyenne (mean reversion speed)
    θ       : inflation cible de long terme
    σ       : volatilité
    dW(t)   : mouvement brownien standard
"""

import numpy as np


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
        
    def simulate(self, nb_periods, nb_scenarios, dt=1/12, seed=None, frequency="auto"):
        """
        Simule des trajectoires d'inflation selon Vasicek (discrétisation d'Euler).
        
        Args:
            nb_periods   : Nombre de périodes (mois)
            nb_scenarios : Nombre de trajectoires parallèles
            dt           : Pas de temps en années (1/12 pour mensuel)
            seed         : Graine aléatoire pour reproductibilité
            frequency    : "auto" (défaut) ou "annual" ou "monthly"
                          Si "annual", convertit automatiquement les params annuels → mensuels
            
        Returns:
            np.array : (nb_periods, nb_scenarios) - taux d'inflation mensuels
        
        IMPORTANT : Si vous utilisez des paramètres annualisés (ex: calibration_default()),
                   assurez-vous que frequency="auto" ou "annual".
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Adapter les paramètres selon la fréquence
        kappa = self.kappa
        theta = self.theta
        sigma = self.sigma
        
        if frequency == "auto" and dt == 1/12:
            # On suppose que les paramètres sont annuels, on convertit en mensuels
            kappa, theta, sigma = self.annualize_to_monthly(kappa, theta, sigma)
        
        inflation = np.zeros((nb_periods, nb_scenarios))
        inflation[0, :] = self.inflation_init
        
        # Pré-calcul des facteurs (optimisation)
        drift_factor = kappa * dt
        diffusion_factor = sigma * np.sqrt(dt)
        
        for t in range(1, nb_periods):
            # Discrétisation d'Euler-Maruyama
            shock = np.random.randn(nb_scenarios)
            inflation[t, :] = (
                inflation[t-1, :] 
                + drift_factor * (theta - inflation[t-1, :])
                + diffusion_factor * shock
            )
        
        # Clamp pour éviter les valeurs négatives ou non-réalistes
        # Min: -2%, Max: 10% (limites raisonnables)
        inflation = np.clip(inflation, -0.02, 0.10)
        
        return inflation
    
    def to_cumulative_factor(self, inflation_monthly):
        """
        Convertit des taux d'inflation mensuels en facteurs cumulatifs d'actualisation.
        
        Utile pour déflatation a posteriori : capital_reel = capital_nominal / factor
        
        Args:
            inflation_monthly : np.array (nb_periods, nb_scenarios) - taux mensuels
            
        Returns:
            np.array : (nb_periods, nb_scenarios) - facteurs cumulatifs (1 + inflation annualisée)^années
        """
        nb_periods = inflation_monthly.shape[0]
        factors = np.ones_like(inflation_monthly)
        
        # Cumul annuel (tous les 12 mois)
        for t in range(1, nb_periods):
            mois_in_year = (t % 12) if t % 12 != 0 else 12
            if t % 12 == 0:  # Fin d'année
                inflation_annual = np.prod(1 + inflation_monthly[t-11:t+1, :], axis=0) - 1
                factors[t, :] = factors[t-1, :] * (1 + inflation_annual)
            else:
                factors[t, :] = factors[t-1, :]
        
        return factors
    
    def to_indexation_factor(self, inflation_monthly):
        """
        Convertit des taux d'inflation mensuels en facteurs d'indexation des revenus.
        
        Utile pour indexation des salaires/apports : salaire(t) = salaire_init * factor(t)
        
        Args:
            inflation_monthly : np.array (nb_periods, nb_scenarios) - taux mensuels
            
        Returns:
            np.array : (nb_periods, nb_scenarios) - facteurs cumulatifs d'indexation
        """
        # Produit cumulatif des (1 + taux mensuel)
        return np.cumprod(1 + inflation_monthly, axis=0)
    
    @staticmethod
    def calibration_default():
        """
        Retourne les paramètres Vasicek calibrés par défaut sur données historiques US.
        
        Basés sur l'inflation PCE (1990-2024) :
        - Moyenne : ~2.3%
        - Écart-type : ~1.1%
        - Autocorrélation (1 an) : ~0.75
        
        IMPORTANT : Les paramètres retournés ici sont ANNUALISÉS et doivent être
        convertis avant utilisation avec une fréquence différente (ex: mensuelle).
        
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
        
        Args:
            kappa_annual : float - vitesse annuelle
            theta_annual : float - cible annuelle
            sigma_annual : float - volatilité annuelle
        
        Returns:
            tuple : (kappa_monthly, theta_monthly, sigma_monthly)
        """
        # kappa ne change pas avec la fréquence (c'est une vitesse de retour à la moyenne)
        kappa_monthly = kappa_annual
        
        # theta doit être divisé par 12 (moyenne mensuelle)
        theta_monthly = theta_annual / 12.0
        
        # sigma doit être divisé par sqrt(12) pour la volatilité mensuelle
        sigma_monthly = sigma_annual / np.sqrt(12)
        
        return kappa_monthly, theta_monthly, sigma_monthly
