"""
MODULE GSE (Generalized Scenario Engine) - Enhanced Version
============================================================
Génère des scénarios économiques sophistiqués avec :
- Processus de retour à la moyenne (mean-reversion)
- Volatilité stochastique
- Régimes de marché (Bull/Bear/Normal)
- Corrélations dynamiques

Utilisé par la stratégie de Faleh pour l'optimisation stochastique.
"""

import numpy as np
from scipy.linalg import sqrtm


class EnhancedGSE:
    """
    Générateur de Scénarios Économiques avancé avec régimes de marché
    et volatilité stochastique (inspiré de Heston).
    """
    
    def __init__(self, mu_e, sigma_e, mu_b, sigma_b, corr_eb, 
                 mean_reversion_speed=0.15, vol_of_vol=0.3):
        """
        Args:
            mu_e: Rendement annuel espéré equity
            sigma_e: Volatilité annuelle equity
            mu_b: Rendement annuel espéré bonds
            sigma_b: Volatilité annuelle bonds
            corr_eb: Corrélation equity-bonds
            mean_reversion_speed: Vitesse de retour à la moyenne
            vol_of_vol: Volatilité de la volatilité (Heston)
        """
        self.mu_e = mu_e
        self.sigma_e = sigma_e
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        self.corr_eb = corr_eb
        self.kappa = mean_reversion_speed
        self.vol_of_vol = vol_of_vol
        
        # États de marché (transition probabilities)
        self.regime_probs = {
            'normal': 0.70,
            'bull': 0.15,
            'bear': 0.15
        }
        
        # Multiplicateurs de rendement par régime
        self.regime_multipliers = {
            'normal': {'mu': 1.0, 'sigma': 1.0},
            'bull': {'mu': 1.5, 'sigma': 0.8},
            'bear': {'mu': -0.5, 'sigma': 1.8}
        }
    
    def generate_scenarios(self, nb_periods, nb_scenarios, seed=None):
        """
        Génère des scénarios avec régimes de marché et volatilité stochastique.
        
        Returns:
            tuple: (r_equity, r_bonds, regimes, volatilities)
                - r_equity: (nb_periods, nb_scenarios)
                - r_bonds: (nb_periods, nb_scenarios)
                - regimes: (nb_periods, nb_scenarios) - 0:normal, 1:bull, 2:bear
                - volatilities: (nb_periods, nb_scenarios)
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = 1.0 / 12.0  # Pas mensuel
        
        # Initialisation
        r_eq = np.zeros((nb_periods, nb_scenarios))
        r_bd = np.zeros((nb_periods, nb_scenarios))
        regimes = np.zeros((nb_periods, nb_scenarios), dtype=int)
        vol_equity = np.ones((nb_periods, nb_scenarios)) * self.sigma_e
        
        # Matrice de corrélation
        s_e = self.sigma_e * np.sqrt(dt)
        s_b = self.sigma_b * np.sqrt(dt)
        cov_matrix = np.array([
            [s_e**2, self.corr_eb * s_e * s_b],
            [self.corr_eb * s_e * s_b, s_b**2]
        ])
        
        # Décomposition de Cholesky pour corrélation
        L = np.linalg.cholesky(cov_matrix)
        
        # Génération des scénarios
        for t in range(nb_periods):
            # 1. Détermination du régime de marché
            regime_draw = np.random.rand(nb_scenarios)
            regimes[t, :] = np.where(
                regime_draw < self.regime_probs['bear'], 2,
                np.where(regime_draw < self.regime_probs['bear'] + self.regime_probs['bull'], 1, 0)
            )
            
            # 2. Volatilité stochastique (processus de Heston simplifié)
            if t > 0:
                # Mean reversion vers la volatilité long-terme
                vol_shock = np.random.randn(nb_scenarios) * self.vol_of_vol * np.sqrt(dt)
                vol_equity[t, :] = vol_equity[t-1, :] + self.kappa * (self.sigma_e - vol_equity[t-1, :]) * dt + vol_shock
                vol_equity[t, :] = np.maximum(vol_equity[t, :], 0.05)  # Plancher à 5%
            
            # 3. Génération des chocs corrélés
            z = np.random.randn(2, nb_scenarios)
            chocs = L @ z  # Applique la corrélation
            
            # 4. Application des régimes
            for s in range(nb_scenarios):
                regime = regimes[t, s]
                regime_name = ['normal', 'bull', 'bear'][regime]
                mult = self.regime_multipliers[regime_name]
                
                # Rendement equity avec régime et vol stochastique
                mu_adj = self.mu_e * mult['mu']
                sigma_adj = vol_equity[t, s] * mult['sigma']
                
                r_eq[t, s] = (mu_adj * dt - 0.5 * (sigma_adj * np.sqrt(dt))**2 + 
                              sigma_adj * np.sqrt(dt) * chocs[0, s])
                
                # Rendement bonds (moins affecté par les régimes)
                r_bd[t, s] = (self.mu_b * dt - 0.5 * (s_b)**2 + chocs[1, s])
        
        return r_eq, r_bd, regimes, vol_equity
    
    def generate_scenarios_simple(self, nb_periods, nb_scenarios, seed=None):
        """
        Version simplifiée sans régimes (pour compatibilité avec le framework existant).
        
        Returns:
            tuple: (r_equity, r_bonds)
        """
        r_eq, r_bd, _, _ = self.generate_scenarios(nb_periods, nb_scenarios, seed)
        return r_eq, r_bd


class MarkovRegimeSwitching:
    """
    Modèle de changement de régime markovien pour améliorer le GSE.
    """
    
    def __init__(self, nb_regimes=3):
        """
        Args:
            nb_regimes: Nombre de régimes (défaut: 3 pour Normal/Bull/Bear)
        """
        self.nb_regimes = nb_regimes
        
        # Matrice de transition par défaut (persistance des régimes)
        self.transition_matrix = np.array([
            [0.85, 0.10, 0.05],  # Normal -> {Normal, Bull, Bear}
            [0.20, 0.70, 0.10],  # Bull -> ...
            [0.15, 0.05, 0.80]   # Bear -> ...
        ])
    
    def simulate_regimes(self, nb_periods, nb_scenarios, initial_regime=0):
        """
        Simule l'évolution des régimes selon une chaîne de Markov.
        
        Returns:
            np.array: (nb_periods, nb_scenarios) - indices de régimes
        """
        regimes = np.zeros((nb_periods, nb_scenarios), dtype=int)
        regimes[0, :] = initial_regime
        
        for t in range(1, nb_periods):
            for s in range(nb_scenarios):
                current_regime = regimes[t-1, s]
                # Tirage selon les probabilités de transition
                draw = np.random.rand()
                cumsum = np.cumsum(self.transition_matrix[current_regime, :])
                regimes[t, s] = np.searchsorted(cumsum, draw)
        
        return regimes
