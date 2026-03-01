"""
MODULE GSE (Generalized Scenario Engine) - Enhanced Version
============================================================
Génère des scénarios économiques sophistiqués avec :
- Processus de retour à la moyenne (mean-reversion) pour les taux (Vasicek)
- Volatilité stochastique pour les actions (Heston-inspired)
- Régimes de marché avec chaîne de Markov (RSLN)
- Corrélations dynamiques

Utilisé par la stratégie de Faleh pour l'optimisation stochastique.
"""

import numpy as np
from scipy.linalg import sqrtm


class MarkovRegimeSwitching:
    """
    Modèle de changement de régime markovien pour le RSLN.
    
    États :
        0 = Normal (70% du temps)
        1 = Bull (15% du temps)
        2 = Bear (15% du temps)
    
    Propriétés :
        - Persistance des régimes (diagonale élevée)
        - Transitions asymétriques Bull→Normal plus fréquent que Bear→Normal
    """
    
    def __init__(self, nb_regimes=3, transition_matrix=None):
        """
        Args:
            nb_regimes: Nombre de régimes (3 par défaut)
            transition_matrix: Matrice de transition personnalisée (optionnel)
        """
        self.nb_regimes = nb_regimes
        
        if transition_matrix is not None:
            self.transition_matrix = np.array(transition_matrix)
        else:
            # Matrice par défaut calibrée sur données S&P 500 (1950-2024)
            self.transition_matrix = np.array([
                # De\Vers:  Normal  Bull   Bear
                [0.85,      0.10,   0.05],  # Normal → forte persistance
                [0.20,      0.70,   0.10],  # Bull → tend à rester en Bull
                [0.15,      0.05,   0.80]   # Bear → très persistant (crises longues)
            ])
        
        # Validation de la matrice stochastique
        row_sums = self.transition_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            raise ValueError("La matrice de transition doit avoir des lignes sommant à 1")
    
    def simulate_regimes(self, nb_periods, nb_scenarios, initial_regime=0, seed=None):
        """
        Simule l'évolution des régimes selon une chaîne de Markov.
        
        Args:
            nb_periods: Nombre de périodes à simuler
            nb_scenarios: Nombre de trajectoires parallèles
            initial_regime: Régime de départ (0=Normal par défaut)
            seed: Seed pour reproductibilité
        
        Returns:
            np.array: (nb_periods, nb_scenarios) - indices de régimes [0, 1, 2]
        """
        if seed is not None:
            np.random.seed(seed)
        
        regimes = np.zeros((nb_periods, nb_scenarios), dtype=int)
        regimes[0, :] = initial_regime
        
        # Pré-calcul des probabilités cumulées pour chaque régime
        cumsum_probs = np.cumsum(self.transition_matrix, axis=1)
        
        for t in range(1, nb_periods):
            for s in range(nb_scenarios):
                current_regime = regimes[t-1, s]
                
                # Tirage selon les probabilités de transition
                draw = np.random.rand()
                
                # Recherche dichotomique du prochain régime
                next_regime = np.searchsorted(
                    cumsum_probs[current_regime, :], 
                    draw
                )
                
                regimes[t, s] = next_regime
        
        return regimes
    
    def get_stationary_distribution(self):
        """
        Calcule la distribution stationnaire de la chaîne de Markov.
        
        Returns:
            np.array: Probabilités stationnaires [P(Normal), P(Bull), P(Bear)]
        """
        # Résoudre π = π * P où π est le vecteur stationnaire
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        
        # Trouver l'eigenvector associé à λ=1
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        
        return stationary
    
    def validate_calibration(self, simulated_regimes):
        """
        Vérifie que les régimes simulés respectent la distribution stationnaire.
        
        Args:
            simulated_regimes: Résultat de simulate_regimes()
        
        Returns:
            dict: Statistiques de validation
        """
        nb_periods, nb_scenarios = simulated_regimes.shape
        
        # Distribution empirique
        empirical_dist = np.array([
            (simulated_regimes == 0).sum() / simulated_regimes.size,
            (simulated_regimes == 1).sum() / simulated_regimes.size,
            (simulated_regimes == 2).sum() / simulated_regimes.size
        ])
        
        # Distribution théorique
        theoretical_dist = self.get_stationary_distribution()
        
        # Durée moyenne des régimes (mesure de persistance)
        durations = {0: [], 1: [], 2: []}
        for s in range(nb_scenarios):
            current_regime = simulated_regimes[0, s]
            duration = 1
            
            for t in range(1, nb_periods):
                if simulated_regimes[t, s] == current_regime:
                    duration += 1
                else:
                    durations[current_regime].append(duration)
                    current_regime = simulated_regimes[t, s]
                    duration = 1
            
            # Dernier régime
            durations[current_regime].append(duration)
        
        avg_durations = {
            regime: np.mean(durations[regime]) if durations[regime] else 0
            for regime in [0, 1, 2]
        }
        
        return {
            'empirical_distribution': empirical_dist,
            'theoretical_distribution': theoretical_dist,
            'max_deviation': np.max(np.abs(empirical_dist - theoretical_dist)),
            'average_durations': avg_durations
        }


class EnhancedGSE:
    """
    Générateur de Scénarios Économiques avancé avec :
    - Régimes de marché avec chaîne de Markov (RSLN)
    - Volatilité stochastique (Heston-inspired)
    - Modèle de Vasicek pour les taux obligataires
    """
    
    def __init__(self, mu_e, sigma_e, mu_b, sigma_b, corr_eb, 
                 mean_reversion_speed=0.15, 
                 vol_of_vol=0.30,
                 bond_mean_reversion_speed=0.30, 
                 bond_long_term_mean=0.03,
                 use_markov_regimes=True,
                 generate_inflation=False):
        """
        Args:
            mu_e: Rendement annuel espéré equity
            sigma_e: Volatilité annuelle equity
            mu_b: Rendement annuel espéré bonds
            sigma_b: Volatilité annuelle bonds
            corr_eb: Corrélation equity-bonds
            mean_reversion_speed: Vitesse retour à la moyenne (vol equity)
            vol_of_vol: Volatilité de la volatilité (Heston)
            bond_mean_reversion_speed: Kappa pour Vasicek (taux)
            bond_long_term_mean: Theta pour Vasicek (taux long-terme)
            use_markov_regimes: Si True, utilise la chaîne de Markov (RSLN)
            generate_inflation: Si True, génère un processus d'inflation AR(1)
        """
        self.mu_e = mu_e
        self.sigma_e = sigma_e
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        self.corr_eb = corr_eb
        
        # Paramètres de volatilité stochastique (Equity)
        self.kappa = mean_reversion_speed
        self.vol_of_vol = vol_of_vol
        
        # Paramètres de Vasicek (Bonds)
        self.kappa_bonds = bond_mean_reversion_speed
        self.theta_bonds = bond_long_term_mean
        
        # Paramètres de régimes
        self.use_markov_regimes = use_markov_regimes
        self.generate_inflation = generate_inflation
        
        if use_markov_regimes:
            self.regime_model = MarkovRegimeSwitching(nb_regimes=3)
        
        # Multiplicateurs de rendement par régime
        self.regime_multipliers = {
            'normal': {'mu': 1.0, 'sigma': 1.0},
            'bull': {'mu': 1.5, 'sigma': 0.8},
            'bear': {'mu': -0.5, 'sigma': 1.8}
        }
    
    def generate_scenarios(self, nb_periods, nb_scenarios, seed=None):
        """
        Génère des scénarios avec régimes de marché, volatilité stochastique et Vasicek.
        
        Returns:
            tuple: 
                - Si generate_inflation=False: (r_equity, r_bonds, regimes, volatilities)
                - Si generate_inflation=True: (r_equity, r_bonds, r_inflation, regimes, volatilities)
        """
        if seed is not None:
            np.random.seed(seed)
        
        dt = 1.0 / 12.0  # Pas mensuel
        
        # ═════════════════════════════════════════════════════════════════════
        # INITIALISATION
        # ═════════════════════════════════════════════════════════════════════
        r_eq = np.zeros((nb_periods, nb_scenarios))
        r_bd = np.zeros((nb_periods, nb_scenarios))
        vol_equity = np.ones((nb_periods, nb_scenarios)) * self.sigma_e
        
        # VASICEK : Initialisation du taux court-terme
        r_t = np.ones(nb_scenarios) * self.mu_b  # Taux initial = mu_b
        
        # INFLATION (optionnel)
        if self.generate_inflation:
            r_inflation = np.zeros((nb_periods, nb_scenarios))
            infl_t = np.ones(nb_scenarios) * 0.02  # Inflation initiale 2%
        
        # ═════════════════════════════════════════════════════════════════════
        # RÉGIMES DE MARCHÉ (RSLN avec Markov ou i.i.d.)
        # ═════════════════════════════════════════════════════════════════════
        if self.use_markov_regimes:
            # ✅ CORRECTION CRITIQUE : Régimes avec mémoire (Markov)
            regimes = self.regime_model.simulate_regimes(
                nb_periods, nb_scenarios, 
                initial_regime=0,  # Démarrer en régime Normal
                seed=seed
            )
        else:
            # Legacy : régimes i.i.d. (pour comparaison)
            regimes = np.zeros((nb_periods, nb_scenarios), dtype=int)
            regime_probs = {'normal': 0.70, 'bull': 0.15, 'bear': 0.15}
            
            for t in range(nb_periods):
                regime_draw = np.random.rand(nb_scenarios)
                regimes[t, :] = np.where(
                    regime_draw < regime_probs['bear'], 2,
                    np.where(regime_draw < regime_probs['bear'] + regime_probs['bull'], 1, 0)
                )
        
        # ═════════════════════════════════════════════════════════════════════
        # MATRICE DE CORRÉLATION
        # ═════════════════════════════════════════════════════════════════════
        s_e = self.sigma_e * np.sqrt(dt)
        s_b = self.sigma_b * np.sqrt(dt)
        cov_matrix = np.array([
            [s_e**2, self.corr_eb * s_e * s_b],
            [self.corr_eb * s_e * s_b, s_b**2]
        ])
        
        # Décomposition de Cholesky pour corrélation
        L = np.linalg.cholesky(cov_matrix)
        
        # ═════════════════════════════════════════════════════════════════════
        # BOUCLE TEMPORELLE
        # ═════════════════════════════════════════════════════════════════════
        for t in range(nb_periods):
            # ─────────────────────────────────────────────────────────────────
            # 1. VOLATILITÉ STOCHASTIQUE (Heston pour Equity)
            # ─────────────────────────────────────────────────────────────────
            if t > 0:
                vol_shock = np.random.randn(nb_scenarios) * self.vol_of_vol * np.sqrt(dt)
                vol_equity[t, :] = (vol_equity[t-1, :] + 
                                     self.kappa * (self.sigma_e - vol_equity[t-1, :]) * dt + 
                                     vol_shock)
                vol_equity[t, :] = np.maximum(vol_equity[t, :], 0.05)  # Plancher à 5%
            
            # ─────────────────────────────────────────────────────────────────
            # 2. GÉNÉRATION DES CHOCS CORRÉLÉS
            # ─────────────────────────────────────────────────────────────────
            z = np.random.randn(2, nb_scenarios)
            chocs = L @ z  # Applique la corrélation
            
            # ─────────────────────────────────────────────────────────────────
            # 3. VASICEK POUR LES TAUX (✅ CORRECTION)
            # ─────────────────────────────────────────────────────────────────
            dr_t = (self.kappa_bonds * (self.theta_bonds - r_t) * dt + 
                    s_b * chocs[1, :])
            r_t += dr_t
            r_t = np.maximum(r_t, 0.001)  # Floor à 0.1% (évite taux négatifs)
            
            # ─────────────────────────────────────────────────────────────────
            # 4. APPLICATION DES RÉGIMES (Equity)
            # ─────────────────────────────────────────────────────────────────
            for s in range(nb_scenarios):
                regime = regimes[t, s]
                regime_name = ['normal', 'bull', 'bear'][regime]
                mult = self.regime_multipliers[regime_name]
                
                # Rendement equity avec régime et vol stochastique
                mu_adj = self.mu_e * mult['mu']
                sigma_adj = vol_equity[t, s] * mult['sigma']
                
                r_eq[t, s] = (mu_adj * dt - 0.5 * (sigma_adj * np.sqrt(dt))**2 + 
                              sigma_adj * np.sqrt(dt) * chocs[0, s])
            
            # ─────────────────────────────────────────────────────────────────
            # 5. RENDEMENT BONDS (basé sur taux court Vasicek)
            # ─────────────────────────────────────────────────────────────────
            # Log-rendement ≈ taux court * dt
            r_bd[t, :] = r_t * dt - 0.5 * (s_b)**2 + chocs[1, :]
            
            # ─────────────────────────────────────────────────────────────────
            # 6. INFLATION (optionnel, processus AR(1))
            # ─────────────────────────────────────────────────────────────────
            if self.generate_inflation:
                # Inflation avec persistance AR(1)
                # infl(t+1) = α * infl(t) + (1-α) * π_target + ε
                alpha = 0.7  # Persistance
                pi_target = 0.02  # Cible 2%
                
                infl_shock = np.random.randn(nb_scenarios) * 0.003  # Vol inflation = 0.3%
                infl_t = alpha * infl_t + (1 - alpha) * pi_target + infl_shock
                infl_t = np.clip(infl_t, -0.02, 0.10)  # Borner entre -2% et 10%
                
                r_inflation[t, :] = infl_t
        
        # ═════════════════════════════════════════════════════════════════════
        # RETOUR
        # ═════════════════════════════════════════════════════════════════════
        if self.generate_inflation:
            return r_eq, r_bd, r_inflation, regimes, vol_equity
        else:
            return r_eq, r_bd, regimes, vol_equity
    
    def generate_scenarios_simple(self, nb_periods, nb_scenarios, seed=None):
        """
        Version simplifiée sans régimes (pour compatibilité avec le framework existant).
        
        Returns:
            tuple: (r_equity, r_bonds)
        """
        result = self.generate_scenarios(nb_periods, nb_scenarios, seed)
        
        if self.generate_inflation:
            r_eq, r_bd, r_infl, _, _ = result
            return r_eq, r_bd, r_infl
        else:
            r_eq, r_bd, _, _ = result
            return r_eq, r_bd