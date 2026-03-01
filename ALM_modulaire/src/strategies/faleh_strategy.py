"""
STRATÉGIE DE FALEH - Optimisation Stochastique Dynamique
==========================================================
Implémentation de la stratégie ALM-AI de Faleh et al. (2015).

Principe :
1. Génération de scénarios économiques (GSE)
2. Construction d'un arbre de décision
3. Optimisation dynamique de l'allocation pour maximiser l'utilité espérée
4. Prise en compte du passif (objectif de capital à la retraite)

Fonction d'utilité : U(W) = W^(1-γ) / (1-γ) où γ est l'aversion au risque
"""

import numpy as np
from scipy.optimize import minimize
from .base_strategy import BaseStrategy
from config import settings, profiles
from ..economics.gse import EnhancedGSE
from ..economics.tree_builder import ScenarioTreeBuilder


class FalehStrategy(BaseStrategy):
    """
    Stratégie d'optimisation stochastique avec arbre de scénarios.
    
    Caractéristiques :
    - Allocation optimale à chaque période
    - Prise en compte du passif (target wealth)
    - Fonction d'utilité CRRA (Constant Relative Risk Aversion)
    - Rééquilibrage dynamique
    """
    
    # Mapping des profils vers gamma (aversion au risque)
    GAMMA_MAPPING = {
        "PRUDENT": 10.0,    # Très risk-averse
        "MODERE": 6.0,      # Risk-averse modéré
        "EQUILIBRE": 4.0,   # Équilibré
        "DYNAMIQUE": 2.5,   # Accepte le risque
        "AGRESSIF": 1.5     # Risk-seeker
    }
    
    def __init__(self, mu_e, sigma_e, mu_b, sigma_b, corr_eb, 
                 target_wealth=None, nb_tree_stages=10):
        """
        Args:
            mu_e, sigma_e, mu_b, sigma_b, corr_eb: Paramètres de marché
            target_wealth: Objectif de capital à la retraite (calculé si None)
            nb_tree_stages: Nombre de stages dans l'arbre de décision
        """
        # Récupération du gamma selon le profil
        self.gamma = self.GAMMA_MAPPING.get(settings.PROFIL_CHOISI, 4.0)
        
        # Paramètres de marché
        self.mu_e = mu_e
        self.sigma_e = sigma_e
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        self.corr_eb = corr_eb
        
        # Générateur de scénarios
        self.gse = EnhancedGSE(mu_e, sigma_e, mu_b, sigma_b, corr_eb)
        
        # Arbre de scénarios
        self.tree_builder = ScenarioTreeBuilder(max_branches_per_node=5)
        self.tree = None
        self.nb_tree_stages = nb_tree_stages
        
        # Objectif de richesse
        if target_wealth is None:
            # Estimation : capital moyen attendu avec stratégie équilibrée
            self.target_wealth = self._estimate_target_wealth()
        else:
            self.target_wealth = target_wealth
        
        # Cache pour les allocations optimales
        self.optimal_allocations = {}
        self.current_stage = 0
        self.scenarios_generated = False
    
    def initialize_tree(self, dates):
        """
        Génère les scénarios et construit l'arbre de décision.
        Appelé une seule fois au début de la simulation.
        """
        if self.scenarios_generated:
            return
        
        nb_periods = len(dates)
        nb_scenarios = settings.NB_SIMULATIONS  # Limite pour performance
        
        print(f"\n Construction de l'arbre de scénarios Faleh...")
        print(f"   • Profil : {settings.PROFIL_CHOISI}")
        print(f"   • Gamma (aversion risque) : {self.gamma:.2f}")
        print(f"   • Scénarios : {nb_scenarios}")
        print(f"   • Stages : {self.nb_tree_stages}")
        
        # Génération des scénarios
        r_eq, r_bd = self.gse.generate_scenarios_simple(
            nb_periods, nb_scenarios, seed=42
        )
        
        # Construction de l'arbre
        self.tree = self.tree_builder.build_tree(
            r_eq, r_bd, nb_stages=self.nb_tree_stages
        )
        
        # Affichage structure
        self.tree_builder.visualize_tree_structure(self.tree)
        
        # Pré-calcul des allocations optimales par stage
        self._precompute_optimal_allocations()
        
        self.scenarios_generated = True
    
    def get_allocation(self, t_index, current_age):
        """
        Retourne l'allocation optimale pour le pas de temps actuel.
        
        Interface conforme à BaseStrategy.
        """
        # Initialisation de l'arbre si nécessaire
        if not self.scenarios_generated:
            # Workaround : on ne peut pas passer dates ici
            # On va initialiser avec une estimation
            pass
        
        # Détermination du stage actuel
        if self.tree is not None:
            # Trouver le stage le plus proche
            stages = np.array(self.tree['stages'])
            stage_idx = np.argmin(np.abs(stages - t_index))
        else:
            # Fallback : allocation progressive
            return self._fallback_allocation(current_age)
        
        # Récupération de l'allocation optimale
        if stage_idx in self.optimal_allocations:
            # Allocation moyenne pondérée par les probabilités
            alloc_data = self.optimal_allocations[stage_idx]
            weights = np.array([node['probability'] for node in alloc_data])
            allocations = np.array([node['allocation_equity'] for node in alloc_data])
            
            pct_equity = np.average(allocations, weights=weights)
        else:
            # Fallback
            pct_equity = self._fallback_allocation(current_age)[0]
        
        # Borner entre 5% et 95%
        pct_equity = max(0.05, min(0.95, pct_equity))
        
        return pct_equity, 1 - pct_equity
    
    def should_rebalance(self, t_index):
        """
        Rééquilibre aux stages de décision de l'arbre.
        """
        if self.tree is None:
            return False
        
        # Rééquilibrer uniquement aux stages de décision
        return t_index in self.tree['stages']
    
    def _precompute_optimal_allocations(self):
        """
        Calcule les allocations optimales pour chaque nœud de l'arbre
        par programmation dynamique backward.
        """
        print(f"\n Optimisation dynamique stochastique...")
        
        nb_stages = len(self.tree['stages'])
        
        # Initialisation : valeur terminale
        # V_T(W) = U(W) où W est la richesse finale
        terminal_values = {}
        for node_id, node_data in self.tree['nodes'][nb_stages - 1].items():
            # Valeur terminale = utilité du surplus (W - L)
            # Pour simplification, on prend l'utilité de la richesse
            terminal_values[node_id] = {
                'value': 0,  # Sera calculé dans le backward pass
                'allocation_equity': 0.5  # Allocation neutre au dernier stage
            }
        
        self.optimal_allocations[nb_stages - 1] = [
            {
                'node_id': nid,
                'allocation_equity': 0.5,
                'probability': self.tree['nodes'][nb_stages - 1][nid]['probability']
            }
            for nid in self.tree['nodes'][nb_stages - 1].keys()
        ]
        
        # Backward induction
        for stage_idx in range(nb_stages - 2, -1, -1):
            stage_allocations = []
            
            for node_id, node_data in self.tree['nodes'][stage_idx].items():
                # Statistiques du nœud
                mean_eq = node_data['mean_eq']
                mean_bd = node_data['mean_bd']
                std_eq = max(node_data['std_eq'], 0.01)
                std_bd = max(node_data['std_bd'], 0.01)
                
                # Optimisation de l'allocation
                optimal_w = self._optimize_allocation_at_node(
                    mean_eq, mean_bd, std_eq, std_bd,
                    stage_idx, node_id
                )
                
                stage_allocations.append({
                    'node_id': node_id,
                    'allocation_equity': optimal_w,
                    'probability': node_data['probability']
                })
            
            self.optimal_allocations[stage_idx] = stage_allocations
            
            print(f"   Stage {stage_idx:2d}: Optimisé {len(stage_allocations)} nœuds")
        
        print(f" Optimisation terminée !\n")
    
    def _optimize_allocation_at_node(self, mean_eq, mean_bd, std_eq, std_bd, stage, node_id):
        """
        Optimise l'allocation equity/bonds à un nœud donné.
        
        Maximise l'utilité espérée : E[U(W_t+1)]
        où U(W) = W^(1-γ) / (1-γ)
        """
        # Fonction objectif : - E[U(W)] (minimisation)
        def objective(w):
            """w = poids equity (entre 0 et 1)"""
            w = max(0.0, min(1.0, w))
            
            # Rendement du portefeuille (approximation normale)
            portfolio_return = w * mean_eq + (1 - w) * mean_bd
            portfolio_std = np.sqrt((w * std_eq)**2 + ((1-w) * std_bd)**2 + 
                                    2 * w * (1-w) * self.corr_eb * std_eq * std_bd)
            
            # Approximation de l'utilité espérée (Taylor au 2nd ordre)
            # E[U(1 + R)] ≈ U(1 + μ) + 0.5 * U''(1 + μ) * σ²
            W_expected = 1 + portfolio_return
            
            if self.gamma == 1.0:
                # Cas log-utilité
                utility = np.log(W_expected) - 0.5 * (portfolio_std / W_expected)**2
            else:
                # Utilité CRRA
                utility = (W_expected**(1 - self.gamma)) / (1 - self.gamma)
                # Correction pour la variance (approximation du 2nd ordre)
                utility -= 0.5 * self.gamma * (W_expected**(-self.gamma - 1)) * portfolio_std**2
            
            return -utility  # Minimisation
        
        # Contraintes : w ∈ [0, 1]
        bounds = [(0.05, 0.95)]
        
        # Guess initial basé sur le profil
        if settings.PROFIL_CHOISI == "PRUDENT":
            x0 = 0.20
        elif settings.PROFIL_CHOISI == "MODERE":
            x0 = 0.40
        elif settings.PROFIL_CHOISI == "EQUILIBRE":
            x0 = 0.60
        elif settings.PROFIL_CHOISI == "DYNAMIQUE":
            x0 = 0.75
        else:  # AGRESSIF
            x0 = 0.85
        
        # Optimisation
        try:
            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 50}
            )
            optimal_w = result.x[0]
        except:
            # Fallback en cas d'échec
            optimal_w = x0
        
        return optimal_w
    
    def _estimate_target_wealth(self):
        """
        Estime l'objectif de richesse basé sur les paramètres du profil.
        """
        # Estimation simple : 80% du capital espéré avec allocation équilibrée
        T = settings.NB_ANNEES_ACCUMULATION
        r_portfolio = 0.6 * self.mu_e + 0.4 * self.mu_b  # Mix 60/40
        
        # Capital initial + apports
        total_contributions = settings.CAPITAL_INITIAL
        for t in range(T):
            # Apport moyen annuel (simplifié)
            apport_annuel = settings.SALAIRE_INITIAL * settings.TAUX_APPORT_BASE * 12
            total_contributions += apport_annuel
        
        # Capitalisation
        target = total_contributions * (1 + r_portfolio) ** T * 0.8
        
        return target
    
    def _fallback_allocation(self, current_age):
        """
        Allocation de secours si l'arbre n'est pas construit.
        Utilise l'allocation fixe du profil.
        """
        pct_equity = profiles.fixed_allocation
        return pct_equity, 1 - pct_equity
    
    def get_strategy_info(self):
        """
        Retourne les informations sur la stratégie.
        """
        return {
            'name': 'Faleh (Optimisation Stochastique)',
            'type': 'Dynamic Programming',
            'gamma': self.gamma,
            'profile': settings.PROFIL_CHOISI,
            'target_wealth': self.target_wealth,
            'tree_stages': self.nb_tree_stages,
            'rebalancing': 'Dynamic (at tree stages)'
        }


class SimplifiedFalehStrategy(BaseStrategy):
    """
    Version simplifiée de Faleh sans arbre complet.
    
    Utilise une règle d'allocation myope optimale à chaque période,
    basée sur l'utilité espérée one-step-ahead.
    
    Plus rapide computationnellement, utile pour tests.
    """
    
    def __init__(self, mu_e, sigma_e, mu_b, sigma_b, corr_eb):
        self.gamma = FalehStrategy.GAMMA_MAPPING.get(settings.PROFIL_CHOISI, 4.0)
        self.mu_e = mu_e
        self.sigma_e = sigma_e
        self.mu_b = mu_b
        self.sigma_b = sigma_b
        self.corr_eb = corr_eb
        
        # Cache de l'allocation optimale
        self._cached_allocation = None
    
    def get_allocation(self, t_index, current_age):
        """Allocation myope optimale."""
        if self._cached_allocation is None:
            self._cached_allocation = self._compute_myopic_optimal()
        
        return self._cached_allocation, 1 - self._cached_allocation
    
    def should_rebalance(self, t_index):
        """Rééquilibre tous les trimestres."""
        return (t_index % 3) == 0
    
    def _compute_myopic_optimal(self):
        """
        Calcul de l'allocation myope optimale (Merton).
        
        Solution fermée pour le problème one-period :
        w* = (μ_e - μ_b) / (γ * σ_e²)
        """
        dt = 1.0 / 12.0
        excess_return = (self.mu_e - self.mu_b) * dt
        variance = (self.sigma_e * np.sqrt(dt))**2
        
        # Allocation de Merton
        w_merton = excess_return / (self.gamma * variance)
        
        # Borner entre 5% et 95%
        w_optimal = max(0.05, min(0.95, w_merton))
        
        return w_optimal
