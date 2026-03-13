"""
MODULE SCENARIO TREE BUILDER
=============================
Construit un arbre de scénarios pour la programmation dynamique stochastique.
Utilisé par la stratégie de Faleh pour optimiser l'allocation dynamique.

L'arbre réduit la complexité computationnelle tout en préservant
les propriétés statistiques clés (moments, corrélations).
"""

import numpy as np
from sklearn.cluster import KMeans


# =============================================================================
# CLASSE SCENARIO TREE BUILDER
# =============================================================================

class ScenarioTreeBuilder:
    """
    Construit un arbre de scénarios par clustering récursif.
    
    Méthode :
    1. Génère N scénarios complets sur T périodes
    2. À chaque nœud, regroupe les scénarios similaires par K-Means
    3. Réduit progressivement le nombre de branches
    """
    
    def __init__(self, max_branches_per_node=5):
        """
        Args:
            max_branches_per_node: Nombre max de branches à chaque nœud
        """
        self.max_branches = max_branches_per_node
    
    def build_tree(self, scenarios_equity, scenarios_bonds, nb_stages=None):
        """
        Construit l'arbre de scénarios par clustering.
        
        Args:
            scenarios_equity: (T, N) - Rendements equity
            scenarios_bonds: (T, N) - Rendements bonds
            nb_stages: Nombre de stages de décision (défaut: T//3)
        
        Returns:
            dict: Structure d'arbre avec:
                - 'stages': Liste de stages
                - 'nodes': Dictionnaire de nœuds {stage: {node_id: data}}
                - 'transitions': Probabilités de transition
        """
        
        nb_periods, nb_scenarios = scenarios_equity.shape
        
        if nb_stages is None:
            # Stage tous les 3 mois (décisions trimestrielles)
            nb_stages = max(2, nb_periods // 3)
        
        # Périodes de décision (équidistantes)
        decision_periods = np.linspace(0, nb_periods-1, nb_stages, dtype=int)
        
        tree = {
            'stages': decision_periods.tolist(),
            'nodes': {},
            'transitions': {},
            'scenario_mapping': {}
        }
        
        # Stage 0 : racine unique
        tree['nodes'][0] = {
            0: {
                'scenarios': np.arange(nb_scenarios),
                'mean_eq': np.mean(scenarios_equity[0, :]),
                'mean_bd': np.mean(scenarios_bonds[0, :]),
                'std_eq': np.std(scenarios_equity[0, :]),
                'std_bd': np.std(scenarios_bonds[0, :]),
                'probability': 1.0
            }
        }
        
        # Mapping initial
        current_mapping = {s: 0 for s in range(nb_scenarios)}
        tree['scenario_mapping'][0] = current_mapping.copy()
        
        # Construction récursive de l'arbre
        for stage_idx in range(1, nb_stages):
            t = decision_periods[stage_idx]
            t_prev = decision_periods[stage_idx - 1]
            
            tree['nodes'][stage_idx] = {}
            tree['transitions'][stage_idx] = {}
            new_mapping = {}
            
            # Pour chaque nœud parent
            for parent_id, parent_data in tree['nodes'][stage_idx - 1].items():
                parent_scenarios = parent_data['scenarios']
                parent_prob = parent_data['probability']
                
                if len(parent_scenarios) <= self.max_branches:
                    # Cas 1 : Pas de clustering (fin de branche)
                    for i, s in enumerate(parent_scenarios):
                        child_id = len(tree['nodes'][stage_idx])
                        
                        transition_prob = 1.0 / len(parent_scenarios)
                        node_prob = parent_prob * transition_prob
                        
                        tree['nodes'][stage_idx][child_id] = {
                            'scenarios': np.array([s]),
                            'mean_eq': scenarios_equity[t, s],
                            'mean_bd': scenarios_bonds[t, s],
                            'std_eq': 0.0,
                            'std_bd': 0.0,
                            'probability': node_prob,
                            'parent': parent_id
                        }
                        new_mapping[s] = child_id
                        
                        if parent_id not in tree['transitions'][stage_idx]:
                            tree['transitions'][stage_idx][parent_id] = {}
                        tree['transitions'][stage_idx][parent_id][child_id] = transition_prob
                
                else:
                    # Cas 2 : Clustering K-Means
                    clusters = self._cluster_scenarios(
                        scenarios_equity[t_prev:t+1, parent_scenarios],
                        scenarios_bonds[t_prev:t+1, parent_scenarios],
                        self.max_branches
                    )

                    for cluster_id in range(self.max_branches):
                        mask = (clusters == cluster_id)
                        cluster_scenarios = parent_scenarios[mask]
                        
                        if len(cluster_scenarios) == 0:
                            continue
                        
                        child_id = len(tree['nodes'][stage_idx])
                        
                        transition_prob = len(cluster_scenarios) / len(parent_scenarios)
                        node_prob = parent_prob * transition_prob

                        tree['nodes'][stage_idx][child_id] = {
                            'scenarios': cluster_scenarios,
                            'mean_eq': np.mean(scenarios_equity[t, cluster_scenarios]),
                            'mean_bd': np.mean(scenarios_bonds[t, cluster_scenarios]),
                            'std_eq': np.std(scenarios_equity[t, cluster_scenarios]),
                            'std_bd': np.std(scenarios_bonds[t, cluster_scenarios]),
                            'probability': node_prob,
                            'parent': parent_id
                        }
                        
                        for s in cluster_scenarios:
                            new_mapping[s] = child_id
                        
                        if parent_id not in tree['transitions'][stage_idx]:
                            tree['transitions'][stage_idx][parent_id] = {}
                        tree['transitions'][stage_idx][parent_id][child_id] = transition_prob
            
            tree['scenario_mapping'][stage_idx] = new_mapping.copy()
        
        return tree
    
    def _cluster_scenarios(self, eq_paths, bd_paths, nb_clusters):
        """
        Clustering K-Means optimisé pour grands ensembles de scénarios.
        
        Args:
            eq_paths: (nb_periods, nb_scenarios) - Trajectoires equity
            bd_paths: (nb_periods, nb_scenarios) - Trajectoires bonds
            nb_clusters: Nombre de clusters cibles
        
        Returns:
            np.array: Labels de cluster pour chaque scénario
        """
        
        nb_scenarios = eq_paths.shape[1]
        
        if nb_scenarios <= nb_clusters:
            return np.arange(nb_scenarios)
        
        # Construction des features (rendements cumulés)
        features = np.vstack([
            eq_paths.sum(axis=0), 
            bd_paths.sum(axis=0)
        ]).T
        
        # Normalisation
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
        
        # K-Means
        kmeans = KMeans(n_clusters=nb_clusters, n_init=5, random_state=42)
        clusters = kmeans.fit_predict(features)
        
        return clusters
    
    def get_representative_scenarios(self, tree, stage):
        """
        Extrait les scénarios représentatifs à un stage donné.
        
        Returns:
            list: Liste de tuples (mean_eq, mean_bd, std_eq, std_bd, prob)
        """
        nodes = tree['nodes'][stage]
        scenarios = []
        
        for node_id, node_data in nodes.items():
            scenarios.append({
                'node_id': node_id,
                'mean_equity': node_data['mean_eq'],
                'mean_bonds': node_data['mean_bd'],
                'std_equity': node_data['std_eq'],
                'std_bonds': node_data['std_bd'],
                'probability': node_data['probability'],
                'nb_scenarios': len(node_data['scenarios'])
            })
        
        return scenarios
    
    def visualize_tree_structure(self, tree):
        """
        Affiche la structure de l'arbre (nombre de nœuds par stage).
        """
        print("\n" + "="*60)
        print("STRUCTURE DE L'ARBRE DE SCÉNARIOS")
        print("="*60)
        
        for stage_idx, stage_t in enumerate(tree['stages']):
            nb_nodes = len(tree['nodes'][stage_idx])
            total_prob = sum(node['probability'] for node in tree['nodes'][stage_idx].values())
            
            print(f"Stage {stage_idx} (t={stage_t:3d}) : {nb_nodes:3d} nœuds | Prob totale: {total_prob:.3f}")
            
            if stage_idx > 0 and stage_idx in tree['transitions']:
                nb_transitions = sum(len(v) for v in tree['transitions'][stage_idx].values())
                print(f"              → {nb_transitions} transitions depuis stage {stage_idx-1}")
        
        print("="*60 + "\n")


# =============================================================================
# CLASSE ADAPTIVE TREE BUILDER
# =============================================================================

class AdaptiveTreeBuilder(ScenarioTreeBuilder):
    """
    Version adaptative qui ajuste le nombre de branches selon
    la dispersion des scénarios.
    """
    
    def __init__(self, min_branches=3, max_branches=7, dispersion_threshold=0.15):
        super().__init__(max_branches_per_node=max_branches)
        self.min_branches = min_branches
        self.dispersion_threshold = dispersion_threshold
    
    def _determine_optimal_branches(self, scenarios, current_nb_branches):
        """
        Détermine le nombre optimal de branches selon la dispersion.
        
        Si les scénarios sont très dispersés → plus de branches
        Si les scénarios sont homogènes → moins de branches
        """
        if len(scenarios) < self.min_branches:
            return len(scenarios)
        
        # Coefficient de variation
        cv = np.std(scenarios) / (np.abs(np.mean(scenarios)) + 1e-8)
        
        if cv > self.dispersion_threshold:
            return min(self.max_branches, current_nb_branches + 1)
        else:
            return max(self.min_branches, current_nb_branches - 1)

