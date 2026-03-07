import pandas as pd
import numpy as np
from datetime import datetime

class Inflation:
    """
    La classe renvoie une matrice de livret A et une matrice inflation.
    Pour chacune des matrices le pas est mensuel masi la valeur annuelle,
    ce sont des matrices "en escalier". Le modele d'inflation a été calibré avec des données de la BCE
    """
    def __init__(self, inflation_file, livret_a_file, date_depart, date_fin,n_sim):
        # Conversion des dates
        self.dt_start = pd.to_datetime(date_depart)
        self.dt_end = pd.to_datetime(date_fin)
        self.n_sim = n_sim
        
        # Chargement des données historiques
        self.hist_inf = pd.read_csv(inflation_file, sep=';')
        self.hist_la = pd.read_csv(livret_a_file, sep=';')
        
        # Nettoyage et normalisation
        self.hist_inf.columns = ['Year', 'Rate']
        self.hist_la.columns = ['Year', 'Rate']
        self.hist_inf['Rate'] /= 100.0
        self.hist_la['Rate'] /= 100.0
        
        # Paramètres de littérature (Vasicek)
        self.theta = 0.02   # Cible long terme (2%)
        self.kappa = 0.30   # Vitesse de retour à la moyenne
        self.sigma = 0.012  # Volatilité annuelle
        
        # Détermination de la fenêtre temporelle
        self.n_months = (self.dt_end.year - self.dt_start.year) * 12 + (self.dt_end.month - self.dt_start.month)
        self.last_hist_year = int(self.hist_inf['Year'].max())

    def _get_quantiles(self, mode='inflation', raw_paths=False): 
        
        # Identification des années à couvrir
        years_needed = []
        temp_date = self.dt_start
        for _ in range(self.n_months):
            years_needed.append(temp_date.year)
            temp_date += pd.DateOffset(months=1)
        
        unique_years = sorted(list(set(years_needed)))
        sim_results = {}
        
        # Point de départ de la simulation (Dernière valeur historique connue)
        last_val = self.hist_inf[self.hist_inf['Year'] == self.last_hist_year]['Rate'].values[0]
        
        # Simulation des années futures
        future_years = [y for y in unique_years if y > self.last_hist_year]
        if future_years:
            n_steps = len(future_years)
            n_paths = self.n_sim  
            
            # Matrice des trajectoires : (Trajectoires, Pas de temps)
            paths = np.zeros((n_paths, n_steps))
            curr_pi = np.full(n_paths, last_val)
            
            # Constantes de discrétisation exacte (dt = 1 an)
            phi = np.exp(-self.kappa)
            vol_term = np.sqrt((self.sigma**2 / (2 * self.kappa)) * (1 - np.exp(-2 * self.kappa)))
            
            # Pré-allocation matricielle des chocs gaussiens
            Z = np.random.standard_normal((n_steps, n_paths))
            
            for t in range(n_steps):
                # Vectorisation sur les n_paths trajectoires
                curr_pi = curr_pi * phi + self.theta * (1 - phi) + vol_term * Z[t]
                
                # Sauvegarde de la coupe transversale à l'instant t
                if mode == 'livret_a':
                    paths[:, t] = np.maximum(curr_pi + 0.005, 0.005)
                else:
                    paths[:, t] = curr_pi

            # Si on demande les trajectoires brutes, on s'arrête ici
            if raw_paths:
                # On crée la matrice "en escalier" (n_sim, n_months)
                raw_matrix = np.zeros((self.n_sim, self.n_months))
                for m in range(self.n_months):
                    yr = years_needed[m]
                    if yr <= self.last_hist_year:
                        # Période historique : on met la même valeur pour toutes les simulations
                        df = self.hist_inf if mode == 'inflation' else self.hist_la
                        val = df[df['Year'] == yr]['Rate'].values[0]
                        raw_matrix[:, m] = val
                    else:
                        # Période simulée : on va chercher l'année correspondante
                        t_idx = future_years.index(yr)
                        raw_matrix[:, m] = paths[:, t_idx]
                
                # On retourne les trajectoires brutes (transposées pour avoir [n_months, n_sims] 
                # si c'est ce format qu'utilise ta fonction Merton, sinon enlève le .T)
                return raw_matrix.T

            # Calcul des statistiques par année simulée (s'exécute uniquement si raw_paths=False)
            stats = {
                future_years[t]: [np.median(paths[:, t]), np.quantile(paths[:, t], 0.05), np.quantile(paths[:, t], 0.95)]
                for t in range(n_steps)
            }
            sim_results.update(stats)

        # Construction de la matrice finale des quantiles (3, n_months)
        matrix = np.zeros((3, self.n_months))
        for m in range(self.n_months):
            yr = years_needed[m]
            if yr <= self.last_hist_year:
                # Partie Historique
                df = self.hist_inf if mode == 'inflation' else self.hist_la
                val = df[df['Year'] == yr]['Rate'].values[0]
                matrix[:, m] = val
            else:
                # Partie Simulée
                matrix[:, m] = sim_results[yr]
                
        return matrix

    def inflation(self):
        """Retourne la matrice d'inflation (Médiane, P5, P95)."""
        return self._get_quantiles(mode='inflation')

    def livret_a(self):
        """Retourne la matrice du taux Livret A (Médiane, P5, P95)."""
        return self._get_quantiles(mode='livret_a')

    def trajectoires_brutes_inflation(self):
        """Retourne les trajectoires simulées (n_sims, n_annees) pour les chocs."""
        return self._get_quantiles(mode='inflation', raw_paths=True)