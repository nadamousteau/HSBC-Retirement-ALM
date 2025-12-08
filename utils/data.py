import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ============================================================================
#  CHARGEMENT DONNÉES (Yield & Assets)
# ============================================================================

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

class YieldCurveBuilder:
    def __init__(self):
        self.rates_data = None
        self.zero_curves = {} # Cache pour stocker les courbes calculées

    def load_from_csv(self, csv_path):
        """Charge et nettoie les données Par Yields."""
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True, na_values=['N/A', ''])
            
            # Mapping standard US Treasury
            col_mapping = {
                '1 Mo': 1/12, '2 Mo': 2/12, '3 Mo': 0.25, '4 Mo': 4/12, '6 Mo': 0.5, 
                '1 Yr': 1, '2 Yr': 2, '3 Yr': 3, '5 Yr': 5, '7 Yr': 7, 
                '10 Yr': 10, '20 Yr': 20, '30 Yr': 30
            }
            df = df.rename(columns=col_mapping)
            
            # Garder uniquement les colonnes numériques
            valid_cols = [c for c in df.columns if isinstance(c, (int, float))]
            
            # Conversion % -> décimal et rééchantillonnage mensuel
            self.rates_data = (df[valid_cols] / 100.0).sort_index(axis=1).resample('ME').last().dropna(how='all')
            
            # --- CORRECTION ICI ---
            # On expose les dates pour que StrategyEngine puisse les lire
            self.dates = self.rates_data.index 
            
            return self
        except Exception as e:
            print(f"Erreur lors du chargement: {e}")
            return None

    def _bootstrap_row(self, date_idx):
        """
        Effectue le bootstrapping pour une date donnée.
        Convertit la courbe Par en courbe Zéro.
        """
        row = self.rates_data.loc[date_idx].dropna()
        if len(row) < 5: # Sécurité : il faut un minimum de points
            raise ValueError(f"Pas assez de données pour bootstrapper à la date {date_idx}")

        known_maturities = row.index.values
        known_par_rates = row.values

        # 1. Création d'une grille semestrielle (US Treasuries paient tous les 6 mois)
        # On va jusqu'à la maturité max (ex: 30 ans) par pas de 0.5
        max_mat = known_maturities[-1]
        grid_maturities = np.arange(0.5, max_mat + 0.5, 0.5)

        # 2. Interpolation des Taux Par sur cette grille pour combler les trous (ex: 4 ans, 8 ans)
        # L'interpolation linéaire sur les taux Par est acceptable ici pour remplir les gaps
        par_interpolator = interp1d(known_maturities, known_par_rates, kind='linear', fill_value="extrapolate")
        grid_par_rates = par_interpolator(grid_maturities)

        # 3. Bootstrapping
        # On calcule les facteurs d'actualisation (Discount Factors - DF)
        discount_factors = np.zeros(len(grid_maturities))
        
        for i, t in enumerate(grid_maturities):
            rate_par = grid_par_rates[i]
            
            # Cas Court Terme (<= 1 an) : Approximation Zero = Par
            # Note: Pour être ultra-précis, il faudrait ajuster les conventions de jour (ACT/360), 
            # mais Par = Zero est suffisant ici.
            if t <= 1.0:
                discount_factors[i] = 1 / ((1 + rate_par)**t)
            
            # Cas Long Terme (> 1 an) : Le vrai bootstrap
            else:
                # Formule : 1 = sum(Coupon * DF_passés) + (1 + Coupon) * DF_actuel
                # Coupon semestriel = rate_par / 2
                coupon = rate_par / 2.0
                
                # Somme des coupons actualisés précédents
                sum_prev_dfs = np.sum(discount_factors[:i])
                
                # Résolution pour le DF actuel
                # 1 = coupon * sum_prev_dfs + coupon * DF_i + 1 * DF_i
                # 1 - coupon * sum_prev_dfs = DF_i * (1 + coupon)
                df_i = (1.0 - coupon * sum_prev_dfs) / (1.0 + coupon)
                discount_factors[i] = df_i

        # Conversion DF -> Taux Zéro
        # DF = 1 / (1+z)^t  =>  (1+z)^t = 1/DF  => 1+z = (1/DF)^(1/t) => z = (1/DF)^(1/t) - 1
        zero_rates = (1.0 / discount_factors)**(1.0 / grid_maturities) - 1.0
        
        return grid_maturities, zero_rates

    def get_zero_rate(self, date, maturity):
        """
        Récupère le taux zéro bootstrappé interpolé.
        """
        # 1. Gestion de la date
        if isinstance(date, str): date = pd.to_datetime(date)
        
        # Trouver la date valide la plus proche (précédente)
        idx_pos = self.rates_data.index.searchsorted(date)
        if idx_pos >= len(self.rates_data): idx_pos = len(self.rates_data) - 1
        # Si la date demandée est avant la première date dispo, on prend la première
        if idx_pos == 0 and date < self.rates_data.index[0]: idx_pos = 0
        elif date < self.rates_data.index[idx_pos]: idx_pos -= 1 # On prend la date précédente connue
            
        target_date = self.rates_data.index[idx_pos]

        # 2. Vérifier si on a déjà calculé cette courbe (Cache)
        if target_date not in self.zero_curves:
            try:
                self.zero_curves[target_date] = self._bootstrap_row(target_date)
            except ValueError:
                # Si le bootstrap échoue, on lève une erreur explicite
                raise ValueError(f"Impossible de calculer la courbe pour {target_date.date()}. Données insuffisantes.")

        mats, zeros = self.zero_curves[target_date]

        # 3. Interpolation Log-Linéaire sur les Discount Factors (Standard Financier)
        # C'est plus stable que d'interpoler les taux directement
        # On calcule les DF aux points connus
        dfs_known = 1 / ((1 + zeros) ** mats)
        
        # On interpole le Log des DF (qui est grosso modo linéaire par rapport au temps)
        log_dfs_known = np.log(dfs_known)
        
        # Interpolation linéaire sur les logs
        # fill_value='extrapolate' permet de gérer si maturity > 30 ans (Flat Forward implicite)
        val_interp = interp1d(mats, log_dfs_known, kind='linear', fill_value="extrapolate")(maturity)
        
        # Retour au Taux Zéro
        # val_interp = ln(DF_target) => DF_target = exp(val_interp)
        df_target = np.exp(val_interp)
        
        if maturity == 0: return 0.0 # Éviter division par zéro
        zero_rate_final = (1 / df_target)**(1 / maturity) - 1
        
        return float(zero_rate_final)

class DataLoader:
    @staticmethod
    def get_asset_returns(csv_path):
        try:
            df = pd.read_csv(csv_path, header=1).iloc[1:]
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            return df.set_index('Date').apply(pd.to_numeric, errors='coerce')
        except: return None