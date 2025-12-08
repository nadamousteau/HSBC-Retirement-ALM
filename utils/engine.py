from utils.constant import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from utils.data import YieldCurveBuilder, DataLoader



# ============================================================================
#  MOTEUR GBI AVEC CAPITAL HUMAIN DYNAMIQUE
# ============================================================================

class GoalPriceIndex:
    def __init__(self, yield_curve, retirement_date, dec_years=20):
        self.yc = yield_curve; self.ret_date = pd.to_datetime(retirement_date); self.dec_years = dec_years
    def calculate(self, date):
        date = pd.to_datetime(date)
        t_ret = (self.ret_date - date).days / 365.25
        rem = max(0, self.dec_years + t_ret) if t_ret < 0 else self.dec_years
        if rem <= 0: return 1.0
        beta = 0.0
        for k in range(int(np.ceil(rem))):
            r = self.yc.get_zero_rate(date, max(0, t_ret) + k)
            beta += np.exp(-r * (max(0, t_ret) + k))
        return beta if beta > 0 else 1.0

class StrategyEngine:
    def __init__(self, gpi, psp, bond, start_date, end_date, initial_wealth, 
                 initial_salary, saving_rate, inflation_salary) :
        self.gpi = gpi
        self.psp = psp
        self.bond = bond
        self.wealth = initial_wealth
        self.initial_salary = initial_salary
        self.saving_rate = saving_rate
        self.inflation_salary = inflation_salary
        self.dates = psp.index.intersection(gpi.yc.dates)
        self.dates = self.dates[(self.dates >= start_date) & (self.dates <= end_date)].sort_values()
        
      
        
        self.history = {'Date': [], 'Wealth': [], 'Allocation_PSP': [], 'Contrib_mensuel': [], 'Floor': []}

    def calculer_apport_quadratique(self, t_annees, apport_init, duree_totale):
        """
        Simule une courbe d'épargne en cloche asymétrique.
        Croissance jusqu'à un 'Age Pic', puis décroissance vers la retraite.
        
        t_annees : temps écoulé en années (0 à 35)
        apport_init : montant du premier versement (ex: 300€)
        duree_totale : durée de la simulation (ex: 35 ans)
        """
        
        # 1. Définition du Pic (Sommet de la carrière)
        # Selon le papier, le pic est souvent vers 50-55 ans.
        # Si on commence à 30 ans sur 35 ans, le pic est aux 2/3 du chemin.
        ratio_pic = 0.55  # Le pic arrive à 65% de la durée (env. 53 ans)
        t_pic = duree_totale * ratio_pic
        
        # 2. Hauteur du Pic
        # Facteur multiplicateur : On suppose qu'au sommet de sa carrière, 
        # on épargne 3x plus qu'au début (ex: 300€ -> 900€).
        facteur_croissance_max = 1.2
        apport_max = apport_init * facteur_croissance_max
        
        # 3. Calcul de la parabole (Forme Canonique : y = a(x-h)^2 + k)
        # On sait que c(0) = apport_init et c(t_pic) = apport_max
        # apport_init = a * (0 - t_pic)^2 + apport_max
        # a = (apport_init - apport_max) / (t_pic^2)
        
        if t_pic > 0:
            a = (apport_init - apport_max) / (t_pic**2)
        else:
            return apport_init

        apport = a * (t_annees - t_pic)**2 + apport_max
        
        # Sécurité : on ne peut pas avoir d'apport négatif (si la baisse est trop forte)
        return max(apport, 0)


    def run_gbi(self, floor_pct, profil_tdf=None):
        print(f" Exécution GBI sur {len(self.dates)} mois...")
        W, W_year_start = self.wealth, self.wealth
        total_injected = 0
        
        for i, date in enumerate(self.dates):
            
            apport_base_init = self.initial_salary * self.saving_rate
            apport_mensuel = self.calculer_apport_quadratique(i / 12, apport_base_init, duree_totale=len(self.dates)/12)

            # Ajout Contribution (Variable selon l'âge)
            W, injected = W + apport_mensuel, apport_mensuel
            total_injected += injected
            
            
            # Indicateurs GBI
            beta = self.gpi.calculate(date)
            beta_start = self.gpi.calculate(pd.Timestamp(f"{date.year}-01-01"))

            if date.month == 1 and self.dates[i-1].month == 12: 
                W_year_start = W
                
                
            if (date.month == 1 and self.dates[i-1].month == 12) or i == 0: 

                floor = floor_pct * (W_year_start / beta_start) * beta
            
            # Multiplicateur
            years_passed = (date - self.dates[0]).days / 365.25
            pct_equity = profil_tdf['allocation_initiale'] - (profil_tdf['decroissance_annuelle'] * years_passed)
            alloc_tdf = max(0.05, min(1.0, pct_equity))
            m = alloc_tdf / (1 - floor_pct + 1e-6)
            
            # Allocation
            cushion = max(0, W - floor)
            w_psp = min(1.0, (m * cushion) / W) if W > 0 else 0
            
            
            r_psp = self.psp.loc[date]
            r_safe = ((self.gpi.calculate(self.dates[i+1]) / beta) - 1) if i < len(self.dates)-1 else 0
            
            W *= (1 + w_psp*r_psp + (1-w_psp)*r_safe)
            fr = (W / beta) / (self.wealth / self.gpi.calculate(self.dates[0]))
            
            self._record(date, W,  w_psp, injected, floor)
        return pd.DataFrame(self.history).set_index('Date')

    def run_tdf(self, profil):
        print(f" Simulation TDF (Profil: {profil})...")
        W = self.wealth
        
        for i, date in enumerate(self.dates):

            apport_base_init = self.initial_salary * self.saving_rate
            apport_mensuel = self.calculer_apport_quadratique(i / 12, apport_base_init, duree_totale=len(self.dates)/12)

            # Ajout Contribution (Variable selon l'âge)
            W, injected = W + apport_mensuel, apport_mensuel
            
            # Calcul Allocation (Formule décroissance linéaire)
            # age_actuel = age_depart + années_écoulées
            years_passed = (date - self.dates[0]).days / 365.25
            
            # Formule: Allocation = Initiale - (Decroissance * Années)
            # On applique les limites : Min 5% Equity, Max 100%
            pct_equity = profil['allocation_initiale'] - (profil['decroissance_annuelle'] * years_passed)
            w_psp = max(0.05, min(1.0, pct_equity))
            
            #  Rendement Portefeuille
            r_port = w_psp * self.psp.loc[date] + (1-w_psp) * self.bond.loc[date]
            W *= (1 + r_port)
            
            
            
            self._record( date, W,  w_psp, injected)
            
        return pd.DataFrame(self.history).set_index('Date')


    def run_fixed_mix(self, fixed_equity_pct):
        print(f" Simulation FIXED MIX ({fixed_equity_pct*100:.0f}/{100-fixed_equity_pct*100:.0f})...")
        
        W = self.wealth
        
        for i, date in enumerate(self.dates):
            
            apport_base_init = self.initial_salary * self.saving_rate
            apport_mensuel = self.calculer_apport_quadratique(i / 12, apport_base_init, duree_totale=len(self.dates)/12)

            W, injected = W + apport_mensuel, apport_mensuel
                    
            r_port = fixed_equity_pct * self.psp.loc[date] + (1-fixed_equity_pct) * self.bond.loc[date]
            W *= (1 + r_port)
            
            
            self._record( date, W, fixed_equity_pct, injected)
            
        return pd.DataFrame(self.history).set_index('Date')

    def _record(self, date, W,  alloc, inj, floor=-1):
        self.history['Date'].append(date); self.history['Wealth'].append(W)
        self.history['Allocation_PSP'].append(alloc)
        self.history['Contrib_mensuel'].append(inj) 
        self.history["Floor"].append(floor)  



