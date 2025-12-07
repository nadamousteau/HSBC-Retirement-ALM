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
        self.gpi = gpi; self.psp = psp; self.bond = bond; self.wealth = initial_wealth
        self.initial_salary = initial_salary; self.saving_rate = saving_rate; self.inflation_salary = inflation_salary
        self.dates = psp.index.intersection(gpi.yc.dates)
        self.dates = self.dates[(self.dates >= start_date) & (self.dates <= end_date)].sort_values()
        
      
        
        self.history = {'Date': [], 'Wealth': [], 'Funding_Ratio': [], 'Allocation_PSP': [], 'Contrib_mensuel': [], 'Floor': []}

    def _process_contribution(self,date,i, current_wealth):
        # On calcule la contribution actuel du client à cette date
        years_passed = (date - self.dates[0]).days / 365.25
            
        # On verse mensuellement 
        monthly_contrib = self.saving_rate * self.initial_salary * ((1+self.inflation_salary)**years_passed)
        
        # On n'ajoute pas le tout premier jour (déjà capital initial)
        if i == 0: return current_wealth, 0
        
        return current_wealth + monthly_contrib, monthly_contrib

    def run_gbi(self, floor_pct):
        print(f" Exécution GBI sur {len(self.dates)} mois...")
        W, W_year_start = self.wealth, self.wealth
        total_injected = 0

        for i, date in enumerate(self.dates):
            # 1. Ajout Contribution (Variable selon l'âge)
            W, injected = self._process_contribution(date, i, W)
            total_injected += injected
            
            
            # 2. Indicateurs GBI
            beta = self.gpi.calculate(date)
            beta_start = self.gpi.calculate(pd.Timestamp(f"{date.year}-01-01"))

            if date.month == 1 and self.dates[i-1].month == 12: 
                W_year_start = W
                # Plancher
            if (date.month == 1 and self.dates[i-1].month == 12) or i == 0: 

                floor = floor_pct * (W_year_start / beta_start) * beta
            
            # Multiplicateur
            yrs_ret = (self.gpi.ret_date - date).days / 365.25
            alloc_tdf = 0.80 if yrs_ret >= 20 else (0.40 if yrs_ret <= 0 else 0.80 - (0.40/20)*(20-yrs_ret))
            m = alloc_tdf / (1 - floor_pct + 1e-6)
            
            # Allocation
            cushion = max(0, W - floor)
            w_psp = min(1.0, (m * cushion) / W) if W > 0 else 0
            
            # 3. Marché
            r_psp = self.psp.loc[date]
            r_safe = ((self.gpi.calculate(self.dates[i+1]) / beta) - 1) if i < len(self.dates)-1 else 0
            
            W *= (1 + w_psp*r_psp + (1-w_psp)*r_safe)
            fr = (W / beta) / (self.wealth / self.gpi.calculate(self.dates[0]))
            
            self._record(date, W, fr, w_psp, injected, floor)
        return pd.DataFrame(self.history).set_index('Date')

    def run_tdf(self):
        print(f" Exécution TDF ...")
        W = self.wealth
        
        for i, date in enumerate(self.dates):
            W, injected = self._process_contribution(date, i, W)
            
            yrs_ret = (self.gpi.ret_date - date).days / 365.25
            w_psp = 0.80 if yrs_ret >= 20 else (0.40 if yrs_ret <= 0 else 0.80 - (0.40/20)*(20-yrs_ret))
            
            r_port = w_psp * self.psp.loc[date] + (1-w_psp) * self.bond.loc[date]
            W *= (1 + r_port)
            
            beta = self.gpi.calculate(date)
            fr = (W / beta) / (self.wealth / self.gpi.calculate(self.dates[0]))
            
            self._record(date, W, fr, w_psp, injected)
        return pd.DataFrame(self.history).set_index('Date')

    def _record(self, date, W, fr, alloc, inj, floor=-1):
        self.history['Date'].append(date); self.history['Wealth'].append(W)
        self.history['Funding_Ratio'].append(fr); self.history['Allocation_PSP'].append(alloc)
        self.history['Contrib_mensuel'].append(inj) 
        self.history["Floor"].append(floor)  



