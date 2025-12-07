"""
Backtest GBI - EDHEC Risk Institute
Version 3.0 : Contributions Quadratiques (Cycle de Vie Réaliste)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 🛠️ ZONE DE CONFIGURATION
# ============================================================================

# Fichiers
YIELD_FILE = 'data/yield-curve-rates-1990-2024.csv'
ASSET_FILE = 'data/HistoricalAssetReturn.csv'

# Période de Simulation (Backtest)
SIMULATION_START = '1990-01-01'
SIMULATION_END   = '2020-12-31'

# Paramètres du Client
INITIAL_WEALTH   = 50000         # Capital de départ
RETIREMENT_DATE  = '2021-01-01'  # Date cible de la retraite
FLOOR_PERCENT    = 0.80          # Protection 80%

# 👇 NOUVEAU : Paramètres de Contribution Quadratique (Hump-Shaped)
# Modèle : La capacité d'épargne augmente jusqu'à un pic puis redescend
CLIENT_AGE_START = 20            # Âge du client au début de la simulation (en 2001)
CONTRIB_START    = 5000          # Épargne annuelle au début (à 35 ans)
CONTRIB_PEAK     = 15000         # Épargne annuelle MAXIMALE (au sommet de la carrière)
AGE_PEAK         = 40           # Âge où l'épargne est maximale

# ============================================================================
# 1. MODÉLISATION DU CAPITAL HUMAIN (CONTRIBUTIONS)
# ============================================================================

class HumanCapitalCurve:
    """
    Génère une courbe de contribution quadratique (parabolique)
    Formule : C(age) = a * (age - age_peak)^2 + peak_amount
    """
    def __init__(self, age_start, contrib_start, age_peak, contrib_peak):
        self.age_peak = age_peak
        self.contrib_peak = contrib_peak
        
        # On calcule le coefficient 'a' de la parabole pour qu'elle passe par le point de départ
        # contrib_start = a * (age_start - age_peak)^2 + contrib_peak
        # a = (contrib_start - contrib_peak) / (age_start - age_peak)^2
        if age_start != age_peak:
            self.a = (contrib_start - contrib_peak) / ((age_start - age_peak)**2)
        else:
            self.a = 0 # Cas plat
            
    def get_contribution(self, current_age):
        # Calcul de la parabole
        amount = self.a * (current_age - self.age_peak)**2 + self.contrib_peak
        # On s'assure que la contribution ne devient pas négative
        return max(0, amount)

    def plot_curve(self, duration_years=30, start_age=30):
        """ Affiche la forme théorique de la courbe pour vérification """
        ages = np.arange(start_age, start_age + duration_years)
        amounts = [self.get_contribution(a) for a in ages]
        plt.figure(figsize=(8, 4))
        plt.plot(ages, amounts, color='green', lw=2)
        plt.title(f"Profil de Contribution (Quadratique)\nPic à {self.age_peak} ans : ${self.contrib_peak:,.0f}")
        plt.xlabel("Âge"); plt.ylabel("Épargne Annuelle ($)")
        plt.grid(True, alpha=0.3); plt.show()

# ============================================================================
# 2. CHARGEMENT DONNÉES (Yield & Assets)
# ============================================================================

class YieldCurveBuilder:
    def load_from_csv(self, csv_path):
        try:
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True, na_values=['N/A', ''])
            col_mapping = {'1 Mo': 1/12, '2 Mo': 2/12, '3 Mo': 0.25, '6 Mo': 0.5, '1 Yr': 1, '2 Yr': 2, '3 Yr': 3, '5 Yr': 5, '7 Yr': 7, '10 Yr': 10, '20 Yr': 20, '30 Yr': 30}
            df = df.rename(columns=col_mapping)
            valid_cols = [c for c in df.columns if isinstance(c, (int, float))]
            self.rates_data = (df[valid_cols] / 100.0).resample('M').last().dropna(how='all')
            self.dates = self.rates_data.index
            return self
        except: return None
    
    def get_zero_rate(self, date, maturity):
        if isinstance(date, str): date = pd.to_datetime(date)
        idx = self.rates_data.index.searchsorted(date)
        if idx >= len(self.rates_data): idx = len(self.rates_data) - 1
        row = self.rates_data.iloc[idx].dropna()
        if len(row) < 3: return 0.03
        
        try: return float(CubicSpline(row.index, row.values)(maturity))
        except: return float(np.interp(maturity, row.index, row.values))

class DataLoader:
    @staticmethod
    def get_asset_returns(csv_path):
        try:
            df = pd.read_csv(csv_path, header=1).iloc[1:]
            df.rename(columns={df.columns[0]: 'Date'}, inplace=True)
            df['Date'] = pd.to_datetime(df['Date'])
            return df.set_index('Date').apply(pd.to_numeric, errors='coerce')
        except: return None

# ============================================================================
# 3. MOTEUR GBI AVEC CAPITAL HUMAIN DYNAMIQUE
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
                 start_age, contrib_model):
        self.gpi = gpi; self.psp = psp; self.bond = bond; self.wealth = initial_wealth
        self.dates = psp.index.intersection(gpi.yc.dates)
        self.dates = self.dates[(self.dates >= start_date) & (self.dates <= end_date)].sort_values()
        
        # Gestion de l'âge et des contributions
        self.start_age = start_age
        self.contrib_model = contrib_model
        
        self.history = {'Date': [], 'Wealth': [], 'Funding_Ratio': [], 'Allocation_PSP': [], 'Contrib_Yearly': []}

    def _process_contribution(self, date, i, current_wealth):
        # On calcule l'âge actuel du client à cette date
        # On approxime : Age = Age_depart + (Date_courante - Date_depart) en années
        years_passed = (date - self.dates[0]).days / 365.25
        current_age = self.start_age + years_passed
        
        # Contribution ANNUELLE théorique pour cet âge
        annual_contrib = self.contrib_model.get_contribution(current_age)
        
        # On verse mensuellement (1/12 de la contribution annuelle)
        monthly_contrib = annual_contrib / 12.0
        
        # On n'ajoute pas le tout premier jour (déjà capital initial)
        if i == 0: return current_wealth, 0
        
        return current_wealth + monthly_contrib, monthly_contrib

    def run_gbi(self, floor_pct):
        print(f"🚀 Exécution GBI (Quadratique) sur {len(self.dates)} mois...")
        W, W_year_start = self.wealth, self.wealth
        total_injected = 0
        
        for i, date in enumerate(self.dates):
            # 1. Ajout Contribution (Variable selon l'âge)
            W, injected = self._process_contribution(date, i, W)
            total_injected += injected
            
            if i > 0 and date.month == 1 and self.dates[i-1].month == 12: W_year_start = W
            
            # 2. Indicateurs GBI
            beta = self.gpi.calculate(date)
            beta_start = self.gpi.calculate(pd.Timestamp(f"{date.year}-01-01"))
            
            # Plancher
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
            
            self._record(date, W, fr, w_psp, injected)
        return pd.DataFrame(self.history).set_index('Date')

    def run_tdf(self):
        print(f"🎯 Exécution TDF (Quadratique)...")
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

    def _record(self, date, W, fr, alloc, inj):
        self.history['Date'].append(date); self.history['Wealth'].append(W)
        self.history['Funding_Ratio'].append(fr); self.history['Allocation_PSP'].append(alloc)
        self.history['Contrib_Yearly'].append(inj * 12) # Projection annuelle pour check

# ============================================================================
# 4. EXÉCUTION & PLOT
# ============================================================================

def plot_sim(gbi, tdf):
    fig, ax = plt.subplots(3, 1, figsize=(12, 14))
    
    # Funding Ratio
    ax[0].plot(gbi.index, gbi['Funding_Ratio'], label='GBI', color='#2E86AB', lw=2)
    ax[0].plot(tdf.index, tdf['Funding_Ratio'], label='TDF', color='#A23B72', ls='--')
    ax[0].axhline(1, c='gray', ls=':'); ax[0].set_title('Funding Ratio')
    ax[0].legend(); ax[0].grid(alpha=0.3)
    
    # Richesse
    ax[1].plot(gbi.index, gbi['Wealth'], label='GBI Wealth', color='#2E86AB')
    ax[1].plot(tdf.index, tdf['Wealth'], label='TDF Wealth', color='#A23B72', ls='--')
    ax[1].set_title('Richesse Totale ($)')
    ax[1].legend(); ax[1].grid(alpha=0.3)
    
    # PROFIL DE CONTRIBUTION (Vérification Visuelle)
    ax[2].plot(gbi.index, gbi['Contrib_Yearly'], color='green', label='Contribution Annuelle (Modèle)')
    ax[2].set_title(f'Trajectoire des Contributions (Quadratique)\nStart: ${CONTRIB_START} | Peak: ${CONTRIB_PEAK} à {AGE_PEAK} ans')
    ax[2].set_ylabel('Contribution ($ / an)')
    ax[2].fill_between(gbi.index, 0, gbi['Contrib_Yearly'], color='green', alpha=0.1)
    ax[2].legend(); ax[2].grid(alpha=0.3)
    
    # 2. Allocation
    ax[1].fill_between(gbi.index, 0, gbi['Allocation_PSP']*100, color='#F18F01', alpha=0.5, label='Actions')
    ax[1].fill_between(gbi.index, gbi['Allocation_PSP']*100, 100, color='#006BA6', alpha=0.5, label='Obligations')
    ax[1].set_title('Allocation Dynamique GBI (%)')
    ax[1].set_ylabel('% Actions'); ax[1].set_ylim(0, 100)
    ax[1].legend(loc='lower left'); ax[1].grid(alpha=0.3)
    
    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    # 1. Setup Modèle Quadratique
    hc_curve = HumanCapitalCurve(CLIENT_AGE_START, CONTRIB_START, AGE_PEAK, CONTRIB_PEAK)
    # Affiche la courbe théorique pour vérifier
    # hc_curve.plot_curve() 
    
    yc = YieldCurveBuilder().load_from_csv(YIELD_FILE)
    assets = DataLoader.get_asset_returns(ASSET_FILE)
    
    if yc and assets is not None:
        gpi = GoalPriceIndex(yc, RETIREMENT_DATE)
        
        # 2. Backtest
        engine = StrategyEngine(gpi, assets['US Equity USD Unhedged'], assets['US Government Bond USD Unhedged'],
                              SIMULATION_START, SIMULATION_END, INITIAL_WEALTH,
                              CLIENT_AGE_START, hc_curve)
        
        gbi_res = engine.run_gbi(FLOOR_PERCENT)
        tdf_res = engine.run_tdf()
        
        print(f"\n💰 Richesse Finale GBI : ${gbi_res['Wealth'].iloc[-1]:,.0f}")
        
        plot_sim(gbi_res, tdf_res)