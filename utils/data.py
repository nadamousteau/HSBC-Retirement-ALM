import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# ============================================================================
#  CHARGEMENT DONNÉES (Yield & Assets)
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