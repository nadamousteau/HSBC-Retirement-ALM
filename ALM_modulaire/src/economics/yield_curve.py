import warnings
import pandas as pd
import numpy as np
from scipy.interpolate import CubicSpline


class YieldCurveBuilder:
    """
    Charge et interpole la courbe des taux zéro-coupon US Treasury.
    Utilisé par GoalPriceIndex pour calculer le prix de la rente cible (GBI).
    """

    # Correspondance colonnes CSV → maturités en années
    _COL_MAP = {
        '1 Mo': 1/12, '2 Mo': 2/12, '3 Mo': 0.25, '6 Mo': 0.5,
        '1 Yr': 1, '2 Yr': 2, '3 Yr': 3, '5 Yr': 5,
        '7 Yr': 7, '10 Yr': 10, '20 Yr': 20, '30 Yr': 30
    }

    def load_from_csv(self, csv_path):
        """
        Charge les données de taux depuis le fichier CSV.
        Retourne self pour permettre le chaînage : YieldCurveBuilder().load_from_csv(...)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = pd.read_csv(csv_path, index_col=0, parse_dates=True, na_values=['N/A', ''])
        df = df.rename(columns=self._COL_MAP)
        valid_cols = [c for c in df.columns if isinstance(c, (int, float))]

        self.rates_data = (df[valid_cols] / 100.0).resample('ME').last().dropna(how='all')
        self.dates = self.rates_data.index
        self._last_known_date = self.dates[-1]
        return self

    def get_zero_rate(self, date, maturity):
        """
        Retourne le taux zéro-coupon interpolé (CubicSpline) pour une maturité donnée.

        Args:
            date      : date (str ou Timestamp) à laquelle on lit la courbe
            maturity  : maturité en années
        Returns:
            float: taux zéro-coupon (ex: 0.045 pour 4.5%)
        """
        if isinstance(date, str):
            date = pd.to_datetime(date)

        # Si la date dépasse les données disponibles, on utilise la dernière observation
        if date > self._last_known_date:
            date = self._last_known_date

        idx = self.rates_data.index.searchsorted(date)
        if idx >= len(self.rates_data):
            idx = len(self.rates_data) - 1

        row = self.rates_data.iloc[idx].dropna()
        if len(row) < 3:
            return 0.03  # Fallback: 3%

        try:
            val = float(CubicSpline(row.index.astype(float), row.values)(maturity))
        except Exception:
            val = float(np.interp(maturity, row.index.astype(float), row.values))

        # Clamp entre -2% et 20% pour éviter les extrapolations absurdes
        return float(np.clip(val, -0.02, 0.20))

    def is_loaded(self):
        return hasattr(self, 'rates_data') and self.rates_data is not None
