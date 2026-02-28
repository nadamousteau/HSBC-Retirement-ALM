import numpy as np
import pandas as pd


class GoalPriceIndex:
    """
    Calcule le Goal Price Index (GPI) = prix de la rente cible.

    Le GPI représente, à la date t, la valeur actualisée d'un flux uniforme
    d'1 € par an pendant `dec_years` années commençant à la retraite.

    Formule (EDHEC Risk Institute) :
        beta(t) = SUM_{k=0}^{dec_years} exp(-r(t, tau_k) * tau_k)
        où tau_k = max(0, t_ret) + k  (années jusqu'au k-ième flux)

    Usage :
        gpi = GoalPriceIndex(yield_curve, retirement_date="2041-12-31", dec_years=20)
        beta = gpi.calculate("2025-01-31")

    Note :
        - Avant la retraite : t_ret > 0 → les flux débutent dans t_ret années
        - Après la retraite : t_ret ≤ 0 → les flux débutent immédiatement
    """

    def __init__(self, yield_curve, retirement_date, dec_years=20):
        """
        Args:
            yield_curve      : instance de YieldCurveBuilder (déjà chargée)
            retirement_date  : date de départ en retraite (str ou Timestamp)
            dec_years        : durée de la phase de décumulation en années
        """
        self.yc = yield_curve
        self.ret_date = pd.to_datetime(retirement_date)
        self.dec_years = dec_years

    def calculate(self, date):
        """
        Calcule beta(date) = valeur actualisée de la rente.

        Returns:
            float: GPI >= 1.0 (minimum 1 pour éviter la division par zéro)
        """
        date = pd.to_datetime(date)

        # Années restantes jusqu'à la retraite (négatif si déjà en retraite)
        t_ret = (self.ret_date - date).days / 365.25

        # Nombre d'années de décumulation restantes
        if t_ret < 0:
            rem = max(0, self.dec_years + t_ret)
        else:
            rem = self.dec_years

        if rem <= 0:
            return 1.0

        n_steps = int(np.ceil(rem))
        beta = 0.0
        for k in range(n_steps):
            tau = max(0.0, t_ret) + k
            r = self.yc.get_zero_rate(date, max(tau, 0.083))  # min 1 mois
            beta += np.exp(-r * tau)

        return max(beta, 1.0)

    def compute_beta_series(self, dates):
        """
        Pré-calcule le vecteur de betas pour une série de dates (backtest).

        Args:
            dates : DatetimeIndex ou liste de dates
        Returns:
            np.ndarray shape (len(dates),)
        """
        return np.array([self.calculate(d) for d in dates])
