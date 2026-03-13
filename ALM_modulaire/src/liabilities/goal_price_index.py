import numpy as np
import pandas as pd


class GoalPriceIndex:
    """
    Calcule le Goal Price Index (GPI) = prix de la rente cible.

    Le GPI représente, a la date t, le facteur d'actualisation d'un objectif
    unique a la date de retraite.

    Formule utilisee ici :
        beta(t) = exp(-r(t, tau_ret) * tau_ret)
        avec tau_ret = max(0, t_ret)

    Usage :
        gpi = GoalPriceIndex(yield_curve, retirement_date="2041-12-31")
        beta = gpi.calculate("2025-01-31")

    Note :
        - Avant la retraite : actualisation sur tau_ret
        - Après la retraite : beta = 1
    """

    def __init__(self, yield_curve, retirement_date):
        """
        Args:
            yield_curve      : instance de YieldCurveBuilder (déjà chargée)
            retirement_date  : date de départ en retraite (str ou Timestamp)
        """
        self.yc = yield_curve
        self.ret_date = pd.to_datetime(retirement_date)

    def calculate(self, date):
        """
        Calcule beta(date) = facteur d'actualisation jusqu'a la retraite.

        Returns:
            float: GPI >= 1e-6
        """
        date = pd.to_datetime(date)

        # Années restantes jusqu'à la retraite (négatif si déjà en retraite)
        t_ret = (self.ret_date - date).days / 365.25

        # A l'echeance (ou apres), le facteur vaut 1.
        if t_ret <= 0:
            return 1.0

        tau = max(t_ret, 1.0 / 12.0)
        r = self.yc.get_zero_rate(date, tau)
        beta = np.exp(-r * tau)

        return max(beta, 1e-6)

    def compute_beta_series(self, dates):
        """
        Pré-calcule le vecteur de betas pour une série de dates (backtest).

        Args:
            dates : DatetimeIndex ou liste de dates
        Returns:
            np.ndarray shape (len(dates),)
        """
        return np.array([self.calculate(d) for d in dates])
