"""
RETIREMENT OBJECTIVE — Modélisation explicite du passif retraite
=================================================================
Vague 2 / Tâche C : nouvelle abstraction unifiée pour le passif retraite.

Avant cette tâche, deux paramètres scalaires (`RETRAIT_MENSUEL_REEL` et
`TAUX_ACTUALISATION_PLANCHER`) servaient à la fois la décumulation et
implicitement la cible Faleh (heuristique 80% × FV portefeuille). Aucun
endroit du code ne représentait *explicitement* ce que le client voulait
financer. Conséquences :
    - `_estimate_target_wealth()` de Faleh ignorait totalement les
      volontés client (revenu, horizon) et calculait la cible à partir
      du portefeuille d'apports, ce qui biaisait l'optimisation.
    - L'horizon de la décumulation (`NB_ANNEES_DECUMULATION`) n'avait
      aucun lien avec l'espérance de vie du client.
    - Toute extension (revenu nominal, horizon actuariel, scénarios
      conjoint, etc.) imposait de propager des paramètres ad hoc dans
      tout le pipeline.

`RetirementLiability` est une dataclass immuable (frozen=True) qui porte
les volontés client de manière explicite et fournit les méthodes :
    - `expected_duration_years()`   → durée espérée du retrait
    - `monthly_income_nominal_at_retirement()` → revenu nominal au moment
      de la liquidation (post-inflation accumulation)
    - `present_value_at_retirement()` → VA des retraits futurs à T_retraite
    - `required_capital_at_retirement()` → capital nécessaire pour financer
      le passif (utilisé par Faleh comme target_wealth)

Le module expose également `build_liability_from_settings(settings)`,
seul point d'entrée du pipeline pour instancier l'objet à partir du
fichier de configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .mortality import annuity_factor_monthly, survival_probability, OMEGA_AGE


IncomeMode = Literal["REAL", "NOMINAL"]
HorizonMode = Literal["FIXED", "ACTUARIAL"]


@dataclass(frozen=True)
class RetirementLiability:
    """
    Passif retraite client (immutable).

    Attributes:
        target_income_monthly : revenu mensuel cible.
            - Si `income_mode == "REAL"` : exprimé en € constants à la
              date de départ (T0). Devient `target_income_monthly *
              (1 + inflation_expected)^age_accum` au moment de la retraite.
            - Si `income_mode == "NOMINAL"` : exprimé en €  futurs au moment
              de la liquidation (pas d'inflation appliquée).
        income_mode : "REAL" ou "NOMINAL" (cf. ci-dessus).
        age_retirement : âge de liquidation du passif (= AGE_DEPART +
                         NB_ANNEES_ACCUMULATION).
        horizon_mode : "FIXED" → durée déterministe de `horizon_years_fixed`
                       années. "ACTUARIAL" → durée = espérance de vie
                       résiduelle calculée sur la table TH 00-02.
        horizon_years_fixed : durée fixe (uniquement si horizon_mode == "FIXED").
        discount_rate : taux d'actualisation annuel pour calculer la VA
                        du passif (anciennement TAUX_ACTUALISATION_PLANCHER).
        inflation_expected : inflation annuelle anticipée (sert à projeter
                             un revenu RÉEL vers le NOMINAL au moment de la
                             retraite, et à indexer la VA si nécessaire).
        nb_annees_accumulation : durée d'accumulation jusqu'à la retraite.
                                 Sert au passage REAL → NOMINAL.
    """

    target_income_monthly: float
    income_mode: IncomeMode
    age_retirement: int
    horizon_mode: HorizonMode
    horizon_years_fixed: int
    discount_rate: float
    inflation_expected: float
    nb_annees_accumulation: int

    # ------------------------------------------------------------------
    # Validation post-init
    # ------------------------------------------------------------------
    def __post_init__(self):
        if self.target_income_monthly <= 0:
            raise ValueError(
                f"target_income_monthly doit être > 0 (reçu {self.target_income_monthly})"
            )
        if self.income_mode not in ("REAL", "NOMINAL"):
            raise ValueError(
                f"income_mode doit être 'REAL' ou 'NOMINAL' (reçu {self.income_mode!r})"
            )
        if self.horizon_mode not in ("FIXED", "ACTUARIAL"):
            raise ValueError(
                f"horizon_mode doit être 'FIXED' ou 'ACTUARIAL' (reçu {self.horizon_mode!r})"
            )
        if self.age_retirement <= 0 or self.age_retirement >= OMEGA_AGE:
            raise ValueError(
                f"age_retirement = {self.age_retirement} hors bornes (1..{OMEGA_AGE - 1})"
            )
        if self.horizon_mode == "FIXED" and self.horizon_years_fixed <= 0:
            raise ValueError(
                f"horizon_years_fixed doit être > 0 en mode FIXED "
                f"(reçu {self.horizon_years_fixed})"
            )
        if self.discount_rate <= -1.0:
            raise ValueError(
                f"discount_rate = {self.discount_rate} : doit être > -1"
            )
        if self.inflation_expected <= -1.0:
            raise ValueError(
                f"inflation_expected = {self.inflation_expected} : doit être > -1"
            )
        if self.nb_annees_accumulation < 0:
            raise ValueError(
                f"nb_annees_accumulation = {self.nb_annees_accumulation} : doit être >= 0"
            )

    # ------------------------------------------------------------------
    # Durée
    # ------------------------------------------------------------------
    def expected_duration_years(self) -> float:
        """
        Durée espérée du passif en années.

        - FIXED     : `horizon_years_fixed` (déterministe).
        - ACTUARIAL : espérance de vie résiduelle à `age_retirement`,
                      calculée sur la table TH 00-02 par sommation
                      Σ_{k=1}^{ω-x} ₖp_x.
        """
        if self.horizon_mode == "FIXED":
            return float(self.horizon_years_fixed)

        # Espérance de vie résiduelle e_x = Σ_{k=1}^{ω-x} (l_{x+k}/l_x)
        x = float(self.age_retirement)
        ages = np.arange(int(np.ceil(x)) + 1, OMEGA_AGE + 1, dtype=np.float64)
        survie = np.array([survival_probability(x, y) for y in ages])
        return float(np.sum(survie))

    # ------------------------------------------------------------------
    # Revenu nominal au moment de la liquidation
    # ------------------------------------------------------------------
    def monthly_income_nominal_at_retirement(self) -> float:
        """
        Revenu mensuel nominal au moment de la retraite.

        - Si income_mode == "NOMINAL" : retourne directement la cible
          (le client a saisi un montant futur).
        - Si income_mode == "REAL" : capitalise par
          (1 + inflation_expected) ** nb_annees_accumulation.
        """
        if self.income_mode == "NOMINAL":
            return float(self.target_income_monthly)

        infl_factor = (1.0 + self.inflation_expected) ** self.nb_annees_accumulation
        return float(self.target_income_monthly * infl_factor)

    # ------------------------------------------------------------------
    # Valeur présente du passif à la date de retraite
    # ------------------------------------------------------------------
    def present_value_at_retirement(self) -> float:
        """
        Capital nécessaire au moment de la retraite pour financer le
        flux de retraits futurs.

        Pour les deux modes d'horizon, on actualise un flux mensuel
        constant (NOMINAL) au taux `discount_rate`. En mode ACTUARIAL,
        on pondère chaque flux par la probabilité de survie ₖp_x — ce
        qui revient à utiliser le facteur de rente viagère mensuelle
        `ä^(12)_x` (cf. mortality.annuity_factor_monthly) et permet
        de financer le revenu strictement tant que le client vit.

        Returns:
            float — capital VA à T_retraite (€ nominaux).
        """
        income_m = self.monthly_income_nominal_at_retirement()

        if self.horizon_mode == "ACTUARIAL":
            # ä^(12)_x : facteur déjà multiplié par 12 mois (somme directe
            # de tous les paiements actualisés). On ne passe PAS de loading
            # ici (le loading assureur est géré par la stratégie ANNUITY,
            # pas par le passif client).
            af = annuity_factor_monthly(
                age_retraite_years=self.age_retirement,
                rate_annual=self.discount_rate,
                loading=0.0,
            )
            return income_m * af

        # FIXED : annuité mensuelle classique (formule actuarielle certaine)
        n_months = int(self.horizon_years_fixed * 12)
        if abs(self.discount_rate) < 1e-12:
            return income_m * n_months
        r_m = (1.0 + self.discount_rate) ** (1.0 / 12.0) - 1.0
        return income_m * (1.0 - (1.0 + r_m) ** (-n_months)) / r_m

    # ------------------------------------------------------------------
    # Cible de capital pour Faleh
    # ------------------------------------------------------------------
    def required_capital_at_retirement(self) -> float:
        """
        Alias sémantique de `present_value_at_retirement`.

        Utilisé par `FalehStrategy._estimate_target_wealth()` comme
        target_wealth, en remplacement de l'ancienne heuristique
        80 % × FV portefeuille (qui ignorait le passif client).
        """
        return self.present_value_at_retirement()


# ----------------------------------------------------------------------
# Builder
# ----------------------------------------------------------------------
def build_liability_from_settings(settings_mod) -> RetirementLiability:
    """
    Construit `RetirementLiability` à partir du module de configuration.

    Convention de nommage des clés (cf. config/settings.py §7c) :
        LIABILITY_TARGET_INCOME_MONTHLY  → target_income_monthly
        LIABILITY_INCOME_MODE            → "REAL" ou "NOMINAL"
        LIABILITY_HORIZON_MODE           → "FIXED" ou "ACTUARIAL"
        LIABILITY_HORIZON_YEARS_FIXED    → horizon en mode FIXED
        LIABILITY_DISCOUNT_RATE          → taux d'actualisation
        LIABILITY_INFLATION_EXPECTED     → inflation anticipée

    Les autres champs (age_retirement, nb_annees_accumulation) sont
    dérivés des paramètres temporels existants (AGE_DEPART,
    NB_ANNEES_ACCUMULATION).

    Args:
        settings_mod : module ou objet exposant les attributs ci-dessus.

    Returns:
        RetirementLiability immuable validé.
    """
    age_retraite = (
        getattr(settings_mod, "AGE_DEPART") + getattr(settings_mod, "NB_ANNEES_ACCUMULATION")
    )

    return RetirementLiability(
        target_income_monthly=float(settings_mod.LIABILITY_TARGET_INCOME_MONTHLY),
        income_mode=settings_mod.LIABILITY_INCOME_MODE,
        age_retirement=int(age_retraite),
        horizon_mode=settings_mod.LIABILITY_HORIZON_MODE,
        horizon_years_fixed=int(settings_mod.LIABILITY_HORIZON_YEARS_FIXED),
        discount_rate=float(settings_mod.LIABILITY_DISCOUNT_RATE),
        inflation_expected=float(settings_mod.LIABILITY_INFLATION_EXPECTED),
        nb_annees_accumulation=int(settings_mod.NB_ANNEES_ACCUMULATION),
    )
