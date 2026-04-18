"""
Tests Vague 2 — Tâche C
========================
Vérifie le comportement de `RetirementLiability` :
    1. Validation des arguments (modes, signes, bornes).
    2. Conversion REAL → NOMINAL via `monthly_income_nominal_at_retirement`.
    3. Cohérence des durées en mode FIXED et ACTUARIAL.
    4. VA du passif : annuité certaine en FIXED, rente viagère en ACTUARIAL.
    5. `funded_ratio` : ratio sans biais sur scalaires et arrays.
    6. `build_liability_from_settings` lève ValueError si une clé manque.
    7. Validation `validate_settings` rejette horizon insuffisant.

Lancement :
    pytest tests/test_retirement_liability.py -v

Ces tests sont indépendants du pipeline de simulation (pas d'import
main.py) et utilisent des modules `types.SimpleNamespace` pour simuler
le module `settings`.
"""

import sys
import types
from pathlib import Path

import numpy as np
import pytest

# Permet de lancer pytest depuis ALM_modulaire/ ou la racine du repo.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.liabilities.retirement_objective import (
    RetirementLiability,
    build_liability_from_settings,
)
from src.liabilities.liability_valuation import funded_ratio
from src.liabilities.mortality import annuity_factor_monthly


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_settings(**overrides):
    """Construit un faux module `settings` avec les LIABILITY_* par défaut."""
    base = dict(
        AGE_DEPART=20,
        NB_ANNEES_ACCUMULATION=40,
        NB_ANNEES_DECUMULATION=30,
        LIABILITY_TARGET_INCOME_MONTHLY=2000,
        LIABILITY_INCOME_MODE="REAL",
        LIABILITY_HORIZON_MODE="ACTUARIAL",
        LIABILITY_HORIZON_YEARS_FIXED=25,
        LIABILITY_DISCOUNT_RATE=0.02,
        LIABILITY_INFLATION_EXPECTED=0.023,
    )
    base.update(overrides)
    return types.SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Validation des arguments
# ---------------------------------------------------------------------------
class TestValidation:
    def test_target_income_doit_etre_positif(self):
        with pytest.raises(ValueError, match="target_income_monthly"):
            RetirementLiability(
                target_income_monthly=0.0,
                income_mode="REAL",
                age_retirement=60,
                horizon_mode="FIXED",
                horizon_years_fixed=25,
                discount_rate=0.02,
                inflation_expected=0.023,
                nb_annees_accumulation=40,
            )

    def test_income_mode_invalide(self):
        with pytest.raises(ValueError, match="income_mode"):
            RetirementLiability(
                target_income_monthly=2000,
                income_mode="EUROS",  # invalide
                age_retirement=60,
                horizon_mode="FIXED",
                horizon_years_fixed=25,
                discount_rate=0.02,
                inflation_expected=0.023,
                nb_annees_accumulation=40,
            )

    def test_horizon_mode_invalide(self):
        with pytest.raises(ValueError, match="horizon_mode"):
            RetirementLiability(
                target_income_monthly=2000,
                income_mode="REAL",
                age_retirement=60,
                horizon_mode="INFINI",  # invalide
                horizon_years_fixed=25,
                discount_rate=0.02,
                inflation_expected=0.023,
                nb_annees_accumulation=40,
            )

    def test_age_retirement_au_dela_omega(self):
        with pytest.raises(ValueError, match="age_retirement"):
            RetirementLiability(
                target_income_monthly=2000,
                income_mode="REAL",
                age_retirement=130,  # > OMEGA_AGE
                horizon_mode="FIXED",
                horizon_years_fixed=25,
                discount_rate=0.02,
                inflation_expected=0.023,
                nb_annees_accumulation=40,
            )

    def test_horizon_fixed_doit_etre_positif(self):
        with pytest.raises(ValueError, match="horizon_years_fixed"):
            RetirementLiability(
                target_income_monthly=2000,
                income_mode="REAL",
                age_retirement=60,
                horizon_mode="FIXED",
                horizon_years_fixed=0,
                discount_rate=0.02,
                inflation_expected=0.023,
                nb_annees_accumulation=40,
            )

    def test_immutable_dataclass(self):
        liab = RetirementLiability(
            target_income_monthly=2000,
            income_mode="REAL",
            age_retirement=60,
            horizon_mode="FIXED",
            horizon_years_fixed=25,
            discount_rate=0.02,
            inflation_expected=0.023,
            nb_annees_accumulation=40,
        )
        with pytest.raises(Exception):
            # frozen=True → FrozenInstanceError (sous-classe de AttributeError)
            liab.target_income_monthly = 3000


# ---------------------------------------------------------------------------
# Conversion REAL → NOMINAL
# ---------------------------------------------------------------------------
class TestIncomeMode:
    def test_nominal_pas_de_capitalisation(self):
        liab = RetirementLiability(
            target_income_monthly=4000,
            income_mode="NOMINAL",
            age_retirement=60,
            horizon_mode="FIXED",
            horizon_years_fixed=25,
            discount_rate=0.02,
            inflation_expected=0.05,  # ignoré en mode NOMINAL
            nb_annees_accumulation=40,
        )
        assert liab.monthly_income_nominal_at_retirement() == 4000.0

    def test_real_capitalise_par_inflation_expected(self):
        liab = RetirementLiability(
            target_income_monthly=2000,
            income_mode="REAL",
            age_retirement=60,
            horizon_mode="FIXED",
            horizon_years_fixed=25,
            discount_rate=0.02,
            inflation_expected=0.023,
            nb_annees_accumulation=40,
        )
        attendu = 2000 * (1.023 ** 40)
        np.testing.assert_allclose(
            liab.monthly_income_nominal_at_retirement(), attendu, rtol=1e-12
        )

    def test_real_inflation_zero_egale_target(self):
        liab = RetirementLiability(
            target_income_monthly=2000,
            income_mode="REAL",
            age_retirement=60,
            horizon_mode="FIXED",
            horizon_years_fixed=25,
            discount_rate=0.02,
            inflation_expected=0.0,
            nb_annees_accumulation=40,
        )
        np.testing.assert_allclose(
            liab.monthly_income_nominal_at_retirement(), 2000.0, rtol=1e-12
        )


# ---------------------------------------------------------------------------
# Durées
# ---------------------------------------------------------------------------
class TestExpectedDuration:
    def test_fixed_retourne_horizon_years_fixed(self):
        liab = RetirementLiability(
            target_income_monthly=2000,
            income_mode="REAL",
            age_retirement=60,
            horizon_mode="FIXED",
            horizon_years_fixed=25,
            discount_rate=0.02,
            inflation_expected=0.023,
            nb_annees_accumulation=40,
        )
        assert liab.expected_duration_years() == 25.0

    def test_actuarial_age_60_environ_21_ans(self):
        """e_60 ≈ 21.6 ans pour TH 00-02."""
        liab = RetirementLiability(
            target_income_monthly=2000,
            income_mode="REAL",
            age_retirement=60,
            horizon_mode="ACTUARIAL",
            horizon_years_fixed=25,
            discount_rate=0.02,
            inflation_expected=0.023,
            nb_annees_accumulation=40,
        )
        e60 = liab.expected_duration_years()
        # Tolérance large (interpolation linéaire des checkpoints)
        assert 19.0 < e60 < 24.0, f"e_60 = {e60}, attendu autour de 21.6"

    def test_actuarial_age_80_inferieur_age_60(self):
        liab_60 = RetirementLiability(
            target_income_monthly=2000,
            income_mode="REAL",
            age_retirement=60,
            horizon_mode="ACTUARIAL",
            horizon_years_fixed=25,
            discount_rate=0.02,
            inflation_expected=0.023,
            nb_annees_accumulation=40,
        )
        liab_80 = RetirementLiability(
            target_income_monthly=2000,
            income_mode="REAL",
            age_retirement=80,
            horizon_mode="ACTUARIAL",
            horizon_years_fixed=25,
            discount_rate=0.02,
            inflation_expected=0.023,
            nb_annees_accumulation=60,
        )
        assert liab_80.expected_duration_years() < liab_60.expected_duration_years()


# ---------------------------------------------------------------------------
# Présent value du passif
# ---------------------------------------------------------------------------
class TestPresentValue:
    def test_fixed_taux_zero_egale_revenu_fois_nb_mois(self):
        liab = RetirementLiability(
            target_income_monthly=1000,
            income_mode="NOMINAL",
            age_retirement=60,
            horizon_mode="FIXED",
            horizon_years_fixed=25,
            discount_rate=0.0,  # pas d'actualisation
            inflation_expected=0.0,
            nb_annees_accumulation=40,
        )
        # 1000 € × 25 ans × 12 mois = 300 000 €
        np.testing.assert_allclose(
            liab.present_value_at_retirement(), 1000 * 25 * 12, rtol=1e-12
        )

    def test_fixed_annuite_decroit_avec_taux(self):
        liab_0 = RetirementLiability(
            target_income_monthly=1000,
            income_mode="NOMINAL",
            age_retirement=60,
            horizon_mode="FIXED",
            horizon_years_fixed=25,
            discount_rate=0.0,
            inflation_expected=0.0,
            nb_annees_accumulation=40,
        )
        liab_4 = RetirementLiability(
            target_income_monthly=1000,
            income_mode="NOMINAL",
            age_retirement=60,
            horizon_mode="FIXED",
            horizon_years_fixed=25,
            discount_rate=0.04,
            inflation_expected=0.0,
            nb_annees_accumulation=40,
        )
        assert liab_4.present_value_at_retirement() < liab_0.present_value_at_retirement()

    def test_actuarial_egale_ax12_fois_revenu(self):
        """En mode ACTUARIAL, PV = revenu × ä^(12)_x (sans loading)."""
        liab = RetirementLiability(
            target_income_monthly=1500,
            income_mode="NOMINAL",
            age_retirement=65,
            horizon_mode="ACTUARIAL",
            horizon_years_fixed=25,
            discount_rate=0.02,
            inflation_expected=0.0,
            nb_annees_accumulation=40,
        )
        af = annuity_factor_monthly(
            age_retraite_years=65, rate_annual=0.02, loading=0.0
        )
        np.testing.assert_allclose(
            liab.present_value_at_retirement(), 1500 * af, rtol=1e-12
        )

    def test_required_capital_alias_de_present_value(self):
        liab = RetirementLiability(
            target_income_monthly=2000,
            income_mode="REAL",
            age_retirement=60,
            horizon_mode="ACTUARIAL",
            horizon_years_fixed=25,
            discount_rate=0.02,
            inflation_expected=0.023,
            nb_annees_accumulation=40,
        )
        assert liab.required_capital_at_retirement() == liab.present_value_at_retirement()


# ---------------------------------------------------------------------------
# funded_ratio
# ---------------------------------------------------------------------------
class TestFundedRatio:
    def test_scalar_simple(self):
        liab = RetirementLiability(
            target_income_monthly=1000,
            income_mode="NOMINAL",
            age_retirement=60,
            horizon_mode="FIXED",
            horizon_years_fixed=25,
            discount_rate=0.0,
            inflation_expected=0.0,
            nb_annees_accumulation=40,
        )
        pv = liab.required_capital_at_retirement()  # = 300_000
        np.testing.assert_allclose(funded_ratio(pv, liab), 1.0, rtol=1e-12)
        np.testing.assert_allclose(funded_ratio(2 * pv, liab), 2.0, rtol=1e-12)
        np.testing.assert_allclose(funded_ratio(0.0, liab), 0.0, rtol=1e-12)

    def test_vectorise(self):
        liab = RetirementLiability(
            target_income_monthly=1000,
            income_mode="NOMINAL",
            age_retirement=60,
            horizon_mode="FIXED",
            horizon_years_fixed=25,
            discount_rate=0.0,
            inflation_expected=0.0,
            nb_annees_accumulation=40,
        )
        pv = liab.required_capital_at_retirement()
        capitaux = np.array([0.5 * pv, pv, 1.5 * pv])
        ratios = funded_ratio(capitaux, liab)
        np.testing.assert_allclose(ratios, np.array([0.5, 1.0, 1.5]), rtol=1e-12)


# ---------------------------------------------------------------------------
# build_liability_from_settings
# ---------------------------------------------------------------------------
class TestBuilderFromSettings:
    def test_construction_par_defaut(self):
        s = _make_settings()
        liab = build_liability_from_settings(s)
        assert liab.target_income_monthly == 2000
        assert liab.income_mode == "REAL"
        assert liab.age_retirement == 60  # AGE_DEPART + NB_ANNEES_ACCUMULATION
        assert liab.horizon_mode == "ACTUARIAL"
        assert liab.discount_rate == 0.02

    def test_clef_manquante_leve_attribute_error(self):
        s = _make_settings()
        delattr(s, "LIABILITY_TARGET_INCOME_MONTHLY")
        with pytest.raises(AttributeError):
            build_liability_from_settings(s)


# ---------------------------------------------------------------------------
# validate_settings (LIABILITY_*)
# ---------------------------------------------------------------------------
class TestValidateSettings:
    def test_horizon_decum_insuffisant_leve(self):
        from src.strategies.enums import validate_settings
        s = _make_settings(NB_ANNEES_DECUMULATION=10)  # << expected_duration
        with pytest.raises(ValueError, match="NB_ANNEES_DECUMULATION"):
            validate_settings(s)

    def test_income_mode_invalide_leve(self):
        from src.strategies.enums import validate_settings
        s = _make_settings(LIABILITY_INCOME_MODE="EUROS")
        with pytest.raises(ValueError, match="LIABILITY_INCOME_MODE"):
            validate_settings(s)

    def test_horizon_mode_invalide_leve(self):
        from src.strategies.enums import validate_settings
        s = _make_settings(LIABILITY_HORIZON_MODE="INFINI")
        with pytest.raises(ValueError, match="LIABILITY_HORIZON_MODE"):
            validate_settings(s)

    def test_target_income_negatif_leve(self):
        from src.strategies.enums import validate_settings
        s = _make_settings(LIABILITY_TARGET_INCOME_MONTHLY=-100)
        with pytest.raises(ValueError, match="LIABILITY_TARGET_INCOME_MONTHLY"):
            validate_settings(s)

    def test_settings_valides_passent(self):
        from src.strategies.enums import validate_settings
        s = _make_settings()
        validate_settings(s)  # ne doit pas lever


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
