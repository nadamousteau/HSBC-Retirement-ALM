"""
ENUMS STRATÉGIES — contrat typé phase par phase
================================================
Fige l'ensemble des stratégies admissibles pour chaque phase du pipeline.
Cela remplace les chaînes libres précédemment validées ad hoc dans
`pipeline/accumulation.py` et `pipeline/decumulation.py`, élimine le
fallback silencieux GBI → FIXED_MIX en décumulation, et fait échouer
tôt toute valeur incohérente dans `config/settings.py`.

GBI n'apparaît pas dans `DecumulationStrategy` : le Goal Price Index
nécessite une date de retraite future. Une fois la retraite atteinte,
le concept de "prix du goal" disparaît ; GBI est donc strictement une
stratégie d'accumulation.

Les Enums héritent de `str` pour rester comparables à des chaînes
existantes (`settings.STRATEGIE_DECUMULATION == "FALEH"` continue à
fonctionner sans adaptation).
"""

from enum import Enum


class AccumulationStrategy(str, Enum):
    GBI = "GBI"
    FALEH = "FALEH"
    FIXED_MIX = "FIXED_MIX"
    TARGET_DATE = "TARGET_DATE"


class DecumulationStrategy(str, Enum):
    FALEH = "FALEH"
    FIXED_MIX = "FIXED_MIX"
    TARGET_DATE = "TARGET_DATE"
    ANNUITY = "ANNUITY"


def _fmt(enum_cls):
    return ", ".join(m.value for m in enum_cls)


_LIABILITY_INCOME_MODES = ("REAL", "NOMINAL")
_LIABILITY_HORIZON_MODES = ("FIXED", "ACTUARIAL")


def validate_settings(settings_mod):
    """
    Valide que les noms de stratégies référencés dans settings.py
    appartiennent bien aux Enums autorisés. Lève ValueError avec un
    message explicite au premier problème détecté.

    Vague 2 / Tâche C — valide aussi les LIABILITY_* :
        - existence et type de chaque clé,
        - mode "REAL"/"NOMINAL" et "FIXED"/"ACTUARIAL",
        - cohérence horizon : NB_ANNEES_DECUMULATION doit couvrir
          `RetirementLiability.expected_duration_years() + 5 ans` de marge
          (sinon les retraits seraient tronqués prématurément).

    À appeler en tout début de main() pour échouer vite.
    """
    methode_defaut = getattr(settings_mod, "METHODE_DEFAUT", None)
    if methode_defaut is not None:
        try:
            AccumulationStrategy(methode_defaut)
        except ValueError:
            raise ValueError(
                f"settings.METHODE_DEFAUT='{methode_defaut}' invalide. "
                f"Valeurs autorisées : {_fmt(AccumulationStrategy)}"
            )

    strategies_a_comparer = getattr(settings_mod, "STRATEGIES_A_COMPARER", None)
    if strategies_a_comparer is not None:
        for name in strategies_a_comparer:
            try:
                AccumulationStrategy(name)
            except ValueError:
                raise ValueError(
                    f"settings.STRATEGIES_A_COMPARER contient '{name}' invalide. "
                    f"Valeurs autorisées : {_fmt(AccumulationStrategy)}"
                )

    strategie_decum = getattr(settings_mod, "STRATEGIE_DECUMULATION", None)
    if strategie_decum is not None:
        try:
            DecumulationStrategy(strategie_decum)
        except ValueError:
            raise ValueError(
                f"settings.STRATEGIE_DECUMULATION='{strategie_decum}' invalide. "
                f"GBI n'est pas une stratégie de décumulation valide (GPI sans "
                f"date de retraite future). Valeurs autorisées : "
                f"{_fmt(DecumulationStrategy)}"
            )

    # ── LIABILITY_* (Vague 2 / Tâche C) ───────────────────────────────────
    _validate_liability_settings(settings_mod)


def _validate_liability_settings(settings_mod):
    """
    Vérifie la présence et la cohérence des clés LIABILITY_* + horizon
    décumulation. Importe `RetirementLiability` localement pour éviter
    une dépendance circulaire (enums.py est importé très tôt).
    """
    required_keys = {
        "LIABILITY_TARGET_INCOME_MONTHLY": (int, float),
        "LIABILITY_INCOME_MODE": str,
        "LIABILITY_HORIZON_MODE": str,
        "LIABILITY_HORIZON_YEARS_FIXED": int,
        "LIABILITY_DISCOUNT_RATE": (int, float),
        "LIABILITY_INFLATION_EXPECTED": (int, float),
    }
    for key, expected_type in required_keys.items():
        if not hasattr(settings_mod, key):
            raise ValueError(
                f"settings manque la clé `{key}` (cf. config/settings.py §7c). "
                f"Vague 2 / Tâche C : cette clé remplace les anciens "
                f"RETRAIT_MENSUEL_REEL et TAUX_ACTUALISATION_PLANCHER."
            )
        val = getattr(settings_mod, key)
        if not isinstance(val, expected_type):
            raise ValueError(
                f"settings.{key} = {val!r} : type {type(val).__name__} "
                f"invalide (attendu {expected_type})."
            )

    income_mode = settings_mod.LIABILITY_INCOME_MODE
    if income_mode not in _LIABILITY_INCOME_MODES:
        raise ValueError(
            f"settings.LIABILITY_INCOME_MODE='{income_mode}' invalide. "
            f"Valeurs autorisées : {', '.join(_LIABILITY_INCOME_MODES)}"
        )

    horizon_mode = settings_mod.LIABILITY_HORIZON_MODE
    if horizon_mode not in _LIABILITY_HORIZON_MODES:
        raise ValueError(
            f"settings.LIABILITY_HORIZON_MODE='{horizon_mode}' invalide. "
            f"Valeurs autorisées : {', '.join(_LIABILITY_HORIZON_MODES)}"
        )

    if settings_mod.LIABILITY_TARGET_INCOME_MONTHLY <= 0:
        raise ValueError(
            f"settings.LIABILITY_TARGET_INCOME_MONTHLY = "
            f"{settings_mod.LIABILITY_TARGET_INCOME_MONTHLY} : doit être > 0."
        )

    # Cohérence horizon : NB_ANNEES_DECUMULATION doit couvrir la durée du
    # passif + marge de sécurité de 5 ans pour absorber la queue de
    # distribution actuarielle.
    from src.liabilities.retirement_objective import build_liability_from_settings
    liability = build_liability_from_settings(settings_mod)
    expected = liability.expected_duration_years()
    nb_decum = getattr(settings_mod, "NB_ANNEES_DECUMULATION", 0)
    marge_min = int(expected) + 5
    if nb_decum < marge_min:
        raise ValueError(
            f"settings.NB_ANNEES_DECUMULATION = {nb_decum} insuffisant : "
            f"le passif client a une durée espérée de {expected:.1f} ans "
            f"(mode {horizon_mode}). Augmenter NB_ANNEES_DECUMULATION à "
            f"au moins {marge_min} ans (durée espérée + 5 ans de marge)."
        )
