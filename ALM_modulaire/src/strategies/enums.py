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


def validate_settings(settings_mod):
    """
    Valide que les noms de stratégies référencés dans settings.py
    appartiennent bien aux Enums autorisés. Lève ValueError avec un
    message explicite au premier problème détecté.

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
