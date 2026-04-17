"""
RNG BUNDLE — point d'entrée unique pour l'aléa
===============================================
Toute source stochastique du pipeline dérive d'un unique `GLOBAL_SEED` via
`numpy.random.SeedSequence.spawn()`. Chaque composant reçoit son propre
`Generator` (isolation garantie entre modules), et la reproductibilité est
contrôlée à un seul endroit : `config.settings.GLOBAL_SEED`.

Utilisation typique :

    from src.utils import make_rng_bundle
    bundle = make_rng_bundle(settings.GLOBAL_SEED)
    inflation = generer_inflation_vasicek(..., rng=bundle["inflation_accum"])

Si `global_seed is None`, le `SeedSequence` tire son entropie de l'OS : le
bundle est alors non-reproductible d'un run à l'autre (comportement voulu
lorsque l'utilisateur demande explicitement un tirage divergent).
"""

import numpy as np


# Liste ordonnée des RNG fournis par le bundle. L'ordre importe : il fixe la
# correspondance entre `spawn()` et la clé, donc modifier cette liste change
# les flux aléatoires de tous les composants.
#
# Note : depuis la Tâche 3, les trajectoires EQ/BD et inflation sont générées
# en une seule passe couvrant accumulation + décumulation — plus besoin de
# RNG séparés pour la phase retraite.
_BUNDLE_KEYS = (
    "equity_bonds",
    "inflation",
    "gbi_ns",
    "faleh_gse",
)


def make_rng_bundle(global_seed):
    """
    Crée un bundle de `numpy.random.Generator` indépendants.

    Args:
        global_seed : int ou None — graine maître.
                      None → entropie OS (non-reproductible).

    Returns:
        dict { str: np.random.Generator } — un Generator par composant.
    """
    ss = np.random.SeedSequence(global_seed)
    children = ss.spawn(len(_BUNDLE_KEYS))
    return {
        key: np.random.default_rng(child)
        for key, child in zip(_BUNDLE_KEYS, children)
    }
