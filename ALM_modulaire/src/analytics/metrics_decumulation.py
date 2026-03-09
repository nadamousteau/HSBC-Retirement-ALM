import numpy as np

# =============================================================================
# MÉTRIQUES DE DÉCUMULATION
# =============================================================================

def annees_couvertes_par_rente(capitaux_finaux, rente_mensuelle_cible, taux_mensuel):
    """
    Calcule, pour chaque simulation, le nombre d'années pendant lesquelles
    le capital final couvre une rente mensuelle cible.

    Si taux_mensuel > 0, on suppose que le capital résiduel continue de fructifier
    (rente viagère à taux constant). On résout alors :
        capital = rente * (1 - (1+r)^(-n)) / r   =>  n = -ln(1 - capital*r/rente) / ln(1+r)

    Parametres
    ----------
    capitaux_finaux    : (N,) capital disponible au début de la retraite, par simulation
    rente_mensuelle_cible : montant mensuel souhaité (€)
    taux_mensuel       : rendement mensuel du capital résiduel (0 = capital non investi)

    Return
    -------
    annees : (N,) durée de couverture en années (np.inf si capital suffisant à vie)
    """
    annees = np.empty(len(capitaux_finaux))

    for i, cap in enumerate(capitaux_finaux):
        if cap <= 0:
            annees[i] = 0.0
            continue

        if taux_mensuel <= 0:
            nb_mois = cap / rente_mensuelle_cible
        else:
            ratio = cap * taux_mensuel / rente_mensuelle_cible
            if ratio >= 1.0:
                annees[i] = np.inf
                continue
            nb_mois = -np.log(1.0 - ratio) / np.log(1.0 + taux_mensuel)

        annees[i] = nb_mois / 12.0

    return annees

def annees_reserve(mat_cap_retraite, rente_mensuelle, age_retraite, esperance_vie: float = 85.0):
    """
    Nombre d'années de réserve au-delà de l'espérance de vie :
    combien de temps supplémentaire le capital couvre-t-il après `esperance_vie` ?

    Une valeur négative signifie que la ruine survient avant l'espérance de vie.

    Parametres
    ----------
    mat_cap_retraite  : (T, N)
    rente_mensuelle   : (N,) rente mensuelle consommée par simulation
    age_retraite      : âge au début de la retraite
    esperance_vie     : âge de référence (défaut = 85 ans)

    Returns
    -------
    annees_reserve : (N,)
    """
    n_sims = mat_cap_retraite.shape[1]
    reserves = np.empty(n_sims)

    for i in range(n_sims):
        trajectoire = mat_cap_retraite[:, i]
        # Période jusqu'à ruine
        idx_ruine = np.where(trajectoire <= 0)[0]
        if len(idx_ruine) > 0:
            age_ruine = age_retraite + idx_ruine[0] / 12.0
            reserves[i] = age_ruine - esperance_vie
        else:
            # Capital résiduel en fin de simulation → on estime les années restantes
            cap_final = trajectoire[-1]
            annees_sup = annees_couvertes_par_rente(
                np.array([cap_final]), rente_mensuelle[i]
            )[0]
            age_fin = age_retraite + len(trajectoire) / 12.0 + annees_sup
            reserves[i] = age_fin - esperance_vie

    return reserves


def rente_viagere_equivalente(
    capitaux_finaux: np.ndarray,
    age_retraite: int,
    horizon_ans: int = 30,
    taux_mensuel: float = 0.0
) -> np.ndarray:
    """
    Calcule la rente mensuelle constante équivalente au capital,
    sur un horizon fixé (ex : jusqu'à 95 ans si retraite à 65).

    Formule (annuité constante) :
        si r > 0 : rente = capital * r / (1 - (1+r)^(-n))
        si r = 0 : rente = capital / n

    Parameters
    ----------
    capitaux_finaux : (N,)
    age_retraite    : âge de départ en retraite
    horizon_ans     : durée en années sur laquelle étaler le capital
    taux_mensuel    : rendement mensuel du capital résiduel

    Returns
    -------
    rente : (N,) rente mensuelle en €
    """
    n_mois = horizon_ans * 12

    if taux_mensuel <= 0:
        return capitaux_finaux / n_mois
    else:
        facteur = taux_mensuel / (1.0 - (1.0 + taux_mensuel) ** (-n_mois))
        return capitaux_finaux * facteur


def taux_remplacement_effectif(
    rentes_mensuelles: np.ndarray,
    derniers_salaires: np.ndarray
) -> np.ndarray:
    """
    Taux de remplacement effectif = rente mensuelle perçue / dernier salaire mensuel.

    Parameters
    ----------
    rentes_mensuelles : (N,) revenu total mensuel en retraite (livret A + capital investi)
    derniers_salaires : (N,) dernier salaire mensuel avant retraite

    Returns
    -------
    taux : (N,) ∈ [0, +∞[
    """
    return np.where(derniers_salaires > 0, rentes_mensuelles / derniers_salaires, 0.0)


def capital_residuel_au_deces(
    mat_cap_retraite: np.ndarray,
    ages_deces: np.ndarray,
    age_retraite: int
) -> np.ndarray:
    """
    Capital résiduel (héritage potentiel) au moment du décès simulé.

    Parameters
    ----------
    mat_cap_retraite : (T, N)
    ages_deces       : (N,)
    age_retraite     : int

    Returns
    -------
    capitaux_residuels : (N,) ≥ 0
    """
    n_sims = mat_cap_retraite.shape[1]
    capitaux_residuels = np.empty(n_sims)

    for i in range(n_sims):
        mois_deces = int((ages_deces[i] - age_retraite) * 12)
        mois_deces = min(mois_deces, mat_cap_retraite.shape[0] - 1)
        capitaux_residuels[i] = max(0.0, mat_cap_retraite[mois_deces, i])

    return capitaux_residuels


def shortfall_gap(
    mat_cap_retraite: np.ndarray,
    rente_mensuelle: np.ndarray,
    ages_deces: np.ndarray,
    age_retraite: int
) -> dict:
    """
    En cas de ruine, calcule le déficit total actualisé manquant pour tenir jusqu'au décès.

    Returns
    -------
    dict avec :
        - gap_moyen      : déficit moyen sur les simulations en ruine (€)
        - gap_median     : déficit médian (€)
        - proba_shortfall: proportion de simulations en déficit
    """
    n_sims = mat_cap_retraite.shape[1]
    gaps = []

    for i in range(n_sims):
        duree_mois = int((ages_deces[i] - age_retraite) * 12)
        duree_mois = min(duree_mois, mat_cap_retraite.shape[0] - 1)
        trajectoire = mat_cap_retraite[:duree_mois + 1, i]

        idx_ruine = np.where(trajectoire <= 0)[0]
        if len(idx_ruine) > 0:
            mois_ruine = idx_ruine[0]
            mois_restants = duree_mois - mois_ruine
            # Déficit = somme des rentes non couvertes
            gaps.append(rente_mensuelle[i] * mois_restants)

    if not gaps:
        return {"gap_moyen": 0.0, "gap_median": 0.0, "proba_shortfall": 0.0}

    return {
        "gap_moyen": float(np.mean(gaps)),
        "gap_median": float(np.median(gaps)),
        "proba_shortfall": len(gaps) / n_sims,
    }


def ratio_confort(
    capitaux_finaux: np.ndarray,
    rente_mensuelle_cible: float,
    age_retraite: int,
    age_reference: int = 95,
    taux_mensuel: float = 0.0
) -> np.ndarray:
    """
    Ratio capital disponible / capital nécessaire pour tenir jusqu'à `age_reference`.

    > 1 : capital suffisant (marge de sécurité)
    < 1 : capital insuffisant (risque de ruine)

    Parameters
    ----------
    capitaux_finaux        : (N,)
    rente_mensuelle_cible  : rente mensuelle souhaitée (€), scalaire ou (N,)
    age_retraite           : âge de départ en retraite
    age_reference          : âge cible de couverture (défaut 95 ans)
    taux_mensuel           : rendement mensuel

    Returns
    -------
    ratio : (N,)
    """
    n_mois = (age_reference - age_retraite) * 12

    if np.isscalar(rente_mensuelle_cible):
        rente_mensuelle_cible = np.full(len(capitaux_finaux), rente_mensuelle_cible)

    if taux_mensuel <= 0:
        capital_necessaire = rente_mensuelle_cible * n_mois
    else:
        facteur = (1.0 - (1.0 + taux_mensuel) ** (-n_mois)) / taux_mensuel
        capital_necessaire = rente_mensuelle_cible * facteur

    return np.where(capital_necessaire > 0, capitaux_finaux / capital_necessaire, np.inf)


# =============================================================================
# SYNTHÈSE GLOBALE
# =============================================================================

def calcul_kpi_complets_decumulation(
    mat_cap_retraite: np.ndarray,
    ages_deces: np.ndarray,
    age_retraite: int,
    rente_mensuelle: np.ndarray,
    derniers_salaires: np.ndarray,
    taux_mensuel: float = 0.0,
    esperance_vie: float = 85.0,
    age_reference_confort: int = 95,
) -> dict:
    """
    Calcule l'ensemble des KPIs de décumulation en un seul appel.

    Parameters
    ----------
    mat_cap_retraite       : (T, N) trajectoires du capital en retraite
    ages_deces             : (N,) âges de décès simulés
    age_retraite           : âge de départ en retraite
    rente_mensuelle        : (N,) rente mensuelle consommée par simulation
    derniers_salaires      : (N,) dernier salaire mensuel avant retraite
    taux_mensuel           : rendement mensuel du capital encore investi
    esperance_vie          : âge de référence pour les années de réserve
    age_reference_confort  : âge cible pour le ratio de confort
    seuil_ruine            : seuil de capital définissant la ruine

    Returns
    -------
    dict de KPIs scalaires (quantiles P5/P50/P95 pour les métriques vectorielles)
    """
    capitaux_finaux = mat_cap_retraite[-1, :]

    reserves = annees_reserve(mat_cap_retraite, rente_mensuelle, age_retraite, esperance_vie)
    ratios = ratio_confort(capitaux_finaux, rente_mensuelle, age_retraite, age_reference_confort, taux_mensuel )

    # -- Niveau de vie --
    rente_equiv = rente_viagere_equivalente(capitaux_finaux, age_retraite,
                                            horizon_ans=age_reference_confort - age_retraite,
                                            taux_mensuel=taux_mensuel)
    taux_remp = taux_remplacement_effectif(rente_mensuelle, derniers_salaires)

    # -- Héritage --
    heritage = capital_residuel_au_deces(mat_cap_retraite, ages_deces, age_retraite)
    prob_heritage = float(np.mean(heritage > 0))

    # -- Déficit --
    gap_stats = shortfall_gap(mat_cap_retraite, rente_mensuelle, ages_deces, age_retraite)

    def _quantiles(arr, name):
        finite = arr[np.isfinite(arr)]
        return {
            f"{name}_p5":  float(np.percentile(finite, 5))  if len(finite) else np.nan,
            f"{name}_p50": float(np.percentile(finite, 50)) if len(finite) else np.nan,
            f"{name}_p95": float(np.percentile(finite, 95)) if len(finite) else np.nan,
        }

    kpis = {
        # Durabilité
        **_quantiles(reserves,    "annees_reserve"),
        **_quantiles(ratios,      "ratio_confort"),

        # Niveau de vie
        **_quantiles(rente_equiv, "rente_mensuelle_equiv"),
        **_quantiles(taux_remp,   "taux_remplacement"),

        # Héritage
        **_quantiles(heritage,    "capital_residuel"),
        "probabilite_heritage":   prob_heritage,

        # Déficit (shortfall)
        **gap_stats,
    }

    return kpis