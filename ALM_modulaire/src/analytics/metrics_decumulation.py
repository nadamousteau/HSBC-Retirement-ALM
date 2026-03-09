import numpy as np

# =============================================================================
# MÉTRIQUES DE DÉCUMULATION
# =============================================================================

def annees_couvertes_par_salaire(
    capitaux_finaux,
    derniers_salaires,
    taux_remplacement=1.0,
    taux_mensuel=0.0
):
    """
    Calcule, pour chaque simulation, le nombre d'années pendant lesquelles
    le capital couvre un revenu de remplacement = taux_remplacement * dernier_salaire.

    Si taux_mensuel > 0, le capital résiduel continue de fructifier :
        capital = rente * (1 - (1+r)^(-n)) / r   =>  n = -ln(1 - capital*r/rente) / ln(1+r)

    Parametres
    ----------
    capitaux_finaux    : (N,) capital au début de la retraite
    derniers_salaires  : (N,) dernier salaire mensuel avant retraite
    taux_remplacement  : fraction du salaire visée (ex: 0.7 = 70% du salaire, défaut = 100%)
    taux_mensuel       : rendement mensuel du capital résiduel (0 = capital non investi)

    Returns
    -------
    annees : (N,) durée de couverture en années (np.inf si le capital suffit à vie)
    """
    rentes_cibles = derniers_salaires * taux_remplacement
    annees = np.empty(len(capitaux_finaux))

    for i, (cap, rente) in enumerate(zip(capitaux_finaux, rentes_cibles)):
        if cap <= 0 or rente <= 0:
            annees[i] = 0.0
            continue

        if taux_mensuel <= 0:
            annees[i] = (cap / rente) / 12.0
        else:
            ratio = cap * taux_mensuel / rente
            if ratio >= 1.0:
                annees[i] = np.inf
                continue
            nb_mois = -np.log(1.0 - ratio) / np.log(1.0 + taux_mensuel)
            annees[i] = nb_mois / 12.0

    return annees


def salaire_retraite_pour_horizon(
    capitaux_finaux,
    horizon_ans,
    taux_mensuel=0.0
):
    """
    Calcule, pour chaque simulation, le revenu mensuel constant que le capital
    permet de verser pendant exactement `horizon_ans` années.

    Formule annuité constante :
        si r > 0 : rente = capital * r / (1 - (1+r)^(-n))
        si r = 0 : rente = capital / n

    Parametres
    ----------
    capitaux_finaux : (N,) capital au début de la retraite
    horizon_ans     : durée cible en années (ex: 25 ans si retraite à 65, décès à 90)
    taux_mensuel    : rendement mensuel du capital résiduel

    Returns
    -------
    rentes : (N,) revenu mensuel en €
    """
    n_mois = horizon_ans * 12

    if taux_mensuel <= 0:
        return capitaux_finaux / n_mois
    else:
        facteur = taux_mensuel / (1.0 - (1.0 + taux_mensuel) ** (-n_mois))
        return capitaux_finaux * facteur


def probabilite_taux_remplacement(
    capitaux_finaux,
    derniers_salaires,
    horizon_ans,
    taux_remplacement=1.0,
    taux_mensuel=0.0
):
    """
    Probabilité (sur N simulations) que le capital permette de couvrir
    un revenu de remplacement pendant toute la durée `horizon_ans`.

    Autrement dit : P(salaire_retraite_pour_horizon >= taux_remplacement * dernier_salaire)

    Exemple : 0.72 → dans 72 simulations sur 100, l'objectif est atteint.

    Parametres
    ----------
    capitaux_finaux   : (N,)
    derniers_salaires : (N,)
    horizon_ans       : durée de retraite à couvrir (années)
    taux_remplacement : fraction du salaire visée (défaut = 1.0, soit 100%)
    taux_mensuel      : rendement mensuel du capital résiduel

    Returns
    -------
    proba : float ∈ [0, 1]
    """
    rentes_possibles = salaire_retraite_pour_horizon(capitaux_finaux, horizon_ans, taux_mensuel)
    rentes_cibles = derniers_salaires * taux_remplacement
    return float(np.mean(rentes_possibles >= rentes_cibles))

# =============================================================================
# SYNTHÈSE GLOBALE
# =============================================================================

def calcul_kpi_complets_decumulation(
    capitaux_finaux: np.ndarray,
    derniers_salaires: np.ndarray,
    horizon_ans: float,
    taux_remplacement: float = 1.0,
    taux_mensuel: float = 0.0,
) -> dict:
    """
    Calcule l'ensemble des KPIs de décumulation en un seul appel,
    en s'appuyant sur les 3 métriques de base.

    Parameters
    ----------
    capitaux_finaux   : (N,) capital au début de la retraite
    derniers_salaires : (N,) dernier salaire mensuel avant retraite
    horizon_ans       : durée de retraite à couvrir (années, ex: 25)
    taux_remplacement : fraction du salaire visée (défaut = 1.0, soit 100%)
    taux_mensuel      : rendement mensuel du capital résiduel

    Returns
    -------
    dict de KPIs scalaires (quantiles P5/P50/P95 pour les métriques vectorielles)
    """

    def _quantiles(arr, name):
        finite = arr[np.isfinite(arr)]
        return {
            f"{name}_p5":  float(np.percentile(finite, 5))  if len(finite) else np.nan,
            f"{name}_p50": float(np.percentile(finite, 50)) if len(finite) else np.nan,
            f"{name}_p95": float(np.percentile(finite, 95)) if len(finite) else np.nan,
        }

    # 1. Nombre d'années couvertes par le capital
    annees = annees_couvertes_par_salaire(
        capitaux_finaux, derniers_salaires,
        taux_remplacement=taux_remplacement,
        taux_mensuel=taux_mensuel,
    )

    # 2. Rente mensuelle atteignable sur l'horizon cible
    rentes = salaire_retraite_pour_horizon(
        capitaux_finaux,
        horizon_ans=horizon_ans,
        taux_mensuel=taux_mensuel,
    )

    # 3. Probabilité d'atteindre le taux de remplacement sur tout l'horizon
    proba = probabilite_taux_remplacement(
        capitaux_finaux, derniers_salaires,
        horizon_ans=horizon_ans,
        taux_remplacement=taux_remplacement,
        taux_mensuel=taux_mensuel,
    )

    kpis = {
        # Durabilité : combien d'années le capital tient-il ?
        **_quantiles(annees, "annees_couvertes"),

        # Niveau de vie : quelle rente mensuelle sur l'horizon ?
        **_quantiles(rentes, "rente_mensuelle"),

        # Probabilité d'atteindre l'objectif de remplacement
        "probabilite_taux_remplacement": proba,
    }

    return kpis


'''
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
'''