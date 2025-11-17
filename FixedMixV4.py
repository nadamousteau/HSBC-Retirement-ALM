"""
Retirement Planner - Fixed-Mix Goal-Based Model
===============================================

Version compatible avec :

    - HistoricalAssetReturn_clean.xlsx
    - AssumptionForSimulation.xlsx

Principes :
-----------
1) Univers d'actifs = feuille 1 de AssumptionForSimulation (colonne "Asset Name").
2) Volatilités annuelles = colonne "Volatility".
3) Rendements espérés annuels = colonne "Expected Return" (sinon fallback sur moyenne historique).
4) Corrélations :
     - calculées sur les actifs qui ont un historique (fichier HistoricalAssetReturn_clean.xlsx),
     - les actifs sans historique ont corrélation 0 avec les autres (diag uniquement).
5) Actif synthétique "ZeroCouponUSD" :
     - mu = taux zéro-coupon 1 an (feuille 2),
     - sigma ≈ 0,
     - corrélation 0 avec les autres.
6) Optimisation de portefeuille pour chaque profil de risque AVEC contraintes d'allocation :
     - profil conservateur : bornes sur equity/property/high-yield + minimum de safe assets.
"""

# ==========================
# IMPORTS
# ==========================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.backends.backend_pdf import PdfPages

# ==========================
# PARAMÈTRES UTILISATEUR
# ==========================

HISTORICAL_FILE = "HistoricalAssetReturn_clean.xlsx"
ASSUMPTIONS_XLSX = "AssumptionForSimulation.xlsx"

RETIREMENT_PROFILE = "pension_constante"

LEGACY_SHARE = 0.20

SALARY_ANNUAL_BRUT = 60_000.0
TAX_RATE = 0.25
SAVINGS_RATE = 0.10
INITIAL_CAPITAL = 20_000.0

AGE_START_SAVING = 30
AGE_RETIREMENT = 65
YEARS_IN_RETIREMENT = 25
AGE_END = AGE_RETIREMENT + YEARS_IN_RETIREMENT

SALARY_GROWTH_REAL = 0.02

PI_ANNUAL_DEFAULT = 0.02
SIGMA_PI_ANNUAL_DEFAULT = 0.01

DT = 1.0 / 12.0
N_SIMS_ACC = 1000
N_SIMS_DEC = 2000

RANDOM_SEED = 12345

RISK_PROFILES = {
    "conservateur": 0.06,
    "equilibre":    0.10,
    "dynamique":    0.15,
}

ESSENTIAL_REPLACEMENT_RATE = 0.70

SECONDARY_HOME_COST = 300_000.0

ZERO_COUPON_ASSET_NAME = "ZeroCouponUSD"


# ==========================
# LECTURE / CALIBRAGE
# ==========================

def load_historical_returns(path):
    """
    Lit HistoricalAssetReturn_clean.xlsx au format :

        Ligne 0 : Type | bonds | Bonds | Equity | ...
        Ligne 1 : Date | US Government Bond | US Inflation Linked Bond | ...
        Ligne 2+ : dates + rendements mensuels

    Retour :
        DataFrame indexé par Date, colonnes = noms d'actifs, valeurs = rendements mensuels en décimal.
    """
    if path.lower().endswith(".xlsx"):
        df_raw = pd.read_excel(path, header=None)
    else:
        df_raw = pd.read_csv(path, header=None)

    first_col = df_raw.iloc[:, 0].astype(str).str.strip().str.lower()
    idx_candidates = np.where(first_col == "date")[0]
    if len(idx_candidates) == 0:
        raise ValueError("Aucune ligne avec 'Date' trouvée en première colonne dans l'historique.")

    idx_header = int(idx_candidates[0])
    header = df_raw.iloc[idx_header]

    df = df_raw.iloc[idx_header + 1:].copy()
    df.columns = header

    if "Date" not in df.columns:
        raise ValueError(f"Ligne d'en-tête trouvée ne contient pas 'Date'. Colonnes : {list(df.columns)}")

    df["Date"] = pd.to_datetime(df["Date"])
    df.set_index("Date", inplace=True)

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Si c'était en %, on normalise
    if df.abs().mean().mean() > 0.5:
        df = df / 100.0

    return df


def load_assumption_data(xlsx_path):
    """
    Lit AssumptionForSimulation.xlsx.

    Feuille 1 :
        - 'Asset Name'
        - 'Expected Return' (annuel, % ou décimal)
        - 'Volatility'     (annuelle, % ou décimal)

    Feuille 2 :
        - 'Tenor'
        - 'Rate Zero Coupon'

    Retour :
      {
        'asset_names':      list[str],
        'asset_vol_annual': np.array,
        'asset_er_annual':  np.array,
        'zc_tenors_years':  np.array,
        'zc_rates_annual':  np.array,
        'rf_rate_annual':   float
      }
    """
    df_assets = pd.read_excel(xlsx_path, sheet_name=0, header=0)

    asset_names = df_assets["Asset Name"].astype(str).str.strip().tolist()
    asset_vol_annual = pd.to_numeric(df_assets["Volatility"], errors="coerce").values
    asset_er_annual = pd.to_numeric(df_assets["Expected Return"], errors="coerce").values

    if np.nanmean(np.abs(asset_vol_annual)) > 1.0:
        asset_vol_annual = asset_vol_annual / 100.0
    if np.nanmean(np.abs(asset_er_annual)) > 1.0:
        asset_er_annual = asset_er_annual / 100.0

    df_zc = pd.read_excel(xlsx_path, sheet_name=1, header=0)

    tenors_raw = df_zc["Tenor"]
    rates_raw = pd.to_numeric(df_zc["Rate Zero Coupon"], errors="coerce")

    tenors_years_list = []
    for x in tenors_raw:
        if isinstance(x, (int, float)):
            tenors_years_list.append(float(x))
        elif isinstance(x, str):
            s = x.strip().lower()
            num = "".join(ch for ch in s if (ch.isdigit() or ch == ".")) or "0"
            val = float(num)
            if "m" in s and "y" not in s:
                tenors_years_list.append(val / 12.0)
            else:
                tenors_years_list.append(val)
        else:
            tenors_years_list.append(np.nan)

    tenors_years = np.array(tenors_years_list, float)
    rates = rates_raw.values.astype(float)

    mask = ~np.isnan(tenors_years) & ~np.isnan(rates)
    tenors_years = tenors_years[mask]
    rates = rates[mask]

    if tenors_years.size > 0 and np.nanmean(np.abs(rates)) > 1.0:
        rates = rates / 100.0

    order = np.argsort(tenors_years)
    tenors_years = tenors_years[order]
    rates = rates[order]

    if tenors_years.size > 0:
        rf_rate_1y = float(np.interp(1.0, tenors_years, rates))
    else:
        rf_rate_1y = 0.0

    return {
        "asset_names": asset_names,
        "asset_vol_annual": asset_vol_annual,
        "asset_er_annual": asset_er_annual,
        "zc_tenors_years": tenors_years,
        "zc_rates_annual": rates,
        "rf_rate_annual": rf_rate_1y,
    }


def align_assets_and_build_cov(returns_hist, assump_dict):
    """
    Aligne les actifs entre l'historique et AssumptionForSimulation,
    puis construit Sigma annuelle.

    - On prend la liste d'actifs de AssumptionForSimulation comme univers de référence.
    - On sépare :
        * actifs avec historique : corrélation empirique,
        * actifs sans historique : seulement variances (corrélation 0).
    - Volatilités annuelles = 'Volatility' (Assumption).
    """
    asset_names_assump = assump_dict["asset_names"]
    asset_vol_annual_assump = assump_dict["asset_vol_annual"]

    # Mapping manuel (au cas où les noms seraient différents dans l'historique)
    HIST_TO_ASSUMP_NAME_MAP = {
        "Liquidity USD": "Liquidity USD",
        "US Government Bond": "US Government Bond",
        "US Inflation Linked Bond": "US Inflation Linked Bond",
        "US High Yield Bond BB-B": "US High Yield Bond BB-B",
        "USD Corporate Bond": "USD Corporate Bond",
        "Asia Pacific ex Japan Equity": "Asia Pacific ex Japan Equity",
        "US Property": "US Property",
        "Global Equity": "Global Equity",
        "Japan Equity": "Japan Equity",
        "US Equity": "US Equity",
    }
    ASSUMP_TO_HIST_NAME_MAP = {v: k for k, v in HIST_TO_ASSUMP_NAME_MAP.items()}

    returns_hist = returns_hist.copy()

    vol_map = {name: vol for name, vol in zip(asset_names_assump, asset_vol_annual_assump)}

    hist_assets = []
    hist_cols = []
    hist_vols = []

    no_hist_assets = []
    no_hist_vols = []

    for ass_name in asset_names_assump:
        hist_name = ASSUMP_TO_HIST_NAME_MAP.get(ass_name, ass_name)
        if hist_name in returns_hist.columns:
            hist_assets.append(ass_name)
            hist_cols.append(hist_name)
            hist_vols.append(vol_map.get(ass_name, np.nan))
        else:
            no_hist_assets.append(ass_name)
            no_hist_vols.append(vol_map.get(ass_name, np.nan))

    # Partie avec historique
    if hist_assets:
        returns_hist_aligned = returns_hist[hist_cols].copy()
        corr_hist = returns_hist_aligned.corr().values
        corr_hist = np.nan_to_num(corr_hist, nan=0.0)
        np.fill_diagonal(corr_hist, 1.0)

        hist_vols = np.array(hist_vols, float)
        D_hist = np.diag(hist_vols)
        Sigma_hist = D_hist @ corr_hist @ D_hist
    else:
        returns_hist_aligned = pd.DataFrame(index=returns_hist.index)
        Sigma_hist = np.zeros((0, 0))
        hist_vols = np.array([], float)

    # Partie sans historique : diag only
    no_hist_vols = np.array(no_hist_vols, float)
    Sigma_no_hist = np.diag(no_hist_vols ** 2) if no_hist_assets else np.zeros((0, 0))

    n_hist = Sigma_hist.shape[0]
    n_no = Sigma_no_hist.shape[0]
    n_tot = n_hist + n_no

    Sigma_full = np.zeros((n_tot, n_tot))
    if n_hist > 0:
        Sigma_full[:n_hist, :n_hist] = Sigma_hist
    if n_no > 0:
        Sigma_full[n_hist:, n_hist:] = Sigma_no_hist

    asset_names_final = hist_assets + no_hist_assets
    asset_vol_final = np.concatenate([hist_vols, no_hist_vols])

    return returns_hist_aligned, asset_names_final, asset_vol_final, Sigma_full


def add_zero_coupon_asset(mu_core, Sigma_annual, asset_names_core, rf_rate_annual):
    """
    Ajoute l'actif synthétique ZeroCouponUSD.
    """
    mu_ext = np.append(mu_core, rf_rate_annual)

    n = Sigma_annual.shape[0]
    Sigma_ext = np.zeros((n + 1, n + 1))
    Sigma_ext[:n, :n] = Sigma_annual

    eps_vol = 1e-6
    Sigma_ext[n, n] = eps_vol ** 2

    asset_names_ext = asset_names_core + [ZERO_COUPON_ASSET_NAME]

    return mu_ext, Sigma_ext, asset_names_ext


def compute_portfolio_vol_annual(w, Sigma):
    w = np.asarray(w)
    return float(np.sqrt(w @ Sigma @ w))


# ==========================
# ACCUMULATION
# ==========================

def simulate_global_inflation(
    pi_annual,
    sigma_pi_annual,
    age_start,
    age_retirement,
    dt=1/12,
    seed=1234
):
    np.random.seed(seed)

    horizon_years = age_retirement - age_start
    N = int(horizon_years / dt)

    I = np.zeros(N + 1)
    I[0] = 1.0

    for t in range(N):
        z = np.random.normal()
        I[t+1] = I[t] * np.exp(
            (pi_annual - 0.5 * sigma_pi_annual**2) * dt
            + sigma_pi_annual * np.sqrt(dt) * z
        )

    t_years = np.linspace(age_start, age_retirement, N + 1)
    return t_years, I


def simulate_accumulation(
    w, mu_vec, Sigma,
    salary_annual_brut,
    tax_rate,
    savings_rate,
    salary_growth_real,
    pi_annual,
    sigma_pi_annual,
    age_start,
    age_retirement,
    initial_capital,
    dt=1/12,
    n_sims=1000,
    seed=123,
):
    np.random.seed(seed)

    horizon_years = age_retirement - age_start
    N = int(horizon_years / dt)

    mu_m = mu_vec * dt
    Sigma_m = Sigma * dt

    try:
        L_chol = np.linalg.cholesky(Sigma_m)
    except np.linalg.LinAlgError:
        L_chol = np.linalg.cholesky(Sigma_m + 1e-8 * np.eye(len(mu_vec)))

    t_years = np.linspace(age_start, age_retirement, N + 1)

    salary_net_annual_0 = salary_annual_brut * (1.0 - tax_rate)

    W_nom_paths = np.zeros((n_sims, N + 1))
    I_paths = np.zeros((n_sims, N + 1))

    for s in range(n_sims):
        W = np.zeros(N + 1)
        I = np.zeros(N + 1)
        W[0] = initial_capital
        I[0] = 1.0

        for t in range(N):
            z_pi = np.random.normal()
            I[t+1] = I[t] * np.exp(
                (pi_annual - 0.5 * sigma_pi_annual**2) * dt
                + sigma_pi_annual * np.sqrt(dt) * z_pi
            )

            years_since_start = t * dt
            salary_net_annual_real = salary_net_annual_0 * (1.0 + salary_growth_real) ** years_since_start
            salary_net_annual_nom = salary_net_annual_real * I[t]
            salary_net_monthly_nom = salary_net_annual_nom / 12.0

            contribution = savings_rate * salary_net_monthly_nom

            z = np.random.normal(size=len(mu_vec))
            r_vec = mu_m + L_chol @ z
            R_p = float(w @ r_vec)

            W[t+1] = W[t] * (1.0 + R_p) + contribution

        W_nom_paths[s, :] = W
        I_paths[s, :] = I

    W_nom_ret = W_nom_paths[:, -1]

    return t_years, W_nom_paths, I_paths, W_nom_ret


# ==========================
# DÉCUMULATION
# ==========================

def build_consumption_profile(
    W0_ret_nom,
    legacy_share,
    profile,
    age_retirement,
    age_end,
    dt=1/12,
    secondary_home_cost=0.0,
):
    N_dec = int(12 * (age_end - age_retirement))
    if N_dec <= 0:
        raise ValueError("Horizon de retraite N_dec doit être positif.")

    W0_legacy = legacy_share * W0_ret_nom
    S0_gross = (1.0 - legacy_share) * W0_ret_nom
    if S0_gross < 0:
        raise ValueError("legacy_share > 1, poche consommable négative.")

    C_real = np.zeros(N_dec)
    secondary_cost = 0.0
    lump_sum_real = 0.0
    rent_real = 0.0

    if profile == "residence_secondaire":
        if secondary_home_cost < 0 or secondary_home_cost > S0_gross:
            raise ValueError("secondary_home_cost invalide.")
        secondary_cost = secondary_home_cost
        S0 = S0_gross - secondary_cost
        C_real[:] = S0 / N_dec

    elif profile == "achat_immo_locatif":
        if secondary_home_cost < 0 or secondary_home_cost > S0_gross:
            raise ValueError("secondary_home_cost invalide.")
        secondary_cost = secondary_home_cost
        S0 = S0_gross - secondary_cost
        C_real[:] = S0 / N_dec
        rent_real = 0.05 * secondary_cost / 12.0

    elif profile == "etudes_enfant":
        secondary_cost = 0.0
        S0 = S0_gross
        N1 = min(60, N_dec)
        N2 = N_dec - N1
        if N2 < 0:
            N2 = 0
        denom = 1.5 * N1 + N2
        if denom <= 0:
            C2 = S0 / max(N_dec, 1)
            C1 = 1.5 * C2
        else:
            C2 = S0 / denom
            C1 = 1.5 * C2
        C_real[:N1] = C1
        C_real[N1:] = C2

    elif profile == "vacances_annuelles":
        secondary_cost = 0.0
        S0 = S0_gross
        n_high = N_dec // 12
        n_low = N_dec - n_high
        denom = n_low + 1.5 * n_high
        if denom <= 0:
            C_base = 0.0
        else:
            C_base = S0 / denom
        C_real[:] = C_base
        for t in range(N_dec):
            if (t + 1) % 12 == 0:
                C_real[t] = 1.5 * C_base

    elif profile == "pension_constante":
        secondary_cost = 0.0
        S0 = S0_gross
        C_real[:] = S0 / N_dec

    elif profile == "retrait_total":
        secondary_cost = 0.0
        S0 = 0.0
        C_real[:] = 0.0
        lump_sum_real = S0_gross

    else:
        raise ValueError(f"Profil '{profile}' non reconnu.")

    return {
        "N_dec": N_dec,
        "S0": S0,
        "W0_legacy": W0_legacy,
        "secondary_cost": secondary_cost,
        "C_real": C_real,
        "lump_sum_real": lump_sum_real,
        "rent_real": rent_real,
    }


def monte_carlo_decumulation_profiles(
    w, mu_vec, Sigma,
    W0_ret_nom,
    pi_annual, sigma_pi_annual,
    age_retirement, age_end,
    legacy_share,
    profile,
    secondary_home_cost=0.0,
    essential_income_real=None,
    N_sims=1000,
    dt=1/12
):
    prof = build_consumption_profile(
        W0_ret_nom=W0_ret_nom,
        legacy_share=legacy_share,
        profile=profile,
        age_retirement=age_retirement,
        age_end=age_end,
        dt=dt,
        secondary_home_cost=secondary_home_cost,
    )

    N_dec = prof["N_dec"]
    S0 = prof["S0"]
    W0_legacy = prof["W0_legacy"]
    secondary_cost = prof["secondary_cost"]
    C_real = prof["C_real"]
    lump_sum_real = prof["lump_sum_real"]
    rent_real = prof["rent_real"]

    mu_m = mu_vec * dt
    Sigma_m = Sigma * dt

    try:
        L_chol = np.linalg.cholesky(Sigma_m)
    except np.linalg.LinAlgError:
        L_chol = np.linalg.cholesky(Sigma_m + 1e-8 * np.eye(len(mu_vec)))

    ruined = np.zeros(N_sims, dtype=bool)
    ruin_age = np.full(N_sims, np.nan)
    W_final_spend_real = np.zeros(N_sims)
    bequest_legacy_real = np.zeros(N_sims)
    bequest_secondary_real = np.zeros(N_sims)
    bequest_total_real = np.zeros(N_sims)

    if essential_income_real is not None:
        shortfall_sum_real = np.zeros(N_sims)
        shortfall_any = np.zeros(N_sims, dtype=bool)
    else:
        shortfall_sum_real = None
        shortfall_any = None

    I_paths_ret = np.zeros((N_sims, N_dec + 1))

    for s in range(N_sims):
        I_ret = np.zeros(N_dec + 1)
        I_ret[0] = 1.0

        for t in range(N_dec):
            z_inf = np.random.normal()
            I_ret[t+1] = I_ret[t] * np.exp(
                (pi_annual - 0.5 * sigma_pi_annual**2) * dt
                + sigma_pi_annual * np.sqrt(dt) * z_inf
            )

        I_paths_ret[s, :] = I_ret

        S = np.zeros(N_dec + 1)
        L = np.zeros(N_dec + 1)
        S[0] = S0
        L[0] = W0_legacy

        is_ruined = False
        ruin_t = None
        sf_total_s = 0.0
        sf_any_s = False

        for t in range(N_dec):
            if not is_ruined:
                income_real_t = C_real[t] + rent_real
            else:
                income_real_t = rent_real

            if essential_income_real is not None:
                sf_t = max(essential_income_real - income_real_t, 0.0)
                if sf_t > 0:
                    sf_any_s = True
                sf_total_s += sf_t

            C_nom = C_real[t] * I_ret[t]

            z = np.random.normal(size=len(mu_vec))
            r_vec = mu_m + L_chol @ z
            R_p = float(w @ r_vec)

            if not is_ruined:
                S[t+1] = S[t] * (1.0 + R_p) - C_nom
                if S[t+1] <= 0:
                    is_ruined = True
                    ruin_t = t + 1
                    S[t+1] = 0.0
            else:
                S[t+1] = 0.0

            L[t+1] = L[t] * (1.0 + R_p)

        ruined[s] = is_ruined
        if is_ruined:
            ruin_age[s] = age_retirement + ruin_t * dt

        S_T = S[-1]
        L_T = L[-1]
        I_T = I_ret[-1]

        W_final_spend_real[s] = S_T / I_T
        bequest_legacy_real[s] = L_T / I_T
        bequest_secondary_real[s] = secondary_cost
        bequest_total_real[s] = (S_T + L_T + secondary_cost * I_T) / I_T

        if essential_income_real is not None:
            shortfall_sum_real[s] = sf_total_s
            shortfall_any[s] = sf_any_s

    return {
        "ruined": ruined,
        "ruin_age": ruin_age,
        "W_final_spend_real": W_final_spend_real,
        "bequest_legacy_real": bequest_legacy_real,
        "bequest_secondary_real": bequest_secondary_real,
        "bequest_total_real": bequest_total_real,
        "C_real": C_real,
        "rent_real": rent_real,
        "S0": S0,
        "W0_legacy": W0_legacy,
        "secondary_cost": secondary_cost,
        "lump_sum_real": lump_sum_real,
        "essential_income_real": essential_income_real,
        "shortfall_sum_real": shortfall_sum_real,
        "shortfall_any": shortfall_any,
        "I_paths_ret": I_paths_ret,
    }


# ==========================
# OPTIMISATION AVEC CONTRAINTES PAR PROFIL
# ==========================

def optimize_portfolio_for_risk(mu_vec, Sigma, target_vol, asset_names, risk_profile_name=None, long_only=True):
    """
    Optimisation de portefeuille avec :
      - cible de volatilité target_vol,
      - contraintes par classe d'actifs (equity / high yield / safe),
      - plafonds par actif,
      - pénalité de concentration.

    Les bornes (eq_max, hy_max, safe_min, plafonds) sont *des fonctions de target_vol*,
    pas du nom de profil, donc on peut modifier la vol cible ou créer de nouveaux profils risque.
    """
    n = len(mu_vec)
    lam_vol = 200.0   # poids de la contrainte de vol
    lam_conc = 1.0    # pénalité de concentration (sum w^2)
    asset_names = list(asset_names)

    # --- Petite fonction d'interpolation linéaire ---
    def interp(tv, xs, ys):
        tv = float(tv)
        if tv <= xs[0]:
            return ys[0]
        if tv >= xs[-1]:
            return ys[-1]
        for i in range(len(xs) - 1):
            if xs[i] <= tv <= xs[i+1]:
                t = (tv - xs[i]) / (xs[i+1] - xs[i])
                return ys[i] + t * (ys[i+1] - ys[i])
        return ys[-1]

    # Points d'ancrage pour les contraintes vs volatilité cible
    # (6%, 10%, 15%) -> (conservateur, équilibre, dynamique typiques)
    tv_knots = [0.06, 0.10, 0.15]

    # max equity+property
    eq_max = interp(target_vol, tv_knots, [0.25, 0.70, 1.00])
    # max high yield
    hy_max = interp(target_vol, tv_knots, [0.10, 0.20, 0.30])
    # min safe (liquidity + zero coupon)
    safe_min = interp(target_vol, tv_knots, [0.40, 0.10, 0.00])

    # --- Classification des actifs ---
    idx_equity_like = [i for i, name in enumerate(asset_names)
                       if ("Equity" in name) or ("Property" in name)]
    idx_high_yield = [i for i, name in enumerate(asset_names)
                      if "High Yield" in name]
    idx_safe = [i for i, name in enumerate(asset_names)
                if ("Liquidity" in name) or (name == ZERO_COUPON_ASSET_NAME)]

    # --- Plafonds par ACTIF, dérivés des bornes globales ---
    max_w = np.ones(n)

    for i in range(n):
        name = asset_names[i]
        if i in idx_equity_like:
            # Limite par ligne d'equity : une fraction de eq_max, plafonnée
            max_w[i] = min(0.35, max(0.08, 0.6 * eq_max))
        elif i in idx_high_yield:
            max_w[i] = min(0.15, max(0.03, 0.5 * hy_max))
        elif i in idx_safe:
            # Safe : on autorise des lignes assez grosses
            max_w[i] = max(0.40, safe_min)
        else:
            # Obligations d'investment grade, inflation-linked...
            max_w[i] = 0.50

    # --- Objectif : rendement max, vol proche de target_vol, peu de concentration ---
    def objective(w):
        w = np.array(w)
        ret = float(w @ mu_vec)
        vol = float(np.sqrt(w @ Sigma @ w))
        conc = float(np.sum(w**2))
        return -ret + lam_vol * (vol - target_vol) ** 2 + lam_conc * conc

    constraints = []

    # Somme des poids = 1
    constraints.append({
        "type": "eq",
        "fun": lambda w: np.sum(w) - 1.0
    })

    # Equity + Property <= eq_max
    if idx_equity_like:
        def eq_constraint(w, idx=tuple(idx_equity_like), eq_max=eq_max):
            return eq_max - np.sum(w[list(idx)])
        constraints.append({"type": "ineq", "fun": eq_constraint})

    # High Yield <= hy_max
    if idx_high_yield:
        def hy_constraint(w, idx=tuple(idx_high_yield), hy_max=hy_max):
            return hy_max - np.sum(w[list(idx)])
        constraints.append({"type": "ineq", "fun": hy_constraint})

    # Safe >= safe_min
    if idx_safe:
        def safe_constraint(w, idx=tuple(idx_safe), safe_min=safe_min):
            return np.sum(w[list(idx)]) - safe_min
        constraints.append({"type": "ineq", "fun": safe_constraint})

    # Bornes par actif
    if long_only:
        bounds = tuple((0.0, float(max_w[i])) for i in range(n))
    else:
        bounds = tuple((-float(max_w[i]), float(max_w[i])) for i in range(n))

    w0 = np.ones(n) / n

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints)
    if not res.success:
        print("⚠️ Optimisation portefeuille n'a pas convergé, on garde la pondération égale.")
        return w0

    return res.x



# ==========================
# KPI PAR PROFIL
# ==========================

def evaluate_profile(
    risk_profile_name,
    mu_vec,
    Sigma,
    asset_names,
    pi_annual,
    sigma_pi_annual,
):
    target_vol = RISK_PROFILES[risk_profile_name]

    w_opt = optimize_portfolio_for_risk(
        mu_vec, Sigma, target_vol, asset_names, risk_profile_name, long_only=True
    )
    port_vol = compute_portfolio_vol_annual(w_opt, Sigma)

    print(f"\n========== PROFIL DE RISQUE : {risk_profile_name.upper()} ==========")
    print("Allocation de portefeuille optimisée :")
    for name, w in zip(asset_names, w_opt):
        print(f" - {name:30s}: {w:.2%}")
    print(f"Volatilité annuelle du portefeuille : {port_vol:.2%} (cible {target_vol:.2%})")

    t_years_acc, W_nom_paths_acc, I_paths_acc, W_nom_ret = simulate_accumulation(
        w=w_opt,
        mu_vec=mu_vec,
        Sigma=Sigma,
        salary_annual_brut=SALARY_ANNUAL_BRUT,
        tax_rate=TAX_RATE,
        savings_rate=SAVINGS_RATE,
        salary_growth_real=SALARY_GROWTH_REAL,
        pi_annual=pi_annual,
        sigma_pi_annual=sigma_pi_annual,
        age_start=AGE_START_SAVING,
        age_retirement=AGE_RETIREMENT,
        initial_capital=INITIAL_CAPITAL,
        dt=DT,
        n_sims=N_SIMS_ACC,
        seed=RANDOM_SEED + hash(risk_profile_name) % 10_000,
    )

    horizon_years = AGE_RETIREMENT - AGE_START_SAVING
    last_salary_net_annual_real = (SALARY_ANNUAL_BRUT * (1.0 - TAX_RATE)
                                   * (1.0 + SALARY_GROWTH_REAL) ** horizon_years)
    last_salary_net_monthly_real = last_salary_net_annual_real / 12.0
    essential_income_real = ESSENTIAL_REPLACEMENT_RATE * last_salary_net_monthly_real

    W0_ret_nom = float(np.mean(W_nom_ret))

    res_dec = monte_carlo_decumulation_profiles(
        w=w_opt,
        mu_vec=mu_vec,
        Sigma=Sigma,
        W0_ret_nom=W0_ret_nom,
        pi_annual=pi_annual,
        sigma_pi_annual=sigma_pi_annual,
        age_retirement=AGE_RETIREMENT,
        age_end=AGE_END,
        legacy_share=LEGACY_SHARE,
        profile=RETIREMENT_PROFILE,
        secondary_home_cost=SECONDARY_HOME_COST,
        essential_income_real=essential_income_real,
        N_sims=N_SIMS_DEC,
        dt=DT,
    )

    ruined = res_dec["ruined"]
    ruin_age = res_dec["ruin_age"]
    bequest_total_real = res_dec["bequest_total_real"]
    bequest_legacy_real = res_dec["bequest_legacy_real"]
    bequest_secondary_real = res_dec["bequest_secondary_real"]
    C_real = res_dec["C_real"]
    rent_real = res_dec["rent_real"]
    shortfall_sum_real = res_dec["shortfall_sum_real"]
    shortfall_any = res_dec["shortfall_any"]
    N_dec = len(C_real)

    prob_ruine = ruined.mean()
    if np.any(ruined):
        age_ruine_mean = float(np.nanmean(ruin_age[ruined]))
    else:
        age_ruine_mean = np.nan

    bequest_mean = float(np.mean(bequest_total_real))
    bequest_median = float(np.median(bequest_total_real))
    bequest_q5 = float(np.percentile(bequest_total_real, 5))
    bequest_legacy_mean = float(np.mean(bequest_legacy_real))
    bequest_immo_mean = float(np.mean(bequest_secondary_real))

    income_real = C_real + rent_real
    pension_real_mean = float(np.mean(C_real))
    income_real_mean = float(np.mean(income_real))

    if shortfall_any is not None:
        prob_sf = shortfall_any.mean()
        avg_sf_per_month_all = float(shortfall_sum_real.mean() / N_dec)
        if prob_sf > 0:
            avg_sf_per_month_cond = float(shortfall_sum_real[shortfall_any].mean() / N_dec)
        else:
            avg_sf_per_month_cond = 0.0
    else:
        prob_sf = np.nan
        avg_sf_per_month_all = np.nan
        avg_sf_per_month_cond = np.nan

    print("\n--- KPI RÉCAPITULATIFS ---")
    print(f"Capital moyen à la retraite (nominal) : {W0_ret_nom:,.0f} €")
    print(f"Probabilité de ruine                 : {prob_ruine:.2%}")
    if not np.isnan(age_ruine_mean):
        print(f"Âge moyen de ruine (si ruine)        : {age_ruine_mean:.1f} ans")
    print(f"Revenu réel moyen total              : {income_real_mean:,.0f} €/mois")
    print(f"Shortfall (P>0)                      : {prob_sf:.2%}")
    print(f"Shortfall moyen (tous scénarios)     : {avg_sf_per_month_all:,.0f} €/mois")
    print(f"Héritage médian (réel)               : {bequest_median:,.0f} €")
    print(f"Héritage Q5% (réel)                  : {bequest_q5:,.0f} €")

    kpi_row = {
        "risk_profile": risk_profile_name,
        "retirement_profile": RETIREMENT_PROFILE,
        "legacy_share": LEGACY_SHARE,
        "secondary_home_cost": SECONDARY_HOME_COST,
        "W0_ret_nom_used": W0_ret_nom,
        "essential_income_real": essential_income_real,
        "prob_ruine": prob_ruine,
        "age_ruine_mean": age_ruine_mean,
        "bequest_mean": bequest_mean,
        "bequest_median": bequest_median,
        "bequest_q5": bequest_q5,
        "bequest_legacy_mean": bequest_legacy_mean,
        "bequest_immo_mean": bequest_immo_mean,
        "pension_real_mean": pension_real_mean,
        "income_real_mean": income_real_mean,
        "prob_shortfall_any": prob_sf,
        "shortfall_avg_per_month_all": avg_sf_per_month_all,
        "shortfall_avg_per_month_cond": avg_sf_per_month_cond,
    }

    profile_data = {
        "risk_profile": risk_profile_name,
        "w_opt": w_opt,
        "asset_names": asset_names,
        "t_years_acc": t_years_acc,
        "W_nom_paths_acc": W_nom_paths_acc,
        "W_nom_ret": W_nom_ret,
        "I_paths_acc": I_paths_acc,
        "res_dec": res_dec,
        "essential_income_real": essential_income_real,
    }

    return kpi_row, profile_data


# ==========================
# PDF
# ==========================

def generate_pdf_report(
    output_path,
    df_kpi,
    profiles_results,
    t_inf,
    I_inf,
    pi_annual,
    sigma_pi_annual,
):
    with PdfPages(output_path) as pdf:
        # PAGE 1
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax0, ax1 = axes

        txt = []
        txt.append("Rapport de simulation retraite – Fixed-Mix\n")
        txt.append(f"Objectif de retraite : {RETIREMENT_PROFILE}")
        txt.append(f"Part d'héritage (legacy_share) : {LEGACY_SHARE:.0%}")
        txt.append("")
        txt.append(f"Salaire brut annuel actuel : {SALARY_ANNUAL_BRUT:,.0f} €")
        txt.append(f"Taux d'imposition approx.  : {TAX_RATE:.0%}")
        txt.append(f"Taux d'épargne (salaire net) : {SAVINGS_RATE:.0%}")
        txt.append(f"Capital initial : {INITIAL_CAPITAL:,.0f} €")
        txt.append("")
        txt.append(f"Âge début épargne : {AGE_START_SAVING} ans")
        txt.append(f"Âge retraite      : {AGE_RETIREMENT} ans")
        txt.append(f"Durée retraite    : {YEARS_IN_RETIREMENT} ans")
        txt.append("")
        txt.append(f"Croissance réelle salaire : {SALARY_GROWTH_REAL:.2%}")
        txt.append(f"Inflation moyenne         : {pi_annual:.2%}")
        txt.append(f"Volatilité de l'inflation : {sigma_pi_annual:.2%}")

        ax0.axis("off")
        ax0.text(0.0, 1.0, "\n".join(txt), fontsize=10, va="top")

        ax1.plot(t_inf, I_inf, lw=2)
        ax1.set_title("Inflation simulée – Trajectoire globale unique")
        ax1.set_xlabel("Âge")
        ax1.set_ylabel("Indice des prix (base 1)")
        ax1.grid(True, alpha=0.3)

        fig.suptitle("Introduction & paramètres globaux", fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

        # PROFILS
        for rp, pdata in profiles_results.items():
            k = df_kpi[df_kpi["risk_profile"] == rp].iloc[0]

            w_opt = pdata["w_opt"]
            asset_names = pdata["asset_names"]
            t_years_acc = pdata["t_years_acc"]
            W_nom_paths_acc = pdata["W_nom_paths_acc"]
            W_nom_ret = pdata["W_nom_ret"]
            res_dec = pdata["res_dec"]
            essential_income_real = pdata["essential_income_real"]

            C_real = res_dec["C_real"]
            rent_real = res_dec["rent_real"]
            income_real = C_real + rent_real
            bequest_total_real = res_dec["bequest_total_real"]
            I_paths_ret = res_dec["I_paths_ret"]

            I_ret_mean = I_paths_ret.mean(axis=0)
            income_nominal_mean = income_real * I_ret_mean[:-1]

            # PAGE KPI
            fig, ax = plt.subplots(figsize=(8.5, 11))
            ax.axis("off")

            lines = []
            lines.append(f"Profil de risque : {rp}")
            lines.append(f"Objectif de retraite : {RETIREMENT_PROFILE}")
            lines.append("")
            lines.append(f"Capital moyen à la retraite : {k['W0_ret_nom_used']:,.0f} € (nominal)")
            lines.append(f"Probabilité de ruine        : {k['prob_ruine']:.2%}")
            if not np.isnan(k["age_ruine_mean"]):
                lines.append(f"Âge moyen de ruine (si ruine) : {k['age_ruine_mean']:.1f} ans")
            lines.append("")
            lines.append(f"Revenu réel moyen total      : {k['income_real_mean']:,.0f} €/mois")
            lines.append(f"Pension réelle moyenne       : {k['pension_real_mean']:,.0f} €/mois")
            lines.append(f"Pension ESSENTIELLE (C_ess)  : {k['essential_income_real']:,.0f} €/mois")
            lines.append("")
            lines.append(f"Probabilité de shortfall     : {k['prob_shortfall_any']:.2%}")
            lines.append(f"Shortfall moyen (tous scén.) : {k['shortfall_avg_per_month_all']:,.0f} €/mois")
            lines.append("")
            lines.append(f"Héritage médian (réel)       : {k['bequest_median']:,.0f} €")
            lines.append(f"Héritage Q5% (réel)          : {k['bequest_q5']:,.0f} €")

            ax.text(0.0, 1.0, "\n".join(lines), fontsize=11, va="top")
            fig.suptitle(f"Profil {rp} – Synthèse & KPI", fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # PAGE ACCUMULATION
            W_med = np.median(W_nom_paths_acc, axis=0)
            W_q5 = np.percentile(W_nom_paths_acc, 5, axis=0)
            W_q95 = np.percentile(W_nom_paths_acc, 95, axis=0)

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            ax1 = axes[0]
            ax1.plot(t_years_acc, W_med, label="Médiane")
            ax1.fill_between(t_years_acc, W_q5, W_q95, alpha=0.2, label="[5%, 95%]")
            ax1.axvline(AGE_RETIREMENT, color="red", linestyle="--", label="Retraite")
            ax1.set_xlabel("Âge")
            ax1.set_ylabel("Capital nominal (€)")
            ax1.set_title("Accumulation : évolution du capital")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            ax2 = axes[1]
            ax2.hist(W_nom_ret, bins=30, alpha=0.7, edgecolor="black")
            ax2.set_title("Distribution du capital à la retraite")
            ax2.set_xlabel("Capital nominal à la retraite (€)")
            ax2.set_ylabel("Fréquence")
            ax2.grid(True, alpha=0.3)

            fig.suptitle(f"Profil {rp} – Phase accumulation", fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # PAGE ALLOCATION
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(w_opt, labels=asset_names, autopct="%1.1f%%")
            ax.set_title(f"Profil {rp} – Allocation de portefeuille")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

            # PAGE DÉCUMULATION
            N_dec = len(C_real)
            t_months = np.arange(N_dec)
            t_years_ret = t_months * DT

            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            ax1 = axes[0]
            ax1.plot(
                t_years_ret,
                income_real,
                label=f"Revenu réel (moy. ≈ {k['income_real_mean']:,.0f} €/mois)"
            )
            ax1.plot(
                t_years_ret,
                income_nominal_mean,
                linestyle="--",
                label="Revenu nominal (moy. trajectoire inflation)"
            )
            ax1.axhline(
                k["essential_income_real"],
                color="red",
                linestyle=":",
                label=f"Pension essentielle ({k['essential_income_real']:,.0f} €/mois réels)"
            )
            ax1.set_xlabel("Années depuis la retraite")
            ax1.set_ylabel("Revenu (€/mois)")
            ax1.set_title("Profil de consommation en retraite")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            ax2 = axes[1]
            ax2.hist(bequest_total_real, bins=30, alpha=0.7, edgecolor="black")
            ax2.set_title("Distribution de l'héritage total (réel)")
            ax2.set_xlabel("Héritage (€/réel)")
            ax2.set_ylabel("Fréquence")
            ax2.grid(True, alpha=0.3)

            fig.suptitle(f"Profil {rp} – Phase décumulation", fontsize=14)
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close(fig)

        # TABLEAU FINAL
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis("off")

        cols_to_show = [
            "risk_profile",
            "W0_ret_nom_used",
            "prob_ruine",
            "prob_shortfall_any",
            "income_real_mean",
            "bequest_median",
            "bequest_q5",
        ]
        df_disp = df_kpi[cols_to_show].copy()
        df_disp["prob_ruine"] = df_disp["prob_ruine"].map(lambda x: f"{x:.1%}")
        df_disp["prob_shortfall_any"] = df_disp["prob_shortfall_any"].map(lambda x: f"{x:.1%}")
        df_disp["income_real_mean"] = df_disp["income_real_mean"].map(lambda x: f"{x:,.0f}")
        df_disp["bequest_median"] = df_disp["bequest_median"].map(lambda x: f"{x:,.0f}")
        df_disp["bequest_q5"] = df_disp["bequest_q5"].map(lambda x: f"{x:,.0f}")
        df_disp["W0_ret_nom_used"] = df_disp["W0_ret_nom_used"].map(lambda x: f"{x:,.0f}")

        table = ax.table(
            cellText=df_disp.values,
            colLabels=df_disp.columns,
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1.1, 1.4)

        fig.suptitle("Comparatif des profils pour l'objectif choisi", fontsize=14)
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"\n📄 Rapport PDF généré dans : {output_path}")


# ==========================
# MAIN
# ==========================

def main():
    print("Chargement de l'historique propre...")
    returns_hist = load_historical_returns(HISTORICAL_FILE)
    print("Colonnes historiques lues :", list(returns_hist.columns))

    print("Chargement de AssumptionForSimulation...")
    assump = load_assumption_data(ASSUMPTIONS_XLSX)
    print("Actifs dans AssumptionForSimulation :", assump["asset_names"])

    returns_aligned, asset_names_core, asset_vol_annual_core, Sigma_annual = align_assets_and_build_cov(
        returns_hist, assump
    )

    # Mu annuel : priorité aux Expected Return, fallback sur moyennes historiques
    er_all = assump["asset_er_annual"]
    assump_names = assump["asset_names"]
    er_map = {name: val for name, val in zip(assump_names, er_all)}

    if returns_aligned.shape[1] > 0:
        mu_hist_map = {
            name: val
            for name, val in zip(
                asset_names_core[:returns_aligned.shape[1]],
                returns_aligned.mean().values * 12.0
            )
        }
    else:
        mu_hist_map = {}

    mu_list = []
    for name in asset_names_core:
        er_val = er_map.get(name, np.nan)
        if not np.isnan(er_val) and er_val != 0.0:
            mu_list.append(er_val)
        else:
            mu_list.append(mu_hist_map.get(name, 0.0))

    mu_core = np.array(mu_list, float)
    print("\nMu annuel utilisé pour chaque actif (avant ZeroCoupon) :")
    for n, m in zip(asset_names_core, mu_core):
        print(f" - {n:30s}: {m:.2%}")

    mu_vec, Sigma, asset_names_ext = add_zero_coupon_asset(
        mu_core, Sigma_annual, asset_names_core, assump["rf_rate_annual"]
    )

    print("\nActifs utilisés dans l'allocation (y compris ZeroCouponUSD) :")
    for name in asset_names_ext:
        print(" -", name)

    pi_annual, sigma_pi_annual = PI_ANNUAL_DEFAULT, SIGMA_PI_ANNUAL_DEFAULT
    t_inf, I_inf = simulate_global_inflation(
        pi_annual=pi_annual,
        sigma_pi_annual=sigma_pi_annual,
        age_start=AGE_START_SAVING,
        age_retirement=AGE_RETIREMENT,
        dt=DT,
        seed=RANDOM_SEED
    )

    kpi_rows = []
    profiles_results = {}

    for rp in RISK_PROFILES.keys():
        row, profile_data = evaluate_profile(
            risk_profile_name=rp,
            mu_vec=mu_vec,
            Sigma=Sigma,
            asset_names=asset_names_ext,
            pi_annual=pi_annual,
            sigma_pi_annual=sigma_pi_annual,
        )
        kpi_rows.append(row)
        profiles_results[rp] = profile_data

    df_kpi = pd.DataFrame(kpi_rows)
    print("\n=========== TABLEAU RÉCAPITULATIF PAR PROFIL DE RISQUE ===========")
    cols_to_show = [
        "risk_profile",
        "retirement_profile",
        "legacy_share",
        "secondary_home_cost",
        "W0_ret_nom_used",
        "prob_ruine",
        "prob_shortfall_any",
        "income_real_mean",
        "bequest_median",
        "bequest_q5",
    ]
    print(df_kpi[cols_to_show])

    df_kpi.to_csv("kpi_retirement_profiles_summary.csv", index=False)
    print("\nKPI sauvegardés dans 'kpi_retirement_profiles_summary.csv'.")

    generate_pdf_report(
        output_path="rapport_retraite_fixedmix.pdf",
        df_kpi=df_kpi,
        profiles_results=profiles_results,
        t_inf=t_inf,
        I_inf=I_inf,
        pi_annual=pi_annual,
        sigma_pi_annual=sigma_pi_annual,
    )


if __name__ == "__main__":
    main()
