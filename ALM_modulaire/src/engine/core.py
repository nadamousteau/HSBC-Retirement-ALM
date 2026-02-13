import numpy as np
from config import settings
from src.liabilities import contributions

def run_simulation(strategy, r_eq, r_bd, dates):
    """
    Exécute la simulation Monte Carlo unifiée (Moteur unique).
    
    Args:
        strategy (BaseStrategy): Instance de la stratégie (TargetDate ou FixedMix).
        r_eq (np.array): Matrice des rendements Equity (T x N).
        r_bd (np.array): Matrice des rendements Bond (T x N).
        dates (pd.DatetimeIndex): Liste des dates.
        
    Returns:
        tuple: (mat_capital, courbe_investi, hist_apport, hist_drawdown, hist_salaire)
    """
    nb_steps, nb_sims = r_eq.shape
    
    # --- Initialisation des structures de données ---
    mat_capital = np.zeros((nb_steps + 1, nb_sims))
    mat_capital[0, :] = settings.CAPITAL_INITIAL
    
    hist_apport = np.zeros(nb_steps)
    hist_salaire = np.zeros(nb_steps) # Pour Fixed Mix uniquement
    hist_drawdown = np.zeros((nb_steps, nb_sims))
    
    courbe_investi = np.zeros(nb_steps + 1)
    courbe_investi[0] = settings.CAPITAL_INITIAL
    
    capital_max = np.ones(nb_sims) * settings.CAPITAL_INITIAL

    # --- Initialisation du Portefeuille (Parts) ---
    eq_price = np.ones(nb_sims) * 100.0
    bd_price = np.ones(nb_sims) * 100.0
    
    # Allocation initiale (t=0)
    pct_eq, pct_bd = strategy.get_allocation(0, settings.AGE_DEPART)
    
    eq_shares = (settings.CAPITAL_INITIAL * pct_eq) / eq_price
    bd_shares = (settings.CAPITAL_INITIAL * pct_bd) / bd_price

    # --- Pré-calculs spécifiques au mode de contribution ---
    # Pour respecter la logique originale : 
    # Target Date -> Apport Quadratique
    # Fixed Mix -> Apport Exponentiel
    
    fm_params = {}
    if settings.METHODE == "FIXED_MIX":
        app_init, app_max, t_pic = contributions.precalculer_parametres_apport_exponentiel(
            settings.SALAIRE_INITIAL, 
            settings.SALAIRE_MAX_CIBLE, 
            settings.NB_ANNEES_ACCUMULATION
        )
        fm_params = {'app_init': app_init, 'app_max': app_max, 't_pic': t_pic}
    
    apport_base_init = settings.SALAIRE_INITIAL * settings.TAUX_APPORT_BASE

    # =========================================================================
    # BOUCLE TEMPORELLE
    # =========================================================================
    for k in range(nb_steps):
        t_annees = k / 12.0
        age_actuel = settings.AGE_DEPART + int(t_annees)
        
        # 1. Mise à jour des prix de marché (Performance)
        eq_price *= (1 + r_eq[k])
        bd_price *= (1 + r_bd[k])
        
        # 2. Valorisation avant flux (pour Drawdown "Marché")
        capital_avant = eq_shares * eq_price + bd_shares * bd_price
        
        # 3. Calcul de l'apport (Logique conditionnelle comme l'original)
        if settings.METHODE == "TARGET_DATE":
            apport_mensuel = contributions.calculer_apport_quadratique(
                t_annees, apport_base_init, settings.NB_ANNEES_ACCUMULATION
            )
        else: # FIXED_MIX
            apport_mensuel = contributions.calculer_apport_exponentiel(
                t_annees, fm_params['app_init'], fm_params['app_max'], fm_params['t_pic']
            )
            # Calcul du salaire théorique pour le reporting
            salaire = contributions.estimer_salaire_saturation(
                t_annees, settings.SALAIRE_INITIAL, settings.SALAIRE_MAX_CIBLE
            )
            hist_salaire[k] = salaire

        hist_apport[k] = apport_mensuel
        courbe_investi[k+1] = courbe_investi[k] + apport_mensuel
        
        # 4. Investissement de l'apport (Buying)
        # On achète selon l'allocation cible à cet instant
        pct_eq_target, pct_bd_target = strategy.get_allocation(k, age_actuel)
        
        eq_buy = apport_mensuel * pct_eq_target
        bd_buy = apport_mensuel * pct_bd_target
        
        eq_shares += eq_buy / eq_price
        bd_shares += bd_buy / bd_price
        
        # 5. Vérification du Rééquilibrage (Rebalancing)
        if strategy.should_rebalance(k):
            total_val = eq_shares * eq_price + bd_shares * bd_price
            
            # Allocation cible pour le pas suivant (ou actuel)
            # Note: TargetDate rebalance vers l'alloc de l'âge suivant (k+1 logic)
            # FixedMix rebalance vers l'alloc constante
            if settings.METHODE == "TARGET_DATE":
                age_next = settings.AGE_DEPART + int((k+1)/12.0) # approx fin d'année
                target_pct_eq, _ = strategy.get_allocation(k, age_next)
            else:
                target_pct_eq, _ = strategy.get_allocation(k, age_actuel)
            
            target_eq_val = total_val * target_pct_eq
            current_eq_val = eq_shares * eq_price
            
            diff_eq = target_eq_val - current_eq_val
            
            # Exécution (Achat/Vente)
            # Si diff > 0 : on doit acheter Equity -> on vend Bond
            # Si diff < 0 : on doit vendre Equity -> on achète Bond
            
            # Transactions vectorisées
            shares_to_trade_eq = diff_eq / eq_price
            cost_in_bonds = diff_eq # Valeur monétaire à transférer
            shares_to_trade_bd = -cost_in_bonds / bd_price
            
            eq_shares += shares_to_trade_eq
            bd_shares += shares_to_trade_bd

        # 6. Valorisation finale et Stockage
        capital_apres = eq_shares * eq_price + bd_shares * bd_price
        mat_capital[k+1, :] = capital_apres
        
        # 7. Calcul du Drawdown
        if settings.DRAWDOWN_AVANT_APPORT:
            capital_ref = capital_avant
        else:
            capital_ref = capital_apres
            
        capital_max = np.maximum(capital_max, capital_ref)
        dd = (capital_ref - capital_max) / capital_max
        # Gestion des divisions par zéro si capital_max = 0 (peu probable ici)
        dd = np.where(capital_max > 1e-9, dd, 0.0)
        hist_drawdown[k, :] = dd

    return mat_capital, courbe_investi, hist_apport, hist_drawdown, hist_salaire