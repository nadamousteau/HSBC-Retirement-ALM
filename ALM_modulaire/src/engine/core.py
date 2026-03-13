import numpy as np
from config import settings
from src.liabilities import contributions

def run_simulation(strategy, r_eq, r_bd, dates, inflation=None):
    """
    Exécute la simulation Monte Carlo unifiée (Moteur unique).
    
    Args:
        strategy : Objet stratégie d'allocation
        r_eq : Rendements actions (nb_steps, nb_sims)
        r_bd : Rendements obligations (nb_steps, nb_sims)
        dates : Liste des dates
        inflation : Inflation stochastique (nb_steps, nb_sims), optionnel
                   Si None, utilise TAUX_INFLATION constant
    
    Returns:
        tuple : (mat_capital, courbe_investi, hist_apport, hist_drawdown, hist_salaire, inflation)
    """
    nb_steps, nb_sims = r_eq.shape
    
    # Générer inflation si non fournie
    if inflation is None:
        inflation = np.ones((nb_steps, nb_sims)) * settings.TAUX_INFLATION
    
    mat_capital = np.zeros((nb_steps + 1, nb_sims))
    mat_capital[0, :] = settings.CAPITAL_INITIAL
    
    hist_apport = np.zeros(nb_steps)
    hist_salaire = np.zeros(nb_steps)
    hist_drawdown = np.zeros((nb_steps, nb_sims))
    
    courbe_investi = np.zeros(nb_steps + 1)
    courbe_investi[0] = settings.CAPITAL_INITIAL
    
    capital_max = np.ones(nb_sims) * settings.CAPITAL_INITIAL

    eq_price = np.ones(nb_sims) * 100.0
    bd_price = np.ones(nb_sims) * 100.0
    
    pct_eq, pct_bd = strategy.get_allocation(0, settings.AGE_DEPART)
    
    eq_shares = (settings.CAPITAL_INITIAL * pct_eq) / eq_price
    bd_shares = (settings.CAPITAL_INITIAL * pct_bd) / bd_price

    app_init, app_max, t_pic = contributions.precalculer_parametres_apport_exponentiel(
        settings.SALAIRE_INITIAL, settings.SALAIRE_MAX_CIBLE, settings.NB_ANNEES_ACCUMULATION
    )
    fm_params = {'app_init': app_init, 'app_max': app_max, 't_pic': t_pic}
    
    # Facteur cumulatif d'indexation (inflation stochastique)
    inflation_factor = np.ones((nb_steps + 1, nb_sims))

    # =========================================================================
    # BOUCLE TEMPORELLE
    # =========================================================================
    for k in range(nb_steps):
        t_annees = k / 12.0
        age_actuel = settings.AGE_DEPART + int(t_annees)
        
        eq_price *= (1 + r_eq[k])
        bd_price *= (1 + r_bd[k])
        
        capital_avant = eq_shares * eq_price + bd_shares * bd_price
        
        # Calcul du facteur d'indexation cumulatif
        inflation_factor[k+1, :] = inflation_factor[k, :] * (1 + inflation[k, :])
        
        # Apports et salaires indexés à l'inflation stochastique
        apport_base = contributions.calculer_apport_exponentiel(
            t_annees, fm_params['app_init'], fm_params['app_max'], fm_params['t_pic']
        )
        salaire_base = contributions.estimer_salaire_saturation(
            t_annees, settings.SALAIRE_INITIAL, settings.SALAIRE_MAX_CIBLE
        )
        
        # Indexation : Multiplier par le taux MENSUEL uniquement, pas le cumul
        apport_mensuel_indexed = apport_base * (1 + inflation[k, :])
        courbe_investi[k+1] = courbe_investi[k] + np.mean(apport_mensuel_indexed)
        
        # Versions moyennes pour l'historique (reporting)
        indexation_moyenne = np.mean(1 + inflation[k, :])
        hist_salaire[k] = salaire_base * indexation_moyenne
        hist_apport[k] = apport_base * indexation_moyenne
        
        pct_eq_target, pct_bd_target = strategy.get_allocation(k, age_actuel)
        
        # Achat d'actifs (distribution à tous les scénarios)
        eq_buy = apport_mensuel_indexed * pct_eq_target
        bd_buy = apport_mensuel_indexed * pct_bd_target
        
        eq_shares += eq_buy / eq_price
        bd_shares += bd_buy / bd_price
        
        # Correction formelle de l'encapsulation : Le moteur évalue l'objet, pas le settings
        if getattr(strategy, 'should_rebalance', lambda x: False)(k):
            total_val = eq_shares * eq_price + bd_shares * bd_price
            
            if strategy.__class__.__name__ == "TargetDateStrategy":
                age_next = settings.AGE_DEPART + int((k+1)/12.0)
                target_pct_eq, _ = strategy.get_allocation(k, age_next)
            else:
                target_pct_eq, _ = strategy.get_allocation(k, age_actuel)
            
            target_eq_val = total_val * target_pct_eq
            current_eq_val = eq_shares * eq_price
            
            diff_eq = target_eq_val - current_eq_val
            
            shares_to_trade_eq = diff_eq / eq_price
            cost_in_bonds = diff_eq
            shares_to_trade_bd = -cost_in_bonds / bd_price
            
            eq_shares += shares_to_trade_eq
            bd_shares += shares_to_trade_bd

        capital_apres = eq_shares * eq_price + bd_shares * bd_price
        mat_capital[k+1, :] = capital_apres
        
        capital_ref = capital_avant if settings.DRAWDOWN_AVANT_APPORT else capital_apres
            
        capital_max = np.maximum(capital_max, capital_ref)
        dd = (capital_ref - capital_max) / capital_max
        dd = np.where(capital_max > 1e-9, dd, 0.0)
        hist_drawdown[k, :] = dd

    return mat_capital, courbe_investi, hist_apport, hist_drawdown, hist_salaire, inflation_factor