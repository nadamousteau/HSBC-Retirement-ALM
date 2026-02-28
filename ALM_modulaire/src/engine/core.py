import numpy as np
from config import settings
from src.liabilities import contributions

def run_simulation(strategy, r_eq, r_bd, dates):
    """Exécute la simulation Monte Carlo unifiée (Moteur unique)."""
    nb_steps, nb_sims = r_eq.shape
    
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

    # =========================================================================
    # BOUCLE TEMPORELLE
    # =========================================================================
    for k in range(nb_steps):
        t_annees = k / 12.0
        age_actuel = settings.AGE_DEPART + int(t_annees)
        
        eq_price *= np.exp(r_eq[k])
        bd_price *= np.exp(r_bd[k])
        
        capital_avant = eq_shares * eq_price + bd_shares * bd_price
        
        apport_mensuel = contributions.calculer_apport_exponentiel(
            t_annees, fm_params['app_init'], fm_params['app_max'], fm_params['t_pic']
        )
        salaire = contributions.estimer_salaire_saturation(
            t_annees, settings.SALAIRE_INITIAL, settings.SALAIRE_MAX_CIBLE
        )
        
        hist_salaire[k] = salaire
        hist_apport[k] = apport_mensuel
        courbe_investi[k+1] = courbe_investi[k] + apport_mensuel
        
        pct_eq_target, pct_bd_target = strategy.get_allocation(k, age_actuel)
        
        eq_buy = apport_mensuel * pct_eq_target
        bd_buy = apport_mensuel * pct_bd_target
        
        eq_shares += eq_buy / eq_price
        bd_shares += bd_buy / bd_price
        
        # Correction formelle de l'encapsulation : Le moteur évalue l'objet, pas le settings
        if getattr(strategy, 'should_rebalance', lambda x: False)(k):
            total_val = eq_shares * eq_price + bd_shares * bd_price
            
            if strategy.__class__.__name__ == "TargetDateStrategy":
                age_next = settings.AGE_DEPART + int((k+1)/12.0)
                target_pct_eq, target_pct_bd = strategy.get_allocation(k, age_next)
            else:
                target_pct_eq, target_pct_bd = strategy.get_allocation(k, age_actuel)
            
            # Réaffectation stricte des parts selon les poids cibles
            eq_shares = (total_val * target_pct_eq) / eq_price
            bd_shares = (total_val * target_pct_bd) / bd_price

        capital_apres = eq_shares * eq_price + bd_shares * bd_price
        mat_capital[k+1, :] = capital_apres
        
        capital_ref = capital_avant if settings.DRAWDOWN_AVANT_APPORT else capital_apres
            
        capital_max = np.maximum(capital_max, capital_ref)
        dd = (capital_ref - capital_max) / capital_max
        dd = np.where(capital_max > 1e-9, dd, 0.0)
        hist_drawdown[k, :] = dd

    return mat_capital, courbe_investi, hist_apport, hist_drawdown, hist_salaire