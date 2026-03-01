"""
Script de validation du RSLN (Regime Switching Log-Normal).
Vérifie que les régimes ont bien une mémoire et respectent la distribution stationnaire.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.economics.gse import EnhancedGSE, MarkovRegimeSwitching

def test_regime_persistence():
    """
    Test 1 : Vérifier que les régimes ont une mémoire (autocorrélation > 0).
    """
    print("\n" + "="*70)
    print("TEST 1 : PERSISTANCE DES RÉGIMES")
    print("="*70)
    
    regime_model = MarkovRegimeSwitching()
    regimes = regime_model.simulate_regimes(
        nb_periods=480,  # 40 ans
        nb_scenarios=1000,
        seed=42
    )
    
    # Calcul de l'autocorrélation à lag 1
    autocorr_lag1 = []
    for s in range(regimes.shape[1]):
        series = regimes[:, s]
        # Autocorr = Cov(X_t, X_{t-1}) / Var(X_t)
        mean_val = series.mean()
        var_val = series.var()
        cov_val = np.mean((series[:-1] - mean_val) * (series[1:] - mean_val))
        autocorr_lag1.append(cov_val / var_val if var_val > 0 else 0)
    
    avg_autocorr = np.mean(autocorr_lag1)
    
    print(f"Autocorrélation moyenne (lag=1) : {avg_autocorr:.4f}")
    print(f"Attendu : > 0.65 (forte persistance)")
    
    if avg_autocorr > 0.65:
        print("✅ PASS : Les régimes ont bien une mémoire")
    else:
        print("❌ FAIL : Régimes trop i.i.d., vérifier la matrice de transition")

def test_stationary_distribution():
    """
    Test 2 : Vérifier que la distribution empirique converge vers la distribution stationnaire.
    """
    print("\n" + "="*70)
    print("TEST 2 : DISTRIBUTION STATIONNAIRE")
    print("="*70)
    
    regime_model = MarkovRegimeSwitching()
    regimes = regime_model.simulate_regimes(
        nb_periods=10000,  # Longue simulation pour convergence
        nb_scenarios=500,
        seed=42
    )
    
    # Validation
    stats = regime_model.validate_calibration(regimes)
    
    print("\nDistribution théorique (stationnaire) :")
    print(f"  Normal : {stats['theoretical_distribution'][0]:.2%}")
    print(f"  Bull   : {stats['theoretical_distribution'][1]:.2%}")
    print(f"  Bear   : {stats['theoretical_distribution'][2]:.2%}")
    
    print("\nDistribution empirique (simulée) :")
    print(f"  Normal : {stats['empirical_distribution'][0]:.2%}")
    print(f"  Bull   : {stats['empirical_distribution'][1]:.2%}")
    print(f"  Bear   : {stats['empirical_distribution'][2]:.2%}")
    
    print(f"\nDéviation maximale : {stats['max_deviation']:.4f}")
    print(f"Attendu : < 0.02 (2%)")
    
    if stats['max_deviation'] < 0.02:
        print("✅ PASS : Distribution respectée")
    else:
        print("⚠️  WARN : Déviation élevée, augmenter nb_periods ou nb_scenarios")
    
    print("\nDurée moyenne des régimes (en périodes) :")
    print(f"  Normal : {stats['average_durations'][0]:.1f} mois")
    print(f"  Bull   : {stats['average_durations'][1]:.1f} mois")
    print(f"  Bear   : {stats['average_durations'][2]:.1f} mois")

def test_impact_on_volatility():
    """
    Test 3 : Vérifier que le RSLN augmente la volatilité totale des rendements.
    """
    print("\n" + "="*70)
    print("TEST 3 : IMPACT SUR LA VOLATILITÉ")
    print("="*70)
    
    # Paramètres de marché
    mu_e, sigma_e = 0.08, 0.18
    mu_b, sigma_b = 0.03, 0.05
    corr_eb = 0.2
    
    # Scénario 1 : Avec RSLN
    gse_with_rsln = EnhancedGSE(mu_e, sigma_e, mu_b, sigma_b, corr_eb, use_markov_regimes=True)
    r_eq_rsln, r_bd_rsln, regimes_rsln, _ = gse_with_rsln.generate_scenarios(480, 1000, seed=42)
    
    # Scénario 2 : Sans RSLN (i.i.d.)
    gse_without_rsln = EnhancedGSE(mu_e, sigma_e, mu_b, sigma_b, corr_eb, use_markov_regimes=False)
    r_eq_iid, r_bd_iid, regimes_iid, _ = gse_without_rsln.generate_scenarios(480, 1000, seed=42)
    
    # Calcul des volatilités réalisées
    vol_rsln = np.std(r_eq_rsln, axis=0).mean() * np.sqrt(12)
    vol_iid = np.std(r_eq_iid, axis=0).mean() * np.sqrt(12)
    
    print(f"Volatilité réalisée (annualisée) :")
    print(f"  Avec RSLN    : {vol_rsln:.2%}")
    print(f"  Sans RSLN    : {vol_iid:.2%}")
    print(f"  Paramètre σ  : {sigma_e:.2%}")
    
    vol_increase = (vol_rsln - vol_iid) / vol_iid
    print(f"\nAugmentation due au RSLN : +{vol_increase:.1%}")
    print(f"Attendu : +10% à +30% (clustering de volatilité)")
    
    if vol_increase > 0.10:
        print("✅ PASS : Le RSLN capture bien le fat-tail risk")
    else:
        print("⚠️  WARN : Impact faible, vérifier les multiplicateurs de régimes")

def plot_regime_evolution():
    """
    Test 4 : Visualisation de l'évolution des régimes sur une trajectoire.
    """
    print("\n" + "="*70)
    print("TEST 4 : VISUALISATION")
    print("="*70)
    
    regime_model = MarkovRegimeSwitching()
    regimes = regime_model.simulate_regimes(
        nb_periods=240,  # 20 ans
        nb_scenarios=1,
        seed=123
    )[:, 0]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # Créer un colormap pour les régimes
    colors = ['green' if r == 0 else 'blue' if r == 1 else 'red' for r in regimes]
    
    ax.scatter(range(len(regimes)), regimes, c=colors, s=20, alpha=0.6)
    ax.plot(regimes, linewidth=0.5, color='black', alpha=0.3)
    
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Normal', 'Bull', 'Bear'])
    ax.set_xlabel('Mois')
    ax.set_title('Évolution des Régimes de Marché (Chaîne de Markov)')
    ax.grid(True, alpha=0.3)
    
    # Légende
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Normal (70%)'),
        Patch(facecolor='blue', label='Bull (15%)'),
        Patch(facecolor='red', label='Bear (15%)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('regime_evolution.png', dpi=150)
    print("✅ Graphique sauvegardé : regime_evolution.png")
    plt.show()

if __name__ == "__main__":
    test_regime_persistence()
    test_stationary_distribution()
    test_impact_on_volatility()
    plot_regime_evolution()
    
    print("\n" + "="*70)
    print("VALIDATION RSLN TERMINÉE")
    print("="*70)