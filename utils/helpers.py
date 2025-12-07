def print_simulation_summary(df, strategy_name, initial_wealth):
    """
    Affiche un résumé financier d'une simulation.
    """
    # 1. Calculs
    # Le capital final est la dernière valeur de la courbe Wealth
    final_wealth = df['Wealth'].iloc[-1]
    
    # Le total investi est le capital de départ + la somme de tous les versements mensuels
    # .fillna(0) est une sécurité au cas où il y aurait des NaN
    total_contributions = df['Contrib_mensuel'].fillna(0).sum()
    total_invested = initial_wealth + total_contributions
    
    # Plus-value (P&L) et Performance (%)
    pnl = final_wealth - total_invested
    roi = (pnl / total_invested) * 100

    # 2. Affichage formaté
    print(f"========================================")
    print(f"  RÉSUMÉ : {strategy_name}")
    print(f"========================================")
    print(f" Capital Initial      : {initial_wealth:,.2f} €")
    print(f" Total Versements     : {total_contributions:,.2f} €")
    print(f" TOTAL INVESTI        : {total_invested:,.2f} €")
    print(f"----------------------------------------")
    print(f" CAPITAL FINAL        : {final_wealth:,.2f} €")
    print(f" Plus-Value           : {pnl:,.2f} €")
    print(f" Performance Globale  : {roi:+.2f} %")
    print(f"========================================\n")

