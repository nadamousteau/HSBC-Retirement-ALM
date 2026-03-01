# HSBC-Retirement-ALM

## Contexte

La viabilité du système de retraite par répartition est menacée par l'évolution démographique.
Le rapport du Conseil d'Orientation des Retraites (COR, 2022) met en évidence une chute 
structurelle du ratio démographique, passant de 2,1 cotisants pour un retraité en 2000 à une 
projection d'environ 1,2 en 2070. Cette tendance impose de compléter les revenus de retraite 
par des mécanismes de capitalisation individuelle.

Ce basculement opère un transfert majeur des risques de longévité et de marché vers les 
individus. La logique financière évolue ainsi d'une simple accumulation de patrimoine vers la 
sécurisation d'un flux de revenus futurs.

C'est dans ce cadre que s'inscrit notre étude des stratégies de retraite par capitalisation. 
L'approche ALM (Asset Liability Management) est mobilisée pour aligner l'horizon de placement 
sur les engagements de passif. L'objectif est de définir des modèles d'allocation offrant un 
compromis optimal entre recherche de performance et sécurisation du capital à long terme.

## Stratégies d'allocation simulées

Ce projet explore quatre stratégies d'accumulation :

1. **Goal-Based Investing** : séparation en deux portefeuilles — un portefeuille de couverture et un portefeuille de performance.
2. **Target Date Fund** : glide path déterministe où l'allocation actions diminue 
   automatiquement avec l'âge.
3. **Fixed-Mix** : maintien d'une allocation constante (ex. 60% actions / 40% obligations), indépendamment de l'âge ou des conditions de marché.
4. **Approche de Faleh** : optimisation globale de l'allocation sur un arbre de scénarios.

## Modélisation financière

### Génération des rendements
Les rendements sont modélisés selon un processus de **Black-Scholes corrélé** (deux actifs : 
actions et obligations). Cinq profils investisseur sont disponibles : Prudent, Modéré, 
Équilibré, Dynamique, Agressif.

### Simulation des crises
Deux modèles de chocs sont implémentés :

- **Jump-Diffusion de Merton** : Des crises surviennent aléatoirement avec une 
  probabilité annuelle λ (défaut : 15%). Chaque crise génère un choc négatif sur les actions 
  (μ = -25%, σ = 10%) et les obligations (μ = -8%, σ = 3%), indépendamment.
- **Crise localisée** : un choc déterministe est injecté à une date précise 
  (ex. mars 2030), combinant une chute instantanée des actions (-35%), une perturbation 
  obligataire (-5%) et une période de volatilité amplifiée (×2,5 sur 18 mois).

### Simulation de l'inflation

L'inflation est simulée grâce au modèle de Vasicek. 

### Stratégies de déccumulation

