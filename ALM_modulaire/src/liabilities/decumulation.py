import numpy as np


class decumulation :

    """
    Différentes stratégies de décumulation, selon le profil, la stratégie.
    Les differents profils à la retraite détermine le pourcentage que la personne veut placer sur lon livret A ( pourcentage déterminé 
    parrapport au salaire final, au taux de remplacement visé ...),veut léguer,
    et le profil de risque. Le capital qui reste investi est investi selon une des stratégies du dossier strategie. 
    Cette information est stockée dans strat_retraite. 
    La classe suppose l existence de la matrice de l evolution du capital jusqu au debut de la retraite.
    Il faudra ensuite à nouveau faire tourner le core pour avoir la matrice de capital à durant la retraite.
    Dans le profil de retraite, il faut qu il y ait les infos sur les differentes stratégies (ex : les allocations fixed mix 
    ou les allocations intitiales du target date), le taux de remplacement visé et le pourcentage de perte qui entraine 
     la consomation du matela de sécurité. (pour le moment, je prends en compte que la volonté de laisser un héritage f
    inancier se traduira par l'aggressivité ou non du profil pour la gestion du capital encore investi selon l'une des stratégies)

    Cette classe permet d'obtenir l age de fin de simulation (en gros l age de la mort), le montant du capital au 
    debut de la retraite qui reste investi selon une strategie du code et le montant minimal qu il doit y avoir sur le matela de securité.
    Les arbitrages sont ensuite gérés dans le fichier core. 
    """

    def __init__(
        self,
        profil_retraite : str,  
        age_retraite: int,
        matrice_capital : np.ndarray,
        matrice_salaire : np.ndarray
       
    ):
        self.profil_retraite = profil_retraite 
        self.age_retraite = age_retraite
        self.matrice_capital = matrice_capital
        self.matrice_salaire = matrice_salaire
        

        

    def age_deces(self) -> int: #renvoie l age de fin de simulation
        # Tirage d'une probabilité uniforme
        u = np.random.uniform(1e-10, 1.0)
        
        # Paramètres calibrés sur la mortalité française
        k, lam = 10.2, 86.5
        
        # Inversion de la fonction de survie conditionnelle : S(x)/S(R) = U
        r_term = (self.age_retraite / lam) ** k
        age_deces = lam * (r_term - np.log(u)) ** (1 / k)
        
        # Retourne l'âge arrondi, plafonné à la limite biologique de 110 ans
        return int(min(round(age_deces), 110))
    
    def capital_retraite(self):

        """On retourne le retrait inital qui est une liste de trois valeurs (médian, P5, P95)
        Je pars du principe que la matrice capital contient aussi ces 3 trajectoires et aussi dans le meme ordre
        Dans le core il faudra faire pour la simulation de retraite mat_capital_retraite[0]=capital_expose_retraite + retrait_inital
        et gerer ses deux parties de maniere differentes (strat/livret A
        )"""
       
        retrait_inital= self.profil_retraite["taux_remplacement_vise"]*12* self.matrice_salaire[:, -1]
        capital_expose_retraite = self.matrice_capital[0]-retrait_inital
        return retrait_inital, capital_expose_retraite
        
