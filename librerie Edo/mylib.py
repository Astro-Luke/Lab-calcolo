'''
libreria mylib contiene:
    STATISTICHE:   (riga 90)  riga in realtà non più precisa perchè ho aggiunto cose
        median
        mean
        variance
        stdev
        stdev_mean
        percentile75
        percentile25
        percentile
    
    sturges    (riga 179)
    
    ESTREMI:   (riga 193)
        bisezione
        bisezione_ricorsiva
    
    ESTREMANTI:    (riga 230)
        sezioneAureaMin
        sezioneAureaMin_ricorsiva
        sezioneAureaEffMin
        sezioneAureaEffMin_ricorsiva
        sezioneAureaMax
        sezioneAureaMax_ricorsiva
    
    GENERATORI NUMERI PSEUDO-CASUALI   (riga 394)
        mygenera_pdf
        mygenera
        generate_uniform 
        rand_range
        generate_range
        rand_TAC
        generate_TAC
        rand_TCL
        generate_TCL
        rand_TCL_ms
        generate_TCL_ms
        inv_exp
        rand_exp
        generate_exp
        rand_poisson
        generate_poisson
    
    INTEGRALI:   (riga 735)
        integral_HOM
        integral_CrudeMC
    
    LIKELIHOOD:    (riga 801)
        likelihood
        loglikelihood
        loglikelihood_prod
        loglikelihood_ratio
        intersect_LLR
        sezioneAureaMAx_LL
        loglikelihood_more_params
    
    CLASSI DEFINITE A LEZIONE:    (riga 999)
        Fraction
        stats
        my_histo

    ALTRE FUNZIONI UTILI:   (riga 1323)
        retta
        polinomio_grad_3
        parabola
        fattoriale
        coeff_binom
        esponenziale
        Fibonacci
        soluz_eq_secondo_grado
        
    DISTRIBUZIONI:      (riga 1414)
        binomial_coefficient
        binomial_distribuition
        bernoulli_trial
        poisson_distribution
        cauchy_distribution
        maxwell_boltzmann_distribution
        breit_wigner_distribution

    FINE LIBRERIE:    
        argomenti lezioni
        pezzi di codice utili
        fit
        opzioni grafiche
        riassunti utili 
        definizione main
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor, log, gcd, factorial, pow
import random
import time

from iminuit import Minuit
from iminuit.cost import LeastSquares, ExtendedBinnedNLL

from IPython.display import display

#------------------------------STATISTICHE----------------------



def median(x):
    '''
    determina la mediana di un array
    '''
    x_sorted = np.sort(x)
    if len(x_sorted)%2 == 0: 
        # se la lunghezza è un numero pari, restituisce la media dei due valori centrali
        m = (x_sorted[int(len(x_sorted)/2)-1] + x_sorted[int(len(x_sorted)/2)])/2
    else: 
        # se dispari restituisce il valore centrale
        m = x_sorted[int((len(x_sorted)-1)/2)]
    return m


def mean(x):
    '''
    calcola la media di un array
    '''
    m = np.sum(x)/len(x)
    return m


def variance(x):
    '''
    calcola la varianzadi un array
    '''
    m = mean(x)
    m2= mean(x*x)
    return m2-m*m


def stdev(x,bessel=True):
    '''
    calcola la deviazione standard con correzione di Bessel di un array
    '''
    m = mean(x)
    r = x-m
    s = np.sqrt( np.sum(r*r)/(len(x)-1) ) if bessel else np.sqrt( np.sum(r*r)/len(x) )
    return s


def stdev_mean(x,bessel=True):
    '''
    calcola la deviazione standard della media con correzione di Bessel di un array
    '''
    s = stdev(x,bessel)
    return s/np.sqrt(len(x))


#--------------------------------statische meno utili-----


def percentile75(x):
    '''
    determina il valore sopra il quale giacciono il 25% dei valori, o sotto il quale trovo il 75% degli elementi
    '''
    x_sorted = np.sort(x)
    idx = int(0.75*len(x))
    pv = x_sorted[idx]
    return pv


def percentile25(x):
    '''
    determina il valore sopra il quale giacciono il 75% dei valori, o sotto il quale trovo il 25% degli elementi
    '''
    x_sorted = np.sort(x)
    idx = int(0.25*len(x))
    pv = x_sorted[idx]
    return pv


def percentile(x,p):
    '''
    determina il valore sopra il quale giacciono una percentuale data p dei valori
    '''
    if not (0 < p and p < 1):
        raise ValueError(f'Il valore percentuale p = {p} non è nell\'intervallo [0,1]')
    x_sorted = np.sort(x)
    idx = int(p*len(x))
    pv = x_sorted[idx]
    return pv



#---------------------------STURGES--------------------------------

#versione che compare nelle correzioni dei temi, non dovrebbe cambiare nulla 
def sturges (N_events) :
    '''
    algoritmo per calcolo del numero di bin ideale data la lunghezza del campione.
    Ideale per array di lunghezza non troppo piccola nè troppo grande
    per array grande (>10000) conviene cercare altri metodi, ad esempio usare la radice quadrata
    '''
    return int( np.ceil( 1 + 3.322 * np.log (N_events) ) )

'''
def sturges (N_events) :
    return int( np.ceil( 1 + np.log2(N_events) ) )
'''
    
    
#----------------------------ESTREMI---------------------------------



def bisezione (
    g,              # funzione di cui trovare lo zero
    xMin,           # minimo dell'intervallo          
    xMax,           # massimo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola zeri
    con il metodo della bisezione
    '''
    xAve = xMin 
    while ((xMax - xMin) > prec) :
        xAve = 0.5 * (xMax + xMin) 
        if (g (xAve) * g (xMin) > 0.): xMin = xAve 
        else                         : xMax = xAve 
    return xAve 
    
    
def bisezione_ricorsiva (
    g,              # funzione di cui trovare lo zero  
    xMin,           # minimo dell'intervallo            
    xMax,           # massimo dell'intervallo          
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola zeri
    con il metodo della bisezione ricorsivo
    '''
    xAve = 0.5 * (xMax + xMin)
    if ((xMax - xMin) < prec): return xAve ;
    if (g (xAve) * g (xMin) > 0.): return bisezione_ricorsiva (g, xAve, xMax, prec) ;
    else                         : return bisezione_ricorsiva (g, xMin, xAve, prec) ;



#-----------------------------ESTREMANTI-----------------------------------
#ovvero punti di massimo o di minimo


def sezioneAureaMin (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    '''
    r = 0.618
    x2 = 0.
    x3 = 0. 
    larghezza = abs (x1 - x0)
     
    while (larghezza > prec):
        x2 = x0 + r * (x1 - x0) 
        x3 = x0 + (1. - r) * (x1 - x0)  
      
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro        
        if (g (x3) > g (x2)): 
            x0 = x3
            # x1 = x1  questo estremo non cambia     
        else :
            x1 = x2
            # x0 = x0  questo estremo non cambia                
        larghezza = abs (x1-x0)                                           
    return (x0 + x1) / 2. 


def sezioneAureaMin_ricorsiva (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    implementata ricorsivamente
    '''
    r = 0.618
    x2 = x0 + r * (x1 - x0)
    x3 = x0 + (1. - r) * (x1 - x0) 
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  : return ( x0 + x1) / 2.
    elif (g (x3) > g (x2)) : return sezioneAureaMin_ricorsiva (g, x3, x1, prec)
    else                   : return sezioneAureaMin_ricorsiva (g, x0, x2, prec)   


def sezioneAureaEffMin (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea,
    riutilizzando nel ciclo uno dei punti calcolati 
    nell'iterazione precedente
    '''
    r = 0.618
    x2 = x0 + r * (x1 - x0)
    g_x2 = g (x2)
    x3 = 0. 
    larghezza = abs (x1 - x0)
     
    while (larghezza > prec):

        x3 = x0 + (1. - r) * (x1 - x0)
        g_x3 = g (x3)
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro        
        if (g_x3 < g_x2):
            # x0 = x0 questo estremo non cambia
            x1 = x2
            x2 = x3
            g_x2 = g_x3
        else :
            x0 = x1
            x1 = x3
            # x2 = x2 questo estremo non cambia

        larghezza = abs (x1-x0)                                            
    return (x0 + x1) / 2. 


def sezioneAureaEffMin_ricorsiva (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo   
    x2,             # punto intermedio dell'intervallo 
    g_x2,           # valore della funzione g in x2     
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    implementata ricorsivamente,
    riutilizzando nel ciclo uno dei punti calcolati 
    nell'iterazione precedente.
    '''
    r = 0.618
    x3 = x0 + (1. - r) * (x1 - x0) 
    g_x3 = g (x3)
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  : return (x0 + x1) / 2.
    elif g_x3 < g_x2 : return sezioneAureaEffMin_ricorsiva (g, x0, x2, x3, g_x3, prec)
    else             : return sezioneAureaEffMin_ricorsiva (g, x1, x3, x2, g_x2, prec)   


def sezioneAureaMax (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    '''
    r = 0.618
    x2 = 0.
    x3 = 0. 
    larghezza = abs (x1 - x0)
     
    while (larghezza > prec):
        x2 = x0 + r * (x1 - x0) 
        x3 = x0 + (1. - r) * (x1 - x0)  
      
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro        
        if (g (x3) < g (x2)): 
            x0 = x3
            # x1 = x1   questo estremo non cambia      
        else :
            x1 = x2
            # x0 = x0   questo estremo non cambia
            
        larghezza = abs (x1-x0)                                           
    return (x0 + x1) / 2. 


def sezioneAureaMax_ricorsiva (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    implementata ricorsivamente
    '''
    r = 0.618
    x2 = x0 + r * (x1 - x0)
    x3 = x0 + (1. - r) * (x1 - x0) 
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  : return ( x0 + x1) / 2.
    elif (g (x3) < g (x2)) : return sezioneAureaMax_ricorsiva (g, x3, x1, prec)
    else                   : return sezioneAureaMax_ricorsiva (g, x0, x2, prec)   



#-------------------GENERATORI NUMERI PSEUDO-CASUALI------------------
#myrand
# di default dalla libreria random
'''
random.random()     # genera numeri pseudocasuali tra 0 ed 1

random.randint(min, max)    # genera numeri pseudocasuali tra min e max

random.seed(seed)       # dove seed va passato nel main o lina di comando o in input, basta chiamare questa funzione una sola volta
'''

# Genera un singolo campione da una PDF arbitraria
def mygenera_pdf(xMin, xMax, yMax, pdf):
    '''
    Genera un singolo valore casuale distribuito secondo una PDF arbitraria
    usando il metodo try-and-catch.
    
    Attenzione: yMin è sempre zero

    Args:
        xMin (float): Estremo inferiore del dominio della PDF.
        xMax (float): Estremo superiore del dominio della PDF.
        yMax (float): Valore massimo stimato della PDF.
        pdf (function): Funzione densità di probabilità (PDF) arbitraria.

    Returns:
        float: Valore generato secondo la PDF.
        int: Numero di tentativi effettuati per accettare il valore
    '''
    num = 0
    while True:
        x = random.uniform(xMin, xMax)  # Genera un x casuale nel dominio [xMin, xMax]
        y = random.uniform(0, yMax)     # Genera un y casuale nell'intervallo [0, yMax]
        num += 1
        if y <= pdf(x):                 # Accetta x se y è sotto la PDF
            return x, num

        
# Genera un campione di N valori da una PDF arbitraria
def mygenera(N, xMin, xMax, yMax, pdf):
    '''
    Genera un campione di N valori distribuiti secondo una PDF arbitraria.

    Attenzione: restituisce una lista. Se si vuole un array, dopo aver ottenuto la lista da questa funzione, scrivo:
    campione = np.array(campione)
    
    Args:
        N (int): Numero di campioni da generare.
        xMin (float): Estremo inferiore del dominio della PDF.
        xMax (float): Estremo superiore del dominio della PDF.
        yMax (float): Valore massimo stimato della PDF.
        pdf (function): Funzione densità di probabilità (PDF) arbitraria.

    Returns:
        list: Lista di N valori generati secondo la PDF.
        float: Stima dell'area sotto la PDF.
    '''
    campione = []
    num = 0

    while len(campione) < N:
        x, count = mygenera_pdf(xMin, xMax, yMax, pdf)  # Genera un valore secondo la PDF
        campione.append(x)                              # Aggiungi il valore al campione
        num += count                                    # Aggiorna il numero totale di tentativi

    # Stima dell'area sotto la PDF
    area = (xMax - xMin) * yMax * len(campione) / num
    return campione, area


def generate_uniform (N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali distribuiti fra 0 ed 1
    a partire da un determinato seed.
    
    Attenzione: restituisce una lista, non necessario passare come argomento un seed
    
    Args:
        N(int): numero campioni da generare
        seed(float): seed da cui far partire la generazione di numeri pseudocasuali, non necessario specificarlo
    
    Returns:
        list: lista di N numeri generati 
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (random.random ())    #aggiungo a randlist un float tra 0 e 1
    return randlist


def rand_range (xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    '''
    return xMin + random.random () * (xMax - xMin)


def generate_range (xMin, xMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali distribuiti fra xMin ed xMax
    a partire da un determinato seed
    
    Attenzione: sfrutta la funzione rand_range definita sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere rand_range nella stessa libreria/script
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_range (xMin, xMax))   #aggiungo un float tra 0 e 1 alla lista
    return randlist


def rand_TAC (f, xMin, xMax, yMax) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo try and catch
    
    Attenzione: sfrutta la funzione rand_range definita sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere rand_range nella stessa libreria/script
    '''
    x = rand_range (xMin, xMax)
    y = rand_range (0, yMax)
    while (y > f (x)) :
        x = rand_range (xMin, xMax)
        y = rand_range (0, yMax)
    return x


def generate_TAC (f, xMin, xMax, yMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo try and catch, in un certo intervallo,
    a partire da un determinato seed
    
    Attenzione: sfrutta le funzioni rand_range e rand_TAC definite sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere anche le altre nella stessa libreria/script
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_TAC (f, xMin, xMax, yMax))
    return randlist


#generano N numeri con metodo try and catch da pdf exp o gaussiana 
def try_and_catch_exp (lamb, N):
    '''
    Genera N numeri pseudo-casuali con il metodo try-and-catch
    a partire da pdf esponenziale.
    
    Attenzione: usare solo per pdf esponenziale e nessun'altra.
                Se si vogliono mettere restrizioni a fino a quante tau (o 1/lamb) di distanza voglio generare numeri seguire i commenti.
    '''
    events = []
    x_max = 1/lamb         #se voglio ad esempio tra 0 e 3 tau al posto dell'1 ci metto 3
    for i in range (N):
        x = rand_range (0., x_max)
        y = rand_range (0., lamb)
        while (y > lamb * exp (-lamb * x)):
            x = rand_range (0., x_max)
            y = rand_range (0., lamb)
        events.append (x)
    return events


def try_and_catch_gau (mean, sigma, N):
    '''
    Genera N numeri pseudo-casuali con il metodo try-and-catch
    a partire da pdf gaussiana.
    
    Attenzione: usare solo per pdf gaussiana e nessun'altra.
                Se si vogliono mettere restrizioni a fino a quante sigma di distanza voglio generare numeri seguire i commenti.
                
    Può essere utile ricordare la forma della funzione gaussiana:
        gauss(x, mu, sigma) = 1/(sqrt(2*np.pi*sigma)) * e**(-1/2*((x*mu)/sigma)**2)
    '''
    events = []
    for i in range (N):
        x = rand_range (mean - 1 * sigma, mean + 1 * sigma)  #come sopra se voglio fino 3 sigma cambio l'1 con 3
        y = rand_range (0., 1.)
        while (y > exp (-0.5 * ( (x - mean)/sigma)**2)):
            x = rand_range (mean - 1 * sigma, mean + 1 * sigma)
            y = rand_range (0, 1.)
        events.append (x)
    return events


def rand_TCL (xMin, xMax, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    su un intervallo fissato
      
    N_sum è un parametro "di precisione", è fissata 10 per avere una precisione buona senza un grande costo computazionale
   
    Attenzione: sfrutta la funzione rand_range definita sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere rand_range nella stessa libreria/script
    '''
    y = 0.
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum ;
    return y ;


def generate_TCL (xMin, xMax, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, in un certo intervallo,
    a partire da un determinato seed
 
    Attenzione: sfrutta le funzioni rand_range e rand_TCL definite sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere anche le altre nella stessa libreria/script
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist


def rand_TCL_ms (mean, sigma, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    note media e sigma della gaussiana
    
    Attenzione: sfrutta la funzione rand_range definita sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere rand_range nella stessa libreria/script
    '''
    y = 0.
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum ;
    return y ;


def generate_TCL_ms (mean, sigma, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, note media e sigma della gaussiana,
    a partire da un determinato seed
    
    Attenzione: sfrutta le funzioni rand_range e rand_TCL_ms definite sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere anche le altre nella stessa libreria/script
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N):
        randlist.append (rand_TCL_ms (xMin, xMax, N_sum))
    return randlist


def inv_exp (y, lamb = 1) :
    '''    
    Calcola l'inversa della funzione cumulativa (CDF) di una distribuzione esponenziale.
    
    Metodo: 
    Dato un numero casuale `y` tra 0 e 1, questa funzione restituisce un valore `x` tale che la probabilità cumulativa 
    F(x) = P(X <= x) sia pari a `y`. Questa operazione è alla base del metodo della trasformata inversa, utilizzato per 
    generare numeri casuali distribuiti esponenzialmente.

    Formula:
    - La distribuzione esponenziale è descritta dalla PDF: f(x) = λ * exp(-λx), x >= 0.
    - La CDF associata è: F(x) = 1 - exp(-λx), x >= 0.
    - L'inversa della CDF è: F^{-1}(y) = -ln(1-y) / λ.

    Args:
        y (float): un valore casuale nell'intervallo [0, 1).
        lamb (float): il parametro della distribuzione esponenziale, detto "tasso" (default: λ = 1).
    
    Returns:
        float: un valore `x` distribuito esponenzialmente con parametro `λ`.
    '''
    return -1 * np.log (1-y) / lamb


def rand_exp (tau) :
    '''
    generazione di un numero pseudo-casuale esponenziale
    con il metodo della funzione inversa
    a partire dal tau dell'esponenziale
    
    Attenzione: sfrutta la funzione inv_exp definita sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere inv_exp nella stessa libreria/script 
    '''
    lamb = 1. / tau
    return inv_exp (random.random (), lamb)


def generate_exp (tau, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali esponenziali
    con il metodo della funzione inversa, noto tau dell'esponenziale,
    a partire da un determinato seed
    
    Attenzione: sfrutta la funzione rand_exp definita sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere rand_exp nella stessa libreria/script
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_exp (tau))
    return randlist


def rand_poisson (mean) :
    '''
    generazione di un numero pseudo-casuale Poissoniano
    a partire da una pdf esponenziale
    
    Attenzione: sfrutta la funzione rand_exp definita sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere rand_exp nella stessa libreria/script
    '''
    total_time = rand_exp (1.)
    events = 0
    while (total_time < mean) :
        events = events + 1
        total_time = total_time + rand_exp (1.)
    return events


def generate_poisson (mean, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali Poissoniani
    a partire da una pdf esponenziale
    
    Attenzione: sfrutta la funzione rand_poisson (che sfrutta rand_exp) definita sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere rand_poisson (e rand_exp) nella stessa libreria/script
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_poisson (mean))
    return randlist



#-----------------------INTEGRALI------------------------------



def integral_HOM (func, xMin, xMax, yMax, N_evt) :
    '''
    Calcola l'integrale di una funzione nell'intervallo [xMin, xMax] 
    utilizzando il metodo "Hit or Miss".
    
   Args:
        func : funzione da integrare.
        xMin (float): limite inferiore dell'integrazione sull'asse x.
        xMax (float): limite superiore dell'integrazione sull'asse x.
        yMax (float): massimo valore atteso di `func(x)` nell'intervallo [xMin, xMax].
        N_evt (int): numero di punti casuali generati per l'approssimazione.

    Returns:
        tuple: 
            - integral (float): stima dell'integrale.
            - integral_unc (float): incertezza associata all'integrale.
    '''
    x_coord = generate_range (xMin, xMax, N_evt)
    y_coord = generate_range (0., yMax, N_evt)

    points_under = 0
    for x, y in zip (x_coord, y_coord):
        if (func (x) > y) : points_under = points_under + 1 

    A_rett = (xMax - xMin) * yMax
    frac = float (points_under) / float (N_evt)
    integral = A_rett * frac
    integral_unc = A_rett**2 * frac * (1 - frac) / N_evt
    return integral, integral_unc


def integral_CrudeMC (g, xMin, xMax, N_rand) :
    '''
    Calcola l'integrale di una funzione g nell'intervallo [xMin, xMax] 
    usando il metodo Monte Carlo "Crude" (diretto).

    Args:
        g (callable): funzione da integrare.
        xMin (float): limite inferiore dell'integrazione sull'asse x.
        xMax (float): limite superiore dell'integrazione sull'asse x.
        N_rand (int): numero di punti casuali generati per l'approssimazione.

    Returns:
        tuple: 
            - integral (float): stima dell'integrale.
            - integral_unc (float): incertezza associata all'integrale.
    '''
    somma     = 0.
    sommaQ    = 0.    
    for i in range (N_rand) :
        x = rand_range (xMin, xMax)
        somma += g(x)
        sommaQ += g(x) * g(x)     
     
    media = somma / float (N_rand)
    varianza = sommaQ /float (N_rand) - media * media 
    varianza = varianza * (N_rand - 1) / N_rand
    lunghezza = (xMax - xMin)
    return media * lunghezza, sqrt (varianza / float (N_rand)) * lunghezza
   
    
    
#-------------------------LIKELIHOOD--------------------------



def likelihood (theta, pdf, sample) :
    '''
    Calcola la funzione di likelihood per un campione dato e una pdf parametrica.

    Args:
        theta (float o array): parametro(i) del modello (esempio: media o lambda della pdf).
        pdf : funzione di densità di probabilità.
        sample (array): campione di dati per cui calcolare la likelihood.

    Returns:
        float: valore della likelihood.
    '''
    risultato = 1.
    for x in sample:
        risultato = risultato * pdf (x, theta)
    return risultato


def loglikelihood (theta, pdf, sample) :
    '''
    Calcola la funzione di log-likelihood per un campione dato e una pdf parametrica.

    Args:
        theta (float o array): parametro(i) del modello (esempio: media o lambda della pdf).
        pdf : funzione di densità di probabilità.
        sample (array): campione di dati per cui calcolare la log-likelihood.

    Returns:
        float: valore della log-likelihood.
        
    Attenzione: usare questa in caso di pdf esponenziali
    '''
    
    risultato = 0.
    for x in sample:
        if (pdf (x, theta) > 0.) : risultato = risultato + log (pdf (x, theta))
    return risultato


def loglikelihood_prod (theta, pdf, sample) :
    '''
    Variante della funzione di log-likelihood che calcola il logaritmo del prodotto delle probabilità.

    Args:
        theta (float o array): parametro(i) del modello (esempio: media o lambda della pdf).
        pdf : funzione di densità di probabilità.
        sample (array): campione di dati per cui calcolare la log-likelihood.

    Returns:
        float: valore della log-likelihood.
    '''
    risultato = 0.
    produttoria = np.prod(pdf(sample, theta))
    risultato= np.log(produttoria)
    return risultato


#per migliorare la visualizzazione
def loglikelihood_ratio(tau, pdf, data, tau_max):
    '''
    Calcola il rapporto di log-likelihood tra due valori di un parametro `tau` e `tau_max`. Può essere utile per confronti.

    Args:
        tau (float): valore del parametro per cui calcolare la log-likelihood.
        pdf : funzione di densità di probabilità.
        data (array): campione di dati.
        tau_max (float): valore massimo del parametro.

    Returns:
        float: valore del log-likelihood ratio.
    '''
    # Calcola la log-likelihood per il valore di tau
    log_likelihood_tau = loglikelihood(tau, pdf, data)

    # Calcola la log-likelihood per il valore massimo tau_max
    log_likelihood_tau_max = loglikelihood(tau_max, pdf, data)

    # Calcola il log-likelihood ratio
    return log_likelihood_tau - log_likelihood_tau_max


#funzione che trova le intersezioni con qualcosa usando il metodo di bisezione
def intersect_LLR (
    g,              # funzione di cui trovare lo zero
    pdf,            # probability density function of the events
    sample,         # sample of the events
    xMin,           # minimo dell'intervallo
    xMax,           # massimo dell'intervallo
    ylevel,         # value of the horizontal intersection
    theta_hat,      # maximum of the likelihood
    prec = 0.0001): # precisione della funzione
    '''
    Trova le intersezioni di una funzione `g` con un valore orizzontale `ylevel` 
    usando il metodo di bisezione.

    Args:
        g (callable): funzione di cui calcolare lo zero.
        pdf (callable): funzione di densità di probabilità.
        sample (array-like): campione di dati.
        xMin (float): limite inferiore dell'intervallo di ricerca.
        xMax (float): limite superiore dell'intervallo di ricerca.
        ylevel (float): valore della retta orizzontale da intersecare.
        theta_hat (float): valore massimo della likelihood.
        prec (float): precisione desiderata (default: 0.0001).

    Returns:
        float: valore di x corrispondente all'intersezione.
    '''
    def gprime (x) :
        return g (x, pdf, sample, theta_hat) - ylevel

    xAve = xMin
    while ((xMax - xMin) > prec) :
        xAve = 0.5 * (xMax + xMin)
        if (gprime (xAve) * gprime (xMin) > 0.) : xMin = xAve
        else                                    : xMax = xAve
    return xAve


#funzione per il calcolo del massimo della likelihood
def sezioneAureaMax_LL(
    g,              # funzione di likelihood trovare il massimo
    pdf,            # probability density function of the events
    sample,         # sample of the events
    x0,             # estremo dell'intervallo
    x1,             # altro estremo dell'intervallo
    prec = 0.0001   # precisione della funzione
):    
    '''
    Massimizza una funzione di log-likelihood `g` utilizzando il metodo della sezione aurea.

    Args:
        g (callable): funzione di log-likelihood da massimizzare.
        pdf : funzione di densità di probabilità.
        sample (array): campione di dati.
        x0 (float): limite inferiore dell'intervallo.
        x1 (float): limite superiore dell'intervallo.
        prec (float): precisione desiderata (default: 0.0001).

    Returns:
        float: valore di x che massimizza la funzione.
    '''
    # Rapporto aureo
    phi = (1 + sqrt(5)) / 2

    # Calcolo dei due punti iniziali interni all'intervallo
    x2 = x1 - (x1 - x0) / phi
    x3 = x0 + (x1 - x0) / phi

    # Funzione per calcolare la log-likelihood data la pdf e il campione
    def likelihood(tau):
        pdf_vals = pdf(sample, tau)
        # Aggiungere un piccolo valore per evitare log(0)
        epsilon = 1e-10
        pdf_vals = np.clip(pdf_vals, epsilon, None)  # Imposta un valore minimo
        return np.sum(np.log(pdf_vals))  # Somma dei logaritmi dei valori di PDF

    # Loop fino a quando la larghezza dell'intervallo non è minore di prec
    while abs(x1 - x0) > prec:
        if likelihood(x3) < likelihood(x2):
            x1 = x3
        else:
            x0 = x2

        # Aggiorno i punti interni
        x2 = x1 - (x1 - x0) / phi
        x3 = x0 + (x1 - x0) / phi

    # Restituisco il punto che massimizza la log-likelihood (punto medio dell'intervallo)
    return (x0 + x1) / 2


#calcola la loglikelihood quando ho più di un parametro da stimare
def loglikelihood_more_params(theta, pdf, sample):
    '''
    Calcola la log-likelihood quando il modello ha più parametri.
    
    Args:
        theta (tuple o list): parametri del modello (esempio: [mu, sigma] per una normale).
        pdf : funzione di densità di probabilità.
        sample (array): campione di dati per cui calcolare la log-likelihood.

    Returns:
        float: valore della log-likelihood.
    '''
    risultato = 0.
    for x in sample:
        prob = pdf(x, *theta)  # Passa tutti gli elementi di theta come parametri separati
        if prob > 0.:
            risultato += log(prob)
    return risultato



#-----------------------------------CLASSI DEFINITE A LEZIONE-------------------
#le classi sono scritte in inglese non ho voglia di tradurle 


def lcm (a, b) :
    '''
    Dati due numeri calcola il minimo comune multiplo (least common multiple).
    '''
    return a * b / gcd (a,b)


class Fraction :
    '''
    a simple class implementing a high-level object
    to handle fractions and their operations
    '''

    def __init__ (self, numerator, denominator) :
        '''the constructor: initialises all the variables needed
        for the high-level object functioning

        Args:
            numerator (int): the numerator of the fraction
            denominator (int): the denominator of the fraction

        Raises:
            ValueError: Denominator cannot be zero
            ValueError: Numerator must be an integer
            ValueError: Denominator must be an integer
        '''
        if denominator == 0 :
            raise ValueError ('Denominator cannot be zero')
        if type(numerator) != int:
            raise TypeError ('Numerator must be an integer')
        if not isinstance(denominator, int ): # alternative way to check the type
            raise TypeError ('Denominator must be an integer')
        
        # this allows to avoid calculating the LCM in the sum and subtraction
        common_divisor = gcd (self.numerator, self.denominator) # greatest common divisor 
        self.numerator = numerator // common_divisor
        self.denominator = denominator // common_divisor
        
    def print (self) :        
        '''
        prints the value of the fraction on screen
        '''
        print (str (self.numerator) + '/' + str (self.denominator))

    def ratio (self) :
        '''
        calculates the actual ratio between numerator and denominator,
        practically acting as a casting to float
        '''
        return self.numerator / self.denominator


    def __add__ (self, other) :
        '''implements the addition of two fractions.
        Note that this function will be callable with the + symbol
        in the program

        Args:
            other (Fraction): the fraction to be added to the current one

        Returns:
            Fraction: the addition of the two fractions
        '''
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __sub__ (self, other) :
        '''implements the subtraction of two fractions.
        Note that this function will be callable with the - symbol
        in the program

        Args:
            other (Fraction): the fraction to be subtracted from the current one

        Returns:
            Fraction: the subtraction of the two fractions
        '''
        new_numerator = self.numerator * other.denominator - other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __mul__ (self, other) :
        '''
        implements the multiplications of two fractions.
        Note that this function will be callable with the * symbol
        in the program

        Args:
            other (Fraction): the fraction to be multiplied from the current one

        Returns:
            Fraction: the multiplication of the two fractions
        '''
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __truediv__ (self, other) :
        '''
        implements the ratio of two fractions.
        Note that this function will be callable with the / symbol
        in the program

        Args:
            other (Fraction): the fraction to be divided from the current one

        Returns:
            Fraction: the ratio of the two fractions
        '''
        if other.numerator == 0 :
            print ('Cannot divide by zero')
            sys.exit (1)
        
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        return Fraction (new_numerator, new_denominator)


def testing ()  :
    '''
    Function to test the class behaviour, called in the main program
    '''
    print ('Initial fractions:')
    frac1 = Fraction (3, 4)
    frac1.print ()
    print ('ratio: ', frac1.ratio ())

    frac2 = Fraction (1, 2)
    frac2.print ()
    print ('ratio: ', frac2.ratio ())
    
    sum_frac = frac1 + frac2
    print ('\nSum :')
    sum_frac.print ()
    
    diff_frac = frac1 - frac2
    print ('\nDifference:')
    diff_frac.print ()
    
    prod_frac = frac1 * frac2
    print ('\nProduct:')
    prod_frac.print ()
    
    div_frac = frac1 / frac2
    print ('\nDivision:')
    div_frac.print ()


#---------------------------


from math import sqrt, pow

class stats :
    '''calculator for statistics of a list of numbers'''

    summ = 0.
    sumSq = 0.
    N = 0
    sample = []

    def __init__ (self, sample):
        '''
        reads as input the collection of events,
        which needs to be a list of numbers
        '''
        self.sample = sample
        self.summ = sum (self.sample)
        self.sumSq = sum ([x*x for x in self.sample])
        self.N = len (self.sample)

    def mean (self) :
        '''
        calculates the mean of the sample present in the object
        '''
        return self.summ / self.N

    def variance (self, bessel = True) :
        '''
        calculates the variance of the sample present in the object
        '''
        var = self.sumSq / self.N - self.mean () * self.mean ()
        if bessel : var = self.N * var / (self.N - 1)
        return var

    def sigma (self, bessel = True) :
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel))


    def sigma_mean (self, bessel = True) :  #errore sulla media
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel) / self.N)

    def skewness (self) :
        '''
        calculate the skewness of the sample present in the object
        '''
        mean = self.mean ()
        asymm = 0.
        for x in self.sample:
            asymm = asymm + pow (x - mean,  3)
        asymm = asymm / (self.N * pow (self.sigma (), 3))
        return asymm    

    def kurtosis (self) :
        '''
        calculate the kurtosis of the sample present in the object
        '''
        mean = self.mean ()
        kurt = 0.
        for x in self.sample:
            kurt = kurt + pow (x - mean,  4)
        kurt = kurt / (self.N * pow (self.variance (), 2)) - 3
        return kurt

    def append (self, x):
        '''
        add an element to the sample
        '''
        self.sample.append (x)
        self.summ = self.summ + x
        self.sumSq = self.sumSq + x * x
        self.N = self.N + 1        
        
    
    '''
    esempio di come si usa:
        stats_calculator = stats (distanze)
        print ('mean    :', stats_calculator.mean ())
        print ('sigma   :', stats_calculator.sigma ())
        print ('skewness:', stats_calculator.skewness ())
        print ('kurtosis:', stats_calculator.kurtosis ())
    '''
 

 #----------------------------------

#in altrelib.py ci sono le funzioni definite per gli istogrammi senza passare da una classe
class my_histo :
    '''calculator for statistics of a list of numbers'''

    summ = 0.
    sumSq = 0.
    N = 0
    sample = []

    def __init__ (self, sample_file_name) :
        '''
        reads as input the file containing the collection of events
        and reads it
        '''
        with open (sample_file_name) as f:
            self.sample = [float (x) for x in f.readlines ()]

        self.summ = sum (self.sample)
        self.sumSq = sum ([x*x for x in self.sample])
        self.N = len (self.sample)

    def mean (self) :
        '''
        calculates the mean of the sample present in the object
        '''
        return self.summ / self.N

    def variance (self, bessel = True) :
        '''
        calculates the variance of the sample present in the object
        '''
        var = self.sumSq / self.N - self.mean () * self.mean ()
        if bessel : var = self.N * var / (self.N - 1)
        return var

    def sigma (self, bessel = True) :
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel))

    def sigma_mean (self, bessel = True) :
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel) / self.N)

    def draw_histo (self, output_file_name) :
        '''
        draw the sample content into an histogram
        '''
        xMin = floor (min (self.sample))
        xMax = ceil (max (self.sample))
        N_bins = sturges (self.N)

        bin_edges = np.linspace (xMin, xMax, N_bins)
        fig, ax = plt.subplots (nrows = 1, ncols = 1)
        ax.hist (self.sample,
                 bins = bin_edges,
                 color = 'orange',
                )
        ax.set_title ('Histogram example', size=14)
        ax.set_xlabel ('variable')
        ax.set_ylabel ('event counts per bin')

        plt.savefig (output_file_name)
     
    
    '''
# -------------------------------------------------- 
    
if __name__ == "__main__" :
    testing ()
    '''

    

# ----------------- ALTRE FUNZIONI UTILI -----------------


# Funzione retta
def retta (x, m, q) :
    return m * x + q

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def polinomio_grad3 (x, a, b, c, d) :
    return a * (x**3) + b * (x**2) + c * x + d

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# funzione parabola
def parabola (x, a, b, c) :
    return a * (x**2) + b * x + c

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#Funzione fattoriale
def fattoriale (N) :
    if N == 0 :
        return 1
    return fattoriale (N-1) * N

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Gaussiana con sigma = 1 e mu = 0
def gaussiana (x) :
    return (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x)**2)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#Funzione coeff. binomiale
def coeff_binom (N, k) :
    if N == 0 & N < k :
        return 0
    return fattoriale (N) / fattoriale (k) * fattoriale (N-1)
    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#Funzione esponenziale
def esponenziale (x, tau) :
    if tau == 0 :
        return 1
    return (np.exp (-1 * x / tau)) / tau

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Funzione che genera n valori con la sequenza di Fibonacci
def Fibonacci (n) :

    # Se il numero fino a cui voglio contare è minore o uguale a zero faccio in modo che il programma mi restituisca una lista vuota.
    if n <= 0 :
        return []
    
    # Se il numero fino a cui voglio contare è 1 allora la lista avrà solo il numero 1
    elif n == 1 :
        return [1]
    
    # definisco una lista iniziale con almeno due numeri (i primi due)
    lista_fibo = [0, 1]
    contatore = len(lista_fibo)            # Conta il numero di elementi nella lista
    
    while (contatore < n) :
        prossimo_num = lista_fibo[contatore-1] + lista_fibo[contatore-2]        # Somma il numero precedente all'i-esimo al i-esimo meno due
        lista_fibo.append(prossimo_num)         # Aggiunge il prossimo numero alla fine della lista
        contatore = contatore + 1       # Incrementa il numero di elementi nella lista alla fine di ofni iterazione
    return lista_fibo

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Funzione per risolvere equazioni di secondo grado
def soluz_eq_secondo_grado (a, b, c) :

    if a == 0 :                                                     # Non sarebbe una eq. di secondo grado e dividerei per 0 (no buono)
        return print("Non è una equazione di econdo grado")
    
    else :
        delta = (b**2) - (4*a*c)                                        # Calcolo il delta per dividere i casi
        if delta > 0 :
            x1 = (-b + np.sqrt(delta))/(2*a)
            x2 = (-b - np.sqrt(delta))/(2*a)
            return print("Le soluzioni sono\nx1 =", x1, "\nx2 =", x2)
    
        elif delta == 0 :                                               # mi basta stampare una sola soluzione (sono uguali)
            x1 = x2 = -b/(2*a)
            return print("La soluzione è x = ", x1)
    
        #elif delta < 0 :                                                # Dovrei introdurre i complessi (che sbatti)
        else :
            return print("Non eiste soluzione per ogni x appartenente ai numeri reali.")



#-----------------------------------------DISTRIBUZIONI----------------------------        
        
        

# Calcolo del coefficiente binomiale
def binomial_coefficient (n, k) :
    '''
    Calcola il coefficiente binomiale (n choose k).
    
    Args:
        n (int): Numero totale di elementi.
        k (int): Numero di elementi scelti.
    
    Returns:
        int: Coefficiente binomiale.
    '''
    if k < 0 or k > n:
        return 0  # Il coefficiente è definito solo per 0 <= k <= n
    return factorial(n) // (factorial(k) * factorial(n - k))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#Funzione per la distribuzione binomiale
def binomial_distribution (n, k, p) :
    '''
    Calcola la probabilità della distribuzione binomiale.
    
    Args:
        n (int): Numero totale di prove.
        k (int): Numero di successi desiderati.
        p (float): Probabilità di successo in una singola prova.
    
    Returns:
        float: Probabilità associata.
    '''
    coeff_binomiale = np.math.comb(n, k)
    return coeff_binomiale * (p ** k) * ((1 - p) ** (n - k))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Funzione Bernulli Trials
def bernoulli_trial (p) :
    '''
    Esegue una singola prova di Bernoulli.
    Args:
        p (float): Probabilità di successo.
    Returns:
        int: 1 per successo, 0 per fallimento.
    '''
    if np.random.random() < p :
        return 1
    else :
        return 0

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#Funzione distribuzione Poisson
def poisson_distribution (lmbda, k) :
    '''
    Calcola la probabilità della distribuzione di Poisson.
    Args:
        lmbda (float): Tasso medio di successo (lambda).
        k (int): Numero di eventi osservati.
    Returns:
        float: Probabilità associata.
    '''
    return (np.exp(-lmbda) * (lmbda ** k)) / np.math.factorial(k)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#Funzione distribuzione di Cauchy
def cauchy_distribution (x, x0, gamma) :
    '''
    Calcola la funzione di densità di probabilità della distribuzione di Cauchy.
    Args:
        x (float): Variabile indipendente.
        x0 (float): Posizione del picco della distribuzione (mediana).
        gamma (float): Larghezza a metà altezza (HWHM).
    Returns:
        float: Valore della densità di probabilità.
    '''
    return (1 / np.pi) * (gamma / ((x - x0)**2 + gamma**2))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#Funzione distribuzione di Maxwell Boltzmann
def maxwell_boltzmann_distribution (v, a) :
    '''
    Calcola la funzione di densità di probabilità della distribuzione di Maxwell-Boltzmann.
    Args:
        v (float): Velocità delle particelle.
        a (float): Parametro della distribuzione legato alla temperatura e alla massa.
    Returns:
        float: Valore della densità di probabilità.
    '''
    return np.sqrt(2 / np.pi) * (v**2) * np.exp(-v**2 / (2 * a**2)) / (a**3)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

#Funzione distribuzione Breit Wigner
def breit_wigner_distribution (x, x0, gamma) :
    '''
    Calcola la funzione di densità di probabilità della distribuzione di Breit-Wigner.
    Args:
        x (float): Variabile indipendente.
        x0 (float): Posizione del picco (massa del risonante, per esempio).
        gamma (float): Larghezza a metà altezza (HWHM).
    Returns:
        float: Valore della densità di probabilità.
    '''
    return (1 / np.pi) * (gamma / 2) / ((x - x0)**2 + (gamma / 2)**2)

        
           
    
#-----------------------------------------FINE LIBRERIE-------------------------------------    
 
    
    
'''
-----------------------------ARGOMENTI LEZIONI-------------------------

1) type(), logical operators, str, list, tuple, dictionaries, python script, funzioni, if else for while, librerie moduli.

2) numpy e array (arange, linspace ecc), matplotlib (come disegnare funzioni su grafico), statistics.py

3) leggere dati da testo, istogrammi, bin, sturges, sample moments, pdf, cdf, integrali

4) numeri pseudo casuali, try-and-catch, funzione inversa, teorema centrale del limite

5) class, overloading, lambda, map, filter

6) trovare zeri delle funzioni e estremi, bisezione (ricorsiva), sezione aurea (ricorsiva)

7) poisson, generare eventi con distribuzione di poisson, con funzione inversa

8) toy experiment, integrazione con pseudo random, hit-or-miss, monte carlo
negli esempi c'è un programma che usa hit or miss senza la funzione della libreria

9) notebook, likelihood, loglikelihood

10) fit iminuit trova parametri con massima verosomiglianza

11) minimi quadrati, fit qualiti Q^2, matrice covarianza
        
12) fit con distribuzioni binnate

esercizi lezione 12:
    1- distribuizone binnata, background esponeneziale e segnale gaussiano
    2- fit distribuzione gaussiana, max likelihood con distr binnana e non binnata
    3- loop dell'esercizio 2 che mostra andamento con numero differente di dati considerati (con toy experiment)
    4- confronto funzioni di costo con likelihood, estesa, minimi quadrati
'''


#--------------------------------PEZZI DI CODICE UTILI----------------------



'''
---------------------DISEGNO DI UN PLOT--------------
    x_axis = np.linspace(x_min, x_max, 100)
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))     # 1 riga, 1 colonna
    ax.plot(x_axis, funzione_da_inserire (x_axis), label="PDF")         
    ax.legend()
    ax.grid()
    ax.set_title("Funzione di densità di probabilità (PDF)")
    #plt.plot(x_del_punto, y_del_punto, marker = "o", color = "red")    # questo serve per mettere un punto con coordinate x ed y
    #plt.savefig("nome dell'immagine.png")
    plt.show()                                                          # da mettere rigorosamente alla fine perchè blocca il porgramma
'''

'''
-------------------DISEGNO FUNZIONE SENO----------------
disegno funzione seno nell'intervallo [0, 2pi]:
    fig, ax = plt.subplots (nrows = 1, ncols = 1)  #crea un'immagine vuota 
    x_coord = np.linspace (0, 2 * np.pi, 10000)  
    y_coord = np.sin (x_coord)
    ax.plot (x_coord, y_coord, label='sin (x)')
    ax.set_title ('title', size=14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.legend ()
    plt.show()


-----------------DATI IN UN FILE .TXT-------------
mettere dati di un file .txt in un array:
    sample = np.loadtxt ('sample.txt')  #mette i dati di sample.txt in un array
se ho dati in più colonne:
    dati, errori = np.loadtxt('sample.txt', unpack = True)

C'è anche metodo più articolato se lo vuoi è in lezione 3


---------------------CREARE ISTOGRAMMA--------------
DUE METODI:

1)  fig, ax = plt.subplots (nrows = 1, ncols = 1)              # crea immagine vuota
    N_bins = sturges (len (sample))
    x_range = (xMin, xMax)      #ricorda di definirli prima :)
    bin_content, bin_edges = np.histogram (sample, bins = N_bins, range = x_range)    
    ax.hist (sample, bins = bin_edges, color = 'orange')
    ax.set_xlabel('x')
    ax.set_ylabel('conteggi')
    ax.grid()
    plt.show()

Qui definisco solo N_bins con sturges e calcolo bin_content e bin_edges con np.histogram(), dove:

N_bins: Il numero totale di bin dell'istogramma. Può essere calcolato direttamente o passato come parametro a funzioni come np.histogram.
bin_edges: array che definisce i bordi (o intervalli) dei bin dell'istogramma.
            Se ad esempio ho 5 bin e i dati vanno da 0 a 10, i bordi potrebbero essere: [0, 2, 4, 6, 8, 10].
bin_content: array che rappresenta il numero di dati in ciascun bin.
             Con i bordi [0, 2, 4, 6, 8, 10], potresti avere un contenuto come [3, 7, 10, 5, 2], 
             dove ogni valore indica quanti dati cadono in quell'intervallo.

        
2)  N_bins = sturges(len(sample))
    bin_edges = np.linspace(x_min, x_max, N_bins)          
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample, bins=bin_edges ,color = 'orange')  # Spesso conviene usare bins = 'auto' evitando di scrivere
                                                        # la linea di codice con bin_edges, per farlo però bisogna importare numpy
    ax.set_title ('Nome istogramma', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()                                          # Se voglio la griglia
    #plt.savefig ('nome_del_grafico.png')
    plt.show ()                                         # Da mettere rigorosamente dopo il savefig

Qui definisco bin_edges e N_bins per conto mio, non utilizzo np.histogram dunque non ho bin_content. 


----------------DISEGNO DI UNA DISTRIBUZIONE UNA ACCANTO ALL'ALTRA-------------
    x_axis = np.linspace(0, 10, 100)

    # Creazione di una figura con due subplot affiancati
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))  # 1 riga, 2 colonne
    
    # Primo grafico: PDF
    ax[0].plot(x_axis, funzione_da_inserire.pdf(x_axis), label="PDF")   # con esponenziale.pdf uso la funzione predefinita nella libreria
    ax[0].legend()
    ax[0].grid()
    ax[0].set_title("Funzione di densità di probabilità (PDF)")

    # Secondo grafico: CDF
    ax[1].plot(x_axis, funzione_da_inserire.cdf(x_axis), label="CDF")     # con esponenziale.cdf uso la funzione predefinita nella libreria
    ax[1].legend()
    ax[1].grid()
    ax[1].set_title("Funzione di distribuzione cumulativa (CDF)")

    plt.tight_layout()  # Questo aggiusta automaticamente spazi tra i grafici
    plt.savefig("grafici_PDF_CDF_affiancati.png")
    plt.show()

----------------TOY EXPERIMENT---------------------
import numpy as np
from scipy.stats import norm

# Parametri del modello
true_mu = 1.0
true_sigma = 0.5
sample_size = 100
N_toys = 1000

# Contenitori per i risultati
mu_estimates = []
sigma_estimates = []

# Esegui N_toys simulazioni
for _ in range(N_toys):
    # 1. Genera un campione simulato
    sample = np.random.normal(loc=true_mu, scale=true_sigma, size=sample_size)
    
    # 2. Stima la media e la deviazione standard
    mu_estimates.append(np.mean(sample))
    sigma_estimates.append(np.std(sample))

# Calcola statistiche sui risultati
print(f"Stima di mu: {np.mean(mu_estimates):.3f} ± {np.std(mu_estimates):.3f}")
print(f"Stima di sigma: {np.mean(sigma_estimates):.3f} ± {np.std(sigma_estimates):.3f}")
'''


#-----------------------FIT------------------
'''
per eseguire i fit devo importare le seguenti librerie:

from iminuit import Minuit
from iminuit.cost import LeastSquares

-------------------MINIMI QUADRATI-----------------------
least_squares = LeastSquares (x, y, sigma, funzione_da_fittare)        # La funzione costo che verrà minimizzata
my_minuit = Minuit (least_squares, par0 = 0, par1 = 0)                 # La classe Minuit va ad effettuare la minimizzazione, 
                                                                       # prende in ingresso la funzione costo ed i parametri iniziali.
è il principale algoritmo di minimizzazione utilizzato da Minuit.

Se ho una distibuzione binnata, per costruire la funzione di costo, uso:
my_cost_func = ExtendedBinnedNLL (bin_content, bin_edges, mod_total)

La funzione di costo è una misura dell'errore o della discrepanza tra i dati osservati e il modello teorico.
Nel caso dei minimi quadrati la funzione di costo è il chi quadro.
Minimizzare la funzione di costo equivale a trovare il miglior adattamento dei dati al modello, massimizzando la probabilità che il modello spieghi i dati.

Questo metodo cerca di trovare i valori ottimali dei parametri (par0 e par1) che minimizzano la funzione di costo (X^2).
Durante l'esecuzione, migrad calcola: 
1- Il valore della funzione di costo (X^2) per i parametri iniziali.
2- La direzione di discesa (gradiente).
3- Itera per trovare i parametri che portano al minimo globale.

Se la minimizzazione ha successo:

my_minuit.values    # conterrà i valori ottimali dei parametri (par0 e par1).
my_minuit.fval      # sarà il valore minimo della funzione di costo (x^2).

my_minuit.hesse ()        # hesse calcola la matrice di Hessian della funzione di costo (X^2) nel punto di minimo.

La matrice Hessiana rappresenta la curvatura della funzione di costo intorno al minimo.
Questo metodo stima le incertezze sui parametri ottimali (delta_par0 e delta_par1), che verranno salvate in my_minuit.errors.

is_valid = my_minuit.valid      # indica se la minimizzazione è stata completata con successo.
Q_squared = my_minuit.fval      # restituisce il valore della funzione di costo (X^2) nel punto di minimo.
N_dof = my_minuit.ndof          # calcola i gradi di libertà (degrees of freedom, N_dof) per l'adattamento.
#La formula per i gradi di libertà è: N_dof = N_dati - N_param

my_minuit.fmin                  # restituisce un oggetto contenente informazioni dettagliate sulla minimizzazione.

#Tra le proprietà più utili di fmin, trovi:
my_minuit.fmin.fval             # Il valore minimo della funzione di costo (X^2).
my_minuit.fmin.is_valid         # Se la minimizzazione è valida.
my_minuit.fmin.edm              # Il valore dell'Expected Distance to Minimum, che misura quanto il risultato è vicino al minimo.
my_minuit.fmin.ngrad            # Numero di valutazioni del gradiente effettuate.

# Posso stampare la matrice di covarianza con:
print (my_minuit.covariance)

# Mentre la matrice di correlazione con:
print (my_minuit.covariance.correlation ())

---------------------VISUALIZZARE RISULTATO FIT CON TABELLOZZE------------

from IPython.display import display

my_minuit.migrad ()
print (my_minuit.valid)
display (my_minuit)
'''

#la lascio commentate perchè mai testate ma promettono bene
'''
# Funzione che esegue il fit con metodo dei minimi quadrati
def esegui_fit (
        x,                  # vettore x (np.array)
        y,                  # vettore y (np.array)
        sigma,              # vettore dei sigma (np.array)
        dizionario_par,     # dizionario con parametri 
        funzione_fit        # funzione del modello da fittare
    ) :

    if not (isinstance(dizionario_par, dict)) :
        print ("Inserisci un dizionario come quarto parametro.\n")
        sys.exit()

    least_squares = LeastSquares (x, y, sigma, funzione_fit)
    my_minuit = Minuit (least_squares, **dizionario_par)
    my_minuit.migrad ()                                 
    my_minuit.hesse ()                                  

    is_valid = my_minuit.valid
    Q_squared = my_minuit.fval
    N_dof = my_minuit.ndof
    matrice_cov = my_minuit.covariance

    diz_risultati = {
        "Validità": is_valid, 
        "Qsquared": Q_squared,
        "Ndof": N_dof,
        "Param": my_minuit.parameters,
        "Value": my_minuit.values,
        "Errori": my_minuit.errors,
        "MatriceCovarianza": matrice_cov
    }

    return diz_risultati

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Funzione per il fit con loglikelihood
def esegui_fit_LL (
        bin_content,        # contenuto dei bin
        bin_edges,          # larghezza dei bin
        dizionario_par,     # dizionario con parametri da determinare
        funzione_fit        # funzione modello da fittare
    ) :

    if not (isinstance (dizionario_par, dict)) :
        print ("Inserisci: bin_content, bin_edges, dizionario parametri e funzione da fittare.\n")
        sys.exit()

    funzione_costo = ExtendedBinnedNLL (bin_content, bin_edges, funzione_fit)
    my_minuit = Minuit (funzione_costo, **dizionario_par)
    my_minuit.migrad ()                                 
    my_minuit.hesse ()                                  

    is_valid = my_minuit.valid
    N_dof = my_minuit.ndof
    matrice_cov = my_minuit.covariance

    diz_risultati = {
        "Validità": is_valid,
        "Ndof": N_dof,
        "Param": my_minuit.parameters,
        "Value": my_minuit.values,
        "Errori": my_minuit.errors,
        "MatriceCovarianza": matrice_cov
    }

    return diz_risultati
'''


#---------------------OPZIONI GRAFICHE---------------------
'''
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))     # numero righe e colonne e dimensione figura

    ax.set_title ("nome grafico", fontsize = 14)
    ax.set_xlabel ("nome asse x", fontsize = 12)
    ax.set_ylabel ("nome asse y", fontsize = 12)
    
    # se voglio la barra degli errori:
    ax.errorbar (sample, valore_y, xerr = 0.0, yerr = errori,                       # nell'ordine: valori x, valori y, errore sulla x, errori sulle y
        markersize = 5,                             # dimensione del punto
        fmt = 'o',                                  # tipo di marker (punto)
        color = 'blue',                             # colore della linea
        linestyle = '--',                           # tipo di linea
        ecolor = 'red',                             # colore della barra di errore
        elinewidth = 1.5,                           # spessore barre errori
        capsize = 5,                                # lunghezza cappello barre errori
        capthick = 1.5,                             # spessore cappello barre errori
        label = "label della linea")                # label

    ax.legend (fontsize = 10, loc = 'best')                         # loc = 'best' mette la legenda dove è c'è spazio
    ax.grid (color = 'gray', linestyle = ':', linewidth = 0.5)      # impostazioni della griglia

    plt.savefig ("nome_file.png")
    plt.show ()                                     # da mettere alla fine se no blocca tutto
'''



#---------------------------------RIASSUNTI UTILI------------------------------



#----------------- OPERAZIONI CON LISTE -----------------

# ------ Funzioni Built-in ------
'''
len(lista): Restituisce il numero di elementi nella lista.
sum(lista): Calcola la somma di tutti gli elementi numerici nella lista.
max(lista): Restituisce il valore massimo nella lista.
min(lista): Restituisce il valore minimo nella lista.
sorted(lista): Restituisce una nuova lista ordinata senza modificare l'originale.
reversed(lista): Restituisce un iteratore per scorrere la lista in ordine inverso.
'''

# ------ Metodi delle Liste ------
'''
lista.append(elemento): Aggiunge un elemento alla fine della lista.
lista.extend(altra_lista): Estende la lista aggiungendo tutti gli elementi di un’altra lista.
lista.insert(indice, elemento): Inserisce un elemento in una posizione specifica.
lista.remove(elemento): Rimuove la prima occorrenza di un elemento.
lista.pop([indice]): Rimuove e restituisce un elemento dalla lista (di default l’ultimo).
lista.clear(): Rimuove tutti gli elementi dalla lista.
lista.index(elemento): Restituisce l’indice della prima occorrenza di un elemento.
lista.count(elemento): Restituisce il numero di occorrenze di un elemento nella lista.
lista.sort(): Ordina la lista in loco (modifica la lista originale).
lista.reverse(): Inverte l’ordine degli elementi nella lista in loco.
'''

# ------ Operatori Utili ------
'''
Concatenazione: lista1 + lista2             # restituisce una nuova lista combinata.
Ripetizione: lista * n restituisce          # una lista ripetuta n volte.
Verifica presenza: elemento in lista        # restituisce True se l’elemento è presente.
Slicing: lista[start:stop:step]             # estrae una porzione della lista. 
                                            # Ad esempio lista[:50] prende solo i primi 50 elementi.    
                                            # lista[50:] gli ultimi 50           
                                            # lista[0:25:1] prende 1, 2, 3,..., 25
'''


# ----------------- OPERAZIONI CON ARRAY -----------------

# ------ Creazione e Inizializzazione ------
'''
np.array([1, 2, 3]): Crea un array da una lista o tupla.
np.zeros((2, 3)): Crea un array di zeri con forma specificata.
np.ones((3, 4)): Crea un array di uni con forma specificata.
np.full((2, 2), 7): Crea un array pieno di un valore specifico.
np.eye(3): Crea una matrice identità.
np.arange(start, stop, step): Crea un array con valori equidistanti.
np.linspace(start, stop, num): Crea un array di valori equidistanti tra start e stop.
'''

# ------ Proprietà degli Array ------
'''
array.shape: Restituisce la forma (dimensioni) dell'array.
array.size: Numero totale di elementi nell'array.
array.ndim: Restituisce il numero di dimensioni dell'array.
array.dtype: Tipo di dato degli elementi dell'array.
'''

# ------ Operazioni Matematiche ------
'''
np.sum(array, axis=None): Calcola la somma degli elementi lungo un asse.
np.mean(array, axis=None): Calcola la media.
np.max(array, axis=None) / np.min(array, axis=None): Restituisce il valore massimo o minimo.
np.std(array) / np.var(array): Calcola la deviazione standard o la varianza.
np.prod(array): Prodotto degli elementi dell'array.
np.cumsum(array): Somma cumulativia.
np.cumprod(array): Prodotto cumulativo
'''

# ------ Operazioni di Modifica ------
'''
array.reshape(new_shape): Cambia la forma dell'array.
array.flatten(): Appiattisce l'array in un array monodimensionale.
np.transpose(array): Calcola la trasposta.
array.T: Alias per trasposta.
np.concatenate([array1, array2], axis=0): Unisce array lungo un asse.
np.split(array, indices): Divide l'array in sotto-array.
'''

# ------ Selezione e Mascheramento ------
'''
array[index]: Accede a un elemento o sotto-array.
array[:, 1]: Slicing; estrae tutti gli elementi della colonna 1.
array[array > 5]: Restituisce gli elementi che soddisfano una condizione.
'''

# ------ Operazioni Logiche ------
'''
np.all(array > 0) / np.any(array > 0): Verifica se tutti o almeno uno degli elementi soddisfano la condizione.
np.where(array > 5): Restituisce gli indici degli elementi che soddisfano una condizione.
np.isin(array, [2, 3]): Verifica se gli elementi appartengono a un insieme.
'''

# ------ Operazioni Avanzate ------
'''
np.dot(array1, array2): Prodotto scalare o matriciale.
np.linalg.inv(array): Calcola l'inversa di una matrice.
np.linalg.eig(array): Restituisce autovalori e autovettori.
np.sort(array, axis=-1): Ordina gli elementi lungo un asse.
'''


# ----------------- OPERAZIONI CON DICTIONARY ----------------- (esempi)
'''
Creazione di un dizionario: my_dict = {"Name": "Alice", "age": 25}

Accedere ai valori: name = my_dict["Name"]                          # mi restituisce "Alice"
Accedere ai valori: my_dict.get("age")                              # mi restituisce 25

Aggiungere oggetti: my_dict["city"] = "New York"
Aggiornare oggetto: my_dict["age"] = 26

Rimuovere oggetti: del my_dict["age"]
Rimuovere e ritornare l'oggetto tolto: value = my_dict.pop ("city") # rimuove city e ritorna New York

Iterare su dizionario: for key, value in my_dict.items() :
                            print (f"{key}: {value}")

Dictionary comprehension: squared_dict = {x: x**2 for x in range (5)}       # restituisce {0: 0, 1: 1, 2:, 4, 3: 9, 4: 16}

Lunghezza dizionario: lunghezza = len (my_dict)

Checking for existence: exists = "name" in my_dict      # True se "name" is a key in my_dict

Copiare un dizionario: copy_dict = my_dict.copy ()

Merging dizionari: merged_dict = {**my_dict, **another_dict}
'''


#-------- Utilizzo di sys.argv--------
'''
import sys 
sys.argv: lista che contiene tutte le parole scritte dopo python3
    esempio: python3 myscript1.py 13 
            sys.argv[0]      #restituisce myscript1.py
            sys.argv[1]      #restituisce 13
'''

#----------------LOGARITMI NELLE VARIE LIBRERIE-------------
'''
math:
    math.log(x, base):
        Permette di specificare una base arbitraria:
        Se fornisci la base, calcola il logaritmo in quella base.
        Se non specifichi la base, il default è il logaritmo naturale in base e
    math.log10(x):
        Calcola il logaritmo in base 10.
    math.log2(x):
        Calcola il logaritmo in base 2.
        
numpy: 
    np.log(x):
        di default calcola il logartimo naturale base e 
        Non permette di specificare direttamente una base diversa.
    np.log10(x):
        Calcola il logaritmo in base 10.
    np.log2(x):
        Calcola il logaritmo in base 2.
    
    per usare base arbitraria in numpysi deve sfruttare la formula del cambio di base, si può fare così:
        def log_base(x, base):
            return np.log(x) / np.log(base)

    se devi farlo a sto punto usa math che è già fatta
'''


# ----------------- TIME -----------------


'''
Posso far partire la libreria time per vedere il tempo di esezuione dell'intero programma o di un pezzo di questo. 
Per farlo importo la libreria time (import time) ed in un punto del programma assegnerò la variabile (che posso definire come voglio)

t_start = time.time()
... funzione di cui si vuole sapere il tempo di esecuzione
t_end = time.time()

# a schermo stampo
print(f"Tempo impiegato per eseguire: {(t_end - t_start):.2f} secondi.")    # il .2 serve a stabilire il numero di cifre significative, 
                                                                            # se ne voglio 3 userò .3
                                                                            
# se il tempo necessario a completare l'operazione è molto breve posso usare
print(f"Tempo impiegato per eseguire: {1000*(t_end - t_start):.6f} millisecondi.")
'''


#-------------DEFINIZIONE MAIN E LIBRERIE DA IMPORTARE---------


'''
import sys
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor, log, gcd
import random
import mylib

from iminuit import Minuit
from iminuit.cost import LeastSquares

def main ():
    # Funzione che implementa il programma principale 
    
    #istruzioni
    
    return

#----------------------------------------------------------

if __name__ == "__main__":
    main () 
'''