'''
libreria myrand contiene: 
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
    
    integral_HOM
    integral_CrudeMC
    
    likelihood
    loglikelihood
    loglikelihood_prod
    loglikelihood_ratio
    intersect_LLR
    sezioneAureaMAx_LL
    loglikelihood_more_params
'''

import random
from math import sqrt, log
import numpy as np



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
        campione.append(x)                            # Aggiungi il valore al campione
        num += count                                  # Aggiorna il numero totale di tentativi

    # Stima dell'area sotto la PDF
    area = (xMax - xMin) * yMax * len(campione) / num
    return campione, area



#-------------



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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_range (xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    '''
    return xMin + random.random () * (xMax - xMin)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TAC (f, xMin, xMax, yMax))
    return randlist

#----------------------------------------------------------

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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TCL (xMin, xMax, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    su un intervallo fissato
    
    Attenzione: sfrutta la funzione rand_range definita sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere rand_range nella stessa libreria/script
    '''
    y = 0.
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum ;
    return y ;


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TCL (xMin, xMax, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, in un certo intervallo,
    a partire da un determinato seed
    
    N_sum è un parametro "di precisione", è fissata 10 per avere una precisione buona senza un grande costo computazionale
    
    Attenzione: sfrutta le funzioni rand_range e rand_TCL definite sopra. 
                Se si vuole copiare questa funzione assicurarsi di avere anche le altre nella stessa libreria/script
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

#jhvbcccuebhveihdjxsgv fechdxjwdkecfjhdjkwocfvhr fckdfjghfjdkskdjhfijbvguikmnbvfui, bm bjmnjjij
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
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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


#integral.py

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


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


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
                 
#---------------------



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

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

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

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

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

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
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

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
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

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

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

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
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
