'''
File unico con altre librerie create per questo esame
dovrebbero essere pressocchè le stesse cose viste a lezione
nel dubbio metto anche queste

non assicuro che questo file funzioni
prendere le funzione che si vuole utilizzare e assicurarsi di importare tutto ciò che serve
e occhio indentazione
'''

#basicstats.py

import numpy as np
import statistics as stats

#calcolare la media aritmetica
def media (sample) :
    somma = sum (sample)
    N = len (sample)
    return somma / N

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# calcola la varianza del campione
def varianza (sample, bessel = True) :
    somma = 0.
    N = len (sample)
    media=stats.mean(sample)
    for elem in sample :
       somma=somma+(elem-media)**2
    var = somma/N
    if bessel : var = N * var / (N - 1)
    return var

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#calcola la deviazione standard
def devstd (sample, bessel = True) :
    return np.sqrt (varianza (sample, bessel))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#calcola la deviazione standard della media
def devstd_media (sample, bessel = True) :
    N = len (sample)
    return np.sqrt (varianza (sample, bessel) / N)



#----------------------------

#estremanti.py


# libreria per trovare gli zeri ed estremanti di una funzione
import numpy as np

'''
Quando utilizzi il metodo della bisezione, devi assicurarti che:

 1) La funzione sia continua nell'intervallo [xMin, xMax]
 2) Gli estremi dell'intervallo abbiano segni opposti
 3) L'intervallo sia scelto in modo appropriato per contenere uno zero della funzione
 4) La precisione sia adeguata rispetto alla dimensione dell'intervallo
'''
#---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# metodo della bisezione (meno veloce)

def bisezione (g, xMin, xMax, prec = 0.0001):

    if g(xMin)*g(xMax)>0 :
      return 'La funzione non ha zeri nell intervallo desiderato'

    xAve = xMin
    while ((xMax - xMin) > prec) :
        xAve = 0.5 * (xMax + xMin)
        if (g (xAve) * g (xMin) > 0.): xMin = xAve
        else                         : xMax = xAve
    return xAve

def coseno (x):
  return np.cos(x)

#---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# metodo della bisezione ricorsiva (più veloce)

def bisezione_ricorsiva (g, xMin, xMax, prec = 0.0001):

    if g(xMin)*g(xMax)>0 :
      return 'La funzione non ha zeri nell intervallo desiderato'

    xAve = 0.5 * (xMax + xMin)
    if ((xMax - xMin) < prec):
      return xAve ;
    if (g (xAve) * g (xMin) > 0.):
      return bisezione_ricorsiva (g, xAve, xMax, prec) ;
    else                         :
      return bisezione_ricorsiva (g, xMin, xAve, prec) ;
    return xAve

#---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# trova il minimo con il metodo della sezione aurea (meno veloce)
#tutti gli algoritmi sezione aurea restituiscono la x del massimo (la y la trovi calcolando la funzione in x max)
def sezioneAureaMin (g, x0, x1, prec = 0.0001):
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
            x1 = x1
        else :
            x1 = x2
            x0 = x0

        larghezza = abs (x1-x0)

    return (x0 + x1) / 2.


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# trova il minimo con il metodo della sezione aurea (più veloce)

def sezioneAureaMin_ricorsiva (g, x0, x1, prec = 0.0001):
    r = 0.618
    x2 = x0 + r * (x1 - x0)
    x3 = x0 + (1. - r) * (x1 - x0)
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  :
      return ( x0 + x1) / 2.
    elif (g (x3) > g (x2)) :
      return sezioneAureaMin_ricorsiva (g, x3, x1, prec)
    else                   :
      return sezioneAureaMin_ricorsiva (g, x0, x2, prec)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#trova il max con il metodo della sezione aurea (meno veloce)

def sezioneAureaMax (g, x0, x1, prec = 0.0001):
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
            x1 = x1
        else :
            x1 = x2
            x0 = x0

        larghezza = abs (x1-x0)

    return (x0 + x1) / 2.


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#trova il max con il metodo della sezione aurea in modo ricorsivo (più veloce)

def sezioneAureaMax_ricorsiva (g, x0, x1, prec = 0.0001) :
    r = 0.618
    x2 = x0 + r * (x1 - x0)
    x3 = x0 + (1. - r) * (x1 - x0)
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  :
      return ( x0 + x1) / 2.
    elif (g (x3) < g (x2)) :
      return sezioneAureaMax_ricorsiva (g, x3, x1, prec)
    else                   :
      return sezioneAureaMax_ricorsiva (g, x0, x2, prec)


#-----------------------------

#fraction.py


from math import gcd
class Fraction :
    '''
    # a simple class implementing a high-level object
    # to handle fractions and their operations
    '''
    #self serve per dire che la funzione è interna alla classe fraction
    def __init__ (self, numerator, denominator) : #costruttore base di tutte le classi python (ci deve essere sempre)
        '''
        the constructor: initialises all the variables needed
        for the high-level object functioning
        '''
        if denominator == 0 :
          raise ValueError ('Il denominatore non può essere zero')
        if type(numerator) != int:
          raise TypeError ('Numerator must be an integer')
        if not isinstance(denominator, int ): # alternative way to check the type
          raise TypeError ('Denominator must be an integer')

        # this allows to avoid calculating the LCM in the sum and subtraction
        common_divisor = gcd (numerator, denominator) # greatest common divisor
        self.numerator = numerator // common_divisor # integer division with floor division
        self.denominator = int(denominator / common_divisor) # integer division with casting

    # funzione che vive nella classe
    def print (self) : #prende tutte le variabili di classe self
        '''
        prints the value of the fraction on screen
        '''
        print (str (self.numerator) + '/' + str (self.denominator))

    #addizione di due frazioni. la funzione si può chiamare con il simbolo +
    def __add__ (self, other) :
      new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
      new_denominator = self.denominator * other.denominator
      return Fraction (new_numerator, new_denominator)

    #sottrazione di due frazioni
    def __sub__ (self, other) :
      new_numerator = self.numerator * other.denominator - other.numerator * self.denominator
      new_denominator = self.denominator * other.denominator
      return Fraction (new_numerator, new_denominator)

    #moltiplicazione di due frazioni
    def __mul__ (self, other) :
      new_numerator = self.numerator * other.numerator
      new_denominator = self.denominator * other.denominator
      return Fraction (new_numerator, new_denominator)

    #divisione di due frazioni
    def __truediv__ (self, other) :
      new_numerator = self.numerator * other.denominator
      new_denominator = self.denominator * other.numerator
      return Fraction (new_numerator, new_denominator)


#-----------------------

#histo.py


import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import math

#calcolare la media aritmetica
def media (sample) :
    somma = sum (sample)
    N = len (sample)
    return somma / N

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
# calcola la varianza del campione
def varianza (sample, bessel = True) :
    somma = 0.
    N = len (sample)
    media=stats.mean(sample)
    for elem in sample :
        somma=somma+(elem-media)**2
    var = somma/N
    if bessel : var = N * var / (N - 1)
    return var

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#calcola la deviazione standard
def devstd (sample, bessel = True) :
    return np.sqrt (varianza (sample, bessel))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#calcola la deviazione standard della media
def devstd_media (sample, bessel = True) :
    N = len (sample)
    return np.sqrt (varianza (sample, bessel) / N)

# ---- ---- ---- ---- --- ---- ---- ---- ---- --- ---- ---- ----
#funzione che determina il bin size dato N (numero eventi)
def sturges (N_events) :
     return int( np.ceil( 1 + 3.322 * np.log(N_events) ) )

# ---- ---- ---- ---- ---- --- ---- ---- ---- ---- ---- ----- ----
#funzione che legge un file.txt
#se None, il range_bins viene calcolato automaticamente
#se voglio estrarre solo le statistiche:
'''
from histo import crea_histo
immagine, statistiche= crea_histo('eventi_gauss.txt')
'''
# e a questo punto prendo solo statistiche (che sarà un vettore con, in ordine:
# media, varianza, deviazione standard, deviazione standard della media)
def crea_histo (nome_file):
    doc=np.genfromtxt(nome_file)
    xx=np.array([media(doc), varianza(doc), devstd(doc), devstd_media(doc) ])
    xmin=math.floor(min(doc))
    xmax=math.ceil(max(doc))
    bin_edges=np.linspace(xmin,xmax,sturges(len(doc)))
    fig, ax = plt.subplots(nrows=1,ncols=1)
    isto= ax.hist(doc, bins= bin_edges, range=(xmin,xmax), color= 'blue')
    plt.xlabel('dati')
    plt.ylabel('frequenza')
    plt.title('istogramma dati')
    plt.savefig('istogramma.png')
    plt.show()
    return isto, xx

def numero_bin (events):
    xMin = max (0., np.ceil (media(events)- 3 * devstd(events)))
    xMax = np.ceil (media(events) + 3 * devstd(events))
    bin_edges = np.linspace (xMin, xMax, int (xMax - xMin) + 1)
    nBins=np.floor (len(events)/10. )+1
    return bin_edges, nBins



#-----------------------------------

#integral.py

import random
import numpy as np

def generate_range (xMin, xMax, N, seed = 0.) :
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_range (xMin, xMax))
    return randlist

def rand_range (xMin, xMax) :
    return xMin + random.random () * (xMax - xMin)

def integral_crudeMC (g, xMin, xMax, N_rand) :
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
    return media * lunghezza, np.sqrt (varianza / float (N_rand)) * lunghezza

# FOR FUNCTIONS THAT ARE: positive, continuous, and defined on a compact and connected interval
def integral_hitormiss (func, xMin, xMax, yMax, N_evt) :
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

#------------------------------

#likelihood.py

import math
import numpy as np
#theta è il parametro

def likelihood (theta, pdf, sample) :
    risultato = 1.
    for x in sample:
        risultato = risultato * pdf (x, theta)
    return risultato

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#meglio usare questo per la pdf esponenziale

def loglikelihood (theta, pdf, sample) :
    risultato = 0.
    for x in sample:
        if (pdf (x, theta) > 0.) : risultato = risultato + math.log (pdf (x, theta))
    return risultato

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def loglikelihood_prod (theta, pdf, sample) :
    risultato = 0.
    produttoria = np.prod(pdf(sample, theta))
    risultato= np.log(produttoria)
    return risultato

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#per migliorare la visualizzazione

def loglikelihood_ratio(tau, pdf, data, tau_max):
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
    # Rapporto aureo
    phi = (1 + math.sqrt(5)) / 2

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
    risultato = 0.
    for x in sample:
        prob = pdf(x, *theta)  # Passa tutti gli elementi di theta come parametri separati
        if prob > 0.:
            risultato += math.log(prob)
    return risultato


#-------------------------------------

#rand_distrib.py


import random
from math import sqrt
import numpy as np

# ---- ---- ---- ---- ---- ----- ---- ---- ---- ---- ---- ---- ---- ----
#funzione che implementa il generatore lineare congruenziale: genera N numeri
#casuali dato un seed

def random_LCG (a,c,m,seed, N):
    randlist=[]
    for i in range(N):
        seed=(a*seed+c)%m
        randlist.append(seed)
    randvec=np.array(randlist)
    return randvec

# ---- ---- ---- ---- --- ---- ---- ---- ----- ----- ---- ---- ---- ----
#generazione di N numeri pseudo-casuali distribuiti fra 0 ed 1
#a partire da un determinato seed

def generate_uniform (N, seed = 0.) :

    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (random.random ())
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#generazione di un numero pseudo-casuale distribuito fra xMin ed xMax

def rand_range (xMin, xMax) :
    return xMin + random.random () * (xMax - xMin)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#generazione di N numeri pseudo-casuali distribuiti fra xMin ed xMax
#a partire da un determinato seed

def generate_range (xMin, xMax, N, seed = 0.) :
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_range (xMin, xMax))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#f è la nostra funzione distribuzione di probabilità, xmin è il minimo della funzione, xmax è il massimo, ymax è il massimo
#non è molto efficiente

def rand_TAC (f, xMin, xMax, yMax) :
    '''
    generazione di un numero pseudo-casuale
    con il metodo try and catch
    '''
    x = rand_range (xMin, xMax)
    y = rand_range (0, yMax)
    while (y > f (x)) :
        x = rand_range (xMin, xMax)
        y = rand_range (0, yMax)
    return x


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#generazione di N numeri pseudo-casuali
#con il metodo try and catch, in un certo intervallo,
#a partire da un determinato seed

def generate_TAC (f, xMin, xMax, yMax, N, seed = 0.) :

    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_TAC (f, xMin, xMax, yMax))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#generazione di un numero pseudo-casuale
#con il metodo del teorema centrale del limite
#su un intervallo fissato
#all'aumentare di N_sum diventa sempre più gaussiana

def rand_TCL (xMin, xMax, N_sum = 10) :
    y = 0.
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax) #se devi generare a partire da un altra funzione (non uniforme) modifica qui 
    y /= N_sum ;
    return y ;


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#generazione di N numeri pseudo-casuali
#con il metodo del teorema centrale del limite, in un certo intervallo,
#a partire da un determinato seed

def generate_TCL (xMin, xMax, N, N_sum = 10, seed = 0.) :
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#generazione di un numero pseudo-casuale
#con il metodo del teorema centrale del limite
#note media e sigma della gaussiana

def rand_TCL_ms (mean, sigma, N_sum = 10) :
    y = 0.
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum ;
    return y ;


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----
#generazione di N numeri pseudo-casuali
#con il metodo del teorema centrale del limite, note media e sigma della gaussiana,
#a partire da un determinato seed

def generate_TCL_ms (mean, sigma, N, N_sum = 10, seed = 0.) :
    if seed != 0. : random.seed (float (seed))
    randlist = []
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N):
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist

#--- --- --- --- --- --- --- --- ---
#genera N numeri casuali distribuiti esponenzialmente usando
#il metodo della funzione inversa

def generate_exp (N, lmbda):
  x=np.array(generate_range(0,1,N))
  distrib = -np.log(1 - x) / lmbda
  return distrib

def random_exp (lmbda):
  x=random.random()
  num = -np.log(1 - x) / lmbda
  return num

def generate_poisson_events(lmbda, T, N):
    eventi = []

    for _ in range(N):
        num_eventi = 0
        tempo_totale = 0

        while tempo_totale <= T:
            # Genera un tempo di interevento esponenziale con parametro lmbda
            intertempo = random_exp(lmbda)
            tempo_totale += intertempo

            # Conta l'evento solo se il tempo totale è entro l'intervallo di osservazione
            if tempo_totale <= T:
                num_eventi += 1

        eventi.append(num_eventi)

    return eventi

#-------------------------------------------------------------------------
#genera numeri casuali come un exp con il metodo try and catch (utile se devi generarli in un range)
def try_and_catch_exp (lamb, N):
    events = []
    x_max = 3/lamb #xmin=0, xmax= 37/lamb. CAMBIA QUI INTERVALLO
    for i in range (N):
      x = rand_range (0., x_max)
      y = rand_range (0., lamb) #LIMITI PER ASSE Y
      while (y > lamb * math.exp (-lamb * x)): #QUI LA FUNZIONE (EXP IN QUESTO CASO)
        x = rand_range (0., x_max)
        y = rand_range (0., lamb)
      events.append (x)
    return events


def try_and_catch_gau (mean, sigma, N):
    events = []
    for i in range (N):
      x = rand_range (mean - 3 * sigma, mean + 3 * sigma) #LIMITI ASSE X
      y = rand_range (0., 1.) #LIMITI ASSE Y
      while (y > math.exp (-0.5 * ( (x - mean)/sigma)**2)):
        x = rand_range (mean - 3 * sigma, mean + 3 * sigma)
        y = rand_range (0, 1.)
      events.append (x)
    return events


#-----------------------------

#stats.py

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

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    def mean (self) :
        '''
        calculates the mean of the sample present in the object
        '''
        return self.summ / self.N

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    def variance (self, bessel = True) :
        '''
        calculates the variance of the sample present in the object
        '''
        var = self.sumSq / self.N - self.mean () * self.mean ()
        if bessel : var = self.N * var / (self.N - 1)
        return var

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    def sigma (self, bessel = True) :
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel))

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    def sigma_mean (self, bessel = True) :
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel) / self.N)

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

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


    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

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

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

    def append (self, x):
        '''
        add an element to the sample
        '''
        self.sample.append (x)
        self.summ = self.summ + x
        self.sumSq = self.sumSq + x * x
        self.N = self.N + 1