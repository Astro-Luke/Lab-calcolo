'''
Librerie usate nei vari temi d'esame dai prof
'''

#----------------------------22 GENNAIO 2024----------------------

#lib.py

import math
import numpy as np
from random import uniform
from scipy.stats import norm


def pdf (x) :
  # if (x < 0) : return 0
  # if (x > 1.5 * math.pi) : return 0
  return np.cos (x)**2


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def genera_pdf (xMin, xMax, yMin, yMax):
  num = 0
  x = uniform (xMin, xMax)
  y = uniform (0, yMax)
  num += 1
  while (y > pdf (x)) :
    x = uniform (xMin, xMax)
    y = uniform (0, yMax)
    num += 1
  return x, num  


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def genera (N) :
  campione = []
  xMin = 0 
  xMax = 1.5 * np.pi
  yMin = 0
  yMax = 1
  num = 0
  while (len (campione) < N):
    x, count = genera_pdf (xMin, xMax, yMin, yMax)
    campione.append (x)
    num += count
  area = (yMax - yMin) * (xMax - xMin) * len (campione) / num
  return campione, area


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def sturges (N) :
     return int( np.ceil( 1 + 3.322 * np.log(N) ) )


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TCL (N_sum = 10) :
  '''
  generazione di un numero pseudo-casuale 
  con il metodo del teorema centrale del limite
  su un intervallo fissato
  '''
  xMin = 0 
  xMax = 1.5 * np.pi
  yMin = 0
  yMax = 1
  y = 0.
  for i in range (N_sum) :
      y = y + genera_pdf (xMin, xMax, yMin, yMax)[0]
  y /= N_sum ;
  return y ;

# the fitting function
def mod_gaus (bin_edges, mu, sigma):
    return norm.cdf (bin_edges, mu, sigma)


#--------------------------

#stats.py


#!/usr/bin/python

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
        

#------------------------------5 FEBBRAIO 2024-----------------------------

import math
import numpy as np
from random import uniform
from math import ceil
# from scipy.stats import norm



def phi (x, a, b, c):
    return a + x * b + x**2 * c


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TCL (xMin, xMax, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    su un intervallo fissato
    '''
    y = 0.
    for i in range (N_sum) :
        y = y + uniform (xMin, xMax)
    y /= N_sum ;
    return y ;


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def rand_TCL_ms (mean, sigma, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    note media e sigma della gaussiana
    '''
    y = 0.
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N_sum) :
        y = y + uniform (xMin, xMax)
    y /= N_sum ;
    return y ;


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def generate_TCL_ms (mean, sigma, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, note media e sigma della gaussiana,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    delta = np.sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (rand_TCL (xMin, xMax, N_sum))
    return randlist
    
    
def sturges (N_events) :
    return ceil (1 + 3.322 * np.log (N_events))


#------------------------------22 FEBBRAIO 2024-------------------

#non sto a copiare il file coi dati
import random
from math import sqrt

def rand_range (xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    '''
    return xMin + random.random () * (xMax - xMin)

def rand_TCL_ms (mean, sigma, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    note media e sigma della gaussiana
    '''
    y = 0.
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N_sum) :
        y = y + rand_range (xMin, xMax)
    y /= N_sum
    return y

#---------------------24 GIUGNO 2024-----------------------------


import random
import math
import numpy as np


def rand_range (xMin, xMax) :
    return xMin + random.random () * (xMax - xMin)


def try_and_catch_exp (lamb, N):
    events = []
    x_max = 3/lamb
    for i in range (N):
      x = rand_range (0., x_max)
      y = rand_range (0., lamb)
      while (y > lamb * math.exp (-lamb * x)):
        x = rand_range (0., x_max)
        y = rand_range (0., lamb)
      events.append (x)
    return events


def try_and_catch_gau (mean, sigma, N):
    events = []
    for i in range (N):
      x = rand_range (mean - 3 * sigma, mean + 3 * sigma)
      y = rand_range (0., 1.)
      while (y > math.exp (-0.5 * ( (x - mean)/sigma)**2)):
        x = rand_range (mean - 3 * sigma, mean + 3 * sigma)
        y = rand_range (0, 1.)
      events.append (x)
    return events


def sturges (N_events) :
     return int( np.ceil( 1 + 3.322 * np.log (N_events) ) )
    
    
    
#-----------------------8 LUGLIO 2024-------------------


#lib.py

import random
import math
import numpy as np


def rand_range (xMin, xMax) :
    return xMin + random.random () * (xMax - xMin)


def try_and_catch_step (mean, sigma):
    x_left = max (mean - 3 * sigma, 0)
    x = rand_range (x_left, mean + 3 * sigma)
    y = rand_range (0., 1.)
    while (y > math.exp (-0.5 * ( (x - mean)/sigma)**2)):
      x = rand_range (x_left, mean + 3 * sigma)
      y = rand_range (0, 1.)
    return x


def norm (position):
    return np.sqrt (position[0]**2 + position[1]**2)


def delta (dopo, prima=[0,0]):
    return norm ([d - p for d, p in zip (dopo, prima)])


def walk (N_steps = 10, start = [0,0], r_cost = False):
    # end = start identificare end e start come associati alla stessa cella di memoria
    end = [start[0], start[1]] # creare una nuova cella di memoria per end e metterci dentro gli stessi
                               # valori di start  
    angle = 0.
    step = 1.
    for i in range (N_steps):
        angle = rand_range (0, 2*np.pi)
        if not r_cost: step = try_and_catch_step (1, 0.2)
        end[0] += step * np.cos (angle)
        end[1] += step * np.sin (angle)
    return end


def sturges (N_events) :
    return int( np.ceil( 1 + 3.322 * np.log (N_events) ) )


#------------

#stats.py

#!/usr/bin/python

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
        
        
#---------------------16 SETTEMBRE 2024----------------------


#lib.py


import random
import math
import numpy as np


class additive_recurrence :

    def __init__ (self, alpha = 0.618034) : # (sqrt(5)-1)/2
        self.alpha = alpha
        self.s_0 = 0.5
        self.s_n = 0.5
        
    def get_number (self) :
        self.s_n = (self.s_n + self.alpha) % 1
        return self.s_n

    def set_seed (self, seed) :
        self.s_0 = seed
        self.s_n = seed
   
    def get_numbers (self, N) :
        lista = []
        for i in range (N) : lista.append (self.get_number ())
        return lista


def rand_range (xMin, xMax) :
    return xMin + random.random () * (xMax - xMin)


def MC_mod (N_points) :
    gen_seq = additive_recurrence ()
    sotto = float (0)
    for i in range (N_points):
        x = gen_seq.get_number ()
        y = rand_range (0., 2.)
        if (y < 2 * x * x) : sotto += 1
    frazione = sotto / N_points
    integrale = 2 * frazione
    sigma = 2 * np.sqrt (frazione * (1 - frazione) / N_points)
    return integrale, sigma


def MC_classic (N_points) :
    gen_seq = additive_recurrence ()
    sotto = float (0)
    for i in range (N_points):
        x = rand_range (0., 1.)
        y = rand_range (0., 2.)
        if (y < 2 * x * x) : sotto += 1
    frazione = sotto / N_points
    integrale = 2 * frazione
    sigma = 2 * np.sqrt (frazione * (1 - frazione) / N_points)
    return integrale, sigma


def sturges (N_events) :
    return int( np.ceil( 1 + 3.322 * np.log (N_events) ) )


def integral_CrudeMC (g, xMin, xMax, x_axis) :
    somma  = 0.
    sommaQ = 0.    
    N_rand = len (x_axis)
    for i in range (N_rand) :
       somma += g(x_axis[i])
       sommaQ += g(x_axis[i]) * g(x_axis[i])     
     
    media = somma / float (N_rand)
    varianza = sommaQ /float (N_rand) - media * media 
    varianza = varianza * (N_rand - 1) / N_rand
    lunghezza = (xMax - xMin)
    return media * lunghezza, np.sqrt (varianza / float (N_rand)) * lunghezza
                                         
    
#-----------

#stats.py

#!/usr/bin/python

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
        

        
#--------------------------10 OTTOBRE 2024----------------

#lib.py
import random
import math
import numpy as np



# def inv_exp (y, lamb = 1) :
#     '''
#     Inverse of the primitive of the exponential PDF.
#     pdf(x) = lambda * exp(-lambda x) x >= 0, 0 otherwise.
#     F(x) = int_{0}^{x} pdf(x)dx = 1 - exp(-lambda * x) for x >= 0, 0 otherwise.
#     F^{-1}(y) = - (ln(1-y)) / lambda
#     '''
#     return -1 * np.log (1-y) / lamb


# def generate_exp (lamb = 1, N = 1) :
#     randlist = []
#     for i in range (N):
#         randlist.append (inv_exp (random.random (), lamb))
#     return randlist


def generate_gaus_bm () :
    X1 = random.random ()
    X2 = random.random ()
    G1 = np.sqrt (-2 * np.log (X1)) * np.cos (2 * np.pi * X2)
    G2 = np.sqrt (-2 * np.log (X1)) * np.sin (2 * np.pi * X2)
    return G1, G2


# def rand_range (xMin, xMax) :
#     return xMin + random.random () * (xMax - xMin)


def sturges (N_events) :
    return int( np.ceil( 1 + 3.322 * np.log (N_events) ) )

#-------------

#stats.py

#!/usr/bin/python

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