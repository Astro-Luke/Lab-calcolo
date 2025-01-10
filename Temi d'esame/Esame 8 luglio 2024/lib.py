import numpy as np
import random
from math import ceil

def sturges (N_eventi) :
    return int (ceil (1 + np.log2 (N_eventi))) 

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Funzione che genera numeri pseudocasuali partendo dal teorema centrale del limite usando media, sigma di una gaussiana
# ed N numero di eventi pseudocasuali
def rand_TCL_par_gauss (mean, sigma, N) :           # par_gauss = parametri gaussiani
    y = 0. ; 
    xMin = np.abs(mean - np.sqrt(3 * N) * sigma)
    xMax = np.abs(mean + np.sqrt(3 * N) * sigma)
    for i in range (N) :
        y += rand_range (xMin, xMax)
    y /= N 
    return y 

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def random_walk (mean, sigma, N_num, N_passi) :
    asse_x = [0.]
    asse_y = [0.]
    for _ in range (N_passi) : 
        theta = 360 * rand_range (0., 1.)
        ro = rand_TCL_par_gauss (mean, sigma, N_num)
        x = ro * np.cos (theta)
        y = ro * np.sin (theta)
        asse_x.append (x)
        asse_y.append (y)
    return asse_x, asse_y

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def calcola_distanza (x_0, x_n, y_0, y_n) :
    return np.sqrt( ((x_n - x_0)**2) + ((y_n - y_0)**2) )

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

def Rayleigh (r, N) :
    return ((2*r)/N) * np.exp(-(r**2)/N)


# -------------- Statistiche --------------

# Media con array
def media (sample) :
    mean = np.sum(sample)/len(sample)
    return mean
    
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Varianza con array
def varianza (sample) :
    somma_quadrata = 0
    somma_quadrata = np.sum( (sample - media (sample))**2 )
    var = somma_quadrata/(len (sample))
    return var

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Varianza con corr. di Bessel con array
def varianza_bessel (sample) :
    somma_quadrata = 0
    somma_quadrata = np.sum( (sample - media(sample))**2 )
    var = somma_quadrata/(len(sample) - 1)
    return var

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Deviaz. standard con array
def dev_std (sample) :
    sigma = np.sqrt (varianza(sample))
    return sigma

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Deviaz. standard della media con array
def dev_std_media (sample) :
    return dev_std(sample) / (np.sqrt( len(sample) ))

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Skewness con array
def skewness (sample) :
    mean = media (sample)  # Calcola la media con la tua funzione
    sigma = dev_std (sample)  # Calcola la deviazione standard con la tua funzione
    n = len(sample)
    skew = np.sum((sample - mean)**3) / (n * sigma**3)
    return skew

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ----

# Curtosi con array
def kurtosis (sample) :
    mean = media (sample)  # Calcola la media con la tua funzione
    variance = varianza (sample)  # Calcola la varianza con la tua funzione
    n = len(sample)
    kurt = np.sum((sample - mean)**4) / (n * variance**2) - 3
    return kurt