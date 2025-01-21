'''
python3 main.py
'''
import numpy as np
from lib import additive_recurrence, MC_mod, function, integral_Crude_MC
import matplotlib.pyplot as plt

# Lib

import numpy as np
import random
from math import sqrt


class additive_recurrence :

    def __init__ (self, alpha) :
        self.alpha = alpha      # a destra la varriabile, a sinistra l'attributo di self
        self.N_0 = None         # esiste l'attributo ma non esiste ancora il valore
        self.N_f = None

    def get_number (self) :
        s = (self.N_f + self.alpha)%1
        self.N_f = s
        return s
    
    def set_seed (self, seed) :
        self.N_0 = seed
        self.N_f = seed

# ----- ----- ----- ----- ----- ----- ----- ----- 

def function (x) :
    return 2 * (x**2)


# Distribuzione uniforme tra x_min e x_max con seed scelto in auto
def rand_range (x_min, x_max) :
    return x_min + random.random() * (x_max - x_min)


#Funzione per il calcolo dell'integrale (area) e scarto secondo il metodo Hit Or Miss
def MC_mod (f, x_min, x_max, y_min, y_max, N_punti, generatore) :
    x_coord = []
    y_coord = []
    for _ in range (N_punti) :
        x_coord.append (generatore.get_number ())
        y_coord.append (rand_range (y_min, y_max))
    
    points_under = 0
    for x, y in zip (x_coord, y_coord) :             #zip per iterare su più liste in contemporanea
        if (f (x) > y) :
            points_under = points_under + 1
    
    A_rett = (x_max - x_min) * (y_max - y_min)
    frac = float (points_under) / float (N_punti)
    integral = A_rett * frac
    integral_incertezza = A_rett**2 * frac * (1-frac) / N_punti
    return integral, integral_incertezza


#Funzione per il calcolo dell'integrale (area ed errore)
def integral_Crude_MC (f, x_min, x_max, N_punti) :
    somma = 0.
    somma_quadrata = 0.0
    for _ in range (N_punti) :
        value = rand_range (x_min, x_max)
        somma = somma + f(value)
        somma_quadrata = somma_quadrata + ( f(value) * f(value) )
    mean = somma / N_punti
    varianza = somma_quadrata / N_punti - (mean)**2
    varianza = ( (N_punti - 1) / N_punti ) * varianza
    lunghezza = x_max - x_min
    return mean * lunghezza, sqrt (varianza / N_punti) * lunghezza


# Main

def main () :

    # Punto 1 e 2
    alpha = ((np.sqrt(5) - 1) / 2)
    seed = 1.
    generatore = additive_recurrence (alpha)
    generatore.set_seed (seed)

    lista_num = []
    for _ in range (0, 1000) :
        number = generatore.get_number ()
        lista_num.append (number)

    #print (lista_num[:10])

    # Punto 3
    int_val, int_incertezza = MC_mod (function, 0., 1., 0., 2., 1000, generatore)
    print ("\nValore dell'integrale: ", int_val, "+/-", int_incertezza, "\n")

    # Punto 4
    N_toys = 100
    N_points = 10
    list_points = []
    incerteza_list = []

    while (N_points < 25000) :
        integral_list = []
        for contatore_toys in range (N_toys) :
            integral_val, incertezza_val = MC_mod (function, 0., 1., 0., 2., N_points, generatore)
            integral_list.append (integral_val)
        incerteza_list.append (np.std (integral_list))         # sono indentati allo stesso modo (infatti hanno la stessa lunghezza)
        list_points.append (N_points)
        N_points = N_points * 2


    # Grafico
    fig, ax = plt.subplots ()
    ax.plot (list_points, incerteza_list, "o-", label = "Incertezza MC_mod", color = "blue")


    # Punto 5
    N_toys = 100
    N_points = 10
    list_points = []
    incerteza_list = []

    while (N_points < 25000) :
        integral_list = []
        for contatore_toys in range (N_toys) :
            integral_val, incertezza_val = integral_Crude_MC (function, 0., 1., N_points)
            integral_list.append (integral_val)
        incerteza_list.append (np.std (integral_list))         # sono indentati allo stesso modo (infatti hanno la stessa lunghezza)
        list_points.append (N_points)
        N_points = N_points * 2

    ax.plot (list_points, incerteza_list, "o-", label = "Incertezza Crude MC", color = "red")
    ax.set_xscale ("log")
    ax.set_yscale ("log")
    ax.legend ()
    ax.grid ()
    
    plt.savefig ("26settembre2024.png")
    plt.show ()
    
if __name__ == "__main__" :
    main ()
