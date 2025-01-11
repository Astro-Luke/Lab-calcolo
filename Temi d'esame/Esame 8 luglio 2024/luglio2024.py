'''
python3 luglio2024.py
'''
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import ExtendedBinnedNLL
from scipy.stats import rayleigh
from lib import random_walk, calcola_distanza, media, varianza_bessel, skewness, kurtosis, sturges, funzione_fit, Rayleigh

def main () :

    # Punto 1 e 2
    mean = 1.
    sigma = 0.2
    N_passi = 10
    coord_x, coord_y = random_walk (mean, sigma, N_passi)
    
    #print ("\n", coord_x, "\n")
    #print (coord_y)
    
    # calcolo la distanza tra il punto (x, y) = (0, 0) ed il punto raggiunto
    coord_x_array = np.array (coord_x)
    coord_y_array = np.array (coord_y)

    distanza = calcola_distanza (0., coord_x_array[10] , 0., coord_y_array[10])
    print ("La distanza dal punto di partenza (0, 0) al punto", "(x, y) = (", coord_x_array[10], ",", 
           coord_y_array[10], ") è: \n", distanza, "\n")

    # Grafico
    fig, ax = plt.subplots ()
    ax.plot (coord_x, coord_y, "o-", color = "blue")
    ax.set_xlabel ("x")
    ax.set_ylabel ("y")
    ax.grid ()
    plt.savefig ("Grafico ubriaco.png")

    # Punto 3
    N_persone = 10000
    list_distanze = []
    for _ in range (N_persone) :
        coord_x, coord_y = random_walk (mean, sigma, N_passi)
        distanza = calcola_distanza (0., coord_x[10], 0., coord_y[10])
        list_distanze.append (distanza)


    Nbin = sturges (len (list_distanze))

    bin_content, bin_edges = np.histogram (list_distanze, bins=Nbin, range = (min (list_distanze), max(list_distanze)))
    #print (bin_edges)
    '''
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (list_distanze, bins = bin_edges, color = 'orange')
    ax.set_title ('Plot persone diversamente sobrie', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('Distanza percorsa')
    ax.grid ()                                        
    plt.savefig ('Persone ubriache.png')
    '''
    # Punto 4
    vett_distanza = np.array (list_distanze)    # casting

    print ("\n----- Statistiche della distribuzione -----\n\nMedia: ", media (list_distanze))
    print ("\nVarianza: ", varianza_bessel (list_distanze))
    print ("\nAsimmetria: ", skewness (list_distanze))
    print ("\nCurtosi: ", kurtosis (list_distanze))

    # punto 5: Fit
    N_passi = 10
    funz_costo = ExtendedBinnedNLL (bin_content, bin_edges, funzione_fit)
    my_minuit = Minuit (funz_costo, N_passi)
    my_minuit.migrad ()

    print ("Esito del Fit: ", my_minuit.valid)
    print ("Valore: ", my_minuit.values[0], "+/-", my_minuit.errors[0])


    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    x = np.linspace (min (bin_edges), max (bin_edges), 500)

    ax.hist (list_distanze, bins = bin_edges, color = 'blue')
    #Normalizzazione tipica delle distribuzioni binnate, prendere la distanza tra due bin edges e moltiplicarla per il numero di eventi/entrate
    ax.plot (x, N_persone * (bin_edges[1] - bin_edges[0]) * Rayleigh (x, *my_minuit.values), label = "Rayleigh Fit", color = "red")          
    ax.plot (x, N_persone * (bin_edges[1] - bin_edges[0]) * Rayleigh (x, N_passi), label = "True Rayleigh", color = "green")
    ax.set_title ('Plot persone diversamente sobrie', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('Distanza percorsa')
    ax.legend ()
    ax.grid ()                                        
    plt.savefig ('Persone ubriache.png')

    plt.show ()

if __name__ == "__main__" :
    main ()