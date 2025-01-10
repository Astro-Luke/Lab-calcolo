'''
python3 luglio2024.py
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rayleigh
from lib import random_walk, calcola_distanza, sturges, media, varianza_bessel, skewness, kurtosis

def main () :

    # Punto 1 e 2
    mean = 1.
    sigma = 0.2
    N_passi = 10
    coord_x, coord_y = random_walk (mean, sigma, 100, N_passi)
    
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
    coord_x_pop = []
    coord_y_pop = []
    list_distanze = []
    for _ in range (N_persone) :
        coord_x, coord_y = random_walk (mean, sigma, 100, N_passi)
        distanza = calcola_distanza (0., coord_x[10], 0., coord_y[10])
        list_distanze.append (distanza)
        coord_x_pop.append (coord_x)
        coord_y_pop.append (coord_y)
        
        '''
        print ("\n", coord_x_pop, "\n")
        print (coord_y_pop)
        '''

    #Nbin = sturges (len (list_distanze))

    #bin_edges = np.linspace (min (coord_x_pop), max (coord_x_pop), Nbin)
    #print (bin_edges)
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (list_distanze, bins = 'auto', color = 'orange')
    ax.set_title ('Plot persone diversamente sobrie', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('Distanza percorsa')
    ax.grid ()                                        
    plt.savefig ('Persone ubriache.png')

    # Punto 4
    vett_distanza = np.array (list_distanze)    # casting

    print ("\n----- Statistiche della distribuzione -----\n\nMedia: ", media (list_distanze))
    print ("\nVarianza: ", varianza_bessel (list_distanze))
    print ("\nAsimmetria: ", skewness (list_distanze))
    print ("\nCurtosi: ", kurtosis (list_distanze))

    # punto 5: Fit


    plt.show ()

if __name__ == "__main__" :
    main ()