'''
python3 main.py
'''

import numpy as np
import matplotlib.pyplot as plt

from lib import sturges, f_cauchy, generate_cauchy, rand_TCL_cauchy

def main () :

    # Punto 1 e 2
    gamma = 1.
    M = 0.5
    N = 1000

    casual_cauchy = []
    for i in range (N) :
        casual_cauchy.append (f_cauchy (gamma, M))

    Nbin = sturges (N)
    bin_content, bin_edges = np.histogram (casual_cauchy, bins=Nbin, range = (min(casual_cauchy), max(casual_cauchy)))

    fig, ax = plt.subplots ()
    ax.hist (casual_cauchy, bins = Nbin, color = 'orange', label = 'f_cauchy')
    ax.set_title('Istogramma distribuzione Cauchy')
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()
    ax.legend ()

    # Punto 4
    lista_medie_cauchy, std_cauchy, lista_contatore = generate_cauchy (M, gamma)

    fig, ax = plt.subplots ()
    ax.plot (lista_contatore, lista_medie_cauchy, color = 'blue', label = 'Media')
    ax.plot (lista_contatore, std_cauchy, color = 'red', label = 'Sigma')
    ax.set_title('Andamento della media e dev_std')
    ax.set_xlabel ('i')
    ax.set_ylabel ('media e sigma')
    ax.grid ()
    ax.legend ()

    # Punto 5
    N_pt5 = 10000
    lista_TCL_cauchy = []
    for i in range (N) :
        lista_TCL_cauchy.append (rand_TCL_cauchy (gamma, M))

    Nbin = sturges (N_pt5)
    #print (lista_TCL_cauchy)
    bin_content, bin_edges = np.histogram (lista_TCL_cauchy, bins=Nbin, range = (min (lista_TCL_cauchy), max (lista_TCL_cauchy)) )
    
    fig, ax = plt.subplots ()
    ax.hist (lista_TCL_cauchy, bins=bin_edges, color = 'orange')
    ax.set_title ('Distribuzione Cauchy con TCL')
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()
    
    plt.show ()

if __name__ == '__main__' :
    main ()