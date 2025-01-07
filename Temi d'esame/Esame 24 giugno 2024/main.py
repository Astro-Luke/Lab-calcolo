'''
python3 main.py
'''

import numpy as np
import matplotlib.pyplot as plt

from lib import rand_exp_inversa, rand_TCL_par_gauss, sturges

# ----- Main -----

def main () :

    # Punto 1
    N_exp = 2000
    tau = 200
    lambd = (1/tau)
    campione_exp = []
    
    N_gauss = 200
    mu = 190
    sigma = 20
    campione_gauss = []
    
    for _ in range (N_exp) :
        campione_exp.append (rand_exp_inversa (lambd))
    '''
    for _ in range (N_gauss) :
        campione_gauss.append (rand_TCL_par_gauss (mu, sigma, 10))

    # controllino:
    # print ("campione uniforme: ", campione_uniforme, "\n", len(campione_uniforme))
    # print ("\ncampione gaussiano: ", campione_gauss, "\n", len(campione_gauss), "\n")

    # Punto 2
    campione_totale = campione_uniforme + campione_gauss
    '''
    Nbin = int (sturges (len (campione_exp)))

    bin_content, bin_edges = np.histogram (campione_exp, bins = Nbin, range = (0, 3 * float(tau)))

    '''
    bin_edges = np.linspace (0, 3 * tau , Nbin)         # Regola la dimensione dei bin e Nbin = numero di bin
    '''
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (campione_exp, bins = bin_edges, color = 'orange')
    ax.set_title ('Istogramma', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()             
    
    plt.savefig ('Istogramma .png')
    plt.show ()   

if __name__ == "__main__" :
    main ()