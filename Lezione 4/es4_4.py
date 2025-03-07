'''
Implementare un generatore di numeri pseudo-casuali secondo una distribuzione uniforme tra due endpoint arbitrari.

Utilizzare la matplotliblibreria per visualizzare la distribuzione dei numeri generati.
'''

import matplotlib.pyplot as plt
import sys
from library import rand_range, sturges
import numpy as np


def controllo_arg() :
    if len(sys.argv) < 3 :
        print("Inserire xmin ed xmax.\n")
        sys.exit()
        
def main () :
    
    #controllo_arg ()
    
    N = 500
    
    x_min = 1. #float(sys.argv[1])
    x_max = 2. #float(sys.argv[2])
    
    sample = []
    for i in range (0, N) :
        sample.append(rand_range(x_min, x_max))
    
    Nbins = sturges(N)
    
    bin_edges = np.linspace(x_min, x_max, Nbins)
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample, bins=bin_edges, color = 'orange')
    ax.set_title ('Istogramma uniforme', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()          #se voglio la griglia
    
    plt.savefig ('istogramma es4_4.png')
    plt.show ()
    
if __name__ == '__main__' :
    main ()
