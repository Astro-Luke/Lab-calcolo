#!/usr/bin/python
'''
 Scrivere un programma Python che legga il file di esempio eventi_unif.txt dell'esercizio 3.2 e, 
 utilizzando la funzione filtro, crei due diversi sottoinsiemi di eventi contenenti rispettivamente 
 quelli maggiori o minori della media, utilizzando lambdadelle funzioni nel processo.
Dimostrare che la sigma dei due sottoinsiemi è la metà di quella del campione padre.

'''

import sys
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor


def sturges (N_events) :
    return ceil (1 + np.log2 (N_events))
    

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    
def main () :
    '''
    Function implementing the main program
    '''

    # read the file
    with open ('../../Lecture_03/exercises/eventi_gauss.txt') as input_file :
        sample = [float (x) for x in input_file.readlines ()]

    for elem in sample[:10]:
        print (elem)
  
    sample_sq = list (map (lambda x: x**2, sample))
    sample_cu = list (map (lambda x: pow (x, 3), sample))

    xMin = floor (min (min (sample), min (sample_sq), min (sample_cu)))
    xMax = ceil (max (max (sample), max (sample_sq), max (sample_cu)))
    N_bins = sturges (len (sample)) * 5

    bin_edges = np.linspace (xMin, xMax, N_bins)
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample,
             bins = bin_edges,
             color = 'orange',
             histtype= 'stepfilled'
            )
    ax.hist (sample_sq,
             bins = bin_edges,
             color = 'red',
             histtype= 'step'
            )
    ax.hist (sample_cu,
             bins = bin_edges,
             color = 'blue',
             histtype= 'step'
            )
    ax.set_yscale ('log')
    ax.set_title ('Histogram example', size=14)
    ax.set_xlabel ('variable')
    ax.set_ylabel ('event counts per bin')

    plt.savefig ('ex_4.5.png')



# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    
if __name__ == "__main__" :
    main ()
