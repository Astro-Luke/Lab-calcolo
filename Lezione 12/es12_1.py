'''
Write a program that fits the events saved in the file dati.txt.
Take care to determine the range and binning of the histogram used for the fit based on the events themselves, 
writing appropriate algorithms to determine the minimum and maximum of the sample and a reasonable estimate of the number of bins to use.
Determine the initial values of the fit parameters using the techniques described in the lesson.
Print the fit result on the screen.
Plot the histogram with the fitted model overlaid.
Which parameters are correlated, and which are anti-correlated with each other?
'''

import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit
from iminuit.cost import LeastSquares, ExtendedBinnedNLL
from scipy.stats import expon, norm

from library import sturges

def mod_total (bin_edges, N_signal, mu, sigma, N_background, tau) :
    return N_signal * norm.cdf (bin_edges, mu, sigma) + \
            N_background * expon.cdf (bin_edges, 0, tau)

def main () :
    
    vettore = np.loadtxt ("dati.txt")

    Nbin = sturges (vettore.size)

    bin_content, bin_edges = np.histogram (vettore, Nbin, (min(vettore), max(vettore) ))     # restituisce le y e le x dell'istogramma

    funzione_costo = ExtendedBinnedNLL (bin_content, bin_edges, mod_total)

    fit = Minuit (funzione_costo, 2000, 10., 1., 2000, 1.)

    fit.migrad()
    print("Fit is valid: ", fit.valid)

    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (vettore, bins=bin_edges ,color = 'orange')  # Spesso conviene usare bins = 'auto' evitando di scrivere la linea di codice con bin_edges, per farlo però bisogna importare numpy
    ax.set_title ('Istogramma', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()                                          # Se voglio la griglia
    
    plt.savefig ('Histo.png')
    plt.show ()   

if __name__ == "__main__" :
    main ()