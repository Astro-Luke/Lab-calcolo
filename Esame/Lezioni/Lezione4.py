
# ----- ----- ----- ----- Lezione 4 ----- ------ ----- ----- 

# Esercizio 4
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


# ----- ----- ----- ----- ----- ------ ----- ----- 


# Esercizio 5
'''
Implementare un generatore di numeri pseudo-casuali che utilizzi il metodo try-and-catch per generare numeri pseudo-casuali in 
base a una distribuzione di probabilità arbitraria.
Prendiamo la funzione di densità di probabilità (pdf) come parametro di input per generare numeri casuali.
Utilizzare la matplotliblibreria per visualizzare la distribuzione dei numeri generati.
'''

import sys
import numpy as np
from library import rand_range, rand_TAC_norm, sturges, seed_range
from scipy.stats import norm
import matplotlib.pyplot as plt
from math import floor


def controllo_arg() :
    if len(sys.argv) < 4 :
        print("Mancano degli argomenti da inserie a riga di comando.\nInserisci x_min, x_max ed N (numero di elementi da generare).")
        sys.exit()


def main () :
    
    x_min = float(sys.argv[1])
    x_max = float(sys.argv[2])
    N = int(sys.argv[3])
    seed = float(sys.argv[4])
    
    loc = 0.
    scale = 1.
    y_max = 1/(2*np.pi)
    
    #sample = seed_range (x_min, x_max, N, seed)
    
    sample = []
    for _ in range (0, N) :
        x = rand_TAC_norm (norm.pdf, x_min, x_max, y_max, loc, scale)
        sample.append(x)
    
    Nbins = sturges (N) + 20     # shift a mano (mio)
    #Nbins = floor (len (sample) / 20.) + 1        # prof
    
    bin_edges = np.linspace(x_min, x_max, Nbins)         # regola la dimensione dei bin e Nbin = numero di bin
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample, bins = bin_edges, color = 'orange')
    ax.set_title ('Eventi con TAC', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()          #se voglio la griglia
    
    plt.savefig ('es4_5.png')
    plt.show ()
    
    
if __name__ == '__main__' :
    main ()


# ----- ----- ----- ----- ----- ------ ----- ----- 


# Esercizio 6
'''
Implementare un generatore di numeri pseudo-casuali che utilizzi il metodo della funzione inversa 
per generare eventi distribuiti secondo una distribuzione di probabilità esponenziale.
Utilizzare la matplotliblibreria per visualizzare la distribuzione dei numeri generati.
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
import random

from library import rand_range, exp_inversa_seed, sturges

# ----- ----- ----- ----- ----- ----- ----- -----
'''
def controllo_arg() :
    if len(sys.argv) < 3 :
        print("Manca il nome del file da passare a linea di comando.\n")
        sys.exit()
'''
# ----- main -----

def main () :
    
    t = 2.4 #float(sys.argv[1])
    N = 10000 #int(sys.argv[2])
    
    if t <= 0 :
        print("Il parametro t dell'esponenziale deve essere positivo!")
        exit()
    
    x_min = 0.
    x_max = 10.
    
    sample = []
    for _ in range (0, N) :
        sample.append (exp_inversa_seed (t, random.random()) )
    
    Nbin = sturges(N) + 5
    print("Numero di bin: ", Nbin)
    
    bin_edges = np.linspace(x_min, x_max, Nbin)
    
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample, bins = bin_edges ,color = 'orange')
    ax.set_title ('Istogramma', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()          #se voglio la griglia
    
    plt.savefig ('grafico e4_6 con seed auto.png')
    plt.show ()                             # da mettere rigorosamente dopo il savefig
    
if __name__ == '__main__' :
    main()


# ----- ----- ----- ----- ----- ------ ----- ----- 


# Esercizio 6 (con seed scelto da me)
'''
Implementare un generatore di numeri pseudo-casuali che utilizzi il metodo della funzione inversa per 
generare eventi distribuiti secondo una distribuzione di probabilità esponenziale.
Utilizzare la matplotliblibreria per visualizzare la distribuzione dei numeri generati.
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
import random

from library import rand_range, exp_inversa_seed, sturges

# ----- ----- ----- ----- ----- ----- ----- -----

def controllo_arg() :
    if len(sys.argv) < 4 :
        print("Manca il nome del file da passare a linea di comando.\n")
        sys.exit()

# ----- main -----

def main () :
    
    t = 2.4 #float(sys.argv[1])
    N = 10000 #int(sys.argv[2])
    seed = 1. #float(sys.argv[3])
    
    #controllo_arg()
    
    if t <= 0 :
        print("Il parametro t dell'esponenziale deve essere positivo!")
        exit()
    
    x_min = 0.
    x_max = 10.
    
    # imposto il seed
    random.seed(seed)
    
    sample = []
    for _ in range (0, N) :
        sample.append (exp_inversa_seed (t, random.random()) )  # Non c'è bisogno di riscrivere random.seed(seed) poichè è già stato inizializzato, se lo richiamassi qui reinizializzerei il seed ogni volta
    
    Nbin = sturges(N) + 5           # shift a mano
    #print("Numero di bin: ", Nbin)
    
    bin_edges = np.linspace(x_min, x_max, Nbin)
    
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample, bins = bin_edges ,color = 'orange')
    ax.set_title ('Istogramma', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()          #se voglio la griglia
    
    plt.savefig ('grafico es4_6.png')
    plt.show ()                             # da mettere rigorosamente dopo il savefig
    
if __name__ == '__main__' :
    main()


# ----- ----- ----- ----- ----- ------ ----- ----- 


# Esercizio 7
'''
Implementare un generatore di numeri pseudo-casuali che utilizzi il metodo del teorema del limite centrale 
per generare eventi distribuiti secondo una distribuzione di probabilità gaussiana.
Come si può ottenere una distribuzione normale, cioè una distribuzione gaussiana centrata sullo zero con varianza unitaria?
Verificare visivamente che all'aumentare del numero di eventi aumenta la similarità tra la distribuzione ottenuta e 
la forma funzionale gaussiana, sia graficamente sia utilizzando i momenti delle distribuzioni calcolati sul campione di eventi generato.
'''

import matplotlib.pyplot as plt
import sys
from library import sturges, rand_TCL
import numpy as np


def controllo_arg() :
    if len(sys.argv) != 4 :
        print("Inserire python3 es4_7.py x_min, x_max ed il numero di numeri N da generare. \n")
        sys.exit()


def main () :
    
    x_min = -1. #float(sys.argv[1])
    x_max = 1. #float(sys.argv[2])
    N = 5000 #int(sys.argv[3])
    
    sample = []
    for _ in range (N) :
        sample.append (rand_TCL (x_min, x_max))
    
    Nbin = sturges (N)
    
    bin_edges = np.linspace (min (sample), max (sample), Nbin)
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample, bins = bin_edges, color = 'orange')
    ax.set_title ('Istogramma', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()          #se voglio la griglia
    
    plt.savefig ('grafico es4_7.png')
    plt.show ()
    
if __name__ == '__main__' :
    main ()
