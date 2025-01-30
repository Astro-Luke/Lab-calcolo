'''
python3 main.py
'''

import numpy as np
import matplotlib.pyplot as plt

from lib import f, rand_TCL_unif, rand_TAC, rand_TCL_para, sturges, skewness, kurtosis

def main () :

    N = 10000
    sample_TCL_unif = []
    sample_TAC = []
    sample_TCL_para = []

    for i in range (N) :
        sample_TCL_unif.append (rand_TCL_unif (0., 3., 1000))
        sample_TAC.append (rand_TAC (f, 0., 3., 1.))
        sample_TCL_para.append (rand_TCL_para (0., 3., 1., 1000))
    
    # Istogramma
    Nbin = sturges (N)

    bin_content_TCL_unif, bin_edges_TCL_unif = np.histogram (sample_TCL_unif, bins=Nbin, range = (min(sample_TCL_unif), max(sample_TCL_unif)))

    bin_content_TAC, bin_edges_TAC = np.histogram (sample_TAC, bins=Nbin, range = (min(sample_TAC), max(sample_TAC)))

    bin_content_TCL_para, bin_edges_TCL_para = np.histogram (sample_TCL_para, bins=Nbin, range = (min(sample_TCL_para), max(sample_TCL_para)))


    fig, ax = plt.subplots (nrows = 1, ncols = 3, figsize = (4, 3))
    
    ax[0].hist (sample_TCL_unif, bins=bin_edges_TCL_unif, color = 'orange')
    ax[0].set_title ('TCL uniforme', size = 14)
    ax[0].set_xlabel ('x')
    ax[0].set_ylabel ('y')
    ax[0].grid ()
    
    ax[1].hist (sample_TAC, bins=bin_edges_TAC, color = 'blue')
    ax[1].set_title ('TAC', size = 14)
    ax[1].set_xlabel ('x')
    ax[1].set_ylabel ('y')
    ax[1].grid ()

    ax[2].hist (sample_TCL_para, bins=bin_edges_TCL_para, color = 'green')
    ax[2].set_title ('TCL para', size = 14)
    ax[2].set_xlabel ('x')
    ax[2].set_ylabel ('y')
    ax[2].grid ()

    plt.savefig ('Istogrammi 15 febbraio 2023.png')
    print ("\nAsimetria a partire da distribuzione uniforme con TCL: ", skewness (sample_TCL_unif))
    print ("\nAsimetria a partire da distribuzione uniforme con TAC: ", skewness (sample_TAC))
    print ("\nAsimetria a partire da distribuzione parabolica con TCL: ", skewness (sample_TCL_para))

    print ("\nCurtosi a partire da distribuzione uniforme con TCL: ", kurtosis (sample_TCL_unif))
    print ("\nCurtosi a partire da distribuzione uniforme con TAC: ", kurtosis (sample_TAC))
    print ("\nCurtosi a partire da distribuzione parabolica con TCL: ", kurtosis (sample_TCL_para))
    
    #Punto 5
    
    N_max = 200
    lista_contatore = []

    lista_asimm_TCL_unif = []
    lista_curt_TCL_unif = []
    lista_asimm_TCL_para = []
    lista_curt_TCL_para = []
    
    for count in range (N_max) :
        asimmetria_TCL_unif = []
        curtosi_TCL_unif = []
        asimmetria_TCL_para = []
        curtosi_TCL_para = []
        
        for i in range (count+2) : 
            val_TCL_unif = rand_TCL_unif (0., 3., 10)
            val_TCL_para = rand_TCL_para (0., 3., 1., 10)
            asimmetria_TCL_unif.append (val_TCL_unif)
            curtosi_TCL_unif.append (val_TCL_unif)
            asimmetria_TCL_para.append (val_TCL_para)
            curtosi_TCL_para.append (val_TCL_para)
        
        lista_asimm_TCL_unif.append (skewness (asimmetria_TCL_unif))
        lista_curt_TCL_unif.append (kurtosis (curtosi_TCL_unif))
        lista_asimm_TCL_para.append (skewness (asimmetria_TCL_para))
        lista_curt_TCL_para.append (kurtosis (curtosi_TCL_para))

        lista_contatore.append (count)

    
    fig, ax = plt.subplots (nrows = 2, ncols = 1)
    ax[0].plot (lista_contatore, lista_asimm_TCL_unif, color = 'red', label = 'Asimmetria uniforme')
    ax[0].plot (lista_contatore, lista_asimm_TCL_para, color = 'blue', label = 'Asimmetria parabolica')
    ax[0].set_title ('TCL uniforme', size = 14)
    ax[0].set_xlabel ('x')
    ax[0].set_ylabel ('y')
    ax[0].legend ()
    ax[0].grid ()

    ax[1].plot (lista_contatore, lista_curt_TCL_unif, color = 'red', label = 'curtosi uniforme')
    ax[1].plot (lista_contatore, lista_curt_TCL_para, color = 'blue', label = 'Curtosi parabolica')
    ax[1].set_title ('TCL parabolica', size = 14)
    ax[1].set_xlabel ('x')
    ax[1].set_ylabel ('y')
    ax[1].legend ()
    ax[1].grid ()

    plt.savefig ('Andamenti asimmetria e curtosi 15 febbraio 2023.png')
    
    plt.show ()
    
if __name__ == '__main__' :
    main ()