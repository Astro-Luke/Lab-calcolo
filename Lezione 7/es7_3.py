'''
Utilizzare il codice sorgente scritto nell'esercizio precedente per aggiungere alla libreria sviluppata per l'esercizio 1 
una funzione che genera numeri casuali secondo la distribuzione di Poisson, con la media degli eventi attesi come parametro di input.

Riscrivi l'esercizio precedente utilizzando questa funzione, disegnando anche l'istogramma della densità di probabilità.

Calcola le statistiche del campione (media, varianza, asimmetria, curtosi) dall'elenco di input utilizzando una libreria.

Utilizzare l'esempio generato per testare la funzionalità della libreria.
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

from library import sturges, rand_pois, media, varianza, dev_std, dev_std_media

def controllo_arg() :
    if len(sys.argv) != 3 :
        print("Inserire il nome del file (compresa l'estensione), numero di eventi da generare e media della distribuzione.\n")
        sys.exit()

def main () :
    
    #controllo_arg()
    
    N = 100 #int(sys.argv[1])        # numero di eventi pseudocasuali da generare
    mean = 2.4 #float(sys.argv[2])   # media da passare
    
    sample = []
    for _ in range (N) :
        sample.append(rand_pois (mean))
    #print(sample)                              # solo per fare un controllo
    
    # stampa dei momenti della distribuzione
    print("La media della distribuzione è: ", media(sample))
    print("La varianca della distribuzione è: ", varianza(sample))
    print("La deviazione standard è: ", dev_std(sample))
    print("La deviazione standard della media è: ", dev_std_media(sample))

    Nbin = sturges(N)
    
    bin_edges = np.linspace (min(sample), max(sample), Nbin)
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample, bins = bin_edges, color = 'orange')
    ax.set_title ('prova', size = 14)
    ax.set_xlabel ('numero di eventi')
    ax.set_ylabel ('Conteggi')
    ax.grid ()                      #se voglio la griglia
    
    plt.savefig ('es7_3.png')
    plt.show ()                     # da mettere rigorosamente alla fine!

if __name__ == "__main__" :
    main ()
