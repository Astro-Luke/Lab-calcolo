
# ----- ----- ----- ----- ----- Lezione 7 ----- ----- ----- ----- -----

'''
Generare un campione di numeri pseudo-casuali distribuiti secondo una distribuzione di densità esponenziale 
con un tempo caratteristico t 0 di 5 secondi.
Visualizzare la distribuzione del campione ottenuto in un istogramma utilizzando il metodo della funzione inversa.
Scrivere tutte le funzioni responsabili della generazione di numeri casuali in una libreria, 
implementate in file separati dal programma principale.
'''

import matplotlib.pyplot as plt
import sys
import numpy as np

from library import rand_exp_inversa, sturges

# ----- ----- Main ----- -----

def main () :
    
    t = 5.      # tempo caratteristico
    N = 5000         # numero di eventi pseudocasuali da generare
    x_min = 0.
    x_max = 25.
    
    sample = []         # genero la lista che conterrà i numeri pseudocasuali generati
    for _ in range (0, N) :                     # ciclo di riempimento
        sample.append(rand_exp_inversa(t))
    
    # Impostazioni istogramma e creazione
    
    Nbin = sturges(N)                           # (comunque secondo me sta funzione funziona un pò a c***o)
    bin_edges = np.linspace(x_min, x_max, Nbin)         # Regola la dimensione dei bin e Nbin = numero di bin
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample, bins = bin_edges ,color = 'orange')  # Da provare anche con bins = 'auto'
    ax.set_title ('Istogramma', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()          #se voglio la griglia
    
    plt.savefig ('es7_1.png')
    plt.show ()                     # da mettere sempre alla fine
    
if __name__ == '__main__' :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 


# Esercizio 2
'''
Utilizzare il risultato del primo esercizio per simulare un esperimento di conteggio con caratteristiche di Poisson:
Scegliere un tempo caratteristico t_0 per un processo di decadimento radioattivo;
Scegliere un tempo di misura t_M per la finestra di conteggio;
In un ciclo, simulare N pseudo-esperimenti di conteggio, dove, per ciascuno di essi, 
viene generata una sequenza di eventi casuali con una caratteristica intertemporale dei fenomeni di Poisson, 
finché il tempo totale trascorso è maggiore del tempo di misurazione, contando il numero di eventi generati che rientrano nell'intervallo.
Compila un istogramma con i conteggi simulati per ciascun esperimento.
'''


import numpy as np
import matplotlib.pyplot as plt
import sys
import random

from library import sturges, rand_pois_new

# ----- ----- Funzioni ----- -----

def controllo_arg() :
    if len(sys.argv) < 2 :
        print("Passare a linea di comando il nome del file (compresa l'estensione) e il numero di eventi da generare.\n")
        sys.exit()
        
# ----- ----- Main ----- -----

def main () :

    controllo_arg()
    
    N = int(sys.argv[1])    # numero di eventi pseudocasuali
    
    t_0 = 3.1       # parametro decadimento
    t_m = 15.       # tempo misura
    
    sample = []
    for _ in range (N) :
        sample.append(rand_pois_new(t_m, t_0))

    Nbin = sturges(N) + 18      # il 18 l'ho messo io a mano 
    
    bin_edges = np.linspace(0., 2*t_m, Nbin)         # Regola la dimensione dei bin e Nbin = numero di bin
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample, bins = bin_edges, color = 'orange')  # Spesso conviene usare bins = 'auto' evitando di scrivere la linea di codice con bin_edges, per farlo però bisogna importare numpy
    ax.set_title ('Istogramma poissoniana', size = 14)
    ax.set_xlabel ('t (s)')
    ax.set_ylabel ('Conteggi')
    ax.grid ()          #se voglio la griglia
    
    plt.savefig ('es7_2.png')
    plt.show ()                     #da mettere rigorosamente alla fine
    
if __name__ == "__main__" :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- ----- ----- 


# Esercizio 3
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
    
    controllo_arg()
    
    N = int(sys.argv[1])        # numero di eventi pseudocasuali da generare
    mean = float(sys.argv[2])   # media da passare
    
    sample = []
    for _ in range (N) :
        sample.append(rand_pois(mean))
    #print(sample)                              # solo per fare un controllo
    
    # stampa dei momenti della distribuzione
    print("La media della distribuzione è: ", media(sample))
    print("La varianca della distribuzione è: ", varianza(sample))
    print("La deviazione standard è: ", dev_std(sample))
    print("La deviazione standard della media è: ", dev_std_media(sample))

    Nbin = sturges(N)
    
    #print(Nbin, max(sample))           # era solo per fare un controllo
    
    bin_edges = np.linspace(-0.5, max(sample) + 0.5, max(sample) + 1)
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
