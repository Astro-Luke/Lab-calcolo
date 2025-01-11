'''
Utilizzare due grafici a dispersione per confrontare l'evoluzione della deviazione standard della media calcolata per ogni singolo 
giocattolo con la al deviazione standard del campione di medie variare del numero di eventi generati in un singolo esperimento 
con un giocattolo.
'''

import numpy as np
import matplotlib.pyplot as plt
import sys

from library import dev_std_media, rand_range, media, dev_std

# ----- Function -----

def controllo_arg() :
    if len(sys.argv) != 3 :
        print("Inserire il nome del file (compresa l'estensione), il numero di eventi ed il numero di esperimenti.\n")
        sys.exit()

# ----- Main -----

def main () :
    
    #controllo_arg ()
    
    x_min = -10.
    x_max = 10.
    
    N_evet = 100 #int (sys.argv[1])
    N_toy = 1000 #int (sys.argv[2])

    std_lista = []
    std_N_toys = []

    for i in range (10, N_evet) :
        N_toy_media = []
        for k in range (N_toy) :
            lista_val_rang_range = []           # conterrà i valori generati con la range range
            for j in range (1, i) :
                val = rand_range (x_min, x_max)
                lista_val_rang_range.append(val)
            mean = media (lista_val_rang_range)
            std = dev_std_media (lista_val_rang_range)
            if (k == 0) :
                std_lista.append(std)
            N_toy_media.append(mean)
        std_N_toys.append(dev_std(N_toy_media))

    fig, ax = plt.subplots ()
    ax.set_title ('sigma of the means', size=14)
    ax.set_xlabel ('number of events')
    ax.set_ylabel ('sigma of the mean')
    ax.plot(range(10, N_evet), std_lista, label = 'dev standard della media singolo toy')
    ax.plot(range(10, N_evet), std_N_toys, label = "deviazione standard della media con N toys")
    ax.set_xscale ('log')
    ax.legend ()
    ax.grid ()

    plt.savefig ("es8_4.png")
    plt.show ()
    
if __name__ == "__main__" :
    main ()
