import matplotlib.pyplot as plt
import numpy as np


from lib import legge_hubble, esegui_fit

def main () :
    
    # parte 1
    redshift, distanza, errore = np.loadtxt ("SuperNovae.txt", unpack=True)
    
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))   # 1 riga, 1 colonna
    ax.grid ()
    ax.set_title ("Plot dei dati della costante di Hubble")
    ax.legend ()
    ax.set_xlabel ("Redshift")
    ax.set_ylabel ("Distanza (Mpc)")
    ax.errorbar (redshift, distanza, xerr = 0.0, yerr = errore,
        markersize = 3,                             # dimensione del punto
        fmt = 'o',                                  # tipo di marker (punto)
        color = 'blue',                             # colore della linea
        ecolor = 'red',                             # colore della barra di error
        )   

    diz_par = {
        "H": 1.,    
    }

    diz_result = esegui_fit (redshift, distanza, errore, diz_par, legge_hubble)

    print ("ok")

    plt.savefig ("Plot_e_fit_Hubble.png")
    plt.show ()  

if __name__ == "__main__" :
    main ()