
# ----- ----- ----- ----- Lezione 8 ----- ----- ----- ----- 

# Esercizio 2
'''
Scrivi un programma che, dato un numero N_max, generi N_toys esperimenti giocattolo, ciascuno contenente un campione di N_max eventi 
che seguono una distribuzione scelta, e ne calcoli la media.
Aggiungere al programma precedente un istogramma che visualizzi la distribuzione delle medie tra gli esperimenti giocattolo.
'''

import sys
import matplotlib.pyplot as plt
import numpy as np

from library import media, rand_TAC, sturges

# ----- ----- Funzioni ----- -----

# Funzione di controllo degli argomenti da modificare di volta in volta nel main
def controllo_arg() :
    if len(sys.argv) != 3 :
        print("Inserire il nome del file (compresa l'estensione), numero massimo di numeri pseudocasuali da generare e numero di toy experiment.\n")
        sys.exit()

# ----- ----- Main ----- -----

def main () :
    
    #controllo_arg ()
    
    N_max = 1000 #int(sys.argv[1])
    N_toys = 100 #int(sys.argv[2])
    
    x_min = -10.
    x_max = 10.
    y_max = 1.
    
    array_mean = []     # lista che conterrà le medie
    
    for j in range (N_toys) :       # primo ciclo sugli experimenti
        sample = []                 # creo ogni volta una lista per ogni esperimento
        for i in range (N_max) :
            value = rand_TAC (np.sin, x_min, x_max, y_max)
            sample.append(value)                                # riempio la lista
        array_mean.append(media (sample))                       # riempio la lista delle medie con le medie dei singoli sample
    
    Nbin = sturges(N_toys)
    print("Numero di bin generati: ", Nbin)     # è solo un controllo
    
    bin_edges = np.linspace(min(array_mean), max(array_mean), Nbin)         # Regola la dimensione dei bin, come minimo ho preso il minimo dell'array e lo stesso per il massimo
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (array_mean, bins = bin_edges, color = 'orange')
    ax.set_title ('Istogramma delle medie', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()          #se voglio la griglia
    
    plt.savefig ('es8_2.png')
    plt.show ()
    
if __name__ == '__main__' :
    main()


# ----- ----- ----- ----- ----- ----- ----- ----- 


# Esercizio 4
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


# ----- ----- ----- ----- ----- ----- ----- ----- 


# Esercizio 5
'''
Si implementi il metodo di integrazione hit-or-miss
con la funzione di esempio f(x) = sin (x).
  * Si scriva l'algoritmo che calcola l'integrale come una funzione esterna al programma `main`,
    facendo in modo che prenda come parametri di ingresso,
    oltre agli estremi lungo l'asse x e l'asse y,
    anche il numero di punti pseudo-casuali da generare.
  * Si faccia in modo che l'algoritmo ritorni due elementi:
    il primo elemento sia il valore dell'integrale,
    il secondo sia la sua incertezza.
'''

import sys
import numpy as np
#import matplotlib.pyplot as plt

from library import integral_HOM

# ----- Function -----

def controllo_arg() :
    if len(sys.argv) != 6 :     # perchè sono 6 argomenti che gli passo in totale
        print("Inserire il nome del file (compresa l'estensione), il numero di punti da generare,, minimo e massimo dell'asse x e minimo e massimo dell'asse y.\n")
        sys.exit()

# ----- Main -----

def main () :
    
    N_punti = 1000 #int(sys.argv[1])      # numero di punti (con più ne metto con più sono preciso)
    x_min = 0. #float (sys.argv[2])     # estremi sull'asse x
    x_max = 3.14 #float (sys.argv[3])
    y_min = 0. #float (sys.argv[4])     # estremi sull'asse y
    y_max = 1. #float (sys.argv[5])
    
    integral, integral_incertezza = integral_HOM (np.sin, x_min, x_max, y_min, y_max, N_punti)
    
    print("Il valore dell'integrale è: ", integral, "+/-", integral_incertezza)
    
    '''
    non sapevo che nel caso la funzione ritornasse due valori, questi possono essere spezzati come fatto a riga 36. il primo prende il primo valore ritornato, il secondo parametro prende il secondo ritornato
    '''
    
if __name__ == "__main__" :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- 


# Esercizio 6
'''
Si inserisca il calcolo dell'integrale dell'esercizio precedente in un ciclo che,
al variare del numero *N* di punti generati, mostri il valore dell'integrale
e della sua incertezza.
  * Si utilizzi uno scatter plot per disegnare gli andamenti del valore dell'integrale
    e della sua incertezza, al variare di *N* con ragione logaritmica.
'''

import sys
import numpy as np
import matplotlib.pyplot as plt
import time

from library import integral_HOM

# ----- Function -----

def controllo_arg() :
    if len(sys.argv) != 2 :
        print("Inserire il nome del file (compresa l'estensione) ed il numero di punti da generare per l'integrazione.\n")
        sys.exit()

# ----- Main -----

def main () :
    
    #controllo_arg ()
    
    N_punti = 10 # int (sys.argv[1])         # punti sotto la curva
    N_punti_max = 1000000
    x_min = 0.          # scelgo tra 0 e pi greco perchè so quanto vale quindi posso vedere al volo se è corretto
    x_max = np.pi
    y_min = 0.
    y_max = 1.          # tanto seno e coseno hanno ampiezza 1
    
    valore = []         # lista che conterrà il valore dell'area
    errori = []         # lista che conterrà l'errore sull'area
    N_lista = []        # lista degli N per il grafico
    
    t_start = time.time ()                      # non è richiesto ma volevo confrontare l'Hit-Or-Miss e Monte Carlo
    
    while (N_punti < N_punti_max) :
        if (N_punti < 5) :                      # se ho meno di 5 punti ne chiedo di inserirne di più
            print("Hai immesso troppi pochi punti, inseriscine un numero compreso tra 10 e 100000")
            sys.exit()
        elif (N_punti > N_punti_max) :          # se ne metto troppi ne chiedo di meno
            print("Hai inserito più punti del massimo consentito dal programma")
            sys.exit()
        
        # calcolo dell'integrale
        val_integral, val_incertezza = integral_HOM (np.sin, x_min, x_max, y_min, y_max, N_punti)
        print ("Valore dell'integrale con", N_punti, " punti :\n", val_integral, "+/-", val_incertezza, "\n")
        
        # riempimento delle liste
        valore.append (val_integral)
        errori.append (val_incertezza)
        N_lista.append (N_punti)
        
        # incremento i numeri di 10 se no ciaonee
        N_punti = N_punti * 2

    t_end = time.time ()    # stoppo il tempo

    # stampa del tempo
    print(f"Tempo impiegato per eseguire: {(t_end - t_start):.2f} secondi.")

    # Creazione del grafico
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))  # 1 riga, 1 colonna

    ax.set_title ("Grafico integrali Hit Or Miss", fontsize = 14)
    ax.set_xlabel ("Punti creati")
    ax.set_ylabel ("Valore dell'integrale con errori")
    ax.set_xscale("log")
    ax.errorbar (N_lista, valore, yerr = errori,            # si mi sono divertito a giocare con il grafico
        markersize = 5,
        fmt = 'o',
        linestyle = '--',
        ecolor = 'red',
        elinewidth = 1.5,
        capsize = 5,
        capthick = 1.5,
        color = 'blue',
        label = "Andamento della precisione")

    ax.legend (fontsize = 10, loc = 'best')
    ax.grid (color = 'gray', linestyle = ':', linewidth = 0.5)

    plt.savefig ("es8_6.png")
    plt.show ()
    
    
if __name__ == "__main__" :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- 


# Esercizio 7
'''
Implementare il metodo di integrazione MC grezzo con la funzione di esempio f(x) = sin(x) .
Scrivere l'algoritmo che calcola l'integrale come una funzione esterna al mainprogramma, 
assicurandosi che accetti come parametri di input i limiti lungo l' asse x e il numero di punti pseudo-casuali da generare.
Assicuratevi che l'algoritmo restituisca un contenitore con due elementi: il primo elemento è il valore dell'integrale, 
il secondo è la sua incertezza.
'''

import numpy as np

from library import integrale_MonteCarlo

# ----- ----- Main ----- -----

def main () :

    N = 10000
    x_min = 0.
    x_max = np.pi

    value_integral, inc_integral = integrale_MonteCarlo (np.sin, x_min, x_max, N)       # di fatto è tutto qui dentro

    print("L'area dell'integrale è: ", value_integral, "+/-", inc_integral)

if __name__ == "__main__" :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- 


# Esercizio 8
'''
Inserire il calcolo dell'integrale dell'esercizio precedente in un ciclo che, al variare del numero N di punti generati, 
visualizzi il valore dell'integrale e la sua incertezza.
Rappresentare graficamente l'andamento del valore integrale e della sua incertezza al variare di N su scala logaritmica.
Sovrapponiamo questo comportamento a quello ottenuto completando l'esercizio 8.6.
'''

import matplotlib.pyplot as plt
import numpy as np
import sys
import time

from library import integrale_MonteCarlo

# ----- ----- Funzioni ----- -----

def controllo_arg() :
    if len(sys.argv) != 2 :
        print("Inserire il nome del file (compresa l'estensione) ed il numero di punti con cui generare l'integrale.\n")
        sys.exit()

# ----- ----- Main ----- ----- 

def main () :

    #controllo_arg ()

    N = 10 #int (sys.argv[1])
    N_max = 1000000

    x_min = 0.          # conosco il risultato tra 0 e pi greco quindi mi faccio furbo
    x_max = np.pi

    value = []          # creo le due liste che dovranno contenere i valori dell'integrale e degli errori
    errori = []
    lista_N = []        # lista degli N per 

    t_start = time.time()       # non è richiesto ma ero curioso
    while (N < N_max) :
        if (N < 5) :            # se sono sotto i 5 punti ne chiedo di più
            print("Inserisci più punti.")
            sys.exit ()

        elif (N > N_max) :      # se input ne passo di più di N_max -> mettere meno punti
            print("Hai inserito troppi punti.")
            sys.exit ()
        
        # calcolo dell'integrale
        val_int, err_int = integrale_MonteCarlo (np.sin, x_min, x_max, N)
        print ("Valore dell'integrale con", N, " punti :\n", val_int, "+/-", err_int, "\n")

        # riempimento delle liste
        value.append (val_int)
        errori.append (err_int)
        lista_N.append(N)
        N = N * 2           # incremento molt. per due per avere un range maggiore

    t_end = time.time()     # stoppo il tempo

    print(f"Tempo impiegato per eseguire: {(t_end - t_start):.2f} secondi.")

    # creazione del grafico
    fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))     # numero righe e colonne e dimensione figura

    ax.set_title ("Grafico integrale Montecarlo", fontsize = 14)
    ax.set_xlabel ("punti sottesi")
    ax.set_ylabel ("valore integrale con errori")
    ax.set_xscale("log")

    ax.errorbar (lista_N, value, xerr = 0.0, yerr = errori,                       # nell'ordine: valori x, valori y, errore sulla x, errori sulle y
        markersize = 5,                             # dimensione del punto
        fmt = 'o',                                  # tipo di marker (punto)
        color = 'blue',                             # colore della linea
        linestyle = '--',                           # tipo di linea
        ecolor = 'red',                             # colore della barra di errore
        elinewidth = 1.5,                           # spessore barre errori
        capsize = 5,                                # lunghezza cappello barre errori
        capthick = 1.5, 
        label = "Andamento della precisione")
    
    ax.legend (fontsize = 10, loc = 'best')
    ax.grid (color = 'gray', linestyle = ':', linewidth = 0.5)      # impostazioni della griglia

    plt.savefig ("es8_8.png")
    plt.show ()


if __name__ == "__main__" :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- 


# Esercizio 9
'''
Use the hit-or-miss method to estimate the integral underlying a Gaussian probability distribution 
with μ=0 and σ=1 within a generic interval [a,b].
Calculate the integral contained within the intervals [-kσ, kσ] as k varies from 1 to 5.
'''

import numpy as np
from library import integral_HOM

# ----- ----- Function ----- -----

# funzione gaussiana normalizzata
def gaussiana (x, mean, sigma) :
    return (1 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(-0.5 * ( (x - mean) / sigma )**2)

# ----- ----- main ----- -----

def main () :

    mean = 0.           # parametri della gaussiana
    sigma = 1.

    y_min = 0.              # per la funzione integral_HOM, in questo caso è normalizzata
    y_max = 1.
    N_punti = 100000

    # Funzione gaussiana con mean e sigma fissati
    gaussiana_fixed = lambda x: gaussiana(x, mean, sigma)

    # itero su k tra 1 e 5 (6 è escluso!)
    for k in range (1, 6) :
        integ_value, integ_error = integral_HOM(gaussiana_fixed, -k*sigma, k*sigma, y_min, y_max, N_punti)      # ritorno 2 valori: area ed errore
        print("Valore dell'integrale entro", k, "sigma: ", integ_value, "+/-", integ_error, "\n")
        k = k + 1

if __name__ == "__main__" :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- 


# Esercizio 9 (Versione 2)
'''
Use the hit-or-miss method to estimate the integral underlying a Gaussian probability distribution 
with μ=0 and σ=1 within a generic interval [a,b].
Calculate the integral contained within the intervals [-kσ, kσ] as k varies from 1 to 5.
'''

import numpy as np

from library import integral_HOM

# ----- ----- Function ----- -----

# funzione gaussiana normalizzata
def gaussiana (x) :
    return (1 / (np.sqrt(2 * np.pi))) * np.exp(-0.5 * (x)**2)

# ----- ----- main ----- -----

def main () :

    mean = 0.           # parametri della gaussiana
    sigma = 1.

    y_min = 0.              # per la funzione integral_HOM, in questo caso è normalizzata
    y_max = 1.
    N_punti = 100000

    # itero su k tra 1 e 5 (6 è escluso!)
    for k in range (1, 6) :
        integ_value, integ_error = integral_HOM (gaussiana, -k*sigma, k*sigma, y_min, y_max, N_punti)      # ritorno 2 valori: area ed errore
        print("Valore dell'integrale entro", k, "sigma: ", integ_value, "+/-", integ_error, "\n")
        k = k + 1

if __name__ == "__main__" :
    main ()