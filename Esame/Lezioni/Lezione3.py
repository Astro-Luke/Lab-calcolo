
# ----- ----- ----- ----- Lezione 3 ----- ----- ----- -----

# Esercizio 1
'''
Crea un istogramma unidimensionale riempito con 5 valori e salva l'immagine dell'istogramma in un pngfile
'''

import matplotlib.pyplot as plt

if __name__ == '__main__' :
    lista_val = [1, 2, 2, 3, 5]
    
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (lista_val, color = 'orange')
    ax.set_title("Histogram", size = 14)
    ax.set_xlabel("samples")
    ax.set_ytitle("counter")
    
    plt.savefig('grafico es3_1.png')
    plt.show()


# ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 2
'''
Leggere il file di testo eventi_unif.txt :
Visualizza sullo schermo i primi 10 elementi positivi.
Conta il numero di eventi contenuti nel file.
Determina i valori minimo e massimo tra i numeri salvati nel file.
'''

import matplotlib.pyplot as plt
import sys
import numpy as np

def controllo_arg() :
    if len(sys.argv) < 2 :
        print("Manca il nome del file da passare a linea di comando.\n")
        sys.exit()

'''
def lettura_file () :
    with open(sys.argv[1]) as file :
        sample = [x for x in file.readlines()]
    print("Elementi nel file: ", len(sample))
'''


#----- MAIN -----

def main() :
    
    # Verifico che siano stati passati tutti gli argomenti
    controllo_arg()

    # Leggo il file
    with open(sys.argv[1]) as file :
        sample = [x for x in file.readlines()]
    print("Elementi nel file: ", len(sample))

    primi = []
    for i in range (0, 10) :
        primi.append(sample[i])
    print("I primi 10 numeri sono: ", primi)

    massimo = max(sample)
    minimo = min(sample)
    
    print("\nMassimo del campione: ", massimo, "\nMinimo del campione: ", minimo)

if __name__ == '__main__' :
    main()


# ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 3
'''
Leggi il file di testo eventi_gauss.txt:
Riempi un istogramma con i primi N numeri contenuti nel file, dove N è un parametro della riga di comando durante l'esecuzione del programma.
Selezionare l'intervallo di definizione dell'istogramma e il suo numero di bin in base ai numeri da rappresentare.
'''

import sys
from math import ceil
import matplotlib.pyplot as plt
import numpy as np

def sturges (N_eventi) :
    return ceil (1 + np.log2 (N_eventi))


def main () :
    
    if len(sys.argv) < 3 :
        print("Mancano degli argomenti da passare a linea di comando.\n")
        sys.exit()
    
    N = int(sys.argv[2])
    
    with open(sys.argv[1]) as file :
        sample = [float(x) for x in file.readlines()]           # Qui il casting è obbligatorio!!!
    print("Numero di elmenti nel file: ", len(sample))
    
    primi_num = sample[:N]
    x_min = min(primi_num)
    x_max = max(primi_num)
    
    Nbin = sturges (N)
    
    bin_edges = np.linspace(x_min, x_max, Nbin)
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (primi_num, bins=bin_edges ,color = 'orange')
    ax.set_title ('Istogramma', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.grid ()          #se voglio la griglia
    
    plt.savefig ('grafico es3_3.png')
    plt.show ()                      #da mettere rigorosamente dopo il savefig
    
    
if __name__ == '__main__' :
    main ()


# ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 5
'''
Leggi il file di testo eventi_unif.txt:
Calcola la media dei numeri nel file di testo.
Calcola la varianza dei numeri nel file di testo.
Calcola la deviazione standard dei numeri nel file di testo.
Calcola la deviazione standard dalla media dei numeri nel file di testo.
'''

import sys
from math import sqrt           # se facessi solo import math dovrei chiamare ogni volta òa radice con math.sqrt()

def controllo_arg() :
    if len(sys.argv) < 2 :
        print("Manca il nome del file da passare a linea di comando.\n")
        sys.exit()


def media (sample) :
    mean = sum(sample) / len(sample)
    return mean


def varianza (sample) :
    var = 0
    somma_quadrata = 0
    for elem in sample :
        somma_quadrata = somma_quadrata + (elem - media(sample))**2
    var = somma_quadrata / (len(sample))
    return var


def varianza_bessel (sample) :
    var = 0
    somma_quadrata = 0
    for elem in sample :
        somma_quadrata = somma_quadrata + (elem - media(sample))**2
    var = somma_quadrata / (len(sample) -1)
    return var


def dev_std (sample) :
     sigma = sqrt(varianza(sample))
     return sigma


def dev_std_media (sample) :
    deviazione_media = dev_std(sample) / sqrt(len(sample))
    return deviazione_media

# ----- MAIN -----

def main () :

    controllo_arg()

    with open(sys.argv[1]) as file :
        sample = [float(x) for x in file.readlines()]           # Qui il casting è obbligatorio!!!
    print("Numero di elmenti nel file: ", len(sample))
    
    print("La media dl campione è: ", media(sample))
    print("La varianza del campione è: ", varianza(sample))
    print("La deviazione standard del campione è: ", dev_std(sample))
    print("La deviazione standard della media: ", dev_std_media(sample))
    
if __name__ == '__main__' :
    main()


# ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 6
'''
Scrivi una pythonlibreria che, dato il nome di un file di testo contenente un campione di eventi come input, 
sia in grado di leggere il campione e salvarlo in un array numpy, quindi calcolarne la media, la varianza, 
la deviazione standard, la deviazione standard dalla media, 
visualizzare il campione in un istogramma con un intervallo di definizione scelto in modo appropriato e un numero di bin. 
Scrivi un programma di test per la libreria creata.
'''

import sys
from library import media, varianza, dev_std, dev_std_media
import numpy as np

def controllo_arg () :
    if len (sys.argv) != 2 :       
        #Super NB! Nel main inserirò una variabile int chiamata num_arg. prima di chiamare la funzione 
        #(Ad esempio: num_arg = int(3) se gli argomenti da passare a linea di comando sono 3 (nome del file compreso) )
        print("Inserire il nome del file (compresa l'estensione) e ... .\n")
        sys.exit()

def main () :

    controllo_arg ()
    
    with open(sys.argv[1]) as file :
        sample = np.array([float(x) for x in file.readlines()])       # Qui il casting è obbligatorio!!!
    print("Numero di elmenti nel file: ", len(sample))
    
    print("La media è: ", media(sample))
    print("La varianza è: ", varianza(sample))
    print("La deviazione standard è: ", dev_std(sample))
    print("La deviazione standard sulla media è: ", dev_std_media(sample))

if __name__ == '__main__' :
    main()


# ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 8
'''
Scrivi un programma Python per disegnare una distribuzione esponenziale e la sua funzione cumulativa
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon

def main():
    x = 5.0
    esponenziale = expon(0., x)               # expon deriva da scipy.stats
    x_axis = np.linspace(0, 10, 100)

    # Creazione di una figura con due subplot affiancati
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 1 riga, 2 colonne
    
    # Primo grafico: PDF
    axes[0].plot(x_axis, esponenziale.pdf(x_axis), label="PDF")   # con esponenziale.pdf uso la funzione predefinita nella libreria
    axes[0].legend()
    axes[0].grid()
    axes[0].set_title("Funzione di densità di probabilità (PDF)")

    # Secondo grafico: CDF
    axes[1].plot(x_axis, esponenziale.cdf(x_axis), label="CDF")     # con esponenziale.cdf uso la funzione predefinita nella libreria
    axes[1].legend()
    axes[1].grid()
    axes[1].set_title("Funzione di distribuzione cumulativa (CDF)")

    plt.tight_layout()  # AQuesto aggiusta automaticamente spazi tra i grafici
    plt.savefig("grafici_PDF_CDF_affiancati.png")
    plt.show()
    
if __name__ == '__main__':
    main()


# ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 9
'''
Use the Python scipy.stat.norm object to determine the area of a normal distribution of its tails outside the range 
included within an interval of 1, 2, 3, 4, and 5standard deviations around its mean.
'''

import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
#import matplotlib.pyplot as plt

def main():
    mean = 0.
    sigma = 1.
    
    # Fattori di deviazione standard
    deviations = [1, 2, 3, 4, 5]
    
    for dev in deviations :
        inf = mean - dev * sigma        # Questo è il limite inferiore dell'intergrazione
        upper = mean + dev * sigma      # limite superiore dell'integrazione
        
        # Calcolo dell'area nelle code
        tail_area = 1 - quad(norm(loc=mean, scale=sigma).pdf, inf, upper)[0]  # faccio 1- ... perchè quello che voglio è l'area delle code e la distribuzione è già normalizzata
        print(f"Deviazioni: {dev}, Area delle code: {tail_area:.5f}")

if __name__ == '__main__':
    main()
