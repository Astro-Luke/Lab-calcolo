
# ----- ----- ----- ----- ----- Lezione 6 ----- ----- ----- ----- -----

# Esercizio 1
'''
Determinare lo zero della funzione g(x) = cos(x) utilizzando il metodo di bisezione nell'intervallo (0, 4).
Quali controlli sono stati omessi nell'implementazione dell'algoritmo descritto nel testo della lezione che potrebbero accelerare il risultato?
'''

import matplotlib.pyplot as plt
from library import bisezione
import numpy as np
import sys
import time

# ----- ----- Funzioni ----- -----

def controllo_arg() :
    if len(sys.argv) != 3 :
        print("Manca il nome del file da passare a linea di comando.\n")
        sys.exit()

# ----- ----- Main ----- -----

def main () :

    x_min = float(sys.argv[1])      # Passaggio argomenti a linea di comando
    x_max = float(sys.argv[2])

    controllo_arg ()                # Controllo che tutti gli argomenti siano stati inseriti
    
    t_start = time.time()           # inizio a misurare il tempo
    zero = bisezione(np.cos, x_min, x_max)      # cerco lo zero con la funzione bisezione
    t_end = time.time()             # fermo il tempo
    
    print("Lo zero della funzione si trova nel punto: (",zero,", 0)")
    print(f"Tempo impiegato per eseguire: {(t_end - t_start):.2f} secondi.")    #Stampa del tempo impiegato a cercare lo zero
    
    
    # Impostazini e creazione grafico
    x_axis = np.linspace(x_min, x_max, 100)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))  # 1 riga, 1 colonna
    axes.plot(x_axis, np.cos(x_axis), label="funzione")   # con esponenziale.pdf uso la funzione predefinita nella libreria scipy
    axes.legend()
    axes.grid()
    axes.set_title("Zero funzione cos(x)")
    plt.plot(zero, 0., marker="o", color="red")
    plt.savefig("es6_1.png")
    plt.show()
    
if __name__ == '__main__' :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# Esercizio 2
'''
Eseguire l'esercizio precedente utilizzando una funzione ricorsiva.
Quale delle due implementazioni è più veloce?
'''

import matplotlib.pyplot as plt
from library import bisezione_ric
import numpy as np
import sys
import time

# ----- ----- Funzioni ----- -----

def controllo_arg() :
    if len(sys.argv) != 3 :
        print("Manca il nome del file da passare a linea di comando.\n")
        sys.exit()

# ----- ----- Main ----- -----

def main () :

    x_min = float(sys.argv[1])      # Passo a linea di comando gli argomenti
    x_max = float(sys.argv[2])

    controllo_arg()                 # controllo che siano stati passati correttamente tutti gli argomenti

    t_start = time.time()           # inizio a misurare il tempo
    zero = bisezione_ric(np.cos, x_min, x_max)      # cerco lo zero con la funzione bisezione ricorsiva
    t_end = time.time()             # fermo il tempo
    
    print("Lo zero della funzione si trova nel punto: (",zero,", 0)")
    print(f"Tempo impiegato per eseguire: {(t_end - t_start):.2f} secondi.")    # stampo il tempo impiegato a cercare lo zero
    
    # Creazione ed impostazioni del grafico
    x_axis = np.linspace(x_min, x_max, 100)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))  # 1 riga, 1 colonna
    axes.plot(x_axis, np.cos(x_axis), label="funzione")   # con esponenziale.pdf uso la funzione predefinita nella libreria scipy
    axes.legend()
    axes.grid()
    axes.set_title("Zero coseno con ricorsione")
    plt.plot(zero, 0., marker="o", color="red")
    plt.savefig("es6_2.png")
    plt.show()
    
if __name__ == '__main__' :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 3
'''
Implementare una funzione che calcoli il fattoriale di un numero utilizzando una funzione ricorsiva.
'''

import sys
from library import fattoriale

# ----- ----- Funzioni ----- -----

# Funzione di controllo degli argomenti da modificare di volta in volta nel main
def controllo_arg() :
    if len(sys.argv) < 2 :
        print("Passare a linea di comando il nome del file ed il valore di cui si vuole calcolare il fattoriale.\n")
        sys.exit()

# ----- ----- Main ----- -----

def main () :
    
    controllo_arg()
    
    N = int (sys.argv[1])
    
    #x = fattoriale(N)
    print("Valore del fattoriale: ", fattoriale(N))
    
if __name__ == '__main__' :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----

# Esercizio 4
'''
Determinare il minimo della funzione g(x) = x^2 + 7,3x + 4 utilizzando il metodo di ricerca della sezione aurea nell'intervallo (-10, 10).
'''

import numpy as np
import matplotlib.pyplot as plt
import time

from library import sezione_aurea_min

# ----- Funzioni -----

def polinomio (x) :
    a = 1.
    b = 7.3
    c = 4.
    return a * (x**2) + b * x + c

def main () :
    
    x_min = -10.
    x_max = 10.
    
    t_start = time.time()
    minimo = sezione_aurea_min(polinomio, x_min, x_max)
    t_end = time.time()
    
    print("Il minimo della funzione è: ", minimo)
    print(f"Tempo impiegato per eseguire: {1000*(t_end - t_start):.6f} millisecondi.")
    
    x_axis = np.linspace(x_min, x_max, 100)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))  # 1 riga, 1 colonna
    axes.plot(x_axis, polinomio(x_axis), label="funzione")   # con esponenziale.pdf uso la funzione predefinita nella libreria scipy
    axes.legend()
    axes.grid()
    axes.set_title("Minimo funzione polinomio")
    plt.plot(minimo, polinomio(minimo), marker="o", color="red")
    plt.savefig("es6_4.png")
    plt.show()
    
if __name__ == '__main__' :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 5
'''
Eseguire l'esercizio precedente utilizzando una funzione ricorsiva.
Quale delle due implementazioni è più veloce?
'''

import numpy as np
import matplotlib.pyplot as plt
import time

from library import sezione_aurea_ric_min

# ----- Funzioni -----

def polinomio (x) :
    a = 1.
    b = 7.3
    c = 4.
    return a * (x**2) + b * x + c

def main () :
    
    x_min = -10.
    x_max = 10.
    
    t_start = time.time()
    minimo = sezione_aurea_ric_min(polinomio, x_min, x_max)
    t_end = time.time()
    
    print("Il minimo della funzione è: ", minimo)
    print(f"Tempo impiegato per eseguire: {1000*(t_end - t_start):.6f} millisecondi.")
    
    x_axis = np.linspace(x_min, x_max, 100)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))  # 1 riga, 1 colonna
    axes.plot(x_axis, polinomio(x_axis), label="funzione")   # con esponenziale.pdf uso la funzione predefinita nella libreria scipy
    axes.legend()
    axes.grid()
    axes.set_title("Minimo funzione polinomio")
    plt.plot(minimo, polinomio(minimo), marker="o", color="red")
    plt.savefig("es6_5.png")
    plt.show()
    
if __name__ == '__main__' :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 6
'''
Completa i due esercizi precedenti trovando il massimo di una funzione scelta.
'''

import numpy as np
import matplotlib.pyplot as plt
import time

from library import sezione_aurea_max

# ----- Funzioni -----

def function (x) :
    return np.exp(-x**2)

def main () :
    
    x_min = -5.
    x_max = 5.
    
    t_start = time.time()
    massimo = sezione_aurea_max(function, x_min, x_max)
    t_end = time.time()
    
    print("Il massimo della funzione è: ", massimo)
    print(f"Tempo impiegato per eseguire: {1000*(t_end - t_start):.6f} millisecondi.")
    
    x_axis = np.linspace(x_min, x_max, 100)
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))  # 1 riga, 1 colonna
    axes.plot(x_axis, function(x_axis), label = "funzione")   # con esponenziale.pdf uso la funzione predefinita nella libreria scipy
    axes.legend()
    axes.grid()
    axes.set_title("Massimo funzione e^(-x^2)")
    plt.plot(massimo, function(massimo), marker="o", color="red")
    plt.savefig("es6_6.png")
    plt.show()
    
if __name__ == '__main__' :
    main ()

