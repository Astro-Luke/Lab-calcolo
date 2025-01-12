
# ----- ----- ----- ----- Lezione 2 ----- ----- ----- -----

# Esercizio 1
'''
Crea array NumPy unidimensionali utilizzando diverse tecniche di generazione
'''

import numpy as np

def crea_array(N):
    # Creo un bell'array di soli zeri
    zero_array = np.zeros(N)

    # Creo un bel vettore di elementi vuoti
    vuoto_array = np.empty(N)

    # Creo un array di float a partire da una lista
    lista = [1.4, 83.0, 274.1, 843.9, 2378.4]
    float_array = np.array(lista)

    # Creo un array contenente tutti i numeri interi tra 0 e 25 con arange
    step = 1
    int_array = np.arange(0, 26, step)

    # Creo un array contenete numeri double con linspace
    doub_array = np.linspace(0., 26., 20)

    print("Array di zeri: ", zero_array)
    print("Array vuoto: ", vuoto_array)
    print("Array di float: ", float_array, "\na partire dalla lista: ", lista)
    print("Array di interi con arange: ", int_array)
    print("Array di double con linspace: ", doub_array)

if __name__ == '__main__':
    
    crea_array(10)          # 10 è il numero di elementi per gli array vuoto e di zeri


# ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 2
'''
Crea un array NumPy unidimensionale contenente una sequenza di numeri interi da 1 a 100
Partendo da questo, crea un array NumPy unidimensionale contenente in ogni voce la somma dei numeri interi da 1 fino all'indice di quella voce
'''

import numpy as np

def somma_voci () :

    array = np.arange (1, 101)                   # Genero l'aray di 100 numeri tra 1 e 101
    somma_array = np.cumsum (array)                 # Cumsum fa la somma di tutti i numeri precedenti
    
    print ("Array con somme voci: ", somma_array)
    return
    
if __name__ == '__main__' :

    somma_voci()


# ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 3
'''
Crea un array unidimensionale contenente la sequenza dei primi 50 numeri naturali pari
Crea un array unidimensionale contenente la sequenza dei primi 50 numeri naturali dispari
Crea un array unidimensionale contenente la somma elemento per elemento dei due array precedenti
'''

import numpy as np

def seq_naturali (N) :
    
    # Gnero due array per ospitare i numeri pari e dispari (attenzione alla lunghezza dei vettori che deve essere la stessa e, poichè uno parte da 0 (pari) l'altro parte da 1)
    array_pari = np.arange(0, N, 2)
    array_disp = np.arange(1, N, 2)
    
    print("Sequenza numeri pari:", array_pari)
    print("Sequenza numeri dispari:", array_disp)

    return array_pari, array_disp                       # questa riga va alla fine dela funzione seq_naturali se no esci dalla funzione!

def somma (array_pari, array_disp) :
    array_somma = array_pari + array_disp
    print("Array con la somma dell'elemento i-esimo dell'array pari con l'elemento i-esimo dell'array dispari", array_somma)
    #return array_somma


if __name__ == '__main__' :
    
    N = int(100)
    
    array_pari, array_disp = seq_naturali(N)   # li metto uno dopo l'altro se no me li scrive due volte
    
    somma(array_pari, array_disp)


# ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 4
'''
All'interno di un programma Python, l'ora corrente può essere ottenuta con la timelibreria:
import time
time_snapshot = time.time ()
print (time_snapshot)

Confronta le prestazioni temporali delle operazioni elemento per elemento eseguite tra due elenchi rispetto alla stessa operazione eseguita in forma compatta tra due array NumPy

A partire da quale dimensione le differenze cominciano a essere significative?
'''

import time
import numpy as np

# Prodotto tra elementi usando le funzionalità di numpy
def prodotto_numpy (N) :                               # N dimensione dell'array
    
    array_uno = np.arange(0, N, 1)
    array_due = np.arange(0, N, 1)

    prodotto_array = array_uno * array_due
    

# Prodotto di elementi usando le liste
def prodotto_liste (N) :

    lista_uno = list(range(N))
    lista_due = list(range(N))
    
    for a, b in zip(lista_uno, lista_due) :
        prodotto_liste = a * b

# ---------- MAIN -----------

if __name__ == '__main__' :
    
    N = 10000000
    
    start_np = time.time()
    prodotto_numpy(N)
    end_np = time.time()
    
    start_list = time.time()
    prodotto_liste(N)
    end_list = time.time()
    
    print("Con", N, "elementi:\n")
    print(f"Tempo impiegato per array numpy: {(end_np - start_np):.3f} secondi.")
    print(f"Tempo impiegato per liste: {(end_list - start_list):.3f} secondi.\n")


# ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 7
'''
Scrivere una libreria Python contenente funzioni per eseguire le seguenti operazioni sugli array NumPy 1D:
Calcola la media dei suoi elementi
Calcola la varianza dei suoi elementi
Calcola la deviazione standard dei suoi elementi
Calcola la deviazione standard dalla media dei suoi elementi
'''

import numpy as np

def media (array) :
    somma = array.sum()
    mean = somma/len(array)
    return mean
    
def varianza (array) :
    mean = media(array)
    somma_quadrata = 0
    for i in array :
        somma_quadrata = somma_quadrata + (i - mean)**2
    return somma_quadrata/len(array)
    
def varianza_bessel (array) :
    mean = media(array)
    somma_quadrata = 0
    for i in array :
        somma_quadrata = somma_quadrata + (i - mean)**2
    return somma_quadrata/ (len(array) - 1)

def dev_standard (array) :
    var = (varianza(array))**0.5
    return var
    
def dev_standard_media (array) :
    sigma_mean = dev_standard(array)/ np.sqrt((len(array)))
    return sigma_mean
    
    
# ----- MAIN ------
if __name__ == '__main__' :

    array_prova = np.array([2.8, 12, 98.1, 126.1, 73.7, 98.3, 72.1, 6, 97.3, 83.9, 23.8])

    print("La media è: ", media(array_prova))
    print("La varianza è: ", varianza(array_prova))
    print("La deviazione standard è: ", dev_standard(array_prova))
    print("La deviazione standard della media è: ", dev_standard_media(array_prova))
