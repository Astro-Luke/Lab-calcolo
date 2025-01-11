
# ----- ----- ----- ----- ----- Lezione 1 ----- ----- ----- ----- -----

# Esercizio 2
'''
Scrivi un programma che, dati i tre lati di un triangolo, determini se il triangolo è acutangolo, rettangolo o ottuso.
'''

import sys

def det_triangolo (a, b, c) :

    # Riordino i valori a, b, c in un elenco per avere c come lato maggiore
    a, b, c = sorted([a, b, c])
        
    if (a == b and b == c) :
        print("Il triangolo è equilatero.\n")
    elif (a**2 + b**2 == c**2) :                # Teorema di Pitagora
        print("Il triangolo è rettangolo.\n")
    elif (a == b or a == c or b == c) :
        print("Il triangolo è isoscele.\n")
    else :
        print("Il triangolo è scaleno.\n")
    
if __name__ == '__main__' :
    
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    c = float(sys.argv[3])

    det_triangolo(a, b, c)


# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 3
'''
Scrivi un programma che, utilizzando un whileciclo, restituisca la sequenza di Fibonacci fino all'n-esimo termine e la memorizzi in un file python list.

N.B.  Di fatto questo programma risolve anche l'esercizio 1.5'''

import sys

def Fibonacci (n) :

    # Se il numero fino a cui voglio contare è minore o uguale a zero faccio in modo che il programma mi restituisca una lista vuota.
    if n <= 0 :
        return []
    
    # Se il numero fino a cui voglio contare è 1 allora la lista avrà solo il numero 1
    elif n == 1 :
        return [1]
    
    # definisco una lista iniziale con almeno due numeri (i primi due)
    lista_fibo = [0, 1]
    contatore = len(lista_fibo)             # Conta il numero di elementi nella lista
    
    while (contatore < n) :
        prossimo_num = lista_fibo[contatore-1] + lista_fibo[contatore-2]        # Somma il numero precedente all'i-esimo al i-esimo meno due
        lista_fibo.append(prossimo_num)         # Aggiunge il prossimo numero alla fine della lista
        contatore = contatore + 1       # Incrementa il numero di elementi nella lista alla fine di ofni iterazione
    return lista_fibo

if __name__ == '__main__' :

    n = int(sys.argv[1])        # Passo a linea di comando il numero di iterazioni
    
    print("Sequenza di fibonacci fino al numero", n, ":", Fibonacci(n))


# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 4
'''
Scrivi un programma che, utilizzando un forciclo, restituisca la sequenza di Fibonacci fino all'n-esimo termine e la memorizzi in un file Python dictionary, dove keyrappresenta l'indice di ciascun elemento e valueil suo valore effettivo.
'''

import sys

def Fibonacci (n) :
    if n == 0 :
        return {}
    
    elif n == 1 :
        return {'a1': 0}
    
    dizionario_fibonacci = {'a1': 0, 'a2': 1}                       # si veda come costruire un dizionario
    elem = len(dizionario_fibonacci)
    
    for i in range (2, n) :
        prossimo_num = dizionario_fibonacci[f'a{i}'] + dizionario_fibonacci[f'a{i-1}']          #si indica con f'...'
        dizionario_fibonacci[f'a{i+1}'] = prossimo_num                                      # attenzione all'assegnazione
    return dizionario_fibonacci
    
    
if __name__ == '__main__' :
    
    n = int(sys.argv[1])
    
    print("Sequenza di Fibonacci fino al numero", n, ":", Fibonacci(n))


# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 7
'''
Scrivi un programma Python che determini la soluzione delle equazioni del secondo ordine.
'''

import sys
import numpy as np

def soluz_eq_secondo_grado (a, b, c) :

    if a == 0 :                                                     # Non sarebbe una eq. di secondo grado e dividerei per 0 (no buono)
        return print("Non è una equazione di econdo grado")
    
    else :
        delta = (b**2) - (4*a*c)                                        # Calcolo il delta per dividere i casi
        if delta > 0 :
            x1 = (-b + np.sqrt(delta))/(2*a)
            x2 = (-b - np.sqrt(delta))/(2*a)
            return print("Le soluzioni sono\nx1 =", x1, "\nx2 =", x2)
    
        elif delta == 0 :                                               # mi basta stampare una sola soluzione (sono uguali)
            x1 = x2 = -b/(2*a)
            return print("La soluzione è x = ", x1)
    
        #elif delta < 0 :                                                # Dovrei introdurre i complessi (che sbatti)
        else :
            return print("Non eiste soluzione per ogni x appartenente ai numeri reali.")
    
if __name__ == '__main__' :
    
    # Passo i coeff. a linea di comando
    
    a = float(sys.argv[1])
    b = float(sys.argv[2])
    c = float(sys.argv[3])
 
    soluz_eq_secondo_grado(a, b, c)     # Richiamo la funzione


# ----- ----- ----- ----- ----- ----- ----- ----- ----- -----


# Esercizio 8
'''
Scrivi un programma Python che trovi l'elenco dei numeri interi primi inferiori a 100, partendo dal presupposto che 2 è un numero primo
'''

def primo (n) :
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):  # Controllo fino alla radice quadrata di n
        if n % i == 0:
            return False
    return True

def numeri_primi_inferiori_a_100 ():
    lista_primi = []
    for num in range (2, 100):  # Controlliamo tutti i numeri da 2 a 99
        if primo (num):
            lista_primi.append (num)
    return lista_primi

if __name__ == '__main__' :

    lista = numeri_primi_inferiori_a_100 ()
    print(lista)
