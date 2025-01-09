'''
python3 main.py
'''
import numpy as np
from lib import additive_recurrence, MC_mod, function

def main () :

    # Punto 1
    alpha = ((np.sqrt(5) - 1) / 2)
    seed = 1.
    generatore = additive_recurrence (alpha)
    generatore.set_seed (seed)

    lista_num = []
    for i in range (0, 1000) :
        number = generatore.get_number ()
        lista_num.append (number)

    print (lista_num[:10])

    # Punto 2
    int_val, int_incertezza = MC_mod (function, 0., 1., 0., 2., 1000, generatore)
    print ("\nValore dell'integrale: ", int_val, "+/-", int_incertezza, "\n")

    # Punto 3
    N_toys = 100
    list_valori_int = []
    list_errori_int = []
    
    for i in range (N_toys) :
        lista = []
        for n in np.arange (10, 25000, 10) :
        

if __name__ == "__main__" :
    main ()