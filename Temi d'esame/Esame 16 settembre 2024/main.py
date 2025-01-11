'''
python3 main.py
'''
import numpy as np
from lib import additive_recurrence, MC_mod, function, integral_Crude_MC
import matplotlib.pyplot as plt


def main () :

    # Punto 1 e 2
    alpha = ((np.sqrt(5) - 1) / 2)
    seed = 1.
    generatore = additive_recurrence (alpha)
    generatore.set_seed (seed)

    lista_num = []
    for _ in range (0, 1000) :
        number = generatore.get_number ()
        lista_num.append (number)

    #print (lista_num[:10])

    # Punto 3
    int_val, int_incertezza = MC_mod (function, 0., 1., 0., 2., 1000, generatore)
    print ("\nValore dell'integrale: ", int_val, "+/-", int_incertezza, "\n")

    # Punto 4
    N_toys = 100
    N_points = 10
    list_points = []
    incerteza_list = []

    while (N_points < 25000) :
        integral_list = []
        for contatore_toys in range (N_toys) :
            integral_val, incertezza_val = MC_mod (function, 0., 1., 0., 2., N_points, generatore)
            integral_list.append (integral_val)
        incerteza_list.append (np.std (integral_list))         # sono indentati allo stesso modo (infatti hanno la stessa lunghezza)
        list_points.append (N_points)
        N_points = N_points * 2


    # Grafico
    fig, ax = plt.subplots ()
    ax.plot (list_points, incerteza_list, "o-", label = "Incertezza MC_mod", color = "blue")


    # Punto 5
    N_toys = 100
    N_points = 10
    list_points = []
    incerteza_list = []

    while (N_points < 25000) :
        integral_list = []
        for contatore_toys in range (N_toys) :
            integral_val, incertezza_val = integral_Crude_MC (function, 0., 1., N_points)
            integral_list.append (integral_val)
        incerteza_list.append (np.std (integral_list))         # sono indentati allo stesso modo (infatti hanno la stessa lunghezza)
        list_points.append (N_points)
        N_points = N_points * 2

    ax.plot (list_points, incerteza_list, "o-", label = "Incertezza Crude MC", color = "red")
    ax.set_xscale ("log")
    ax.set_yscale ("log")
    ax.legend ()
    ax.grid ()

    plt.show ()
    
if __name__ == "__main__" :
    main ()