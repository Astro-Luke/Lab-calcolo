'''
python3 main.py
'''
import numpy as np
from lib import additive_recurrence, MC_mod, function
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

    print (lista_num[:10])

    # Punto 3
    int_val, int_incertezza = MC_mod (function, 0., 1., 0., 2., 1000, generatore)
    print ("\nValore dell'integrale: ", int_val, "+/-", int_incertezza, "\n")

    # Punto 4
    N_toys = 1000
    N_points = 10
    N_points_max = 25000

    seq_N = []
    seq_sigma = []
    seq_sigma_t = []
    seq_mean = []

    while N_points < N_points_max :
        print ('running with', N_points, 'points')
        integrals = []
        for i_toy in range (N_toys): 
            result = MC_mod (function, 0., 1., 0., 2., N_points, generatore)
            integrals.append (result[0])
            if i_toy == 0 : 
                seq_sigma.append (result[1])  
        seq_N.append (N_points)
        seq_sigma_t.append (np.std (integrals))
        seq_mean.append (np.mean (integrals))
        N_points *= 2
    print ('DONE')

    # Grafico
    fig, ax = plt.subplots ()
    ax.plot (seq_N, seq_mean, "o-", label = "mean", color = "blue")
    ax.plot (seq_N, seq_sigma, "o-", label = "estimate", color = "red")
    ax.plot (seq_N, seq_sigma_t, "o-", label = "toys", color = "orange")
    ax.set_xscale ("log")
    #ax.set_yscale ("log")
    ax.grid ()

    plt.show ()
    
if __name__ == "__main__" :
    main ()