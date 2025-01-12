
# ----- ----- ----- ----- Lezione 9 ----- ----- ----- ----- 

# Esercizio 4
'''
Esercizio 9.1 
Scrivi un programma che generi numeri pseudo-casuali distribuiti secondo una funzione esponenziale e li memorizzi in un elenco.

9.10.2. Esercizio 9.2 
Aggiungere al programma precedente il codice sorgente che riempie un istogramma con i numeri presenti nell'elenco in cui 
sono stati trasferiti e visualizza l'istogramma sullo schermo.

9.10.3. Esercizio 9.3 
Scrivere un programma che traccia il grafico della distribuzione di probabilità esponenziale con un parametro fisso t_0 .

9.10.4. Esercizio 9.4 
Scrivere una funzione likelihoodche calcoli la verosimiglianza al variare del parametro t_0, per un campione di eventi pseudo-casuali 
generati secondo le istruzioni dell'esercizio 1.

In che modo il risultato dipende dal numero di eventi nel campione?

9.10.5. Esercizio 9.5 
Scrivi una funzione loglikelihood che calcola il logaritmo della verosimiglianza al variare del parametro t_0, per un campione di 
eventi pseudo-casuali generati secondo le istruzioni dell'Esercizio 1. 
Ricorda che il logaritmo della verosimiglianza è definito solo quando la verosimiglianza è strettamente positiva.
'''

import numpy as np
import matplotlib.pyplot as plt

from library import rand_exp_inversa, sturges, loglikelihood

# funzione esponenziale

def funz_exp (x, t_0) :
    return (1 / t_0) * np.exp(- (x / t_0))

def main() :

    N = 10000  # numero di numeri pseudocasuali da generare
    t_0 = 2.4
    elenco_pseudocasual = []

    x_min = 0.
    x_max = 10.

    for _ in range (N):
        elenco_pseudocasual.append (rand_exp_inversa (t_0))

    tau_list = []
    array_var_loglikelihoo = []
    for t in np.arange (1, 6, 0.2) :
        var_loglikelihood = loglikelihood (elenco_pseudocasual, funz_exp, t)
        tau_list.append (t)
        array_var_loglikelihoo.append (var_loglikelihood)


    Nbin = sturges (N)

    x_axis = np.linspace (x_min, x_max, 100)
    bin_edges = np.linspace (0., 10., Nbin)  # Regola la dimensione dei bin

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize = (24, 9))

    ax[0].hist (elenco_pseudocasual, bins=bin_edges, color='orange')
    ax[0].set_title ('Istogramma', size=14)
    ax[0].set_xlabel ('Tempo decadimento (s)')
    ax[0].set_ylabel ('Conteggi')
    ax[0].grid ()

    ax[1].plot (x_axis, (N/t_0)*funz_exp(x_axis, t_0), label="PDF", color="blue")
    ax[1].set_title ('PDF', size=14)
    ax[1].set_xlabel ('Tempo decadimento (s)')
    ax[1].set_ylabel ('Probabilità')
    ax[1].grid ()

    ax[2].plot (tau_list, array_var_loglikelihoo, color = "orange")
    ax[2].set_title ('Loglikelihood', size=14)
    ax[2].set_xlabel ('tau (s)')
    ax[2].set_ylabel ('Valori loglikelihood')
    ax[2].grid ()

    plt.savefig ('es9_4.png')
    plt.show ()

if __name__ == "__main__" :
    main()
