'''
Rappresenta graficamente il profilo della funzione di verosimiglianza e 
il punto identificato come suo massimo.
'''

import numpy as np
import matplotlib.pyplot as plt

from library import rand_exp_inversa, loglikelihood, media, esponenziale, sezioneAureaMax_LL

def main () :

    tau = 2.4
    N_eventi = 10000

    elenco_n_pseudocasuali = []     # questo conterrà i numeri pseudocasuali
    raccolta_likelihood = []        # raccoglie tutti i valori della loglikelihood al variare di tau
    lista_tau = []                  # raccoglie i tau per fare l'asse x del grafico


    for _ in range (N_eventi) :
        elenco_n_pseudocasuali.append (rand_exp_inversa(tau))

    for t in np.arange(0.5, 8., 0.2) :              # arange per aver step di 0.2
        val_loglikelihood = loglikelihood (elenco_n_pseudocasuali, esponenziale, t)
        raccolta_likelihood.append (val_loglikelihood)
        lista_tau.append (t)
    
    '''
    per fare il grafico mi servono le coordinate del punto del massimo, ottengo con la sezione aurea il valore del parametro tau 
    (quindi asse x) in cui si trova il massimo. la coordinata y sarà invece la loglikelihood calcolata in tau_max
    '''

    tau_max_loglike = sezioneAureaMax_LL (loglikelihood, esponenziale, elenco_n_pseudocasuali, 0.5, 8.)
    log_like_max = loglikelihood (elenco_n_pseudocasuali, esponenziale, tau_max_loglike)        # attenzione qui a passare tau_max_loglike!

    # Creazione del grafico
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    ax.plot (lista_tau, raccolta_likelihood, label = "loglikelihood")
    ax.set_title ("Grafico loglikelihood")
    ax.set_xlabel ("tau")
    ax.set_ylabel ("valori loglikelihood")
    ax.grid ()
    ax.scatter (tau_max_loglike, log_like_max, marker="o", color="red")         # aggiungo il punto di massimo

    plt.savefig ("es10_2.png")

    print ("Confronto tra tau trovato con loglikelihood e media:\n")
    print ("Media: ", media (elenco_n_pseudocasuali), "\n")
    print ("Massimo della loglikelihood: ", tau_max_loglike, "\n")

if __name__ == "__main__" :
    main ()