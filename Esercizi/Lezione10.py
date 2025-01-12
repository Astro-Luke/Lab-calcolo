
# ----- ----- ----- ----- Lezione 10 ----- ----- ----- ----- 

# Esercizio 1
'''
Scrivere una libreria di funzioni per determinare il parametro τ di una distribuzione esponenziale da un elenco di numeri 
riempito con numeri pseudo-casuali distribuiti secondo una distribuzione di densità di probabilità esponenziale.
Confrontare il risultato ottenuto con la media dei numeri salvati nell'elenco.
In che modo il risultato dipende dall'intervallo iniziale passato alla sezione_aurea_max_LLfunzione?
'''

from library import rand_exp_inversa, loglikelihood_single_para, media, esponenziale, sezioneAureaMax_LL

def main () :

    tau = 2.4
    N_eventi = 10000

    elenco_n_pseudocasuali = []     # questo mi serve per salvare i numeri pseudocasuali

    for _ in range (N_eventi) :
        elenco_n_pseudocasuali.append (rand_exp_inversa(tau))       # li genero seguendo una distribuzione esponenziale

    '''
    nella prossima riga calcolo il massimo della loglikelihood, qui passo la loglikelihood della quale voglio il massimo, 
    la distribuzione (PDF), il mio sample di numeri creati e gli estremi in cui cercare il massimo
    '''
    value_max_loglike = sezioneAureaMax_LL (loglikelihood_single_para, esponenziale, elenco_n_pseudocasuali, 0.5, 5.)

    print ("Confronto tra tau trovato con loglikelihood e media:\n")
    print ("Media: ", media (elenco_n_pseudocasuali), "\n")
    print ("Massimo della loglikelihood: ", value_max_loglike, "\n")

    # la media ed il valore massimo della loglikelihood combaciano

if __name__ == "__main__" :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- 


# Esercizio 2
'''
Rappresenta graficamente il profilo della funzione di verosimiglianza e 
il punto identificato come suo massimo.
'''

import numpy as np
import matplotlib.pyplot as plt

from library import rand_exp_inversa, loglikelihood_single_para, media, esponenziale, sezioneAureaMax_LL

def main () :

    tau = 2.4
    N_eventi = 10000

    elenco_n_pseudocasuali = []     # questo conterrà i numeri pseudocasuali
    raccolta_likelihood = []        # raccoglie tutti i valori della loglikelihood al variare di tau
    lista_tau = []                  # raccoglie i tau per fare l'asse x del grafico


    for _ in range (N_eventi) :
        elenco_n_pseudocasuali.append (rand_exp_inversa(tau))

    for t in np.arange(0.5, 8., 0.2) :              # arange per aver step di 0.2
        val_loglikelihood = loglikelihood_single_para (elenco_n_pseudocasuali, esponenziale, t)
        raccolta_likelihood.append (val_loglikelihood)
        lista_tau.append (t)
    
    '''
    per fare il grafico mi servono le coordinate del punto del massimo, ottengo con la sezione aurea il valore del parametro tau 
    (quindi asse x) in cui si trova il massimo. la coordinata y sarà invece la loglikelihood calcolata in tau_max
    '''

    tau_max_loglike = sezioneAureaMax_LL (loglikelihood_single_para, esponenziale, elenco_n_pseudocasuali, 0.5, 8.)
    log_like_max = loglikelihood_single_para (elenco_n_pseudocasuali, esponenziale, tau_max_loglike)        # attenzione qui a passare tau_max_loglike!

    # Creazione del grafico
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 5))

    ax.plot (lista_tau, raccolta_likelihood, label = "loglikelihood")
    ax.set_title ("Grafico loglikelihood")
    ax.set_xlabel ("tau")
    ax.set_ylabel ("valori loglikelihood")
    ax.grid ()
    ax.scatter (tau_max_loglike, log_like_max, marker="o", color="red")         # aggiungo il punto di massimo

    print ("Confronto tra tau trovato con loglikelihood e media:\n")
    print ("Media: ", media (elenco_n_pseudocasuali), "\n")
    print ("Massimo della loglikelihood: ", tau_max_loglike, "\n")

    plt.savefig ("es10_2.png")
    plt.show ()

if __name__ == "__main__" :
    main ()


# ----- ----- ----- ----- ----- ----- ----- ----- 


# Esercizio 6