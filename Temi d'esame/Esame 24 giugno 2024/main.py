'''
python3 main.py
'''

import numpy as np
import matplotlib.pyplot as plt
from iminuit.cost import ExtendedBinnedNLL
from iminuit import Minuit
from tabulate import tabulate

from lib import sturges, rand_exp_inversa, rand_TCL_par_gauss, mod_total, loglikelihood, pdf_tot, esegui_fit_LL

# ----- Main -----

def main () :

    # Punto 1
    N_exp = 2000
    tau = 200
    lambd = (1/tau)
    campione_exp = []
    
    for _ in range (N_exp) :
        campione_exp.append (rand_exp_inversa (tau))

    N_gau = 200
    mu = 190
    sigma = 20
    campione_gauss = []

    for _ in range (N_gau) :
        campione_gauss.append (rand_TCL_par_gauss (mu, sigma, 10000))

    # controllino:
    # print ("campione uniforme: ", campione_uniforme, "\n", len(campione_uniforme))
    # print ("\ncampione gaussiano: ", campione_gauss, "\n", len(campione_gauss), "\n")

    # Punto 2
    campione_totale = campione_gauss + campione_exp
    
    Nbin = int (sturges (len (campione_totale))) + 15

    bin_content, bin_edges = np.histogram (campione_totale, bins = Nbin, range = (0., 3*tau) )  
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (campione_totale, bins = bin_edges, color = 'orange') 
    ax.grid ()                                      
    
    plt.savefig ('Istogramma esame giugno24.png')

    # P_arte 3: Fit
    # calcolo i parametri per passare qualcosa di sensato al fit
    media_campione = np.mean (campione_totale)
    sigma_campione = np.std (campione_totale)
    N_eventi = np.sum (bin_content)
    
    diz_variabili = {
        "N_signal": N_eventi,
        "mu_function": media_campione,
        "sigma_function": sigma_campione,
        "N_background": N_eventi,
        "tau_function": tau
    }

    diz_result = esegui_fit_LL (bin_content, bin_edges, diz_variabili, mod_total)

    #print (diz_result)

    for param, value, errore in zip (diz_result["Param"], diz_result["Value"], diz_result["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    
    # Parte 4 
    campione_tot_arrau = np.array(campione_totale)
    log_value = loglikelihood (campione_totale, pdf_tot, diz_result["Value"])
    print ("valore della loglikelihood: ", log_value)
    
    valori_loglike = []
    for mu in np.arange (30, 300, 0.5) :
        diz_result["Value"][1] = mu
        valori_loglike.append (loglikelihood (campione_totale, pdf_tot, diz_result["Value"]))
    

    fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (10, 5))   # 1 riga, 1 colonna
    axes.plot (np.arange (30, 300, 0.5), valori_loglike, label="loglikelihood")       # con esponenziale.pdf uso la funzione predefinita nella libreria scipy
    axes.legend ()
    axes.grid ()
    axes.set_title ("Plot loglikelihood al variare di mu")
    plt.savefig ("Plot loglikelihood.png") 

    # parte 5

    plt.show ()

if __name__ == "__main__" :
    main ()