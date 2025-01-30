'''
python3 3luglio23.py
'''

import numpy as np
import matplotlib.pyplot as plt

from lib import funz, funzione_fit, rand_TCL_par_gauss, esegui_fit, funz_quadra, funz_quadra_fit, sturges 


def main () :
    
    x_list = [0.5, 1.5, 2.5, 3.5]
    x = np.array (x_list)
    y = np.zeros(4)
    sigma = np.zeros(4)
    
    # dizionario per la funzione esegui fit
    diz_para = {
        "a": 1., 
        "b": 1.
    }
    
    for i in range (4) :
        sigma[i] = 0.2
        y[i] = funz(x[i]) + rand_TCL_par_gauss (0., 0.2, 100)


    # Punto 2 e 3 (Grafico e Fit)
    diz_result = esegui_fit (x, y, sigma, diz_para, funzione_fit)
    
    # Per vedere i risultati del fit
    print ("\nEsito del Fit: ", diz_result["Validità"])
    print ("\nNumero di gradi di libertà: ", diz_result["Ndof"])
    print ("\nValore del Q-quadro: ", diz_result["Qsquared"])
    print ("\nP-value: ", diz_result["Pvalue"])
    print("Matrice di covarianza:\n", diz_result["MatriceCovarianza"])

    for param, value, errore in zip (diz_result["Param"], diz_result["Value"], diz_result["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')
    
    # Grafico
    x_fit = np.linspace (0., 4., 500)
    y_fit = funzione_fit (x_fit, *diz_result["Value"])
    
    fig, ax = plt.subplots()
    ax.errorbar (x, y, xerr = 0.0, yerr = sigma, marker = 'o', linestyle = 'None', capsize = 5, capthick = 1.5, color = 'blue')
    ax.plot (x_fit, y_fit, color = 'red')
    ax.grid ()
    
    plt.savefig ("3luglio2023.png")

    # Punto 4: Toy experiment
    
    N_toy = 1000
    lista_Q2 = []
    lista_Pval = []
    
    for _ in range (N_toy) :
        sigma_toy = np.full (4, 0.2)
        y_toy = np.zeros (4)
        
        for i in range (4) :
            y_toy[i] = funz(x[i]) + rand_TCL_par_gauss (0., 0.2, 100)
        
        diz_result = esegui_fit (x, y_toy, sigma_toy, diz_para, funzione_fit)
        
        lista_Q2.append (diz_result["Qsquared"])
        lista_Pval.append (diz_result["Pvalue"])
    

    lista_quadra_Q2 = []
    lista_quadra_Pval = []
    
    for _ in range (N_toy) :
        sigma_toy = np.full (4, 0.2)
        y_toy = np.zeros (4)
        
        for i in range (4) :
            y_toy[i] = funz(x[i]) + rand_TCL_par_gauss (0., 0.2, 100)
        
        diz_result = esegui_fit (x, y_toy, sigma_toy, diz_para, funz_quadra_fit)
        
        lista_quadra_Q2.append (diz_result["Qsquared"])
        lista_quadra_Pval.append (diz_result["Pvalue"])

    
    Nbin = sturges (N_toy)
    
    bin_content_Q2, bin_edges_Q2 = np.histogram (lista_Q2, bins=Nbin, 
                                                 range = (min(lista_Q2), max(lista_Q2)))      
    bin_content_Pval, bin_edges_Pval = np.histogram (lista_Pval, bins=Nbin, 
                                                     range = (min(lista_Pval), max(lista_Pval)))

    bin_content_quadra_Q2, bin_edges_quadra_Q2 = np.histogram (lista_quadra_Q2, bins=Nbin, 
                                                               range = (min(lista_quadra_Q2), max(lista_quadra_Q2)))      
    bin_content_quadra_Pval, bin_edges_quadra_Pval = np.histogram (lista_quadra_Pval, bins=Nbin, 
                                                                   range = (min(lista_quadra_Pval), max(lista_quadra_Pval)))
    
    fig, ax = plt.subplots (nrows = 2, ncols = 2)
    ax[0][0].hist (lista_Q2, bins=bin_edges_Q2, color = 'orange', density = True)
    ax[0][0].set_title ('distribuzione Q2', size = 14)
    ax[0][0].set_xlabel ('Q2')
    ax[0][0].set_ylabel ('frequenza')
    ax[0][0].grid ()                                          # Se voglio la griglia

    ax[0][1].hist (lista_Pval, bins=bin_edges_Pval, color = 'orange', density = True)
    ax[0][1].set_title ('distribuzione P-value', size = 14)
    ax[0][1].set_xlabel ('P-value')
    ax[0][1].set_ylabel ('frequenza')
    ax[0][1].grid ()

    ax[1][0].hist (lista_quadra_Q2, bins=bin_edges_quadra_Q2, color = 'blue', density = True)
    ax[1][0].set_title ('distribuzione Q2 funz quadra', size = 14)
    ax[1][0].set_xlabel ('Q2')
    ax[1][0].set_ylabel ('frequenza')
    ax[1][0].grid ()

    ax[1][1].hist (lista_quadra_Pval, bins=bin_edges_quadra_Pval, color = 'blue', density = True)
    ax[1][1].set_title ('distribuzione P-value funz quadra', size = 14)
    ax[1][1].set_xlabel ('P-value')
    ax[1][1].set_ylabel ('frequenza')
    ax[1][1].grid ()
    
    plt.tight_layout()
    plt.savefig ('Istogrammi Q2-Pval.png')

    array_sigma = np.arange (0.1, 2., 0.1)
    lista_medie_Q2 = []
    lista_medie_quadri_Q2 = []
    for j in array_sigma :
        sigma_toy = np.full (4, j)
        list_Q2_errori = []
        list_Q2_quadri_errori = []
        
        for _ in range (N_toy) :
            y_toy = np.zeros (4)
        
            for i in range (4) :
                y_toy[i] = funz(x[i]) + rand_TCL_par_gauss (0., j, 100)
                
            diz_result = esegui_fit (x, y_toy, sigma_toy, diz_para, funzione_fit)
            diz_result_quadra = esegui_fit (x, y_toy, sigma_toy, diz_para, funz_quadra_fit)
            list_Q2_errori.append (diz_result["Qsquared"])
            list_Q2_quadri_errori.append (diz_result_quadra["Qsquared"])
            
        lista_medie_Q2.append (np.mean (list_Q2_errori))
        lista_medie_quadri_Q2.append (np.mean (list_Q2_quadri_errori))
   
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.plot (array_sigma, lista_medie_Q2, label = 'fit con funz. cubica')
    ax.plot (array_sigma, lista_medie_quadri_Q2, label = 'fit con funz. quadratica')
    ax.set_title ('distribuzione medie Q2', size = 14)
    ax.set_xlabel ('sigma')
    ax.set_ylabel ('media Q2')
    ax.set_yscale ('log')
    ax.grid ()
    ax.legend ()
    
    plt.show ()

if __name__ == '__main__' :
    main ()