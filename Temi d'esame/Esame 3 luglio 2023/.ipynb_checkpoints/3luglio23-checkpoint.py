'''
python3 3luglio23.py
'''

import numpy as np
import matplotlib.pyplot as plt

from lib import funz, funzione_fit, rand_TCL_par_gauss, esegui_fit, funz_quadra, funz_quadra_fit


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
    print ("\nValore del Q-quadro: ", diz_result["Qsquared"], "\n")

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
    '''
    N_toy = 100
    N = 100
    list_n = []
    
    for i in range (N_toy) :
        lista_Q2 = []
        lista_Pval = []
        for i in range (N) :
                sigma_toy[i] = 0.2
                y_toy[i] = funz(x[i]) + rand_TCL_par_gauss (0., 0.2, 100)
                diz_result = esegui_fit (x, y_toy, sigma_toy, diz_para, funzione_fit)
        lista_Q2.append (diz_result["Qsquared"])
        lista_Pval.append (diz_result["Pvalue"])
    
    print (lista_Q2)     # controllo
    print (lista_Pval)
    '''
    plt.show ()

if __name__ == '__main__' :
    main ()