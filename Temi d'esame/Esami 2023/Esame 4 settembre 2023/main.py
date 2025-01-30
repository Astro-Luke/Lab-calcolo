'''
python3 main.py
'''
import numpy as np
import sys
import matplotlib.pyplot as plt

from lib import rand_range, funzione, rand_TCL_par_gauss, esegui_fit, funzione_fit

def main () :
    
    sigma = 0.2
    x = np.array([0.5, 2.5, 4.5, 6.5, 8.5, 10.5])
    y = np.zeros(6)
    epsilon = np.zeros(6)
    errori = []

    diz_parametri = {
        "p_0": 2.,
        "p_1": 0.5,
        "p_2": 0.78,
        "p_3": 0.8,
    }
    
    for i in range (6) :
        epsilon[i] = rand_TCL_par_gauss (0., sigma, 10)
        y[i] = funzione(x[i]) + epsilon[i]
        errori.append(sigma)
    
    np_errori = np.array(errori)

    #print ("y: \n", y, "\nerrori: \n", np_errori)

    # Parte 4: Fit
    diz_result = esegui_fit (x, y, errori, diz_parametri, funzione_fit)   
    
    print ("\nEsito del Fit: ", diz_result["Validità"])
    print ("\nNumero di gradi di libertà: ", diz_result["Qsquared"])
    print ("\nValore del Q-quadro: ", diz_result["Ndof"], "\n")
    print("Matrice di covarianza:\n", diz_result["MatriceCovarianza"])

    for param, value, errore in zip (diz_result["Param"], diz_result["Value"], diz_result["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    x_fit = np.linspace (min(x), max(x), 500)
    y_fit = funzione_fit (x_fit, *diz_result["Value"])            

    fig, ax = plt.subplots ()
    ax.set_title ('Grafico e fit', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.errorbar (x, y, xerr = 0.0, yerr = np_errori, linestyle = 'None', marker = 'o') 
    ax.plot (x_fit, y_fit, color = 'red', label = 'Fit')
    ax.grid ()
    plt.savefig ("Settembre2023pt3.png")

    # Parte 4 e 5
    delta = np.zeros (6)
    errore_tot = np.zeros (6)
    y_new = np.zeros (6)
    
    for i in range (6) :
        delta[i] = rand_range (0, 1)
        errore_tot[i] = np.sqrt ( (epsilon[i])**2 + (delta[i])**2 )
        y_new[i] = funzione(x[i]) + errore_tot[i]

    second_diz_result = esegui_fit (x, y_new, errore_tot, diz_parametri, funzione_fit)   
    
    print ("\nEsito del Fit: ", diz_result["Validità"])
    print ("\nValore del Q-quadro: ", diz_result["Qsquared"])
    print ("\nNumero di gradi di libertà: ", diz_result["Ndof"], "\n")
    print ("Matrice di covarianza:\n", diz_result["MatriceCovarianza"])

    for param, value, errore in zip (diz_result["Param"], diz_result["Value"], diz_result["Errori"]) : 
        print (f'{param} = {value:.6f} +/- {errore:.6f}\n')

    second_x_fit = np.linspace (min(x), max(x), 500)
    second_y_fit = funzione_fit (second_x_fit, *second_diz_result["Value"])
    
    fig, ax = plt.subplots ()
    ax.set_title ('Grafico e fit dei nuovi punti', size = 14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.errorbar (x, y_new, xerr = 0.0, yerr = errore_tot, linestyle = 'None', marker = 'o') 
    ax.grid ()
    ax.plot (second_x_fit, second_y_fit, color = 'red', label = 'Fit con errori diversi')
    plt.savefig ("Settembre2023pt4.png")

    
    
    plt.show ()                        

if __name__ == '__main__' :
    main ()