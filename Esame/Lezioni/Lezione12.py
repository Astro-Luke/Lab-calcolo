
# ----- ----- ----- ----- Lezione 12 ----- ----- ----- -----


# Esercizio 1
'''
Write a program that fits the events saved in the file dati.txt.
Take care to determine the range and binning of the histogram used for the fit based on the events themselves, 
writing appropriate algorithms to determine the minimum and maximum of the sample and a reasonable estimate of the number of bins to use.
Determine the initial values of the fit parameters using the techniques described in the lesson.
Print the fit result on the screen.
Plot the histogram with the fitted model overlaid.
Which parameters are correlated, and which are anti-correlated with each other?
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon, norm
from library import sturges, esegui_fit_LL

# ----- Funzioni -----

def mod_total(bin_edges, N_signal, mu_function, sigma_function, N_background, tau_function):
    return N_signal * norm.cdf(bin_edges, mu_function, sigma_function) + N_background * expon.cdf(bin_edges, 0, tau_function)

# ----- Main -----

def main():
    
    vettore = np.loadtxt("dati.txt")

    # Calcolo numero di bin con il criterio di Sturges
    Nbin = sturges(vettore.size)

    # Creazione dell'istogramma
    bin_content, bin_edges = np.histogram(vettore, Nbin, (min(vettore), max(vettore)))

    # Parametri iniziali
    media_campione = np.mean(vettore)
    sigma_campione = np.std(vettore)
    N_eventi = np.sum(bin_content)
    tau = 1.0

    diz_variabili = {
        "N_signal": N_eventi,
        "mu_function": media_campione,
        "sigma_function": sigma_campione,
        "N_background": N_eventi,
        "tau_function": tau
    }

    # Esecuzione del fit
    diz_result = esegui_fit_LL (bin_content, bin_edges, diz_variabili, mod_total)

    # Stampa dei risultati in formato leggibile
    print("\nRisultati del fit:\n")
    print(f"Fit valido: {diz_result['Validità']}")
    print(f"Gradi di libertà (Ndof): {diz_result['Ndof']}")
    print("\nParametri:\n")

    for param, value, errore in zip(diz_result["Param"], diz_result["Value"], diz_result["Errori"]):
        print(f"{param}: {value:.6f} ± {errore:.6f}")

    print("\nMatrice di covarianza:\n")
    print(diz_result["MatriceCovarianza"])

    # Plot dell'istogramma con il modello fittato
    fig, ax = plt.subplots()
    ax.hist(vettore, bins=bin_edges, color='orange', label='Dati', alpha=0.7)
    ax.set_title('Istogramma con fit', size=14)
    ax.set_xlabel('x')
    ax.set_ylabel('Conteggi')
    ax.grid()

    x_fit = np.linspace(min(vettore), max(vettore), 500)
    y_fit = mod_total (x_fit, *diz_result["Value"])
    #ax.plot (x_fit, y_fit, color='blue', label='Modello fittato')

    ax.legend()
    plt.savefig('Histo_with_fit.png')
    plt.show()

if __name__ == "__main__":
    main()