'''
python3 main.py
'''

import numpy as np
import matplotlib.pyplot as plt
from iminuit.cost import ExtendedBinnedNLL
from iminuit import Minuit
from tabulate import tabulate

from lib import sturges, rand_exp_inversa, rand_TCL_par_gauss, mod_total, loglikelihood, pdf_tot, esegui_fit_LL

# Funzione per il calcolo del massimo della loglikelihood
def sezioneAureaMax_LL(
    f,              # funzione di likelihood da massimizzare
    pdf,            # probability density function degli eventi
    sample,         # campione degli eventi
    x0,             # estremo dell'intervallo
    x1,             # altro estremo dell'intervallo
    prec=0.0001,    # precisione della funzione
    params=None     # parametri aggiuntivi per la funzione f
):
    r = 0.618  # Costante aurea
    x2 = 0.
    x3 = 0.
    larghezza = abs(x1 - x0)

    if params is None:
        params = []

    while larghezza > prec:
        x2 = x0 + r * (x1 - x0)
        x3 = x0 + (1. - r) * (x1 - x0)

        # Restringimento dell'intervallo
        if f(sample, pdf, params + [x3]) < f(sample, pdf, params + [x2]):
            x0 = x3
        else:
            x1 = x2
        larghezza = abs(x1 - x0)

    return (x0 + x1) / 2.   # Ritorna il punto medio dell'intervallo finale

# ----- Main -----

def main():
    # Punto 1
    N_exp = 2000
    tau = 200
    lambd = (1 / tau)
    campione_exp = [rand_exp_inversa(tau) for _ in range(N_exp)]

    N_gau = 200
    mu = 190
    sigma = 20
    campione_gauss = [rand_TCL_par_gauss(mu, sigma, 10000) for _ in range(N_gau)]

    # Punto 2
    campione_totale = campione_gauss + campione_exp
    
    Nbin = int(sturges(len(campione_totale))) + 15
    bin_content, bin_edges = np.histogram(campione_totale, bins=Nbin, range=(0., 3 * tau))

    fig, ax = plt.subplots()
    ax.hist(campione_totale, bins=bin_edges, color='orange')
    ax.grid()
    plt.savefig('Istogramma_esame_giugno24.png')

    # Punto 3: Fit
    media_campione = np.mean(campione_totale)
    sigma_campione = np.std(campione_totale)
    N_eventi = np.sum(bin_content)

    diz_variabili = {
        "N_signal": N_eventi,
        "mu_function": media_campione,
        "sigma_function": sigma_campione,
        "N_background": N_eventi,
        "tau_function": tau
    }

    diz_result = esegui_fit_LL(bin_content, bin_edges, diz_variabili, mod_total)

    for param, value, errore in zip(diz_result["Param"], diz_result["Value"], diz_result["Errori"]):
        print(f'{param} = {value:.6f} +/- {errore:.6f}\n')

    # Punto 4
    valori_parametri = list(diz_result["Value"])
    log_value = loglikelihood(campione_totale, pdf_tot, valori_parametri)
    print("Valore della loglikelihood:", log_value)

    valori_loglike = []
    for mu in np.arange(30, 300, 0.5):
        valori_parametri[1] = mu  # Aggiorna solo il parametro mu
        valori_loglike.append(loglikelihood(campione_totale, pdf_tot, valori_parametri))

    # Punto 5
    maximum = sezioneAureaMax_LL(loglikelihood, pdf_tot, campione_totale, 120, 250, params=valori_parametri[:-1])
    print("Massimo della loglikelihood trovato a mu:", maximum)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(30, 300, 0.5)
    ax.plot(x, valori_loglike, label="loglikelihood")
    ax.legend()
    ax.grid()
    ax.set_title("Plot loglikelihood al variare di mu con massimo evidenziato")
    plt.savefig("Plot_loglikelihood_con_massimo.png")

    plt.show()

if __name__ == "__main__":
    main()
