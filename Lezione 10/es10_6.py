import numpy as np
import matplotlib.pyplot as plt

from library import loglikelihood, esponenziale, rand_exp_inversa, sezioneAureaMax_LL, intersect_LLR

def main():

    tau = 2.4
    N_eventi = 1000

    elenco_n_pseudocasuali = []     # questo conterrà i numeri pseudocasuali
    raccolta_likelihood = []        # raccoglie tutti i valori della loglikelihood al variare di tau
    lista_tau = []                  # raccoglie i tau per fare l'asse x del grafico

    for _ in range (N_eventi) :
        elenco_n_pseudocasuali.append (rand_exp_inversa(tau))

    for t in np.arange (0.5, 5., 0.02) :              # arange per aver step di 0.2
        val_loglikelihood = loglikelihood (elenco_n_pseudocasuali, esponenziale, t)
        raccolta_likelihood.append (val_loglikelihood)
        lista_tau.append (t)

    tau_max_loglike = sezioneAureaMax_LL (loglikelihood, esponenziale, elenco_n_pseudocasuali, 0.5, 5.)

    # Livello di confidenza: log-likelihood massima - 0.5
    ylevel = loglikelihood(elenco_n_pseudocasuali, esponenziale, tau_max_loglike) - 0.5

    lista_parameter = [tau_max_loglike]

    # Calcolo dei punti di intersezione
    val_left_intercept = intersect_LLR (loglikelihood, esponenziale, elenco_n_pseudocasuali, 0.5, tau_max_loglike, ylevel, lista_parameter)
    val_right_intercept = intersect_LLR (loglikelihood, esponenziale, elenco_n_pseudocasuali, tau_max_loglike, 5., ylevel, lista_parameter)


    print(f"Massimo della log-likelihood (tau): {tau_max_loglike:.2f}")
    print(f"Intervallo di confidenza a sinistra (tau - sigma): {val_left_intercept:.2f}")
    print(f"Intervallo di confidenza a destra (tau + sigma): {val_right_intercept:.2f}")

    # Grafico
    plt.figure(figsize = (10, 6))
    plt.plot(lista_tau, raccolta_likelihood, label='Profilo di verosimiglianza')
    plt.axvline(tau_max_loglike, color='r', linestyle='--', label=r'$\tau_{max}$')
    plt.axhline(ylevel, color='g', linestyle='--', label='Livello di confidenza')
    plt.axvline(val_left_intercept, color='b', linestyle='--', label=r'$\tau - \sigma_\tau$')
    plt.axvline(val_right_intercept, color='b', linestyle='--', label=r'$\tau + \sigma_\tau$')
    plt.xlabel(r'$\tau$')
    plt.ylabel('Log-Likelihood')
    plt.legend()
    plt.grid()
    plt.title('Profilo di verosimiglianza e intervallo di confidenza')
    plt.savefig ("es10_6.png")
    plt.show()

if __name__ == "__main__":
    main()
