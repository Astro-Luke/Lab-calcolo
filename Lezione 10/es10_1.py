'''
Scrivere una libreria di funzioni per determinare il parametro τ di una distribuzione esponenziale da un elenco di numeri 
riempito con numeri pseudo-casuali distribuiti secondo una distribuzione di densità di probabilità esponenziale.

Confrontare il risultato ottenuto con la media dei numeri salvati nell'elenco.

In che modo il risultato dipende dall'intervallo iniziale passato alla sezione_aurea_max_LLfunzione?
'''

from library import rand_exp_inversa, loglikelihood, media, esponenziale, sezioneAureaMax_LL

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
    value_max_loglike = sezioneAureaMax_LL (loglikelihood, esponenziale, elenco_n_pseudocasuali, 0.5, 5.)

    print ("Confronto tra tau trovato con loglikelihood e media:\n")
    print ("Media: ", media (elenco_n_pseudocasuali), "\n")
    print ("Massimo della loglikelihood: ", value_max_loglike, "\n")

    # la media ed il valore massimo della loglikelihood combaciano

if __name__ == "__main__" :
    main ()