'''
libreria mylib contiene:

    median
    percentile75
    percentile25
    percentile
    mean
    variance
    stdev
    stdev_mean
    sturges

    bisezione
    bisezione_ricorsiva
    sezioneAureaMin
    sezioneAureaMin_ricorsiva
    sezioneAureaEffMin
    sezioneAureaEffMin_ricorsiva
    sezioneAureaMax
    sezioneAureaMax_ricorsiva

'''


import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor
import random



#statistics.py

def median(x):
    '''
    determina la mediana di un array

    Args:
        x (numpy.ndarray): array del quale voglio calcolare la mediana

    Returns:
        float: mediana dell'array
    '''
    x_sorted = np.sort(x)
    if len(x_sorted)%2 == 0: 
        # se la lunghezza è un numero pari, restituisce la media dei due valori centrali
        m = (x_sorted[int(len(x_sorted)/2)-1] + x_sorted[int(len(x_sorted)/2)])/2
    else: 
        # se dispari restituisce il valore centrale
        m = x_sorted[int((len(x_sorted)-1)/2)]
    return m

def percentile75(x):
    '''
    determina il valore sopra il quale giacciono il 25% dei valori, o sotto il quale trovo il 75% degli elementi
    '''
    x_sorted = np.sort(x)
    idx = int(0.75*len(x))
    pv = x_sorted[idx]
    return pv

def percentile25(x):
    '''
    determina il valore sopra il quale giacciono il 75% dei valori, o sotto il quale trovo il 25% degli elementi
    '''
    x_sorted = np.sort(x)
    idx = int(0.25*len(x))
    pv = x_sorted[idx]
    return pv

def percentile(x,p):
    '''
    determina il valore sopra il quale giacciono una percentuale data p dei valori
    '''
    if not (0 < p and p < 1):
        raise ValueError(f'Il valore percentuale p = {p} non è nell\'intervallo [0,1]')
    x_sorted = np.sort(x)
    idx = int(p*len(x))
    pv = x_sorted[idx]
    return pv

def mean(x):
    '''
    calcola la media di un array
    '''
    m = np.sum(x)/len(x)
    return m

def variance(x):
    '''
    calcola la varianzadi un array
    '''
    m = mean(x)
    m2= mean(x*x)
    return m2-m*m

def stdev(x,bessel=True):
    '''
    calcola la deviazione standard con correzione di Bessel di un array
    '''
    m = mean(x)
    r = x-m
    s = np.sqrt( np.sum(r*r)/(len(x)-1) ) if bessel else np.sqrt( np.sum(r*r)/len(x) )
    return s

def stdev_mean(x,bessel=True):
    '''
    calcola la deviazione standard della media con correzione di Bessel di un array
    '''
    s = stdev(x,bessel)
    return s/np.sqrt(len(x))


#--------------



def sturges (N_events) :
    '''
    algoritmo per calcolo del numero di bin ideale data la lunghezza del campione.
    Ideale per array di lunghezza non troppo piccola nè troppo grande
    per array grande (>10000) conviene cercare altri metodi, ad esempio usare la radice quadrata
    '''
     return int( np.ceil( 1 + np.log2(N_events) ) )

    
    
#-------------



#bisezione.py 



def bisezione (
    g,              # funzione di cui trovare lo zero
    xMin,           # minimo dell'intervallo          
    xMax,           # massimo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola zeri
    con il metodo della bisezione
    '''
    xAve = xMin 
    while ((xMax - xMin) > prec) :
        xAve = 0.5 * (xMax + xMin) 
        if (g (xAve) * g (xMin) > 0.): xMin = xAve 
        else                         : xMax = xAve 
    return xAve 
    

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def bisezione_ricorsiva (
    g,              # funzione di cui trovare lo zero  
    xMin,           # minimo dell'intervallo            
    xMax,           # massimo dell'intervallo          
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola zeri
    con il metodo della bisezione ricorsivo
    '''
    xAve = 0.5 * (xMax + xMin)
    if ((xMax - xMin) < prec): return xAve ;
    if (g (xAve) * g (xMin) > 0.): return bisezione_ricorsiva (g, xAve, xMax, prec) ;
    else                         : return bisezione_ricorsiva (g, xMin, xAve, prec) ;



#---------------



#sezione_aurea.py

def sezioneAureaMin (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti(= punti di max o min)
    con il metodo della sezione aurea
    '''

    r = 0.618
    x2 = 0.
    x3 = 0. 
    larghezza = abs (x1 - x0)
     
    while (larghezza > prec):
        x2 = x0 + r * (x1 - x0) 
        x3 = x0 + (1. - r) * (x1 - x0)  
      
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro        
        if (g (x3) > g (x2)): 
            x0 = x3
            # x1 = x1  this extreme does not change       
        else :
            x1 = x2
            # x0 = x0  this extreme does not change         
            
        larghezza = abs (x1-x0)             
                                   
    return (x0 + x1) / 2. 


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def sezioneAureaMin_ricorsiva (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    implementata ricorsivamente
    '''

    r = 0.618
    x2 = x0 + r * (x1 - x0)
    x3 = x0 + (1. - r) * (x1 - x0) 
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  : return ( x0 + x1) / 2.
    elif (g (x3) > g (x2)) : return sezioneAureaMin_ricorsiva (g, x3, x1, prec)
    else                   : return sezioneAureaMin_ricorsiva (g, x0, x2, prec)   


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def sezioneAureaEffMin (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea,
    riutilizzando nel ciclo uno dei punti calcolati 
    nell'iterazione precedente
    '''

    r = 0.618
    x2 = x0 + r * (x1 - x0)
    g_x2 = g (x2)
    x3 = 0. 
    larghezza = abs (x1 - x0)
     
    while (larghezza > prec):

        x3 = x0 + (1. - r) * (x1 - x0)
        g_x3 = g (x3)
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro        
        if (g_x3 < g_x2):
            # x0 = x0 this extreme does not change
            x1 = x2
            x2 = x3
            g_x2 = g_x3
        else :
            x0 = x1
            x1 = x3
            # x2 = x2 this point does not change

        larghezza = abs (x1-x0)             
                                   
    return (x0 + x1) / 2. 


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def sezioneAureaEffMin_ricorsiva (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo   
    x2,             # punto intermedio dell'intervallo 
    g_x2,           # valore della funzione g in x2     
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    implementata ricorsivamente,
    riutilizzando nel ciclo uno dei punti calcolati 
    nell'iterazione precedente.
    '''

    r = 0.618
    x3 = x0 + (1. - r) * (x1 - x0) 
    g_x3 = g (x3)
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  : return (x0 + x1) / 2.
    elif g_x3 < g_x2 : return sezioneAureaEffMin_ricorsiva (g, x0, x2, x3, g_x3, prec)
    else             : return sezioneAureaEffMin_ricorsiva (g, x1, x3, x2, g_x2, prec)   


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def sezioneAureaMax (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    '''

    r = 0.618
    x2 = 0.
    x3 = 0. 
    larghezza = abs (x1 - x0)
     
    while (larghezza > prec):
        x2 = x0 + r * (x1 - x0) 
        x3 = x0 + (1. - r) * (x1 - x0)  
      
        # si restringe l'intervallo tenendo fisso uno dei due estremi e spostando l'altro        
        if (g (x3) < g (x2)): 
            x0 = x3
            # x1 = x1   this extreme does not change      
        else :
            x1 = x2
            # x0 = x0   this extreme does not change
            
        larghezza = abs (x1-x0)             
                                   
    return (x0 + x1) / 2. 


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def sezioneAureaMax_ricorsiva (
    g,              # funzione di cui trovare lo zero
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    '''
    Funzione che calcola estremanti
    con il metodo della sezione aurea
    implementata ricorsivamente
    '''

    r = 0.618
    x2 = x0 + r * (x1 - x0)
    x3 = x0 + (1. - r) * (x1 - x0) 
    larghezza = abs (x1 - x0)

    if (larghezza < prec)  : return ( x0 + x1) / 2.
    elif (g (x3) < g (x2)) : return sezioneAureaMax_ricorsiva (g, x3, x1, prec)
    else                   : return sezioneAureaMax_ricorsiva (g, x0, x2, prec)   


#--------------



'''
riassunto lezioni:
1) type(), logical operators, str, list, tuple, dictionaries, python script, funzioni, if else for while, librerie moduli.

2) numpy e array (arange, linspace ecc), matplotlib (come disegnare funzioni su grafico), statistics.py

3) leggere dati da testo, istogrammi, bin, sturges, sample moments, pdf, cdf, integrali

4) numeri pseudo casuali, try-and-catch, funzione inversa, teorema centrale del limite

5) class, overloading, lambda, map, filter

6) trovare zeri delle funzioni e estremi, bisezione (ricorsiva), sezione aurea (ricorsiva)

7) poisson, generare eventi con distribuzione di poisson, con funzione inversa

8) toy experiment, integrazione con pseudo random, hit-or-miss, monte carlo
negli esempi c'è un programma che usa hit or miss senza la funzione della libreria

9) notebook, likelihood loglikelihood

10) fit iminuit trova parametri con massima verosomiglianza

11) minimi quadrati, fit qualiti Q^2, matrice covarianza
        
12) fit con distribuzioni binnate
'''

'''
Pezzi di codice utili:

disegno funzione seno nell'intervallo [0, 2pi]:
    fig, ax = plt.subplots (nrows = 1, ncols = 1)  #crea un'immagine vuota 
    x_coord = np.linspace (0, 2 * np.pi, 10000)  
    y_coord = np.sin (x_coord)
    ax.plot (x_coord, y_coord, label='sin (x)')
    ax.set_title ('title', size=14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.legend ()
    plt.show()

mettere dati di un file .txt in un array:
    sample = np.loadtxt ('sample.txt')  #mette i dati di sample.txt in un array
se ho dati in più colonne:
    dati, errori = np.loadtxt('sample.txt', unpack = True)

creare istogramma:
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    N_bins = sturges (len (sample))
    x_range = (xMin, xMax)      #ricorda di definirli prima :)
    bin_content, bin_edges = np.histogram (sample, bins = N_bins, range = x_range)
    #bin_edges = np.linspace (xMin, xMax, N_bins)   #alternativa a usare np.histogram
    ax.hist (unione, bins = bin_edges, color = 'orange')
    ax.set_xlabel('x')
    ax.set_ylabel('conteggi')
    plt.show()

''' 

    '''
riassunti utili
 
import sys 
sys.argv: lista che contiene tutte le parole scritte dopo python3
    esempio: python3 myscript1.py 13 
            sys.argv[0]      #restituisce myscript1.py
            sys.argv[1]      #restituisce 13
 
bin_edges: Un array che definisce i bordi (o intervalli) dei bin dell'istogramma.
            Se ad esempio hai 5 bin e i dati vanno da 0 a 10, i bordi potrebbero essere: [0, 2, 4, 6, 8, 10].
bin_content: Un array che rappresenta il numero di dati in ciascun bin.
            Con i bordi [0, 2, 4, 6, 8, 10], potresti avere un contenuto come [3, 7, 10, 5, 2], dove ogni valore indica quanti dati cadono in quell'intervallo.
N_bins: Il numero totale di bin dell'istogramma. Può essere calcolato direttamente o passato come parametro a funzioni come np.histogram.
        
'''


'''
definizione del main:

def main ():
    # Funzione che implementa il programma principale 
    
    #istruzioni
    
    return

#----------------------------------------------------------

if __name__ == "__main__":
    main () 
'''