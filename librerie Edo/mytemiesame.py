'''
Raccolta temi d'esame svolti da me medesimo stesso.
Spesso non completi e mancheranno pezzi, soluzione potrebbe non combaciare con quella fornita dai professori
delle note indicheranno se funzionano comunque o no

le librerie non verranno implementate,
perchè ho ridefinito le funzioni direttamente nello script o ho usato quelle nelle mie librerie prima che venissero compresse in una unica
in caso ne dovessi creare una apposita la lascio commentata
'''


#-------------------------------------22 GENNAIO 2024-------------------

#fallito la normalizzazione e lultimo punto non ho idea di come si faccia per il resto bene
'''
distribuzione di probabilità
f(x) = {Acos^2(x) se x in (0, 3/2 pi)
       {0         altrimenti
1) con hit-or-miss calcola valore A per normalizzare pdf
2) genera 10.000 punti pseudo casuali distibuiti secondo la pdf f(x) con metodo try-and-catch
3) mostra in istogramma gli eventi generati
4) calcola media, deviazione standard e curtosi
5) mostra che vale teorema centrale  del limite
'''

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor, log, gcd
import random
import mylib

from iminuit import Minuit
from iminuit.cost import LeastSquares

#funzione densità di probabilità
def pdf(x):
    return np.cos(x)**2

def rand_range (xMin, xMax) :
    '''
    generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    '''
    return xMin + random.random () * (xMax - xMin)


def generate_range (xMin, xMax, N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali distribuiti fra xMin ed xMax
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (rand_range (xMin, xMax))   #aggiungo un float tra 0 e 1 alla lista
    return randlist

#integrale con metodo hit-or-miss
def integral_HOM (func, xMin, xMax, yMax, N_evt) :
    '''
    Calcola l'integrale di una funzione nell'intervallo [xMin, xMax] 
    utilizzando il metodo "Hit or Miss".
    
   Args:
        func : funzione da integrare.
        xMin (float): limite inferiore dell'integrazione sull'asse x.
        xMax (float): limite superiore dell'integrazione sull'asse x.
        yMax (float): massimo valore atteso di `func(x)` nell'intervallo [xMin, xMax].
        N_evt (int): numero di punti casuali generati per l'approssimazione.

    Returns:
        tuple: 
            - integral (float): stima dell'integrale.
            - integral_unc (float): incertezza associata all'integrale.
    '''
    x_coord = generate_range (xMin, xMax, N_evt)
    y_coord = generate_range (0., yMax, N_evt)

    points_under = 0
    for x, y in zip (x_coord, y_coord):
        if (func (x) > y) : points_under = points_under + 1 

    A_rett = (xMax - xMin) * yMax
    frac = float (points_under) / float (N_evt)
    integral = A_rett * frac
    integral_unc = A_rett**2 * frac * (1 - frac) / N_evt
    return integral, integral_unc

# Genera un singolo campione da una PDF arbitraria
def mygenera_pdf(xMin, xMax, yMax, pdf):
    '''
    Genera un singolo valore casuale distribuito secondo una PDF arbitraria
    usando il metodo try-and-catch.
    
    Attenzione: yMin è sempre zero

    Args:
        xMin (float): Estremo inferiore del dominio della PDF.
        xMax (float): Estremo superiore del dominio della PDF.
        yMax (float): Valore massimo stimato della PDF.
        pdf (function): Funzione densità di probabilità (PDF) arbitraria.

    Returns:
        float: Valore generato secondo la PDF.
        int: Numero di tentativi effettuati per accettare il valore
    '''
    num = 0
    while True:
        x = random.uniform(xMin, xMax)  # Genera un x casuale nel dominio [xMin, xMax]
        y = random.uniform(0, yMax)     # Genera un y casuale nell'intervallo [0, yMax]
        num += 1
        if y <= pdf(x):                 # Accetta x se y è sotto la PDF
            return x, num

        
# Genera un campione di N valori da una PDF arbitraria
def mygenera(N, xMin, xMax, yMax, pdf):
    '''
    Genera un campione di N valori distribuiti secondo una PDF arbitraria.

    Attenzione: restituisce una lista. Se si vuole un array, dopo aver ottenuto la lista da questa funzione, scrivo:
    campione = np.array(campione)
    
    Args:
        N (int): Numero di campioni da generare.
        xMin (float): Estremo inferiore del dominio della PDF.
        xMax (float): Estremo superiore del dominio della PDF.
        yMax (float): Valore massimo stimato della PDF.
        pdf (function): Funzione densità di probabilità (PDF) arbitraria.

    Returns:
        list: Lista di N valori generati secondo la PDF.
        float: Stima dell'area sotto la PDF.
    '''
    campione = []
    num = 0

    while len(campione) < N:
        x, count = mygenera_pdf(xMin, xMax, yMax, pdf)  # Genera un valore secondo la PDF
        campione.append(x)                              # Aggiungi il valore al campione
        num += count                                    # Aggiorna il numero totale di tentativi

    # Stima dell'area sotto la PDF
    area = (xMax - xMin) * yMax * len(campione) / num
    return campione, area



def main ():
    ''' Funzione che implementa il programma principale '''
    #punto 1 
    #definisco gli estremi del rettangolo su cui voglio integrare
    xMin = 0.
    xMax = 1.5*np.pi
    yMax = 1.
    N = 10000     #N eventi che voglio generare con hom
    Area = (xMax-xMin)*yMax
    hom = integral_HOM(pdf, xMin, xMax, yMax, N)    #tupla con risultato integrale, errore e n_hit
    I = hom[0]
    errI = hom[1]
    Norm = (Area* n_hit)/N*I   #normalizzazione, non sono sicuro sia venuta bene ...
    #print(I, errI, n_hit, Norm)    
    
    
    #punto 2
    #genera 10000punti distribuiti con pdf con try and catch
    #come N sfrutto quella definita sopra
    campione, area = mygenera(N, xMin, xMax, yMax, pdf)
    #fattore normalizzazione è 1/area 
    norm = 1./area
    
    #punto 3
    #mostro dati in istogramma
    fig, ax = plt.subplots (nrows = 1, ncols = 1)              # crea immagine vuota
    N_bins = mylib.sturges (len (campione))
    x_range = (xMin, xMax)      
    bin_content, bin_edges = np.histogram (campione, bins = N_bins, range = x_range)    
    ax.hist (campione, bins = bin_edges, color = 'orange')
    ax.set_title('10000 punti con TAC')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    plt.show()
    
    #punto 4
    #calcolo statistiche dell'istogramma
    stats_calculator = mylib.stats (campione)
    print ('media: \t', stats_calculator.mean ())
    print ('deviazione standard: \t', stats_calculator.sigma ())
    print ('asimmetria: \t', stats_calculator.skewness ())
    print ('curtosi: \t', stats_calculator.kurtosis ())    
    
    #punto 5
    #non ho idea di come farlo
    
    #forse mostrare che vale tcl vuol dire che le statistiche tendono a quelle di una gaussiana ??
    #e come lo mostro??
    #e se generassi i numeri con tcl??
    
    tcl = mylib.generate_TCL(xMin, xMax, N)
    #li metto in altro istogramma???
    fig, ax = plt.subplots (nrows = 1, ncols = 1)              
    N_bins = mylib.sturges (len (tcl))
    x_range = (xMin, xMax)      
    bin_content, bin_edges = np.histogram (tcl, bins = N_bins, range = x_range)    
    ax.hist (tcl, bins = bin_edges, color = 'orange')
    ax.set_title('tcl')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    plt.show()    
    #ok ora è molto gaussiano ma dovrebbe?
    #boh robe strane che non ho capito
    
    
    return

#----------------------------------------------------------

if __name__ == "__main__":
    main () 

    
#-------------------------5 FEBBRAIO 2024---------------------


#bene i primi tre punti poi mi sono perso con i toy experiment
'''
1) Si definisca una funzione phi(x, a, b, c) che traccia un andamento parabolico in funzione di x e se ne
disegni l’andamento nell’intervallo (0, 10):
phi(x, a, b, c) = a + bx + cx^2
con: a = 3 b = 2 c = 1
2) Si generino N = 10 punti x_i distribuiti in modo pseudo-casuale secondo una distribuzione uniforme
sull’intervallo orizziontale e si associ a ciascuno di essi una coordinata
y_i = phi(x_i, a, b, c) + ε_i 
dove ε_i è un numero pseudo casuale generato, con il metodo del teorema centrale del limite, secondo
una distribuzione Gaussiana di media 0 e deviazione standard σ_y = 10.
3. Si faccia un fit della funzione ϕ(x, a, b, c) sul campione così generato (che tecnica bisogna utilizzare?).
4. Si costruisca la distribuzione del Q^2 a partire dal fit effettuato, ripetendolo molte volte utilizzando
toy experiment.
5. Si svolgano i punti precedenti generando gli scarti εi secondo una distribuzione uniforme che abbia la stessa deviazione standard della Gaussiana, disegnando poi la distribuzione del Q2
così ottenuto sovrapposta a quella precedente (per una visualizzazione migliore, si può utilizzare l’opzione
histtype=’step’).
6. In funzione della distribuzione ottenuta per il Q2
, si determini la soglia oltre la quale rigettare il
risultato del fit, dato il suo valore di Q2
, per ottenere un p-value maggiore o uguale di 0.10.
'''

import numpy as np
import matplotlib.pyplot as plt
import mylib
import myrand
import random
from math import sqrt
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import chi2


#funzione phi 
def phi (x,a, b, c):
    return a + b*x +c*x*x

# Genera un singolo campione da una PDF arbitraria
def mygenera_pdf(xMin, xMax, yMax, pdf ,a ,b, c):
    num = 0
    while True:
        x = random.uniform(xMin, xMax)  # Genera un x casuale nel dominio [xMin, xMax]
        y = random.uniform(0, yMax)     # Genera un y casuale nell'intervallo [0, yMax]
        num += 1
        if y <= pdf(x,a , b, c):          # Accetta x se y è sotto la PDF
            return x, num

# Genera un campione di N valori da una PDF arbitraria
def mygenera(N, xMin, xMax, yMax, pdf, a,b ,c):

    campione = []
    num = 0

    while len(campione) < N:
        x, count = mygenera_pdf(xMin, xMax, yMax, pdf, a, b, c)  # Genera un valore secondo la PDF
        campione.append(x)                            # Aggiungi il valore al campione
        num += count                                  # Aggiorna il numero totale di tentativi

    # Stima dell'area sotto la PDF
    area = (xMax - xMin) * yMax * len(campione) / num
    return campione, area



def epsilon (mean, sigma, N_sum = 10) :
    '''
    generazione di un numero pseudo-casuale 
    con il metodo del teorema centrale del limite
    note media e sigma della gaussiana
    '''
    y = 0.
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N_sum) :
        y = y + myrand.rand_range (xMin, xMax)
    y /= N_sum ;
    return y ;

def generate_epsilon (mean, sigma, N, N_sum = 10, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali
    con il metodo del teorema centrale del limite, note media e sigma della gaussiana,
    a partire da un determinato seed
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    delta = sqrt (3 * N_sum) * sigma
    xMin = mean - delta
    xMax = mean + delta
    for i in range (N):
        # Return the next random floating point number in the range 0.0 <= X < 1.0
        randlist.append (myrand.rand_TCL (xMin, xMax, N_sum))
    return randlist





def main():
    '''Funzione che implementa il programma principale '''
    
    #mostro l'andamento di phi nell'intervallo dato con i parametri dati
    a= 3.
    b= 2.
    c= 1.
    xMax = 10.
    xMin = 0.
    x_coord = np.linspace(xMin, xMax, 10000)
    y_coord = phi (x_coord, a,b,c)
    
    '''
    fig, ax = plt.subplots (nrows = 1, ncols = 1)  
    ax.plot (x_coord, y_coord, label='phi')
    ax.set_title ('phi', size=14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.legend ()
    #plt.show()
    '''
    
    #genero 10 numeri pseudocasuali secondo la pdf phi
    N = 10
    yMax = phi(10, a,b, c)
    x_i =np.array ([random.uniform (0, 10) for i in range (10)])
    x_i.sort() 
    media = 0.
    devstd = 10.  #errore della y
    
    
    #######
    #ma che cazzo fa sta cosa
    #y_i = list( map (lambda k:sum(k), zip (phi (x_coord, a, b, c), myrand.generate_TCL_ms (0., devstd, 10))))
    #la faccio a modo mio e inculati
    y_phi = phi(x_i,a,b,c)  #array di 10 elementi
    epsilon = np.array(generate_epsilon(media, devstd, N))   #altro array
    
    y_i = y_phi + epsilon
    #print(y_i, type(y_i))    
    
   
    #mo devo fare fit, provo con minimi quadrati
    fig, ax = plt.subplots ()
    ax.set_title ('fit phi', size=14)
    ax.plot (x_i, y_i, label='phi')
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    ax.errorbar (x_i, y_i, xerr = 0.0, yerr = devstd , linestyle = 'None', marker = 'o') 
    plt.show()

    
    #minimi quadrati
    least_squares = LeastSquares (x_i, y_i, devstd, phi)
    my_minuit = Minuit (least_squares, a= 3, b = 2,c= 1)  # starting values for m and q
    my_minuit.migrad ()  # perche non metti tutte le tabelle fighe????

    N_toys = 10000
    
    Q2_list = []
    for i in range (N_toys):
        x_i =np.array ([random.uniform (0, 10) for i in range (10)])
        x_i.sort() 
        y_phi = phi(x_i,a,b,c)  #array di 10 elementi
        epsilon = np.array(generate_epsilon(media, devstd, N))   #con TCL
        y_i = y_phi + epsilon
        least_squares = LeastSquares (x_i, y_i, devstd, phi)
        my_minuit = Minuit (least_squares, a= 3, b = 2,c= 1)  
        my_minuit.migrad ()  
        Q2_list.append (my_minuit.fval)
    
    
    Q2_unif = []     #tu fai casini
    for i in range (N_toys):
        x_i =np.array ([random.uniform (0, 10) for i in range (10)])
        x_i.sort() 
        y_phi = phi(x_i,a,b,c)  #array di 10 elementi
        epsilon = np.array(random.uniform(0,10))   #uniforme, non so come impostare la devstd
        y_i = y_phi + epsilon
        least_squares = LeastSquares (x_i, y_i, devstd, phi)
        my_minuit = Minuit (least_squares, a= 3, b = 2,c= 1)  
        my_minuit.migrad ()  
        Q2_unif.append (my_minuit.fval)
    
    N_bins = mylib.sturges (len (Q2_list))
    xMin = 0
    xMax = 20
    bin_edges = np.linspace (xMin, xMax, N_bins)
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (Q2_list,
             bins = bin_edges,
             color = 'orange',
             label = 'gaus',
            )
    ax.hist (Q2_unif,
             bins = bin_edges,
             color = 'blue',
             label = 'unif',
             histtype='step',
            )
    ax.set_title ('Q2 distributions', size=14)
    ax.set_xlabel ('Q2')
    ax.set_ylabel ('event counts per bin')
    ax.legend ()
    plt.show ()
    
    return



#----------------------
if __name__ == '__main__':
    main()


#--------------------------19 FEBBRAIO 2024-----------------------

#bene l'inizio poi mi sono perso coi fit
'''
L’espansione dell’universo fu determinata da Edwyn Hubble nel 1929 osservando, nelle galassie visibili
dalla TERRA, un legame matematico fra la loro distanza D_L ed il loro redshift z, che è uno spostamento
Doppler verso frequenze più basse del loro colore naturale (Il redshift è definito come il rapporto fra lo spostamento doppler e la frequenza a riposo z = ∆f /f).
Oggi queste misure vegono svolte osservando un tipo particolare di supernova, dette A1 e le misure correnti vengono utilizzate per decidere se
l’universo sia destinata a continuare o se prima o poi tornerà a contrarsi, con decelerazione q. In assenza
di decelerazione, l’espansione dell’universo è descritta da un andamento lineare:
D_L = (z*c)/H0
dove c = 3 · 105 km/s è la velocità della luce nel vuoto, mentre H0 è una costante di proporzionalità detta
costante di Hubble.
1. Si legga il file SuperNovae.txt e si salvino i dati in tre liste (o array). La prima colonna è il redshift,
la seconda la distanza e la terza l’errore sulla distanza.
2. Si faccia il grafico dei dati mettendo sull’asse x il redshift z e sull’asse y la distanza DL, includendo
gli errori nel grafico.
3. Si esegua un fit dei dati utilizzando il modello lineare e si stampi la costante di Hubble incluso il suo
errore.
4. Si esegua il fit dei dati utilizzando un modello che preveda decelerazione dell’universo:
DL = c/H0 * (z + 1/2 *(1 − q)*z^2)
facendo un grafico con i dati ed i due modelli sovrapposti (utilizzando ax.legend() per identificarli),
decidendo quale dei due si adatti meglio ai dati. Si determinino il valore della costante di Hubble
ed il valore medio della densità dell’universo Ωm ed il loro errore, sapendo che:
q = (3 · Ωm) /2 − 1
'''

import numpy as np
import matplotlib.pyplot as plt
import mylib
import myrand
from iminuit import Minuit
from iminuit.cost import LeastSquares


c=float( 3e5 )#km/s velocità della luce

def lineare(x, m, q):
    y=m*x+q
    return y

def nonlin(x, a, m, q):
    y = a*x*x + m* x + q
    return y
    
def main():
    
    SuperNovae = np.loadtxt ("SuperNovae.txt") #array di array, ogni elemento è una riga del txt
    redshift = []
    dist =[]
    errdist =[]
    
    #metto il primo elemento di ogni elemento in redshift ecc
    for i in range(len(SuperNovae)):
        redshift.append(SuperNovae[i][0])    #x
        dist.append(SuperNovae[i][1])        #y
        errdist.append(SuperNovae[i][2])     #errore y
    
    #un po' fantasioso ma ha funzionato
    #metodo più veloce è semplicemente:
    #redshift, distanza, sigma = np.loadtxt('SuperNovae.txt', unpack=True)
    
    redshift = np.array(redshift)
    dist = np.array (dist)
    errdist = np.array(errdist)

    #metto i dati nel grafico 
    xMin = min(redshift)
    xMax = max(redshift)
    yMin = min(dist)
    yMax = max(dist)
    
    fig, ax = plt.subplots (nrows = 1, ncols = 1)  

    #ax.plot (redshift, dist, label= 'Hubble')
    ax.set_title ('title', size=14)
    ax.set_xlabel ('Redshift')
    ax.set_ylabel ('Distanza')
    ax.errorbar (redshift, dist, xerr = 0.0, yerr = errdist , linestyle = 'None', marker = 'o') 
    #ax.legend ()
    plt.show()
            
    #faccio fit con modello lineare e stampo cost Hubble

    # generate a least-squares cost function
    least_squares = LeastSquares (redshift, dist, errdist, lineare)
    linear = Minuit (least_squares, m = 0., q = 0.)  # m e q non è uno dei parametri ??
    linear.migrad ()  # finds minimum of least_squares function
    #non mi mostra le immaginettte ma forse perchè su qui e non sulla loro macchina virtuale
    #devo stampare hubble
    #for par, val, err in zip (my_minuit.parameters, my_minuit.values, my_minuit.errors) :
    #    print(f'{par} = {val:.3f} +/- {err:.3f}') # formatted output

    m_fit = linear.values[0]
    q_fit = linear.values[1]  
    m_err = linear.errors[0]
    q_err = linear.errors[1]

    H0 = c/m_fit
    H0_err = 1/m_err
    print('costante di Hubble: ', H0, 'e suo errore: ', H0_err) #facciamo finta si faccia così
        
    #ora rifaccio stessa cosa ma con modello non lineare
        # generate a least-squares cost function
    least_squares_nonlin = LeastSquares (redshift, dist, errdist, nonlin)
    nonlinear = Minuit (least_squares_nonlin, a= 0., m = 0., q = 0.)  # m e q non è uno dei parametri ??
    nonlinear.migrad ()  # finds minimum of least_squares function
    
    a_fit = nonlinear.values[0]
    m_fit_nonlin = nonlinear.values[1]
    q_fit_nonlin = nonlinear.values[2] 
    
    a_err = nonlinear.errors[0]
    m_err_nonlin = nonlinear.errors[1]
    q_err_nonlin = nonlinear.errors[2]
    
    #print(m_fit, m_err, q_fit, q_err, a_fit,a_err)
    
    H0_nonlin = c/m_fit_nonlin
    H0_err_nonlin = 1/m_err_nonlin
    print('costante di Hubble non lin: ', H0_nonlin, 'e suo errore: ', H0_err_nonlin) #facciamo finta si faccia così
        
    #q = ((3*omega_m)/2 ) -1 
    #inverto la formula e trovo omega   
    omega_m = (2/3) * (q_fit_nonlin +1)
    omega_m_err = q_err_nonlin
    print(omega_m,omega_m_err)   #il valore ha senso l'errore no :/
    
    #devo mettere i fit in un grafico
    fig, ax = plt.subplots()
    ax.set_title ('sovrapposizione lin-non lin', size=14)
    ax.plot(redshift, lineare(redshift, m_fit, q_fit), label = 'Lineare')
    ax.plot(redshift, nonlin(redshift, a_fit, m_fit_nonlin, q_fit_nonlin), label = 'Non lineare')
    ax.set_xlabel ('Redshift')
    ax.set_ylabel ('Distanza')
    ax.errorbar (redshift, dist, xerr = 0.0, yerr = errdist , linestyle = 'None', marker = 'o') 
    ax.legend ()
    plt.show()
    
    return




#------------------

if __name__ == '__main__':
    main()



#---------------------24 GIUGNO 2024-----------------------------

#anche questo lo ho corretto molto e preso le funzioni definite dal prof
'''
genera N_exp eventi con pdf esponenziale compresi tra 0 e 3 tau, poi genera N_gau eventi con pdf gaussiana con mu e sigma
mettili insieme e facci istogramma
fai un fit
crea funzione che fa log max veros
giocando con la media calca max log likelihood
'''


import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor, log, exp
import random
import myrand
import mylib
from iminuit import Minuit
from scipy.stats import expon, norm
from iminuit.cost import ExtendedBinnedNLL
from IPython.display import display


def rand_range (xMin, xMax) :
    return xMin + random.random () * (xMax - xMin)

#questi poi me li copio
def try_and_catch_exp (lamb, N):
    events = []
    x_max = 3/lamb         #mi prende fino a 3 tau
    for i in range (N):
        x = rand_range (0., x_max)
        y = rand_range (0., lamb)
        while (y > lamb * exp (-lamb * x)):
            x = rand_range (0., x_max)
            y = rand_range (0., lamb)
        events.append (x)
    return events


def try_and_catch_gau (mean, sigma, N):
    events = []
    for i in range (N):
        x = rand_range (mean - 3 * sigma, mean + 3 * sigma)  #mi prende fino a 3 sigma
        y = rand_range (0., 1.)
        while (y > exp (-0.5 * ( (x - mean)/sigma)**2)):
            x = rand_range (mean - 3 * sigma, mean + 3 * sigma)
            y = rand_range (0, 1.)
        events.append (x)
    return events
    

#modello esponenziale + gaussiana
def mod_total (bin_edges, N_signal, mu, sigma, N_background, tau):
    return N_signal * norm.cdf (bin_edges, mu, sigma) + \
            N_background * expon.cdf (bin_edges, 0, tau )


def pdf (x, mean, sigma, f_exp, f_gau, gau_norm, lam):    #ma è identica a mod total, si ma no questa è la funzione l'altra istogramma
    return f_exp * lam * np.exp (-x * lam) + \
             f_gau * gau_norm * np.exp (-0.5 * ((x - mean)/sigma )**2)  
           # il simbolo \ serve per andare accapo senza terminare la linea di istruzione


def loglikelihood (mean, sigma, f_exp, f_gau, gau_norm, lam, pdf, sample) :
    risultato = 0.
    for x in sample:
        if (pdf (x, mean, sigma, f_exp, f_gau, gau_norm, lam) > 0.) : risultato = risultato + log (pdf (x, mean, sigma, f_exp, f_gau, gau_norm, lam))
    return risultato



#trova massimo della loglikelihood (solo lei però, non è generalizzata)
def sezioneAureaMax (
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo       
    sigma, f_exp, f_gau, gau_norm, lam, pdf, sample,
    prec = 0.0001): # precisione della funzione        
    r = 0.618
    x2 = 0.
    x3 = 0. 
     
    while (abs (x1 - x0) > prec):  # x0, x3, x2, x1
        x2 = x0 + r * (x1 - x0) 
        x3 = x0 + (1. - r) * (x1 - x0)  
        if loglikelihood (x3, sigma, f_exp, f_gau, gau_norm, lam, pdf, sample) < loglikelihood (x2, sigma, f_exp, f_gau, gau_norm, lam, pdf, sample):
            x0 = x3
        else :
            x1 = x2
    return (x0 + x1) / 2.



def main ():
    
    N_exp = 2000
    tau = 200
    lamb = 1/tau
    
    N_gau = 200
    mu = 190.
    sigma = 20.
    '''
    pdfexp = myrand.generate_exp(tau, N_exp)
    pdfgau = myrand.generate_TCL_ms (mu, sigma, N_gau)
    per exp non sono riuscito a dirgli che lo coglio tra 0 e 3 tau
    problema potrebbe essere che ho usato due metodi diversi per generarli, 
    uno è funzione inversa, l'altro teorema centrale limite
    provo con entrambi try and catch ma devo creare le funzioni
    '''
    numexp = try_and_catch_exp(lamb, N_exp)
    numgau = try_and_catch_gau(mu, sigma, N_gau)
    
    unione = numexp + numgau
    #print(type(unione))
    #random.shuffle(unione)  ??? a che pro?
    
    
    #ci faccio istogramma

    N_bins = mylib.sturges (len (unione))
    bin_content, bin_edges = np.histogram (unione, bins = N_bins, range = (0.,3*200.))
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (unione, bins = bin_edges, color = 'orange')
    ax.set_xlabel('x')
    ax.set_ylabel('conteggi')
    #plt.show()

    
    #fitto l'istogramma
    
    my_cost_func = ExtendedBinnedNLL (bin_content, bin_edges, mod_total)

    mediatot = np.mean (unione)
    sigmatot = np.std (unione)

    N_events = sum (bin_content)
    my_minuit = Minuit (my_cost_func, 
                        N_signal = N_events, mu = mediatot, sigma = sigmatot, 
                        N_background = N_events, tau = mediatot)                  

    my_minuit.migrad ()   
    print (my_minuit.valid)   #a volte funziona e a volte no non so perchè
    #display (my_minuit)

    
    '''
    mu_tot = mylib.mean(unione)
    sigma_tot = mylib.stdev(unione)

    N_events = sum (bin_content)

    # the cost function for the fit
    my_cost_func = ExtendedBinnedNLL (bin_content, bin_edges, expgau)

    # the fitting algoritm
    my_minuit = Minuit (my_cost_func, 
                        N_gau = N_events, mu = mu_tot, sigma = sigma_tot, 
                        N_exp = N_events, tau = 1.) #ah ok probabilmente problema era in definizione a cazzo di tau...                        

    # bounds the following parameters to being positive
    my_minuit.limits['N_gau', 'N_exp', 'sigma', 'tau'] = (0, None)
    

    my_minuit.migrad ()
    print (my_minuit.valid) #non ha funzionato esce false
    display (my_minuit)
    '''
    #funzione gaussiana gauss(x, mu, sigma) = 1/(sqrt(2*np.pi*sigma)) * e**(-1/2*((x*mu)/sigma)**2)
    #funzione max log verosimi con pdf = a* exp(x, lambda) + b* gauss(x, mu, sigma)
    # questi sono i parametri del modello che rimangono fissati dal fit
    tau = my_minuit.values['tau']
    lam = 1/tau
    sigma = my_minuit.values['sigma']
    gau_norm = 1. / (np.sqrt (2 * np.pi) * sigma)
    f_exp = my_minuit.values['N_background']
    f_gau = my_minuit.values['N_signal']
    f_tot = f_exp + f_gau
    f_exp = f_exp / f_tot
    f_gau = f_gau / f_tot

    
    #si calcoli il valore del logaritmo della verosimiglianza per il campione dato il modello,
    #variando il valore del parametro mean, fra 30 e 300, con passo costante e se ne disegni l'andamento
    
    #plotto la pdf
    fig, ax = plt.subplots ()
    x = np.linspace (30, 300, 100)
    ax.plot (x, pdf(x, my_minuit.values['mu'], my_minuit.values['sigma'],f_exp, f_gau, gau_norm, lam), color = 'blue')
    ax.set_xlabel ('x')
    ax.set_ylabel ('pdf')
    #plt.show ()

    
    #plotto la loglikelihood
    fig, ax = plt.subplots ()

    x_coord = np.linspace (30, 300, 100)
    l_like = []
    for x in x_coord: l_like.append (loglikelihood (x, sigma, f_exp, f_gau, gau_norm, lam, pdf, unione))   
    y_coord = np.array (l_like)

    ax.plot (x_coord, y_coord, color = 'red')
    ax.set_xlabel ('media')
    ax.set_ylabel ('log-likelihood')
    #plt.show()
    
    #cerca massimo della loglikelihood
    mean_maxll = sezioneAureaMax (20, 300,sigma, f_exp, f_gau, gau_norm, lam, pdf, unione)
    print('Il massimo della loglikelihood è: ',mean_maxll)       #viene un valore sensato :))) 
    return

#------------------

if __name__ == "__main__":
    main () 


#-----------------------8 LUGLIO 2024-------------------

#questo lo avevo fatto abbastanza bene ero soddisfatto ma mancava punto 5
'''
1. Si scriva una funzione che simuli il cammino degli abitanti del villaggio dopo aver bevuto la grappa,
assumendo che si spostino in piano, che ogni passo abbia direzione casuale uniforme angolarmente
ed una lunghezza distribuita secondo una distribuzione Gaussiana con media 1 e larghezza 0.2,
troncata a valori positivi.
2. Immaginando che il calderone si trovi alle coordinate (0, 0) sul piano, si scriva una funzione che
calcoli la posizione (x, y) raggiunta da Asterix dopo N = 10 passi e si disegni il suo percorso.
3. Si consideri ora l’intera popolazione: si determini la posizione (x, y) di ogni abitante dopo N =
10 passi a partire dal calderone e si disegni le distribuzione della distanza raggiunta dal punto di
partenza, assumendo la popolazione totale composta da 10000 persone.
4. Si determinino media, varianza, asimmetria e curtosi della distribuzione ottenuta.
5. Se la lunghezza dei passi è costante uguale ad 1, la distribuzione delle distanze r dopo N passi segue
una distribuzione di Rayleigh:
f(r) = 2r/N * e**(-r^2/N)
Si utilizzi un fit per determinare, a partire dalla distribuzione di distanze costruita in queste ipotesi,
il numero di passi effettuati, sapendo che la distribuzione di Rayleigh è presente in scipy come
scipy.stats.rayleigh e che per ottenere la forma funzionale di interesse per il problema questa
distribuzione ha come parametri loc = 0 e scale =
p
N/2 (dove N è il numero di passi).
'''

import numpy as np 
import matplotlib.pyplot as plt
import math
import random
import mylib
from myclasses import stats
import myrand
from iminuit import Minuit
from iminuit.cost import LeastSquares
from scipy.stats import rayleigh
from iminuit.cost import ExtendedBinnedNLL

def Rayleigh(r,N):
    return 2*r/N * e**(-(r**2)/N)


def main ():
    ''' Funzione che implementa il programma principale '''
    
    #angolo e lunghezza di un singolo passo di un singolo abitante
    #limiti angolari
    aMin = 0.
    aMax = 360.
    angolo = myrand.rand_range(aMin,aMax)  
    
    #limiti lunghezza e gaussiana
    xMin = 0.
    media = 1.
    sigma = 0.2
    lungh = myrand.rand_TCL_ms(media, sigma)
    
    x = lungh * np.cos(angolo)
    y = lungh *np.sin(angolo)
    
    print('Dopo un passo Asterix si trova nella posizione: (', x,',',y,')')   
    
    N = 10

    r = np.array(myrand.generate_TCL_ms(media, sigma, N) )    #array di 10 elementi che sono le lunghezze di ogni passo
    theta = np.array(myrand.generate_range(aMin, aMax, N ))   #array di 10 elementi che sono gli angoli di ogni passo
    #trasformo r e theta in x e y
    x_Asterix = r*np.cos(theta) 
    y_Asterix = r*np.sin(theta)
    
    #distanza totale percorsa dopo 10 passi da singola persona
    somma_x = sum (x_Asterix)
    somma_y = sum(y_Asterix)
    print('Dopo 10 passi Asterix si trova nella posizione: (', somma_x,',',somma_y,')')
    
    
    #considero l'intera popolazione
    N_tot = 10000
    popolazione_x = []
    popolazione_y = []
    
    for i in range (N_tot):
        r_pop = np.array(myrand.generate_TCL_ms(media, sigma, N) )    #array di 10 elementi che sono le lunghezze di ogni passo
        theta_pop = np.array(myrand.generate_range(aMin, aMax, N ))   #array di 10 elementi che sono gli angoli di ogni passo

        x_pop = r_pop*np.cos(theta_pop) 
        y_pop = r_pop*np.sin(theta_pop)

        somma_popx = sum (x_pop)
        somma_popy = sum(y_pop)

        popolazione_x.append(somma_popx)
        popolazione_y.append(somma_popy)
    
    print(len(popolazione_x), len(popolazione_y))
    
    #devo disegnare la distribuzione della distanza raggiunta dal punto di partenza
    #faccio prova con uno solo
    
    dist = math.sqrt(x**2 + y**2)
    print('Asterix è distante ', dist, ' dal calderone')
    #dopo 10 passi:
    dist10= math.sqrt(somma_x**2 + somma_y**2)
    print('dopo 10 passi invece è distante: ', dist10)
    
    #ora lintera popolazione....
    distpop = []   
    for j in range(N_tot):
        popdist = math.sqrt(popolazione_x[j]**2 + popolazione_y[j]**2)
        distpop.append(popdist)
    
    
    distpop = np.array(distpop)
    #print(len(distpop), type(distpop))
    
    #devo disegnare la distribuzione ora
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    N_bins = mylib.sturges (len (distpop))
    xMin = 0.
    xMax = max(distpop)    
    x_range = (xMin, xMax)      
    bin_content, bin_edges = np.histogram (distpop, bins = N_bins, range = x_range)
    ax.hist (distpop, bins = bin_edges, color = 'orange')
    ax.set_xlabel('Distanza dal calderone')
    ax.set_ylabel('Abitanti a quella distanza')
    #plt.show()
    
    #media devstd kurtosi e skewness
    stats_calculator = stats (distpop)
    print ('mean    :', stats_calculator.mean ())
    print ('sigma   :', stats_calculator.sigma ())
    print ('skewness:', stats_calculator.skewness ())
    print ('kurtosis:', stats_calculator.kurtosis ())
    
    #quinto punto lo faccio domani è arrviato domani
    '''
    5. Se la lunghezza dei passi è costante uguale ad 1, la distribuzione delle distanze r dopo N passi segue
    una distribuzione di Rayleigh:
    f(r) = 2r/N * e**(-r^2/N)
    Si utilizzi un fit per determinare, a partire dalla distribuzione di distanze costruita in queste ipotesi,
    il numero di passi effettuati, sapendo che la distribuzione di Rayleigh è presente in scipy come
    scipy.stats.rayleigh e che per ottenere la forma funzionale di interesse per il problema questa
    distribuzione ha come parametri loc = 0 e scale = sqrt(N/2) (dove N è il numero di passi).

    '''
    my_cost_func = ExtendedBinnedNLL (bin_content, bin_edges, Rayleigh)  #non funziona lei ...
    my_minuit = Minuit (my_cost_func, loc = 0, scale = math.sqrt(N/2))
    
    is_valid = my_minuit.valid
    print(is_valid)
    #my_minuit.values    # conterrà i valori ottimali dei parametri (par0 e par1).
    #my_minuit.fval      # sarà il valore minimo della funzione di costo (x^2).

    #my_minuit.hesse ()        # hesse calcola la matrice di Hessian della funzione di costo (X^2) nel punto di minimo.
    #La matrice Hessiana rappresenta la curvatura della funzione di costo intorno al minimo.
    #Questo metodo stima le incertezze sui parametri ottimali (delta_par0 e delta_par1), che verranno salvate in my_minuit.errors.

    
    return

#----------------------------------------------------------

if __name__ == "__main__":
    main () 

#---------------------16 SETTEMBRE 2024----------------------


#qui avevo fatto pena non avevo idea di come fare la classe e utilizzarla poi, anche con integrali male
#giusto qui avevo creato libreria apposta, la lascio in fondo a questo tema commentata
'''
L’ottimizzazione dell’integrazione numerica con il metodo Monte Carlo si ottiene, fra le altre cose, con una
scelta oculata delle coordinate x dei punti generati casualmente. Infatti, più essi ricoprono in maniera
ottimale l’insieme di definizione della funzione da integrare, migliore è la precisione ottenuta nella sua
stima, a parità di punti generati. La sequenza sn generata secondo il seguente algoritmo:
sn+1 = (sn + α) mod 1 (1)
produce un insieme di punti, distribuiti fra 0 ed 1, che hanno la proprietà di ben riempire questo insieme
di definizione, in particolare se α = (√5 − 1)/2.
1. Si scriva una libreria che contenga una classe di python, chiamata additive_recurrence, che generi
la sequenza di numeri sn dell’equazione (1), che abbia come variabili membro il parametro α, il
numero di partenza della sequenza e l’ultimo numero generato, che assegni un valore ad α durante
l’inizializzazione della classe ed implementi i metodi seguenti:
• get_number per ottenere un numero della sequenza
• set_seed per inizializzare la sequenza
2. Si faccia un test del funzionamento della classe generando una sequenza di 1000 numeri e scrivendone i primi 10 a schermo.
3. Si aggiunga alla libreria una funzione chiamata MC_mod che calcoli l’integrale definito di f(x) = 2x^2
nell’intervallo (0, 1), utilizzando il metodo crude Montecarlo dove la generazione dei punti lungo
l’asse x non sia fatta in modo pseudo-casuale, ma utilizzando la classe additive_recurrence.
4. Utilizzando il metodo dei toy experiment, si determini l’incertezza del calcolo dell’integrale in funzione del numero totale N_points di punti generati per la stima di un singolo integrale, disegnandone l’andamento dell’errore in funzione di N_points al variare fra 10 e 25000.
5. Si rifaccia il medesimo test con l’algoritmo crude Montecarlo studiato a lezione e si confrontino i due
risultati: quale è più efficiente?
'''


import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor, log, gcd
import random
import mylib
from libesame import additive_recurrence as ar
import libesame

from iminuit import Minuit
from iminuit.cost import LeastSquares

#punto 1 da vedere nella libreria

def f(x):
    return 2*x**2

def main ():
    ''' Funzione che implementa il programma principale '''
    #punto 2 faccio test libreria e classe generare 1000 numeri e scrivi i primi 10
    
    alpha = (sqrt(5)-1)/2
    
    seq = ar(alpha)
    myseq =[]
    for i in range (1000):
        myseq.append(seq.get_number())
    print(myseq[:10])
      
    #punto 3
    xMin = 0.
    xMax = 1.
    x_range = (xMin, xMax)
    N = 1000
    integrale = libesame.MC_mod(N)
    
    print(integrale)
    
    return

#----------------------------------------------------------

if __name__ == "__main__":
    main () 


    
'''
------------------LIBRERIA ESAME SETTEMBRE------------

#questa è la libreria
#ci sarebbero cose commentate ma le ho tolte dagli apici
#deve contenere una classe di python additive_recurrence

import math
import numpy as np
import random

class additive_recurrence :

    def __init__ (self, alpha = 0.618034) : # (sqrt(5)-1)/2
        self.alpha = alpha
        self.s_0 = 0.5
        self.s_n = 0.5
        #ma perche 0.5???
        
    def get_number (self) :
        #tu mi generi il numero n-esimo della sequenza
        self.s_n = (self.s_n + self.alpha) % 1    # %1 mod 1 resto ?? ocsa cazoz faceva il mod 
        return self.s_n

    def set_seed (self, seed) :
        self.s_0 = seed
        self.s_n = seed
   
    def get_numbers (self, N) :
        #tu generi l'intera lista fino a N numeri
        lista = []
        for i in range (N) : lista.append (self.get_number ())
        return lista

#3. Si aggiunga alla libreria una funzione chiamata MC_mod che calcoli l’integrale definito di f(x) = 2x^2
#nell’intervallo (0, 1), utilizzando il metodo crude Montecarlo dove la generazione dei punti lungo
#l’asse x non sia fatta in modo pseudo-casuale, ma utilizzando la classe additive_recurrence.


def rand_range (xMin, xMax) :
    
    #generazione di un numero pseudo-casuale distribuito fra xMin ed xMax
    return xMin + random.random () * (xMax - xMin)


def MC_mod (N_points) :
    gen_seq = additive_recurrence ()
    sotto = float (0)
    for i in range (N_points):
        x = gen_seq.get_number ()
        y = rand_range (0., 2.)
        if (y < 2 * x * x) : sotto += 1
    frazione = sotto / N_points
    integrale = 2 * frazione
    sigma = 2 * np.sqrt (frazione * (1 - frazione) / N_points)
    return integrale, sigma



def integral_CrudeMC (g, xMin, xMax, N_rand) :
    
    #Calcola l'integrale di una funzione g nell'intervallo [xMin, xMax] 
    #usando il metodo Monte Carlo "Crude" (diretto).
#
    #Args:
    #    g (callable): funzione da integrare.
    #    xMin (float): limite inferiore dell'integrazione sull'asse x.
    #    xMax (float): limite superiore dell'integrazione sull'asse x.
    #    N_rand (int): numero di punti casuali generati per l'approssimazione.
#
    #Returns:
    #    tuple: 
    #        - integral (float): stima dell'integrale.
    #        - integral_unc (float): incertezza associata all'integrale.
  
    somma     = 0.
    sommaQ    = 0.    
    for i in range (N_rand) :
        x = rand_range (xMin, xMax)
        somma += g(x)
        sommaQ += g(x) * g(x)     
     
    media = somma / float (N_rand)
    varianza = sommaQ /float (N_rand) - media * media 
    varianza = varianza * (N_rand - 1) / N_rand
    lunghezza = (xMax - xMin)
    return media * lunghezza, sqrt (varianza / float (N_rand)) * lunghezza
   
'''


#-------------------------10 OTTOBRE 2024----------------

#era uscito bene per me, non completo e molti passaggi semplificabili ma funzionava fino a un certo punto, no punto 5
'''
Secondo l’algoritmo di Box-Müller, dati due numeri pseudo-casuali x1 ed x2 generati uniformemente
nell’intervallo (0, 1), si dimostra che i due numeri g1 e g2 calcolati con le equazioni seguenti:
g1 = sqrt(−2 log(x_1)) cos (2πx_2) (1)
g2 = sqrt(−2 log(x_1)) sin (2πx_2) (2)
possano essere considerati due numeri pseudo-casuali distribuiti secondo una distribuzione di densità di
probabilità normale.
1. Si scriva una funzione chiamata generate_gaus_bm che generi coppie di numeri pseudo-casuali
distribuiti secondo una densità di probabilità Gaussiana utilizzando l’algoritmo di Box-Müller, implementata in una libreria dedicata.
2. Si generino N = 1000 numeri pseudo-casuali utilizzando la funzione appena sviluppata e li si disegni
in un istogramma, scegliendone con un algoritmo opportuno gli estremi ed il binnaggio.
3. Si determinino media e varianza della distribuzione ottenuta e relativi errori.
4. Si mostri graficamente che, al variare del numero N di eventi generati, la sigma della distribuzione
non cambia, mentre l’errore sulla media si riduce.
5. Si trasformi l’algoritmo in modo che generi numeri pseudo-casuali con densità di probabilità Gaussiana con media µ = 5 e varianza σ
2 = 4. Si generi un nuovo campione di N = 1000 eventi con il
nuovo algoritmo e se ne disegni la distribuzione, sempre scegliendo in modo opportuno gli estremi
ed il binnaggio dell’istogramma corrispondente.
'''


import mylib
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, ceil, floor, log, gcd
import random

#secondo me qui devo usare la base 10 o almeno io faccio così

#punto 1 scrivo la funzione
#dovrei farlo in una libreria separata ma non ho voglia scrivo tutto qui

#genera numeri psuedocasuali in [0;1] uniformemente  , questo lavoto lo fa già random.random() ...
def generate_uniform (N, seed = 0.) :
    '''
    generazione di N numeri pseudo-casuali distribuiti fra 0 ed 1
    a partire da un determinato seed.
    
    Attenzione: restituisce una lista, non necessario passare come argomento un seed
    
    Args:
        N(int): numero campioni da generare
        seed(float): seed da cui far partire la generazione di numeri pseudocasuali, non necessario specificarlo
    
    Returns:
        list: lista di N numeri generati 
    '''
    if seed != 0. : random.seed (float (seed))
    randlist = []
    for i in range (N):
        randlist.append (random.random ())    #aggiungo a randlist un float tra 0 e 1
    return randlist

#genera numeri pc con bm
def generate_gaus_bm (x_1, x_2):
    '''
    Genera coppie di numeri pseudo-casuali con pdf gaussiana sfruttando l'algoritmo di Box-Mueller
    
    Args:
        x_1, x_2 (float) : numeri generati da pdf uniforme
    
    Returns:
        bm_list (lista): lista contenente i due valori g_1 e g_2 generati con algoritmo bm e aventi pdf gaussiana 
    '''
    bm_list = []
    g_1 = sqrt(-2*np.log(x_1))* np.cos(2*np.pi*x_2)
    g_2 = sqrt(-2*np.log(x_1))* np.sin(2*np.pi*x_2)
    bm_list.append(g_1)
    bm_list.append(g_2)
    return bm_list

#si poteva evitare sto casino, vedi soluzione che è piu compatta

#non ho copiato qua sturges
#anche la classe stats sbatti rimane in mylib


#def punto5 ():  non lho fatto alla fine
    '''
    genera coppie di numeri con algormitmo bm ma con media 5 e varianza 4, quindi simga 2 
    '''
#    return

def main():
    #per utilizzare l'algoritmo bm innnanzitutto devo generare x_1 e x_2 uniformemente
    n_unif = generate_uniform(2)
    x_1 = n_unif[0]
    x_2 = n_unif[1]
    #print(n_unif, x_1, x_2)
    
    #ora dati questi due numeri posso sfruttare bm per avere due numeri con pdf gaussiana
    n_gaus = generate_gaus_bm(x_1, x_2)
    g_1 = n_gaus[0]
    g_2 = n_gaus[1]
    #print(n_gaus, len(n_gaus), g_1, g_2)
    
    #punto 2
    #genera 1000 numeri tramite bm e inseriscili in istogramma
    #io ho funzione che mi genera coppie di numeri, quindi se voglio 1000 numeri devo chiamare la funzione 500 volte
    bm = []
    for i in range (500):
        N_unif = generate_uniform(2)    #genero coppia numeri distribuiti uniformemente
        x1 = N_unif[0]
        x2 = N_unif[1]

        N_gaus = generate_gaus_bm(x1, x2)   #genero coppia con bm
        bm.extend(N_gaus)
    
    #print(len(bm), max(bm), min(bm))     #dovrebbe aver avuto senso
    
    #ora metto i numeri generati in un istogramma
    xMin = min(bm)          #scelgo direttamente il min e max della lista come estremi
    xMax = max(bm)
    fig, ax = plt.subplots (nrows = 1, ncols = 1)             
    N_bins = mylib.sturges (len (bm))    #sfrutto l'algoritmo di sturges per il numero di bin
    x_range = (xMin, xMax)     
    bin_content, bin_edges = np.histogram (bm, bins = N_bins, range = x_range)    
    ax.hist (bm, bins = bin_edges, color = 'orange')
    ax.set_title('1000 numeri generati', fontsize = 10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid()
    plt.show()
    
    #punto 3
    #determina media e varianza con relativi errori
    stat_histo = mylib.stats(bm)
    print('Media con 1000 numeri: ', stat_histo.mean())
    print('Varianza con 1000 numeri: ', stat_histo.variance())
    #mi serve l'errore però che non so calcolare, da soluzione si calcola come sotto:
    
    #media_err = statistiche.sigma_mean ()    #errore sulla media 
    #varia_err = 2 * varia * varia / (len (lista_G) - 1)    #errore sulla varianza

    
    
    #punto 4
    #Si mostri graficamente che, al variare del numero N di eventi generati, la sigma della distribuzione
    #non cambia, mentre l’errore sulla media si riduce.
    #ehehe quindi mi serve saper calcolare lerrore 
    
    bm_2 = []
    for i in range (1000):
        N_unif_2 = generate_uniform(2)    #genero coppia numeri distribuiti uniformemente
        x1_2 = N_unif_2[0]
        x2_2 = N_unif_2[1]

        N_gaus_2 = generate_gaus_bm(x1_2, x2_2)   #genero coppia con bm
        bm_2.extend(N_gaus_2)    
        if i == 10:
            xMin = min(bm_2)          #scelgo direttamente il min e max della lista come estremi
            xMax = max(bm_2)
            fig, ax = plt.subplots (nrows = 1, ncols = 1)             
            N_bins = mylib.sturges (len (bm))    #sfrutto l'algoritmo di sturges per il numero di bin
            x_range = (xMin, xMax)     
            bin_content, bin_edges = np.histogram (bm, bins = N_bins, range = x_range)    
            ax.set_title('20 numeri generati', fontsize = 10)
            ax.hist (bm, bins = bin_edges, color = 'orange')
            stat_histo_10 = mylib.stats(bm_2)
            print('Sigma con 20 numeri: ', stat_histo_10.sigma())
            #plt.show()
        if i == 100:
            xMin = min(bm_2)          #scelgo direttamente il min e max della lista come estremi
            xMax = max(bm_2)
            fig, ax = plt.subplots (nrows = 1, ncols = 1)             
            N_bins = mylib.sturges (len (bm))    #sfrutto l'algoritmo di sturges per il numero di bin
            x_range = (xMin, xMax)     
            bin_content, bin_edges = np.histogram (bm, bins = N_bins, range = x_range)    
            ax.set_title('200 numeri generati', fontsize = 10)
            ax.hist (bm, bins = bin_edges, color = 'orange')            
            stat_histo_100 = mylib.stats(bm_2)
            print('Sigma con 200 numeri: ', stat_histo_100.sigma())            
            #plt.show()
        if i == 999:
            xMin = min(bm_2)          #scelgo direttamente il min e max della lista come estremi
            xMax = max(bm_2)
            fig, ax = plt.subplots (nrows = 1, ncols = 1)             
            N_bins = mylib.sturges (len (bm))    #sfrutto l'algoritmo di sturges per il numero di bin
            x_range = (xMin, xMax)     
            bin_content, bin_edges = np.histogram (bm, bins = N_bins, range = x_range)    
            ax.set_title('2000 numeri generati', fontsize = 10)
            ax.hist (bm, bins = bin_edges, color = 'orange')
            stat_histo_1000 = mylib.stats(bm_2)
            print('Sigma con 2000 numeri: ', stat_histo_1000.sigma())            
            #plt.show()
     
    #fa cagare sta roba e non era neanche quello che ha chiesto
    #non so calcolare l'errore quindi mi manca quel ppunto   che era semplicemente sigma mean
    
    #5. Si trasformi l’algoritmo in modo che generi numeri pseudo-casuali con densità di probabilità Gaussiana con media µ = 5 e varianza 
    #sifma^2 = 4. Si generi un nuovo campione di N = 1000 eventi con il
    #nuovo algoritmo e se ne disegni la distribuzione, sempre scegliendo in modo opportuno gli estremi
    #ed il binnaggio dell’istogramma corrispondente.
    
    #ok quindi creo un'altra funzione, in realta no...
    
    return

#----------------------------------------------------------

if __name__ == "__main__":
    main () 