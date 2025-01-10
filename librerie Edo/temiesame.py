'''
Raccolta di tutti i temi d'esame svolti in python
Viene mostrata la versione corretta fornita dai professori 
se io ho svolto il tema con altri metodi saranno presenti commentati

Temi distribuiti in ordine cronologico dal più vecchio al più giovane 
'''


#-------------------------------------22 GENNAIO 2024-------------------


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
from lib import pdf, genera, sturges, rand_TCL
from stats import stats


if __name__ == '__main__':

    # draw the pdf
    # ---- ---- ---- ---- ---- ---- ---- 

    fig, ax = plt.subplots (nrows = 1, ncols = 1)

    # preparing the set of points to be drawn 
    x_coord = np.linspace (0, 1.5 * np.pi, 10000)
    y_coord_1 = pdf (x_coord)

    # visualisation of the image
    ax.plot (x_coord, y_coord_1, label='pdf')
    ax.set_title ('probability density function', size=14)
    ax.set_xlabel ('x')
    ax.set_ylabel ('y')
    plt.savefig ('pdf.png')
    #  plt.show ()

    # generate the sample and calculate the integral
    # ---- ---- ---- ---- ---- ---- ---- 

    campione, area = genera (10000)

    # draw the histogram of the sample
    # ---- ---- ---- ---- ---- ---- ---- 

    ax.set_title ('generated sample', size=14)
    print ('generati', len (campione),'eventi')
    print ("l'area della pdf prima della normalizzazione è",area)
    print ('il fattore di normalizzazione è', 1./area)
    N_bins = sturges (len (campione))
    bin_edges = np.linspace (0, 1.5 * np.pi, N_bins)
    ax.hist (campione, bin_edges, color = 'orange')
    plt.savefig ('histo.png')

    # calculate moments
    # ---- ---- ---- ---- ---- ---- ---- 

    my_stats = stats (campione)
    print ('mean    :', my_stats.mean ())
    print ('sigma   :', my_stats.sigma ())
    print ('skewness:', my_stats.skewness ())
    print ('kurtosis:', my_stats.kurtosis ())

    # study the Gaussian behaviour
    # ---- ---- ---- ---- ---- ---- ---- 

    N_events = 10000
    means = []
    sigmas = []
    skews = []
    kurts = []
    x_axis = [2**j for j in range(0,6)]
    for N_sum in x_axis:
        campione_loc = [rand_TCL (N_sum) for j in range (N_events)]
        my_stats = stats (campione_loc)
        means.append (my_stats.mean ())
        sigmas.append (my_stats.sigma ())
        skews.append (my_stats.skewness ())
        kurts.append (my_stats.kurtosis ())

      fig, ax = plt.subplots (nrows = 4, ncols = 1)
      ax[0].plot (x_axis, means, label='mean')
      ax[1].plot (x_axis, sigmas, label='sigma')
      ax[2].plot (x_axis, skews, label='skewness')
      ax[3].plot (x_axis, kurts, label='kurtosis')
      plt.savefig ('stats.png')

      campione_gaus = [rand_TCL (32) for j in range (N_events)]

      fig, ax = plt.subplots (nrows = 1, ncols = 1)
      N_bins = sturges (len (campione_gaus))
      bin_edges = np.linspace (0, 1.5 * np.pi, N_bins)
      bin_content, _, _ = ax.hist (campione_gaus,
               bin_edges,
               color = 'orange',
              )
      plt.savefig ('gauss.png')

      from iminuit import Minuit
      from iminuit.cost import BinnedNLL
      from scipy.stats import norm
      from lib import mod_gaus

      my_stats_gaus = stats (campione_gaus)

      # the cost function for the fit
    #  my_cost_func = BinnedNLL (bin_content, bin_edges, gaus_model)
      my_cost_func = BinnedNLL (bin_content, bin_edges, mod_gaus)

      my_minuit = Minuit (my_cost_func, 
                          mu = my_stats_gaus.mean (), 
                          sigma = my_stats_gaus.sigma ())

      my_minuit.migrad ()
      my_minuit.minos ()
      print (my_minuit.valid)
      from scipy.stats import chi2
      print ('associated p-value: ', 1. - chi2.cdf (my_minuit.fval, df = my_minuit.ndof))
      if 1. - chi2.cdf (my_minuit.fval, df = my_minuit.ndof) > 0.10:
        print ('the event sample is compatible with a Gaussian distribution')

        
#----------------------------5 FEBBRAIO 2024---------------------------


#soluzione è in formato notebook qui ricopio codice e commento parte scritta

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

from lib import phi, rand_TCL_ms, generate_TCL_ms, sturges

#punto 1

import matplotlib.pyplot as plt
import numpy as np
from random import uniform

fig, ax = plt.subplots (nrows = 1, ncols = 1)

# preparing the set of points to be drawn 
x_func = np.linspace (0, 10, 10000)
theta = [3, 2, 1]
a = 3
b = 2
c = 1
y_func = phi (x_func, a, b, c)

# visualisation of the image
ax.plot (x_func, y_func, label='phi')
ax.set_xlabel ('x')
ax.set_ylabel ('y')
#  plt.savefig ('pdf.png')

            #immmagine grafico
    
#punto 2

x_coord = np.array ([uniform (0, 10) for i in range (10)])
x_coord.sort () # i punti vanno ordinati perché iminuit faccia il fit nel range che li contiene tutti
sigma_y = 10
y_coord = list( map (lambda k:sum(k), zip (phi (x_coord, a, b, c), generate_TCL_ms (0., sigma_y, 10))))

# fig, ax = plt.subplots()
ax.errorbar (x_coord, y_coord, xerr=0, yerr=10,  marker='o', linestyle = '')
plt.show()

plt.show ()    #non so perche sia doppio lol


#punto 3

from iminuit import Minuit
from iminuit.cost import LeastSquares

# generate a least-squares cost function
least_squares = LeastSquares (x_coord, y_coord, sigma_y, phi)
my_minuit = Minuit (least_squares, a = 0, b = 1, c = 1)  # starting values for m and q
my_minuit.migrad ()  # finds minimum of least_squares function

            #tabellozze fighe e immagine grafico
                #dovrebbero comparire con display che qui non ha scritto 

#punto 4


#costruisco Q^2
N_toys = 100000

Q2_list = []
for i in range (N_toys):
    x_coord = np.array ([uniform (0, 10) for i in range (10)])
    x_coord.sort ()
    y_coord = list( map (lambda k:sum(k), zip (phi (x_coord, a, b, c), generate_TCL_ms (0., sigma_y, 10))))
    least_squares = LeastSquares (x_coord, y_coord, sigma_y, phi)
    my_minuit = Minuit (least_squares, a = 0, b = 1, c = 1)  # starting values for m and q
    my_minuit.migrad ()  # finds minimum of least_squares function
    Q2_list.append (my_minuit.fval)
    
#punto 5
Q2_list_unif = []
for i in range (N_toys):
    x_coord = np.array ([uniform (0, 10) for i in range (10)])
    x_coord.sort ()
    y_coord = list( map (lambda k:sum(k), zip (phi (x_coord, a, b, c), [uniform (- 1.732 * sigma_y, 1.732 * sigma_y) for i in range (10)])))
    least_squares = LeastSquares (x_coord, y_coord, sigma_y, phi)
    my_minuit = Minuit (least_squares, a = 0, b = 1, c = 1)  # starting values for m and q
    my_minuit.migrad ()  # finds minimum of least_squares function
    Q2_list_unif.append (my_minuit.fval)    
    
    
N_bins = sturges (len (Q2_list))
xMin = 0
xMax = 20
bin_edges = np.linspace (xMin, xMax, N_bins)
fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (Q2_list,
         bins = bin_edges,
         color = 'orange',
         label = 'gaus',
        )
ax.hist (Q2_list_unif,
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

#punto 6
# ordina i valori di Q2
# scorri i valori fino a che la frazione è maggiore di 0.9... cioè prendi l'elemento che sta all'indice 90% del totale
from math import floor
N_threshold = floor (N_toys * 0.9)
Q2_list_unif.sort ()
print ('soglia al 90%:', Q2_list_unif[N_threshold])

Q2_list_unif_rigettati = [val for val in Q2_list_unif if val > Q2_list_unif[N_threshold]]

fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (Q2_list_unif,
         bins = bin_edges,
         color = 'blue',
         label = 'test statistics',
         histtype='step',
        )
ax.hist (Q2_list_unif_rigettati,
         bins = bin_edges,
         color = 'lightblue',
         label = 'rigettati',
        )
ax.set_title ('Q2 distributions', size=14)
ax.set_xlabel ('Q2')
ax.set_ylabel ('event counts per bin')
ax.legend ()
plt.show ()

                #graficozzo
    
    
#------------------------------22 FEBBRAIO 2024-------------------

#soluzione in notebook

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
    
#1
import numpy as np
redshift, distanza, sigma = np.loadtxt('SuperNovae.txt', unpack=True)
print (len (distanza))

#2
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.errorbar(redshift, distanza, yerr=sigma, marker='o', linestyle='none')
ax.set_xlabel('Redshift')
ax.set_ylabel('Distanza [MPc]')

#3
from iminuit import Minuit
from iminuit.cost import LeastSquares
def distance_linear(z, H0):
    c = 3e5
    return c * z / H0

def distance_quadratic(z, H0, q):
    c = 3e5
    return c / H0 * (z + 0.5 * z**2 * (1-q))

def distance_cubic(z, H0, q):
    c = 3e5

    fact = c * z / H0
    mult = 1 + 0.5 * (1-q) * z - 1 / 6 * (1 - q - 3*q**2 + 1) * z**2

    return fact * mult
def init_H0(d, z):
    c = 3e5
    return c * z / d

H0_start = np.mean(init_H0(distanza[redshift<0.1], redshift[redshift<0.1]))


#fit lineare
least_squares = LeastSquares(redshift, distanza, sigma, distance_linear)
linear = Minuit(least_squares, H0 = H0_start)
linear.migrad()
linear.hesse()
        #grafico
    
#fit quadratico
least_squares = LeastSquares(redshift, distanza, sigma, distance_quadratic)
quad = Minuit(least_squares, H0 = H0_start, q=-1)
quad.migrad()
quad.hesse()
    #grafico
    
#fit cubico    che non era richiesto ma ok
least_squares = LeastSquares(redshift, distanza, sigma, distance_cubic)
cubic = Minuit(least_squares, H0 = H0_start, q=-1)
cubic.migrad()
cubic.hesse()

#anche qua mancano tutti i display per vedere risultati fit con tabellozze

#4
z = np.linspace(0, np.amax(redshift), 1000)

fig, ax = plt.subplots()
ax.errorbar(redshift, distanza, yerr=sigma, marker='o', linestyle='none')
ax.plot(z, distance_linear(z, *linear.values), label = 'Lineare')
ax.plot(z, distance_quadratic(z, *quad.values), label = 'Quadratico')
ax.plot(z, distance_cubic(z, *cubic.values), label = 'Cubico')
ax.set_xlabel('Redshift')
ax.set_ylabel('Distanza [MPc]')
ax.legend()

from scipy.stats import chi2
H0_finale = 0
p_value_finale = 0

modelli = [linear, quad, cubic]

count = 0

for mod in modelli:
    p_value = 1. - chi2.cdf (mod.fval, df = mod.ndof)

    if p_value > p_value_finale and p_value < 0.95:
        p_value_finale = p_value
        H0_finale = mod.values['H0']
        count += 1

print(f'Il Valore della Costante di Hubble è {H0_finale}')


#5
import lib
Ntoys = 10000

omega = []

q = modelli[count].values['q']
delta_q = modelli[count].errors['q']

for _ in range(Ntoys):

    qt = lib.rand_TCL_ms (q, delta_q, N_sum = 50)
     
    omega.append(
        2 / 3 * ( qt + 1 )
    )

omega.sort()

print(f'Il valore mediano di Omega è {omega[int(len(omega) * 0.5)]:.4f}')
print(f'Il valore di Omega è compreso tra {omega[int(len(omega) * 0.1)]:.4f} e {omega[int(len(omega) * 0.9)]:.4f}')



#---------------------24 GIUGNO 2024-----------------------------



'''
La scoperta di nuove particelle instabili, come nel caso del bosone di Higgs, si basa molto spesso sullo
studio dell’istogramma della massa invariante dei loro prodotti di decadimento, per trovare un eccesso
rispetto al rumore di fondo localizzato in un intorno di un valore specifico. In questo tema vi è richiesto di
costruire una simulazione di questa ricerca, utilizzando un modello esponenziale di rumore di fondo ed
uno Gaussiano per il segnale.
1. Si generi un campione {xi} di N_exp=2000 eventi distribuiti secondo una distribuzione di densità di
probabilità esponenziale, con λ = 1/200, compresi fra 0 e 3 volte τ , ed uno {xj} di N_gau=200 eventi
distribuiti secondo una distribuzione di densità di probabilità Gaussiana, con µ = 190 e σ = 20.
2. Si costruisca un campione pari all’unione dei due precedenti e se ne disegni l’istogramma, scegliendo
opportunamente il numero di bin.
3. Si effettui un fit del campione per determinare i parametri del modello.
4. Si costruisca una funzione che calcoli il logaritmo della verosimiglianza associata al campione, dato
il seguente modello di densità di probabilità:
f(x) = a ∗ Exp(x, λ) + b ∗ Gaus(x, µ, σ) (1)
5. Fissati i parametri del modello al risultato ottenuto dal fit, si calcoli il valore del logaritmo della
verosimiglianza per il campione dato il modello, variando il valore del parametro µ fra 30 e 300 con
passo costante e se ne disegni l’andamento.
6. Si determini il massimo della funzione di verosimiglianza in funzione del parametro µ, utilizzando
l’algoritmo della sezione aurea.
'''

#punto 1
import random
import math
from lib import try_and_catch_exp, try_and_catch_gau

N_exp = 2000
N_gau = 200
sample_exp = try_and_catch_exp (1./200., N_exp)
sample_gau = try_and_catch_gau (190., 20., N_gau)   

#2
sample_tot = sample_exp + sample_gau
print (len (sample_exp))
print (len (sample_gau))
print (len (sample_tot))

random.shuffle (sample_tot)


import matplotlib.pyplot as plt
import numpy as np
from lib import sturges

N_bins = sturges (len (sample_tot))

# build a numpy histogram containing the data counts in each bin
bin_content, bin_edges = np.histogram (sample_tot, bins = N_bins, range = (0, 3 * 200.))

fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (sample_tot,
         bins = bin_edges,
         color = 'orange',
        )
            #istogrammone

#3
from iminuit import Minuit
from scipy.stats import expon, norm
from iminuit.cost import ExtendedBinnedNLL

def mod_total (bin_edges, N_signal, mu, sigma, N_background, tau):
    return N_signal * norm.cdf (bin_edges, mu, sigma) + \
            N_background * expon.cdf (bin_edges, 0, tau )

from iminuit.cost import ExtendedBinnedNLL
my_cost_func = ExtendedBinnedNLL (bin_content, bin_edges, mod_total)

sample_mean = np.mean (sample_tot)
sample_sigma = np.std (sample_tot)

N_events = sum (bin_content)
my_minuit = Minuit (my_cost_func, 
                    N_signal = N_events, mu = sample_mean, sigma = sample_sigma, # signal input parameters
                    N_background = N_events, tau = sample_mean)                  # background input parameters

my_minuit.migrad ()
print (my_minuit.valid)
display (my_minuit)


#4
# questi sono i parametri del modello che rimangono fissati
tau = my_minuit.values['tau']
lam = 1/tau
sigma = my_minuit.values['sigma']
gau_norm = 1. / (np.sqrt (2 * np.pi) * sigma)
f_exp = my_minuit.values['N_background']
f_gau = my_minuit.values['N_signal']
f_tot = f_exp + f_gau
f_exp = f_exp / f_tot
f_gau = f_gau / f_tot

def pdf (x, mean):
    return f_exp * lam * np.exp (-x * lam) + \
             f_gau * gau_norm * np.exp (-0.5 * ((x - mean)/sigma )**2)  
           # il simbolo \ serve per andare accapo senza terminare la linea di istruzione

def loglikelihood (theta, pdf, sample) :
    logL = 0.
    for x in sample:
      if (pdf (x, theta) > 0.) : logL = logL + np.log (pdf (x, theta))    
    return logL

#5
fig, ax = plt.subplots ()
x = np.linspace (30, 300, 100)
ax.plot (x, pdf(x, my_minuit.values['mu']), color = 'blue')
ax.set_xlabel ('x')
ax.set_ylabel ('pdf')
plt.show ()
        #grafico
fig, ax = plt.subplots ()

x_coord = np.linspace (30, 300, 100)
l_like = []
for x in x_coord: l_like.append (loglikelihood (x, pdf, sample_tot))   
y_coord = np.array (l_like)

ax.plot (x_coord, y_coord, color = 'red')
ax.set_xlabel ('mean')
ax.set_ylabel ('log-likelihood')
plt.show()
        #altro grafico
#6    
def sezioneAureaMax (
    x0,             # estremo dell'intervallo          
    x1,             # altro estremo dell'intervallo         
    prec = 0.0001): # precisione della funzione        
    r = 0.618
    x2 = 0.
    x3 = 0. 
     
    while (abs (x1 - x0) > prec):  # x0, x3, x2, x1
        x2 = x0 + r * (x1 - x0) 
        x3 = x0 + (1. - r) * (x1 - x0)  
        if loglikelihood (x3, pdf, sample_tot) < loglikelihood (x2, pdf, sample_tot):
            x0 = x3
        else :
            x1 = x2
    return (x0 + x1) / 2.
mean_maxll = sezioneAureaMax (20, 300, 0.1)
print (f'il valore del parametro mean che massimizza la verosimiglianza è: {mean_maxll:.1f}')

            
#-----------------------8 LUGLIO 2024-------------------


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
distribuzione ha come parametri loc = 0 e scale = sqrt(N/2) (dove N è il numero di passi).
'''
#mia soluzione completamente diversa da questa

#1 è nella libreria, qui test della funzione
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from lib import try_and_catch_step

N_galois = 10000 # numero di abitanti del villaggio
N_steps_galois = 10

#2
# testing a single walk
from lib import walk, delta

asterix = [0,0]
path = []
path.append (asterix)
print (asterix)
for i_step in range (N_steps_galois): path.append (walk (1, path[-1]))

fig, ax = plt.subplots ()
ax.plot ([x for x,y in path], [y for x,y in path], 'bo-')
ax.set_xlabel ('x')
ax.set_ylabel ('y')
plt.show () 

    
#3

from lib import walk, norm

abitanti = []
distanze = []
for galois in range (N_galois):
    abitanti.append (walk (N_steps_galois, [0,0]))

print ('generati', len (abitanti), 'abitanti')


distanze = [delta(galois) for galois in abitanti]
from lib import sturges

N_bins = sturges (len (distanze))
h_max = np.ceil (max (distanze))
# build a numpy histogram containing the data counts in each bin
bin_content, bin_edges = np.histogram (distanze, bins = N_bins, range = (0, h_max))

fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (distanze,
         bins = bin_edges,
         color = 'deepskyblue',
        )


#4

from stats import stats

stats_calculator = stats (distanze)
print ('mean    :', stats_calculator.mean ())
print ('sigma   :', stats_calculator.sigma ())
print ('skewness:', stats_calculator.skewness ())
print ('kurtosis:', stats_calculator.kurtosis ())

#5
abitanti_R1 = []
for galois in range (N_galois): abitanti_R1.append (walk (N_steps_galois, [0,0], True))
distanze_R1 = [delta(galois) for galois in abitanti_R1]
def Rayleigh (r, N_steps, norm):
    return norm * 2 * r * np.exp (-1 * r**2 / N_steps) / N_steps

N_bins = sturges (len (distanze_R1))
h_max = np.ceil (max (distanze_R1))
bin_content, bin_edges = np.histogram (distanze_R1, bins = N_bins, range = (0, h_max))

norm = N_galois * (bin_edges[2] - bin_edges[1])

x_coord = np.linspace (0, h_max, 1000)
y_coord = Rayleigh (x_coord, N_steps_galois, norm)
# for i in range (x_coord.size):
#     y_coord[i] = Rayleigh (x_coord[i], 10000)


fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.plot (x_coord, y_coord, label='Rayleigh function')
ax.hist (distanze_R1,
         bins = bin_edges,
         color = 'deepskyblue',
         label = 'distanze',
        )
ax.set_xlabel ('distanza')
ax.legend ()

# y_coord_2 = np.arange (0., x_coord.size)
# for i in range (x_coord.size):
#     y_coord_2[i] = func (x_coord[i])

#6

from iminuit import Minuit
from scipy.stats import rayleigh
from iminuit.cost import BinnedNLL

# the fitting function
def model (bin_edges, N_steps):
    return rayleigh.cdf (bin_edges, 0, np.sqrt (N_steps/2))

N_events = sum (bin_content)

# the cost function for the fit
my_cost_func = BinnedNLL (bin_content, bin_edges, model)

# the fitting algoritm
my_minuit = Minuit (my_cost_func, 
                    N_steps = h_max) # la stima migliore che si può fare di N_step è la distanza massima

# bounds the following parameters to being positive
#my_minuit.limits['N_signal', 'N_background', 'sigma', 'tau'] = (0, None)

# my_minuit.values['N_galois'] = N_events
# my_minuit.fixed['N_galois'] = True

my_minuit.migrad ()
my_minuit.minos ()
print (my_minuit.valid)
display (my_minuit)

print ('Value of Q2: ', my_minuit.fval)
print ('Number of degrees of freedom: ', my_minuit.ndof)
from scipy.stats import chi2
print ('associated p-value: ', 1. - chi2.cdf (my_minuit.fval, df = my_minuit.ndof))


#---------------------16 SETTEMBRE 2024----------------------


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


#punto 1 è nella libreria 
#punto 2 è qui che è test

from lib import additive_recurrence as ar

gen_seq = ar ()
my_seq = []
for i in range (1000) : my_seq.append (gen_seq.get_number ())
for i in range (10) : print (my_seq[i], ' ', end = '')
print ()


from lib import sturges
import numpy as np
import matplotlib.pyplot as plt

N_bins = sturges (len (my_seq))
# build a numpy histogram containing the data counts in each bin
bin_content, bin_edges = np.histogram (my_seq, bins = N_bins, range = (0, 1))

fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (my_seq,
         bins = bin_edges,
         color = 'deepskyblue',
        )

#3 e 4

from lib import integral_CrudeMC
from statistics import mean, stdev
import numpy as np

N_toys = 1000
N_points = 10
N_points_max = 25000

seq_N = []
seq_sigma = []
seq_sigma_t = []
seq_mean = []

func = lambda x : 2 * x * x

while N_points < N_points_max :
    print ('running with', N_points, 'points')
    integrals = []
    for i_toy in range (N_toys): 
        x_axis = gen_seq.get_numbers (N_points)
        result = integral_CrudeMC (func, 0, 1, x_axis)
        integrals.append (result[0])
        if i_toy == 0 : seq_sigma.append (result[1])  
    seq_N.append (N_points)
    seq_sigma_t.append (stdev (integrals))
    seq_mean.append (mean (integrals))
    N_points *= 2
print ('DONE')


fig, ax = plt.subplots ()
ax.plot (seq_N, seq_sigma, 'bo-', label = 'estimate')
ax.plot (seq_N, seq_sigma_t, 'bo-', label = 'toys')
ax.set_xlabel ('integration points')
ax.set_xscale ('log')
ax.set_yscale ('log')
ax.set_ylabel ('integral uncertainty')
plt.show () 

fig, ax = plt.subplots ()
ax.plot (seq_N, seq_mean, 'bo-')
ax.set_xlabel ('integration points')
ax.set_ylabel ('integral value')
plt.show ()    

#5
from lib import MC_classic, rand_range

N_points = 10
seq_sigma_cl = []
seq_sigma_cl_t = []
seq_mean_cl = []

while N_points < N_points_max :
    print ('running with', N_points, 'points')
    integrals = []

    for i_toy in range (N_toys): 
        x_axis = []
        for i in range (N_points): x_axis.append (rand_range (0., 1.))
        result = integral_CrudeMC (func, 0, 1, x_axis)
        integrals.append (result[0])
        if i_toy == 0 : seq_sigma_cl.append (result[1])  
    seq_sigma_cl_t.append (stdev (integrals))
    seq_mean_cl.append (mean (integrals))
    N_points *= 2
print ('DONE')

fig, ax = plt.subplots ()
ax.plot (seq_N, seq_sigma, 'bo-', color = label = 'sequential')
ax.plot (seq_N, seq_sigma_cl, 'ro-', color = label = 'classic')
ax.plot (seq_N, seq_sigma_t, 'bo-', color = 'red', label = 'sequential')
ax.plot (seq_N, seq_sigma_cl_t, 'ro-', color = 'blue', label = 'classic')
ax.set_xlabel ('integration points')
ax.set_ylabel ('integral uncertainty')
ax.set_xscale ('log')
ax.set_yscale ('log')
ax.legend ()
plt.show ()   

fig, ax = plt.subplots ()
# ax.plot (seq_N, seq_mean, 'bo-', label = 'sequential')
# ax.plot (seq_N, seq_mean_cl, 'ro-', label = 'classic')
# ax.errorbar (seq_N, seq_mean, seq_sigma, 'bo-', label = 'sequential')
# ax.errorbar (seq_N, seq_mean_cl, seq_sigma_cl, 'ro-', label = 'classic')
ax.errorbar (seq_N, seq_mean, seq_sigma, color = 'blue', capsize=3, label = 'sequential',zorder=5)
ax.errorbar (seq_N, seq_mean_cl, seq_sigma_cl, color = 'red', capsize=3, label = 'classic', zorder=0)
ax.plot (seq_N, [2./3. for i in seq_mean], 'g-', label = 'expected', zorder=10)
ax.set_xlabel ('integration points')
ax.set_ylabel ('integral value')
ax.legend ()
plt.show ()



#--------------------------10 OTTOBRE 2024----------------


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

#1 è funzione in libreria
#2
from lib import generate_gaus_bm, sturges
import matplotlib.pyplot as plt
import numpy as np

N = 1000

lista_G = []
for i in range (N//2):
    G1, G2 = generate_gaus_bm ()
    lista_G.append (G1)
    lista_G.append (G2)

lista_min = np.floor (min (lista_G))
lista_max = np.ceil (max (lista_G))

N_bins = sturges (len (lista_G))
bin_content, bin_edges = np.histogram (lista_G, bins = N_bins, range = (lista_min, lista_max))

fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (lista_G,
         bins = bin_edges,
         color = 'deepskyblue',
        )
        #grafico
    
#3
from stats import stats

statistiche = stats (lista_G)

media = statistiche.mean ()
varia = statistiche.variance ()
media_err = statistiche.sigma_mean ()
varia_err = 2 * varia * varia / (len (lista_G) - 1)

print ('media:', media, '±', media_err) 
print ('varianza:', varia, '±', varia_err) 

#4
from statistics import stdev

N_max = 1000000

N = 10
sigmas = []
sigmas_mean = []
events = []
while N <= N_max:
    lista = []
    for i in range (N//2):
        print ('N:', N, end = '\r')
        G1, G2 = generate_gaus_bm ()
        lista.append (G1)
        lista.append (G2)
    sigmas.append (stdev (lista))
    sigmas_mean.append (stdev (lista) / np.sqrt (len (lista)))
    events.append (len (lista))
    N *= 2

fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.plot (events, sigmas, label = 'single measurement error')
ax.plot (events, sigmas_mean, label = 'mean error')
ax.set_xscale ('log')
ax.set_yscale ('log')
ax.legend ()
plt.show ()

#5
from statistics import mean

N = 10000
mu = 5
sigma = 2

lista_G2 = []
transform = lambda x : x * sigma + mu
for i in range (N//2):
    G1, G2 = generate_gaus_bm ()
    lista_G2.append (transform (G1))
    lista_G2.append (transform (G2))

print ('media:', mean (lista_G2))
print ('sigma:', stdev (lista_G2))

lista_min = np.floor (min (lista_G2))
lista_max = np.ceil (max (lista_G2))

N_bins = sturges (len (lista_G2))
bin_content, bin_edges = np.histogram (lista_G2, bins = N_bins, range = (lista_min, lista_max))

fig, ax = plt.subplots (nrows = 1, ncols = 1)
ax.hist (lista_G2,
         bins = bin_edges,
         color = 'deepskyblue',
        )