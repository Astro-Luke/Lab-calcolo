
# ----- ----- ----- ----- Lezione 5 ----- ----- ----- -----

# Esercizio 1
'''
Creare una libreria Python che implementi la Fractionclasse, 
contenente il suo costruttore, i membri dati per salvare numeratore e denominatore e il metodo della classe 
che restituisce la divisione tra numeratore e denominatore.
'''

from math import gcd
import sys


def lcm (a, b) : 
    '''
    least common multiple 
    '''
    return a * b / gcd (a,b)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


class Fraction :
    '''
    a simple class implementing a high-level object
    to handle fractions and their operations
    '''


    def __init__ (self, numerator, denominator) :
        '''
        the constructor: initialises all the variables needed
        for the high-level object functioning
        '''
        if denominator == 0 :
            print ('Denominator cannot be zero')
            sys.exit (1)
        
        # this allows to avoid calculating the LCM in the sum and subtraction
        common_divisor = gcd (numerator, denominator) # greatest common divisor 
        self.numerator = numerator // common_divisor
        self.denominator = denominator // common_divisor
        
    def print (self) :
        '''
        prints the value of the fraction on screen
        '''
        print (str (self.numerator) + '/' + str (self.denominator))

    def ratio (self) :
        '''
        calculates the actual ratio between numerator and denominator,
        practically acting as a casting to float
        '''
        return self.numerator / self.denominator

    def __add__ (self, other) :
        '''
        implements the addition of two fractions.
        Note that this function will be callable with the + symbol
        in the program
        '''
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __sub__ (self, other) :
        '''
        implements the subtraction of two fractions.
        Note that this function will be callable with the - symbol
        in the program
        '''
        new_numerator = self.numerator * other.denominator - other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __mul__ (self, other) :
        '''
        implements the multiplications of two fractions.
        Note that this function will be callable with the * symbol
        in the program
        '''
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __truediv__ (self, other) :
        '''
        implements the ratio of two fractions.
        Note that this function will be callable with the / symbol
        in the program
        '''
        if other.numerator == 0 :
            print ('Cannot divide by zero')
            sys.exit (1)
        
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        return Fraction (new_numerator, new_denominator)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def testing_1 ()  :
    '''
    Function to test the class behaviour, called in the main program, ex. 2
    '''

    print ('Initial fractions:')
    frac1 = Fraction (3, 4)
    frac1.print ()
    print ('numerator: ', frac1.numerator )
    print ('denominator: ', frac1.denominator )
    print ('ratio: ', frac1.ratio ())
    

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    

def testing_2 () :
    '''
    Function to test the class behaviour, called in the main program, ex. 3
    '''

    print ('Initial fractions:')
    frac1 = Fraction (3, 4)
    frac2 = Fraction (1, 2)
    frac1.print ()
    frac2.print ()
    
    sum_frac = frac1 + frac2
    print ('\nSum :')
    sum_frac.print ()
    
    diff_frac = frac1 - frac2
    print ('\nDifference:')
    diff_frac.print ()
    
    prod_frac = frac1 * frac2
    print ('\nProduct:')
    prod_frac.print ()
    
    div_frac = frac1 / frac2
    print ('\nDivision:')
    div_frac.print ()
    

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    
if __name__ == "__main__" :
    testing_1 ()
    testing_2 ()


# ----- ----- ----- ----- ----- ----- ----- -----

# Esercizio 2
'''
Implementare una funzione di test della classe all'interno del file di 
libreria stesso, che verifichi l'output di ciascun metodo della classe 
e che stampi sullo schermo il valore del numeratore e del denominatore di una frazione.
a simple class to handle fractions of integer numbers
'''

from math import gcd
import sys


def lcm (a, b) : 
    '''
    least common multiple 
    '''
    return a * b / gcd (a,b)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


class Fraction :
    '''
    a simple class implementing a high-level object
    to handle fractions and their operations
    '''


    def __init__ (self, numerator, denominator) :
        '''
        the constructor: initialises all the variables needed
        for the high-level object functioning
        '''
        if denominator == 0 :
            print ('Denominator cannot be zero')
            sys.exit (1)
        
        # this allows to avoid calculating the LCM in the sum and subtraction
        common_divisor = gcd (numerator, denominator) # greatest common divisor 
        self.numerator = numerator // common_divisor
        self.denominator = denominator // common_divisor
        
    def print (self) :
        '''
        prints the value of the fraction on screen
        '''
        print (str (self.numerator) + '/' + str (self.denominator))

    def ratio (self) :
        '''
        calculates the actual ratio between numerator and denominator,
        practically acting as a casting to float
        '''
        return self.numerator / self.denominator

    def __add__ (self, other) :
        '''
        implements the addition of two fractions.
        Note that this function will be callable with the + symbol
        in the program
        '''
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __sub__ (self, other) :
        '''
        implements the subtraction of two fractions.
        Note that this function will be callable with the - symbol
        in the program
        '''
        new_numerator = self.numerator * other.denominator - other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __mul__ (self, other) :
        '''
        implements the multiplications of two fractions.
        Note that this function will be callable with the * symbol
        in the program
        '''
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __truediv__ (self, other) :
        '''
        implements the ratio of two fractions.
        Note that this function will be callable with the / symbol
        in the program
        '''
        if other.numerator == 0 :
            print ('Cannot divide by zero')
            sys.exit (1)
        
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        return Fraction (new_numerator, new_denominator)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def testing_1 ()  :
    '''
    Function to test the class behaviour, called in the main program, ex. 2
    '''

    print ('Initial fractions:')
    frac1 = Fraction (3, 4)
    frac1.print ()
    print ('numerator: ', frac1.numerator )
    print ('denominator: ', frac1.denominator )
    print ('ratio: ', frac1.ratio ())
    

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    

def testing_2 () :
    '''
    Function to test the class behaviour, called in the main program, ex. 3
    '''

    print ('Initial fractions:')
    frac1 = Fraction (3, 4)
    frac2 = Fraction (1, 2)
    frac1.print ()
    frac2.print ()
    
    sum_frac = frac1 + frac2
    print ('\nSum :')
    sum_frac.print ()
    
    diff_frac = frac1 - frac2
    print ('\nDifference:')
    diff_frac.print ()
    
    prod_frac = frac1 * frac2
    print ('\nProduct:')
    prod_frac.print ()
    
    div_frac = frac1 / frac2
    print ('\nDivision:')
    div_frac.print ()
    

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    
if __name__ == "__main__" :
    testing_1 ()
    testing_2 ()


# ----- ----- ----- ----- ----- ----- ----- -----

# Esercizio 3
'''
Aggiungere alla Fractionclasse il sovraccarico delle operazioni +, -, *, / in modo 
che ciascuna di esse restituisca un oggetto di tipo Fraction.
Aggiungere alla funzione di test la chiamata a tutti i nuovi metodi e 
la verifica del loro comportamento.
'''


from math import gcd
import sys


def lcm (a, b) : 
    '''
    least common multiple 
    '''
    return a * b / gcd (a,b)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


class Fraction :
    '''
    a simple class implementing a high-level object
    to handle fractions and their operations
    '''


    def __init__ (self, numerator, denominator) :
        '''
        the constructor: initialises all the variables needed
        for the high-level object functioning
        '''
        if denominator == 0 :
            print ('Denominator cannot be zero')
            sys.exit (1)
        
        # this allows to avoid calculating the LCM in the sum and subtraction
        common_divisor = gcd (numerator, denominator) # greatest common divisor 
        self.numerator = numerator // common_divisor
        self.denominator = denominator // common_divisor
        
    def print (self) :
        '''
        prints the value of the fraction on screen
        '''
        print (str (self.numerator) + '/' + str (self.denominator))

    def ratio (self) :
        '''
        calculates the actual ratio between numerator and denominator,
        practically acting as a casting to float
        '''
        return self.numerator / self.denominator

    def __add__ (self, other) :
        '''
        implements the addition of two fractions.
        Note that this function will be callable with the + symbol
        in the program
        '''
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __sub__ (self, other) :
        '''
        implements the subtraction of two fractions.
        Note that this function will be callable with the - symbol
        in the program
        '''
        new_numerator = self.numerator * other.denominator - other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __mul__ (self, other) :
        '''
        implements the multiplications of two fractions.
        Note that this function will be callable with the * symbol
        in the program
        '''
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __truediv__ (self, other) :
        '''
        implements the ratio of two fractions.
        Note that this function will be callable with the / symbol
        in the program
        '''
        if other.numerator == 0 :
            print ('Cannot divide by zero')
            sys.exit (1)
        
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        return Fraction (new_numerator, new_denominator)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def testing_1 ()  :
    '''
    Function to test the class behaviour, called in the main program, ex. 2
    '''

    print ('Initial fractions:')
    frac1 = Fraction (3, 4)
    frac1.print ()
    print ('numerator: ', frac1.numerator )
    print ('denominator: ', frac1.denominator )
    print ('ratio: ', frac1.ratio ())
    

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 
    

def testing_2 () :
    '''
    Function to test the class behaviour, called in the main program, ex. 3
    '''

    print ('Initial fractions:')
    frac1 = Fraction (3, 4)
    frac2 = Fraction (1, 2)
    frac1.print ()
    frac2.print ()
    
    sum_frac = frac1 + frac2
    print ('\nSum :')
    sum_frac.print ()
    
    diff_frac = frac1 - frac2
    print ('\nDifference:')
    diff_frac.print ()
    
    prod_frac = frac1 * frac2
    print ('\nProduct:')
    prod_frac.print ()
    
    div_frac = frac1 / frac2
    print ('\nDivision:')
    div_frac.print ()
    

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    
if __name__ == "__main__" :
    testing_1 ()
    testing_2 ()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


# Esercizio 4
'''
 Scrivere un programma Python che legga il file di esempio eventi_unif.txt dell'esercizio 3.2 e, 
 utilizzando la funzione filtro, crei due diversi sottoinsiemi di eventi contenenti rispettivamente 
 quelli maggiori o minori della media, utilizzando lambdadelle funzioni nel processo.
Dimostrare che la sigma dei due sottoinsiemi è la metà di quella del campione padre.

'''

import sys
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor


def sturges (N_events) :
    return ceil (1 + np.log2 (N_events))
    

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    
def main () :
    '''
    Function implementing the main program
    '''

    # read the file
    with open ('../../Lecture_03/exercises/eventi_gauss.txt') as input_file :
        sample = [float (x) for x in input_file.readlines ()]

    for elem in sample[:10]:
        print (elem)
  
    sample_sq = list (map (lambda x: x**2, sample))
    sample_cu = list (map (lambda x: pow (x, 3), sample))

    xMin = floor (min (min (sample), min (sample_sq), min (sample_cu)))
    xMax = ceil (max (max (sample), max (sample_sq), max (sample_cu)))
    N_bins = sturges (len (sample)) * 5

    bin_edges = np.linspace (xMin, xMax, N_bins)
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample,
             bins = bin_edges,
             color = 'orange',
             histtype= 'stepfilled'
            )
    ax.hist (sample_sq,
             bins = bin_edges,
             color = 'red',
             histtype= 'step'
            )
    ax.hist (sample_cu,
             bins = bin_edges,
             color = 'blue',
             histtype= 'step'
            )
    ax.set_yscale ('log')
    ax.set_title ('Histogram example', size=14)
    ax.set_xlabel ('variable')
    ax.set_ylabel ('event counts per bin')

    plt.savefig ('ex_4.5.png')



# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    
if __name__ == "__main__" :
    main ()


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


# Esercizio 5
'''
Scrivere un programma Python che legga il file di esempio eventi_gauss.txt dell'esercizio 3.3 e, utilizzando la funzione map, 
crei rispettivamente la distribuzione dei quadrati e dei cubi dei numeri gaussiani casuali, utilizzando lambdadelle funzioni nel processo.
Rappresenta graficamente la loro distribuzione, insieme a quella del campione originale, il tutto nello stesso frame.
'''

import sys
import matplotlib.pyplot as plt
import numpy as np
from math import ceil, floor


def sturges (N_events) :
    return ceil (1 + np.log2 (N_events))
    

# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    
def main () :
    '''
    Function implementing the main program
    '''

    # read the file
    with open ('../../Lecture_03/exercises/eventi_gauss.txt') as input_file :
        sample = [float (x) for x in input_file.readlines ()]

    for elem in sample[:10]:
        print (elem)
  
    sample_sq = list (map (lambda x: x**2, sample))
    sample_cu = list (map (lambda x: pow (x, 3), sample))

    xMin = floor (min (min (sample), min (sample_sq), min (sample_cu)))
    xMax = ceil (max (max (sample), max (sample_sq), max (sample_cu)))
    N_bins = sturges (len (sample)) * 5

    bin_edges = np.linspace (xMin, xMax, N_bins)
    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample,
             bins = bin_edges,
             color = 'orange',
             histtype= 'stepfilled'
            )
    ax.hist (sample_sq,
             bins = bin_edges,
             color = 'red',
             histtype= 'step'
            )
    ax.hist (sample_cu,
             bins = bin_edges,
             color = 'blue',
             histtype= 'step'
            )
    ax.set_yscale ('log')
    ax.set_title ('Histogram example', size=14)
    ax.set_xlabel ('variable')
    ax.set_ylabel ('event counts per bin')

    plt.savefig ('ex_4.5.png')



# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    
if __name__ == "__main__" :
    main ()
