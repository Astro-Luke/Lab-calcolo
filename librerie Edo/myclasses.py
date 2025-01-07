'''
myclasses contiene tutte le classi che abbiamo scritto durante le lezioni:
    Fraction
    stats
    my_histo
'''



from math import gcd, sqrt, ceil, floor
import sys
import numpy as np
import matplotlib.pyplot as plt



def lcm (a, b) :
    """least common multiple 

    Args:
        a (int): the first number
        b (int): the second number

    Returns:
        int: the least common multiple of the two numbers
    """
    return a * b / gcd (a,b)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


class Fraction :
    '''
    a simple class implementing a high-level object
    to handle fractions and their operations
    '''

    def __init__ (self, numerator, denominator) :
        """the constructor: initialises all the variables needed
        for the high-level object functioning

        Args:
            numerator (int): the numerator of the fraction
            denominator (int): the denominator of the fraction

        Raises:
            ValueError: Denominator cannot be zero
            ValueError: Numerator must be an integer
            ValueError: Denominator must be an integer
        """
        if denominator == 0 :
            raise ValueError ('Denominator cannot be zero')
        if type(numerator) != int:
            raise TypeError ('Numerator must be an integer')
        if not isinstance(denominator, int ): # alternative way to check the type
            raise TypeError ('Denominator must be an integer')
        
        # this allows to avoid calculating the LCM in the sum and subtraction
        common_divisor = gcd (self.numerator, self.denominator) # greatest common divisor 
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
        """implements the addition of two fractions.
        Note that this function will be callable with the + symbol
        in the program

        Args:
            other (Fraction): the fraction to be added to the current one

        Returns:
            Fraction: the addition of the two fractions
        """
        new_numerator = self.numerator * other.denominator + other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __sub__ (self, other) :
        """implements the subtraction of two fractions.
        Note that this function will be callable with the - symbol
        in the program

        Args:
            other (Fraction): the fraction to be subtracted from the current one

        Returns:
            Fraction: the subtraction of the two fractions
        """
        new_numerator = self.numerator * other.denominator - other.numerator * self.denominator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __mul__ (self, other) :
        """
        implements the multiplications of two fractions.
        Note that this function will be callable with the * symbol
        in the program

        Args:
            other (Fraction): the fraction to be multiplied from the current one

        Returns:
            Fraction: the multiplication of the two fractions
        """
        new_numerator = self.numerator * other.numerator
        new_denominator = self.denominator * other.denominator
        return Fraction (new_numerator, new_denominator)
    
    def __truediv__ (self, other) :
        '''
        implements the ratio of two fractions.
        Note that this function will be callable with the / symbol
        in the program

        Args:
            other (Fraction): the fraction to be divided from the current one

        Returns:
            Fraction: the ratio of the two fractions
        '''
        if other.numerator == 0 :
            print ('Cannot divide by zero')
            sys.exit (1)
        
        new_numerator = self.numerator * other.denominator
        new_denominator = self.denominator * other.numerator
        return Fraction (new_numerator, new_denominator)


# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 


def testing ()  :
    '''
    Function to test the class behaviour, called in the main program
    '''

    print ('Initial fractions:')
    frac1 = Fraction (3, 4)
    frac1.print ()
    print ('ratio: ', frac1.ratio ())

    frac2 = Fraction (1, 2)
    frac2.print ()
    print ('ratio: ', frac2.ratio ())
    
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



#------------



#classe stats contiene le statistiche anche se le ho definite come funzioni nell'altra libreria

class stats :
    '''calculator for statistics of a list of numbers'''

    summ = 0.
    sumSq = 0.
    N = 0
    sample = []

    def __init__ (self, sample):
        '''
        reads as input the collection of events,
        which needs to be a list of numbers
        '''
        self.sample = sample
        self.summ = sum (self.sample)
        self.sumSq = sum ([x*x for x in self.sample])
        self.N = len (self.sample)

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def mean (self) :
        '''
        calculates the mean of the sample present in the object
        '''
        return self.summ / self.N

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def variance (self, bessel = True) :
        '''
        calculates the variance of the sample present in the object
        '''
        var = self.sumSq / self.N - self.mean () * self.mean ()
        if bessel : var = self.N * var / (self.N - 1)
        return var

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def sigma (self, bessel = True) :
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel))

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def sigma_mean (self, bessel = True) :
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel) / self.N)

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def append (self, x):
        '''
        add an element to the sample
        '''
        self.sample.append (x)
        self.summ = self.summ + x
        self.sumSq = self.sumSq + x * x
        self.N = self.N + 1

        
        
#--------



#my_histo.py  calcola statistiche dell'istogramma

class my_histo :
    '''calculator for statistics of a list of numbers'''

    summ = 0.
    sumSq = 0.
    N = 0
    sample = []

    def __init__ (self, sample_file_name) :
        '''
        reads as input the file containing the collection of events
        and reads it
        '''
        with open (sample_file_name) as f:
            self.sample = [float (x) for x in f.readlines ()]

        self.summ = sum (self.sample)
        self.sumSq = sum ([x*x for x in self.sample])
        self.N = len (self.sample)

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def mean (self) :
        '''
        calculates the mean of the sample present in the object
        '''
        return self.summ / self.N

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def variance (self, bessel = True) :
        '''
        calculates the variance of the sample present in the object
        '''
        var = self.sumSq / self.N - self.mean () * self.mean ()
        if bessel : var = self.N * var / (self.N - 1)
        return var

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def sigma (self, bessel = True) :
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel))

    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def sigma_mean (self, bessel = True) :
        '''
        calculates the sigma of the sample present in the object
        '''
        return sqrt (self.variance (bessel) / self.N)
    
    # ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    def draw_histo (self, output_file_name) :
        '''
        draw the sample content into an histogram
        '''
        xMin = floor (min (self.sample))
        xMax = ceil (max (self.sample))
        N_bins = sturges (self.N)

        bin_edges = np.linspace (xMin, xMax, N_bins)
        fig, ax = plt.subplots (nrows = 1, ncols = 1)
        ax.hist (self.sample,
                 bins = bin_edges,
                 color = 'orange',
                )
        ax.set_title ('Histogram example', size=14)
        ax.set_xlabel ('variable')
        ax.set_ylabel ('event counts per bin')

        plt.savefig (output_file_name)


        
    '''
# ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- ---- 

    
if __name__ == "__main__" :
    testing ()
    '''