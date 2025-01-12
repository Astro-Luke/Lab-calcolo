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
