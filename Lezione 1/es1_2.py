'''
Scrivi un programma che, dati i tre lati di un triangolo, determini se il triangolo è acutangolo, rettangolo o ottuso.
'''

import sys

def det_triangolo (a, b, c) :

    # Riordino i valori a, b, c in un elenco per avere c come lato maggiore
    a, b, c = sorted([a, b, c])
    if ( a == 0 or b == 0 or c == 0) :
        print ("Inserire dei lati validi\n")
        sys.exit ()

    if (a == b and b == c) :
        print("Il triangolo è equilatero.\n")
    elif (a**2 + b**2 == c**2) :                # Teorema di Pitagora
        print("Il triangolo è rettangolo.\n")
    elif (a == b or a == c or b == c) :
        print("Il triangolo è isoscele.\n")
    else :
        print("Il triangolo è scaleno.\n")
    
if __name__ == '__main__' :
    
    a = 1. #float(sys.argv[1])
    b = 5. #float(sys.argv[2])
    c = 2. #float(sys.argv[3])

    det_triangolo(a, b, c)
