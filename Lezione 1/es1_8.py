'''
Scrivi un programma Python che trovi l'elenco dei numeri interi primi inferiori a 100, partendo dal presupposto che 2 è un numero primo
'''

def primo (n) :
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):  # Controllo fino alla radice quadrata di n
        if n % i == 0:
            return False
    return True

def numeri_primi_inferiori_a_100 ():
    lista_primi = []
    for num in range (2, 100):  # Controlliamo tutti i numeri da 2 a 99
        if primo (num):
            lista_primi.append (num)
    return lista_primi

if __name__ == '__main__' :

    lista = numeri_primi_inferiori_a_100 ()
    print(lista)
