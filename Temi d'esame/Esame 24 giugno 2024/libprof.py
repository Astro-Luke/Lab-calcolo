import random
import math
import numpy as np


# Funzione sturges per il binnaggio (funziona discretamente bene, ma conviene sempre veerificare)
def sturges (N_eventi) :
    return np.ceil (1 + np.log2 (N_eventi))


def rand_range (xMin, xMax) :
    return xMin + random.random () * (xMax - xMin)


def try_and_catch_exp (lamb, N):
    events = []
    x_max = 3/lamb
    for i in range (N):
      x = rand_range (0., x_max)
      y = rand_range (0., lamb)
      while (y > lamb * math.exp (-lamb * x)):
        x = rand_range (0., x_max)
        y = rand_range (0., lamb)
      events.append (x)
    return events