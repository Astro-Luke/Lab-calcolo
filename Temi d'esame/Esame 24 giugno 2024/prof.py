import random
import math
import matplotlib.pyplot as plt
import numpy as np

from libprof import try_and_catch_exp, sturges


def main () :

    N_exp = 2000
    N_gau = 200

    sample_exp = try_and_catch_exp (1./200., N_exp)

    N_bins = int (sturges (len (sample_exp)))

    bin_content, bin_edges = np.histogram (sample_exp, bins = N_bins, range = (0, 3 * 200.))

    fig, ax = plt.subplots (nrows = 1, ncols = 1)
    ax.hist (sample_exp,
         bins = bin_edges,
         color = 'orange',
        )

    plt.savefig ("prof_histo.png")
    plt.show ()

if __name__ == "__main__" :
    main ()