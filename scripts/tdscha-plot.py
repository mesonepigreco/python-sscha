#!python
from __future__ import print_function
from __future__ import division

import sys, os

import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

import cellconstructor as CC
import cellconstructor.Phonons

import sscha
import tdscha, tdscha.DynamicalLanczos as DL
import sscha.Ensemble

HEADER = """

TDSCHA
------

"""


if __name__ == "__main__":
    print(HEADER)

    if len(sys.argv) not in [2, 4, 5]:
        print("""
Plot the spectrum of a TDSCHA calculation.

Usage: tdscha-plot.py <file.abc> [w_start w_end [smearing]]

Pass a .abc or .npz file resulting from a linear response calculation.
Optionally you can pass a range of frequencies (cm-1) and the smearing.
""")
        exit()
    
    fname = sys.argv[1]
    assert os.path.exists(fname), "Error, file {} does not exist".format(fname)
    

    lanc = DL.Lanczos()

    print("Loading file {}".format(fname))

    if fname.endswith(".abc"):
        lanc.load_abc(fname)
    elif fname.endswith(".npz"):
        lanc.load_status(fname)
    else:
        print("ERROR, the specified file must either be a .abc, or .npz file.")
    

    w_start = 0
    w_end = 5000
    n_w = 50000
    smearing = 5

    if len(sys.argv) > 2:
        w_start = float(sys.argv[2])
        w_end = float(sys.argv[3])
    if len(sys.argv) == 5:
        smearing = float(sys.argv[4])
    
    w = np.linspace(w_start, w_end, n_w)
    w_ry = w / CC.Units.RY_TO_CM
    smearing /= CC.Units.RY_TO_CM

    gf = lanc.get_green_function_continued_fraction(w_ry, smearing = smearing, use_terminator=False)
    spectrum = - np.imag(gf)

    # Print some info about the calculation
    print()
    print("Number of poles: {}".format(len(lanc.a_coeffs)))
    
    plt.plot(w, spectrum)
    plt.xlabel("Frequency [cm-1]")
    plt.ylabel("Spectrum [a.u.]")
    plt.tight_layout()
    plt.show()


