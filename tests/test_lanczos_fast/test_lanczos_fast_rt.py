from __future__ import print_function

import numpy as np

import cellconstructor as CC
import cellconstructor.Phonons

import sscha, sscha.Ensemble
import tdscha, tdscha.DynamicalLanczos as DL

import scipy, scipy.sparse

from tdscha.Parallel import pprint as print

import sys, os
import mpi4py

CDIR= "../../CModules"
EXE = os.path.join(CDIR, "tdscha-lanczos.x")

def checkcompile():
    if not os.path.exists(EXE):
        os.system("D=$PWD; cd {}; make; cd $D;".format(CDIR))


def test_lanczos():
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    # Compile the C/C++ code if it does not exists
    checkcompile()

    T = 250
    NQIRR = 3

    dyn = CC.Phonons.Phonons("data/dyn_gen_pop1_", NQIRR)
    ens = sscha.Ensemble.Ensemble(dyn, T)
    ens.load_bin("data", 1)
    

    # Prepare the test c code
    dirname = "test_c_code_lanczos"

    lanc = DL.Lanczos(ens)
    N_STEPS = 5
    lanc.ignore_harmonic = False
    lanc.ignore_v3 = False
    lanc.ignore_v4 = False
    lanc.init(use_symmetries = True)
    lanc.prepare_mode(10)

    # Save everything for the c code execution
    lanc.prepare_input_files("mode10", N_STEPS, directory = dirname)
    lanc.run_FT(2*N_STEPS, save_dir = 'normal', debug = False)

#     # Run the lanczos with the C++ program
#     os.system("cd {} && ../{} mode10 > log && cd ..".format(dirname, EXE))

#     # HARDCODE the last c value
#     C_LAST_GOOD = 6.59744344e-07

#     assert np.abs(lanc.c_coeffs[4] - C_LAST_GOOD) / np.abs(C_LAST_GOOD) < 1e-6, "CVALUE CALCULATED: {} | EXPECTED C VALUE: {}".format(lanc.c_coeffs[4], C_LAST_GOOD)

#     # Load the abc value from the C++ programm
#     abcfile = np.loadtxt(os.path.join(dirname, "mode10.abc"))
#     cc_value = abcfile[-1, 2]
#     c8_value = lanc.c_coeffs[7]
    
#     assert np.abs(lanc.c_coeffs[4] - cc_value) / np.abs(cc_value) < 1e-6
    
#     lanc.load_from_input_files("mode10", directory = dirname)
    

    
#     gf = lanc.get_green_function_continued_fraction(np.array([0]))
#     w2 = 1 / np.real(gf)
#     w = np.sign(w2) * np.sqrt(np.abs(w2))
#     w *= CC.Units.RY_TO_CM

#     print("From C/C++ I get: {} cm-1".format(w))
#     lanc.run_FT(N_STEPS, debug = False)

#     assert np.abs(lanc.c_coeffs[7] - c8_value) / np.abs(c8_value) < 1e-6


if __name__ == "__main__":
    test_lanczos()
