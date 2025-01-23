import pytest
import cellconstructor as CC, cellconstructor.Phonons
import cellconstructor.Manipulate
import sscha, sscha.Ensemble

import sys, os
import chain, MyFunctions
from MyFunctions import ToyModelCalculator

import numpy as np
import matplotlib.pyplot as plt
import tdscha, tdscha.Perturbations

def test_effective_charges_derivative(verbose = False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)

    dyn = CC.Phonons.Phonons("final_dyn")
    dyn.structure.coords[0,0] = 0
    dyn.structure.coords[1,0] = dyn.structure.unit_cell[0,0] / 2


    np.random.seed(0)
    N_RANDOM = 400
    N = 12
    _x_ = np.linspace(0, 0.4, N)
    _x_bohr_ = _x_ * CC.Units.A_TO_BOHR

    z0 = np.zeros(N, dtype = np.double)
    dz_dR = np.zeros(N, dtype = np.double)

    if verbose:
        print("Computing effective charges...")
        print()
    
    for i, x in enumerate(_x_):
        if verbose:
            sys.stdout.write("\rProgress {} / {}".format(i+1, N))
            sys.stdout.flush()
            
        dyn.structure.coords[0,0] = x
        
        ensemble = sscha.Ensemble.Ensemble(dyn, 0)
        ensemble.ignore_small_w = True
        ensemble.generate(N_RANDOM)
        
        # Get the effective charges
        eff_charges = get_effective_charges(ensemble.structures)
        z_av = tdscha.Perturbations.get_d1M_dR_av(ensemble, eff_charges)
        dz_dR_av = tdscha.Perturbations.get_d2M_dR_av(ensemble, eff_charges)

        z0[i] = z_av[0, 0]
        dz_dR[i] = dz_dR_av[0, 0, 0]

    if verbose:
        print()
        plt.figure()
        plt.plot(_x_bohr_, z0, label = "<Z>")
        plt.legend()
        plt.figure()
        plt.plot(_x_bohr_, np.gradient(z0, _x_bohr_), label = "Numeric grad")
        plt.plot(_x_bohr_, dz_dR, label = "anal grad")
        plt.legend()
        plt.tight_layout()
        plt.show()

    grad = np.gradient(z0, _x_bohr_)
    assert np.max(np.abs(dz_dR - grad)[1:-1]) < 8

    
    

    
    



def get_effective_charges(structs, ncell = 1, uc = 2.5):
    eff_charges = []

    Delta = 0.1 # onsite energy in eV
    t = 1 # hopping energy in eV
    beta = 1. # el-ph coupling parameter (in eV/A)
    # number of k-points to use in the electronic calculation
    N_k = 101
    k_ela = 1.1 # elastic constant of the ion-ion interaction (in eV/A^2)
    carbon_mass = 10947. # mass of the carbon atom (in Ry)

    calc = ToyModelCalculator(N_cells = ncell,
                              atoms_per_cell = 2,
                              cell_length = uc,
                              onsite = Delta,
                              hopping = t,
                              N_k = N_k,
                              mass = carbon_mass,
                              el_ph = beta,
                              k_ela = k_ela,
                              default = True)

    for s in structs:
        atm = s.get_ase_atoms()
        calc.calculate(atm)
        eff_charges.append(calc.results["effective charges"])

    return eff_charges
    
    
def test_eff_charges_with_calc(plot = False):
    total_path = os.path.dirname(os.path.abspath(__file__))
    os.chdir(total_path)
    
    dyn = CC.Phonons.Phonons("final_dyn")
    s = dyn.structure.copy()

    s.coords[0,:] = 0
    s.coords[1,0] = s.unit_cell[0,0] / 2

    
    N = 50
    _x_ = np.linspace(0, 0.2, N)
    z0 = np.zeros(N, dtype = np.double)

    ss = []
    for i, x in enumerate(_x_):
        s.coords[0,0] = x
        ss.append(s.copy())

    eff = [x[0, 0, 0] for x in get_effective_charges(ss)]


    if plot:
        plt.plot(_x_, eff)
        plt.show()
        

if __name__ == "__main__":
    test_effective_charges_derivative(True)
    test_eff_charges_with_calc(True)    
