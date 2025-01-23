from __future__ import print_function
from __future__ import division

"""
This module performs the Lanczos algorithm in order to compute the responce function
of a particular perturbation.
"""

import sys, os
import time
import warnings, difflib
import numpy as np

import warnings

from timeit import default_timer as timer

import json

# Import the scipy Lanczos modules
import scipy, scipy.sparse.linalg

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries

import sscha.Ensemble as Ensemble
import tdscha.Tools as Tools
import sscha_HP_odd

# Override the print function to print in parallel only from the master
import cellconstructor.Settings as Parallel
from tdscha.Parallel import pprint as print
from tdscha.Parallel import *
import tdscha.Perturbations as perturbations


# Try to import the julia module
__JULIA_EXT__ = False
try:
    import julia, julia.Main

    # Compile the tdscha code
    julia.Main.include(os.path.join(os.path.dirname(__file__), "tdscha_core.jl"))
    __JULIA_EXT__ = True
except:
    pass

# Try to import the julia module
__JULIA_EXT__ = False
try:
    import julia, julia.Main
    julia.Main.include(os.path.join(os.path.dirname(__file__), 
        "tdscha_core.jl"))
    __JULIA_EXT__ = True
except:
    try:
        import julia
        from julia.api import Julia
        jl = Julia(compiled_modules=False)
        import julia.Main
        try:
            julia.Main.include(os.path.join(os.path.dirname(__file__),
                "tdscha_core.jl"))
            __JULIA_EXT__ = True
        except:
            # Install the required modules
            julia.Main.eval("""
using Pkg
Pkg.add("SparseArrays")
Pkg.add("InteractiveUtils")
""")
            try:
                julia.Main.include(os.path.join(os.path.dirname(__file__),
                    "tdscha_core.jl"))
                __JULIA_EXT__ = True
            except Exception as e:
                warnings.warn("Julia extension not available.\nError: {}".format(e))
    except Exception as e:
        warnings.warn("Julia extension not available.\nError: {}".format(e))
    pass


# Try to import the julia module
__JULIA_EXT__ = False
try:
    import julia, julia.Main
    julia.Main.include(os.path.join(os.path.dirname(__file__), 
        "tdscha_core.jl"))
    __JULIA_EXT__ = True
except:
    try:
        import julia
        from julia.api import Julia
        jl = Julia(compiled_modules=False)
        import julia.Main
        try:
            julia.Main.include(os.path.join(os.path.dirname(__file__),
                "tdscha_core.jl"))
            __JULIA_EXT__ = True
        except:
            # Install the required modules
            julia.Main.eval("""
using Pkg
Pkg.add("SparseArrays")
Pkg.add("InteractiveUtils")
""")
            try:
                julia.Main.include(os.path.join(os.path.dirname(__file__),
                    "tdscha_core.jl"))
                __JULIA_EXT__ = True
            except Exception as e:
                warnings.warn("Julia extension not available.\nError: {}".format(e))
    except Exception as e:
        warnings.warn("Julia extension not available.\nError: {}".format(e))
    pass


# Define a generic type for the double precision.
TYPE_DP = np.double
__EPSILON__ = 1e-12
N_REP_ORTH = 1

try:
    from ase.units import create_units
    units = create_units("2006")#Rydberg, Bohr
    Rydberg = units["Ry"]
    Bohr = units["Bohr"]
    __RyToK__ =  Rydberg / units["kB"]
    
except:
    Rydberg = 13.605698066 #RY->eV
    Bohr = 1.889725989 #Angstrom -> Bohr
    __RyToK__ = 157887.32400374097


__SPGLIB__ = True
try:
    import spglib
except:
    __SPGLIB__ = False
    

def f_ups(w, T):
    r"""
    The eigenvalue of the upsilon matrix as a function of the frequency and the
    temperature. This is (xi^2_\mu)^-1.
    
    Parameters:
    ----------
        -w, frequencies in Rydberg
        -T, temperature in Kelvin
    """

    n_w = bose_occupation(w, T)
    return 2*w / (1 + 2 *n_w)

def bose_occupation(w, T):
    """ The Bose-Einstein occupation. Assumes T in K and w in Ry."""
    n_w = 0
    if T > 0:
        n_w = 1 / (np.exp(w * __RyToK__ / T) - 1)
    return n_w

# Modes for the calculation
MODE_FAST_JULIA = 3
MODE_FAST_MPI = 2
MODE_FAST_SERIAL = 1
MODE_SLOW_SERIAL = 0

def is_julia_enabled():
    return __JULIA_EXT__

def is_julia_enabled():
    return __JULIA_EXT__


def is_julia_enabled():
    return __JULIA_EXT__


class Lanczos(object):
    def __init__(self, ensemble = None, mode = None, unwrap_symmetries = False, select_modes = None, use_wigner = False, lo_to_split = "random"):
        """
        INITIALIZE THE LANCZOS
        ======================

        This function extracts the weights, the X and Y arrays for the d3 and d4
        computation as well as the polarization vectors and frequencies.

        Parameters
        ----------
            ensemble : Ensemble.Ensemble()
                The ensemble upon which you want to compute the DynamicalResponce
            mode : int
                The mode of the speedup.
                   0) Slow python implementation 
                      Use this just for testing
                   1) Fast C serial code
                   2) Fast C parallel (MPI)
                   3) Fast Julia multithreading (only if julia is available)
            unwrap_symmetries : bool
                If true (default), the ensemble is unwrapped to respect the symmetries.
                This requires SPGLIB installed.
            select_modes : ndarray(size = n_modes, dtype = bool)
                A mask for each mode, if False, the mode is neglected. Use this to exclude some modes that you know are not
                involved in the calculation. If not specified, all modes are considered by default.  
            use_wigner: bool, if True Wigner equations are used.
                involved in the calculation. If not specified, all modes are considered by default.
            lo_to_split : string or ndarray
                Mode of lo_to_splitting. If empty or none, it is LO-TO splitting correction is neglected.
                If a ndarray is provided, it is the direction of q on which the LO-TO splitting is computed.
        """

        if is_julia_enabled():
            self.mode = MODE_FAST_JULIA
        else:
            self.mode = MODE_FAST_SERIAL

        if mode is not None:
            self.mode = mode

        # Define the order
        order = "C"
        #if self.mode >= 1:
        #    order = "F"

        # HERE DEFINE ALL THE VARIABLES FOR THE Dynamical Lanczos
        # The temperature
        self.verbose = True
        self.T = 0
        # Number of atoms in the supercell
        self.nat = 0
        # The array of masses in the supercell, np.shape = 3 * N_at_sc
        self.m = []
        # Auxiliary eigenmodes of SCHA
        self.w = []
        # Auxiliary eigenvectors of SCHA
        self.pols = []
        # The numbero of modes, translations excluded
        self.n_modes = 0
        self.ignore_harmonic = False 
        # Ignore D3 and D4
        self.ignore_v3 = False
        self.ignore_v4 = False
        # Number of configurations
        self.N = 0
        # The weights from the static calculations
        self.rho = []
        # Effective number of configurations
        self.N_eff = 0
        # The satic displacements in the polarization basis
        self.X = []
        # The static forces in the polarization basis
        self.Y = []
        # The vector on which we apply Lanczos
        self.psi = []
        self.eigvals = None
        self.eigvects = None
        # In the custom lanczos mode
        self.a_coeffs = [] # Coefficients on the diagonal
        self.b_coeffs = [] # Coefficients close to the diagonal
        self.c_coeffs = [] # Coefficients in the case of the biconjugate Lanczos
        self.krilov_basis = [] # The basis of the krilov subspace
        self.basis_P = [] # The basis of the P vectors for the biconjugate Lanczos (normalized)
        self.basis_Q = [] # The basis of the Q vectors for the biconjugate Lanczos
        self.s_norm = [] # Store the normalization of the s vector, this allows to rebuild the correct p when needed. 
        self.arnoldi_matrix = [] # If requested, the upper triangular arnoldi matrix
        self.reverse_L = False
        self.shift_value = 0
        self.symmetrize = False
        self.symmetries = None
        self.degenerate_space = None
        self.N_degeneracy = None
        self.initialized = False
        self.perturbation_modulus = 1
        self.dyn = None
        # Unit cell structure
        self.uci_structure = None
        # Structure for the supercell
        self.super_structure = None
        # Symmetries
        self.qe_sym = None
        # The application of L as a linear operator
        self.L_linop = None
        self.M_linop = None
        self.unwrapped = False
        self.sym_julia = None
        self.deg_julia = None
        self.n_syms = 1

        self.u_tilde = None
        # The static forces divided by the sqrt(masses)
        self.f_tilde = None

        self.sym_block_id = None
        
        # Set to True if we want to use the Wigner equations
        self.use_wigner = use_wigner
        
        # This flag is usefull to work with 1D or 2D systems
        # Default is False meaning that we ignore only translational modes
        self.ignore_small_w = False

        # Setup the attribute control
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        self.fixed_attributes = True # This must be the last attribute to be setted

        # Perform a bare initialization if the ensemble is not provided
        if ensemble is None:
            return


        # ========== END OF VARIABLE DEFINITION (EACH NEW DEFINITION FROM NOW ON RESULTS IN AN ERROR) =======
        self.dyn = ensemble.current_dyn.Copy() 
        self.uci_structure = ensemble.current_dyn.structure.copy()
        self.super_structure = self.dyn.structure.generate_supercell(self.dyn.GetSupercell())#superdyn.structure

        self.T = ensemble.current_T

        ws, pols = self.dyn.DiagonalizeSupercell(lo_to_split = lo_to_split)

        self.nat = self.super_structure.N_atoms
        n_cell = np.prod(self.dyn.GetSupercell())

        self.qe_sym = CC.symmetries.QE_Symmetry(self.dyn.structure)
        self.qe_sym.SetupQPoint()

        # Get the masses
        m = self.super_structure.get_masses_array()
        self.m = np.tile(m, (3,1)).T.ravel()

        # Remove the translations
        if not ensemble.ignore_small_w:
            trans_mask = CC.Methods.get_translations(pols, m)
            good_mask  = ~trans_mask
        else:
            self.ignore_small_w = True
            trans_mask = np.abs(ws) < CC.Phonons.__EPSILON_W__
            good_mask  = ~trans_mask
        
        # If requested, isolate only the specified modes.
        if select_modes is not None:
            if len(select_modes) != len(trans_mask):
                raise ValueError("""
Error, 'select_modes' should be an array of the same lenght of the number of modes.
 n_modes = {} | len(select_modes) = {}
""".format(len(ws), len(select_modes)))
            print()
            print('Selecting some of the modes...')
            print()
            good_mask = (~trans_mask) & select_modes

        # Get the frequencies in Ry and polarization vectors
        self.w = ws[good_mask]
        self.pols = pols[:, good_mask]

        # Correctly reshape the polarization in case only one mode is selected
        if len(self.w) == 1:
            self.pols = self.pols.reshape((len(self.m), 1))
        
        # Get the number of modes
        self.n_modes = len(self.w)

        # Prepare the list of q point starting from the polarization vectors
        #q_list = CC.symmetries.GetQForEachMode(self.pols, self.uci_structure, self.super_structure, self.dyn.GetSupercell())
        # Store the q vectors in crystal space
        #bg = self.uci_structure.get_reciprocal_vectors() / 2* np.pi
        #self.q_vectors = np.zeros((self.n_modes, 3), dtype = np.double, order = "C")
        #for iq, q in enumerate(q_list):
        #    self.q_vectors[iq, :] = CC.Methods.covariant_coordinate(bg, q)
        
        # Ignore v3 or v4. You can set them for testing
        # This is no longer implemented in the fast Lanczos
        self.ignore_v3 = False
        self.ignore_v4 = False

        # The number of configurations
        self.N = ensemble.N
        rho = ensemble.rho.copy() 
        # Transform Angstrom -> Bohr
        u = ensemble.u_disps  / Ensemble.Bohr
        # Forces are in Ry/Angstrom for now only
        f = ensemble.forces.reshape(self.N, 3 * self.nat).copy()
        f -= ensemble.sscha_forces.reshape(self.N, 3 * self.nat)

        # Get the average force in the unit cell, (N_at_uc, 3)
        f_mean = ensemble.get_average_forces(get_error = False)

        # Perform the symmetrization of the average force
        qe_sym = CC.symmetries.QE_Symmetry(self.dyn.structure)
        qe_sym.SetupQPoint()
        qe_sym.SymmetrizeVector(f_mean)

        # Reproduce the average force on the full supercell
        f_mean = np.tile(f_mean, (np.prod(ensemble.current_dyn.GetSupercell()), 1)).ravel()

        # Transform forces in Ry/Bohr
        f_mean *= Ensemble.Bohr
        
        # Subtract also the average force to clean more the stochastic noise
        #av_force = ensemble.get_average_forces(get_error = False).ravel()
        #new_av_force = np.tile(av_force, (n_cell, 1)).ravel()
        
        #f -= np.tile(new_av_force, (self.N, 1)) 
        
        # Transform in Ry/Bohr
        f *= Ensemble.Bohr

        if unwrap_symmetries:
            u, f, rho = ensemble.get_unwrapped_ensemble()
            self.N = len(rho)
            self.unwrapped = True

            u /= Ensemble.Bohr
            f *= Ensemble.Bohr

        # Subtract the SSCHA GRADIENT on average position
        # In this way the calculation works even if the system is not in equilibrium
        #print(np.shape(f), np.shape(f_mean))
        f[:, :] -= np.tile(f_mean, (self.N, 1))

        # Perform the mass rescale to get the tilde variables
        u *= np.tile(np.sqrt(self.m), (self.N, 1)) 
        f /= np.tile(np.sqrt(self.m), (self.N, 1)) 

        # Get the info about the ensemble
        self.rho = rho
        self.N_eff = np.sum(self.rho)

        # Mass rescaled quantities
        self.u_tilde = u
        self.f_tilde = f

        # The dispalcements in BOHR mass resclaed and in polarization basis
        self.X = np.zeros((self.N, self.n_modes), order = order, dtype = TYPE_DP)
        # The forces in RY/BOHR mass resclaed and in polarization basis
        self.Y = np.zeros((self.N, self.n_modes), order = order, dtype = TYPE_DP)

        # Convert in the polarization space the displacements and the forces
        self.X[:, :] = self.u_tilde.dot(self.pols) #.T.dot(self.u_tilde)
        self.Y[:, :] = self.f_tilde.dot(self.pols) #self.pols.T.dot(self.f_tilde)

        # Prepare the variable used for the working
        # The len of psi = N_modes + 0.5 * N_modes * (N_modes + 1) + 0.5 * N_modes * (N_modes + 1)
        len_psi = self.n_modes
        #if self.T < __EPSILON__:
        #    len_psi += self.n_modes**2
        #else:
        len_psi += self.n_modes * (self.n_modes + 1)
        #print("N MODES:", self.n_modes)
        #print("LEN PSI:", len_psi)
        
        # In Wigner the variables are a'^(1) and b'^(1) 
        # In Standard the variables are Y^(1) and ReA^(1)
        
        ##########################################################
        # Psi contains R^(1), Upsilon^(1)-a'^(1), ReA^(1)-b'^(1) #
        ##########################################################
        
        # Everything is in the polarization basis
        self.psi = np.zeros(len_psi, dtype = TYPE_DP)
        
        ################################################################
        # For the matrices the code will store only the upper triangle #
        ################################################################

        # Prepare the L as a linear operator 
        # Prepare the possibility to transpose the matrix with L_transp
        def L_transp(psi):
            return self.apply_full_L(psi, transpose = True)
        self.L_linop = scipy.sparse.linalg.LinearOperator(shape = (len(self.psi), len(self.psi)),\
                                                          matvec = self.apply_full_L, rmatvec = L_transp, dtype = TYPE_DP)

        # Define the preconditioner
        def M_transp(psi):
            return self.apply_L1_inverse_FT(psi, transpose = True)
        self.M_linop = scipy.sparse.linalg.LinearOperator(shape = (len(self.psi), len(self.psi)),\
                                                          matvec = self.apply_L1_inverse_FT, rmatvec = M_transp, dtype = TYPE_DP)


        # Prepare the solution of the Lanczos algorithm
        self.eigvals  = None
        self.eigvects = None 

        # Store the basis and the coefficients of the Lanczos procedure
        # In the custom lanczos mode
        self.krilov_basis = [] # The basis of the krilov subspace
        self.arnoldi_matrix = [] # If requested, the upper triangular arnoldi matrix

        # These are some options that can be used to properly reverse and shift the L operator to
        # fasten the convergence of low energy modes
        self.reverse_L = False
        self.shift_value = 0



    def __setattr__(self, name, value):
        """
        This method is used to set an attribute.
        It will raise an exception if the attribute does not exists (with a suggestion of similar entries)
        """
        
        if "fixed_attributes" in self.__dict__:
            if name in self.__total_attributes__:
                super(Lanczos, self).__setattr__(name, value)
            elif self.fixed_attributes:
                similar_objects = str( difflib.get_close_matches(name, self.__total_attributes__))
                ERROR_MSG = """
        Error, the attribute '{}' is not a member of '{}'.
        Suggested similar attributes: {} ?
        """.format(name, type(self).__name__,  similar_objects)

                raise AttributeError(ERROR_MSG)
        else:
            super(Lanczos, self).__setattr__(name, value)

            if "ignore_v" in name:
                warnings.warn("Setting {} is deprecated. It will always be True.".format(name), DeprecationWarning)
        


    def reset(self):
        """
        RESET THE LANCZOS
        =================

        This function reset the Lanczos algorithm, allowing for a new responce function calculation
        with the same ensemble and the same settings.
        """

        # Prepare the variable used for the working
        len_psi = self.n_modes
        len_psi += self.n_modes * (self.n_modes + 1)
        self.psi = np.zeros(len_psi, dtype = TYPE_DP)


        # Prepare the solution of the Lanczos algorithm
        self.eigvals = None
        self.eigvects = None 

        # Store the basis and the coefficients of the Lanczos procedure
        # In the custom lanczos mode
        self.a_coeffs = [] # Coefficients on the diagonal
        self.b_coeffs = [] # Coefficients close to the diagonal
        self.c_coeffs = []

        # The krilov basis for the symmetric and unsymmetric Lanczos
        self.basis_P = []
        self.basis_Q = []
        self.s_norm = []
        self.krilov_basis = [] # The basis of the krilov subspace
        self.arnoldi_matrix = [] # If requested, the upper triangular arnoldi matrix


    def init(self, use_symmetries = True):
        """
        INITIALIZE THE CALCULATION
        ==========================

        Perform everithing needed to initialize the calculation.

        Parameters
        ----------
            use_symmetries : bool
                if False (default True) symmetries are neglected (unless the ensemble has been unwrapped).
    
        """
        # Prepare the variable used for the working
        len_psi  = self.n_modes
        len_psi += self.n_modes * (self.n_modes + 1)
        self.psi = np.zeros(len_psi, dtype = TYPE_DP)
        
        self.prepare_symmetrization(no_sym = not use_symmetries)
        self.initialized = True


    def interpolate(self, q_mesh, support_dyn = None, auto_init = True):
        """
        INTERPOLATION
        =============

        This subroutine prepare the Lanczos algorithm to run on a bigger mesh than the one defined on the original supercell.
        This is fundamental to correctly converge resonances.

        This method automatically initializes with symmetries the new Lanczos. 
        You can disable this behaviour setting auto_init = False.

        Parameters
        ----------
            q_mesh : list or ndarray (size=3, dtye = np.intc)
                The mesh of q points on which you want to perform the simulation.
                It should be bigger than the original supercell size.
            support_dyn : CC.Phonons.Phonons, optional
                By default, the original dynamical matrix will be Fourier interpolated in the new q_mesh.
                However, you can provide a new dynamical matrix in the q_mesh already interpolated.
                This is usefull if you want to use a custom interpolation
                (for example to interpolate only the differences between the SSCHA force constat matrix and the harmonic one).
            auto_init: bool
                If True, after the interpolation is performed, the new Lanczos object will be initialized (with symmetries).
                If you disable it, you must call the init function manually
        
        Results
        -------
            interpolated_lanczos : Lanczos()
                A new Lanczos class object, with interpolated data.

        """
        interpolated_lanczos = Lanczos()

        # Interpolate the original dynamical matrix
        if support_dyn is not None:
            interpolated_lanczos.dyn = support_dyn.Copy()
        else:
            interpolated_lanczos.dyn = self.dyn.Interpolate(q_mesh)

        # Prepare the new structure
        interpolated_lanczos.uci_structure = interpolated_lanczos.dyn.structure.copy()
        interpolated_lanczos.super_structure = interpolated_lanczos.uci_structure.generate_supercell(q_mesh)

        interpolated_lanczos.T = self.T 

        ws, pols = interpolated_lanczos.dyn.DiagonalizeSupercell()


        interpolated_lanczos.nat = interpolated_lanczos.super_structure.N_atoms
        n_cell = np.prod(q_mesh)

        interpolated_lanczos.qe_sym = CC.symmetries.QE_Symmetry(interpolated_lanczos.uci_structure)
        interpolated_lanczos.qe_sym.SetupQPoint()

        # Get the masses
        m = interpolated_lanczos.super_structure.get_masses_array()
        interpolated_lanczos.m = np.tile(m, (3,1)).T.ravel()

        # Remove the translations
        trans_mask = CC.Methods.get_translations(pols, m)

        # Isolate only translational modes
        good_mask = ~trans_mask

        # Get the polarization vectors
        interpolated_lanczos.w = ws[good_mask]
        interpolated_lanczos.pols = pols[:, good_mask]

        # Correctly reshape the polarization in case only one mode is selected
        if len(self.w) == 1:
            self.pols = self.pols.reshape((len(self.m), 1))

        interpolated_lanczos.n_modes = len(interpolated_lanczos.w)


        # Prepare the list of q point starting from the polarization vectors
        #q_list = CC.symmetries.GetQForEachMode(self.pols, self.uci_structure, self.super_structure, self.dyn.GetSupercell())
        # Store the q vectors in crystal space
        bg = interpolated_lanczos.uci_structure.get_reciprocal_vectors() / 2* np.pi
        interpolated_lanczos.q_vectors = np.zeros((interpolated_lanczos.n_modes, 3), dtype = np.double, order = "C")
        #for iq, q in enumerate(q_list):
        #    self.q_vectors[iq, :] = CC.Methods.covariant_coordinate(bg, q)


        # Prepare the interpolation of the ensemble
        interpolated_lanczos.rho = self.rho.copy()
        interpolated_lanczos.N_eff = np.sum(self.rho)

        interpolated_lanczos.u_tilde = self.u_tilde.copy()
        interpolated_lanczos.f_tilde = self.f_tilde.copy()

        interpolated_lanczos.X = np.zeros((interpolated_lanczos.N, interpolated_lanczos.n_modes), order = order, dtype = TYPE_DP)
        interpolated_lanczos.Y = np.zeros((interpolated_lanczos.N, interpolated_lanczos.n_modes), order = order, dtype = TYPE_DP)

        # TODO: Interpolation of the q vectors with the Tetrahedral method.
        raise NotImplementedError("Error, interpolation is not yet implemented.")
        interpolated_lanczos.X[:, :] = self.u_tilde.dot(self.pols) #.T.dot(self.u_tilde)
        interpolated_lanczos.Y[:, :] = self.f_tilde.dot(self.pols) #self.pols.T.dot(self.f_tilde)

        
        

        if auto_init:
            interpolated_lanczos.init()
        return interpolated_lanczos
        


    def prepare_symmetrization(self, no_sym = False, verbose = True, symmetries = None):
        """
        PREPARE THE SYMMETRIZATION
        ==========================

        This function analyzes the character of the symmetry operations for each polarization vectors.
        This will allow the method do know how many modes are degenerate.

        If the ensemble has been unwrapped, then the symmetries are not initialized.

        Parameters
        ----------
            no_sym : bool
                If True, the symmetries are neglected.
            symmetries : list of 3x4 matrices
                If None, spglib is employed to find the symmetries,
                         otherwise, the symmetries here contained are employed.
        """

        self.initialized = True

        # All the rest is deprecated in the Fast Lanczos implementation
        # As the symmetrization is performed by unwrapping the ensemble

        # Generate the dynamical matrix in the supercell
        super_structure = self.dyn.structure.generate_supercell(self.dyn.GetSupercell())
        w, pols = self.dyn.DiagonalizeSupercell()

        # Get the symmetries of the super structure
        if not __SPGLIB__ and not no_sym:
            raise ImportError("Error, spglib module required to perform symmetrization in a supercell. Otherwise, use no_sym")
        
        # Neglect the symmetries
        if no_sym or self.unwrapped:
            self.symmetries = [np.ones( (1,1,1), dtype = np.double)] * self.n_modes
            self.N_degeneracy = np.ones(self.n_modes, dtype = np.intc)
            self.degenerate_space = [np.array([i], dtype = np.intc) for i in range(self.n_modes)]
            self.sym_block_id = np.arange(self.n_modes).astype(np.intc)
            return

        t1 = time.time()
        if symmetries is None:
            super_symmetries = CC.symmetries.GetSymmetriesFromSPGLIB(spglib.get_symmetry(super_structure.get_ase_atoms()), False)

        else:
            super_symmetries = symmetries
        t2 = time.time()


        if verbose:
            print("Time to get the symmetries [{}] from spglib: {} s".format(len(super_symmetries), t2-t1))


        # Get the symmetry matrix in the polarization space
        # Translations are needed, as this method needs a complete basis.
        pol_symmetries, basis = CC.symmetries.GetSymmetriesOnModesDeg(super_symmetries, super_structure, self.pols, self.w)
        #pol_symmetries = CC.symmetries.GetSymmetriesOnModes(super_symmetries, super_structure, pols)
        t1 = time.time()
        if verbose:
            print("Time to convert symmetries in the polarizaion space: {} s".format(t1-t2))

        self.symmetries = pol_symmetries
        self.degenerate_space = [np.array(x, dtype = np.intc) for x in basis]
        self.N_degeneracy = np.array([len(x) for x in basis], dtype = np.intc)
        self.sym_block_id = -np.ones(self.n_modes, dtype = np.intc)
        self.n_syms =  self.symmetries[0].shape[0]

        if self.mode is MODE_FAST_JULIA:
            # Get the max length
            max_val = 0
            nblocks = len(self.symmetries)
            for s in self.symmetries:
                m = s.shape[1]
                if m > max_val:
                    max_val = m
            self.sym_julia = np.zeros((nblocks, self.n_syms, max_val, max_val), dtype = TYPE_DP)
            self.deg_julia = np.zeros((nblocks, max_val), dtype = np.int32)

            # Now fill the array
            for i, sblock in enumerate(self.symmetries):
                nsym, c, _ = np.shape(sblock)
                self.sym_julia[i, :, :c, :c] = sblock
                self.deg_julia[i, :c] = self.degenerate_space[i]

        # Create the mapping between the modes and the block id.
        t1 = time.time()
        for i in range(self.n_modes):
            for j, block in enumerate(self.degenerate_space):
                if i in block:
                    self.sym_block_id[i] = j 
                    break
            
            assert self.sym_block_id[i] >= 0, "Error, something went wrong during the symmetrization"
        t2 = time.time()

        if verbose:
            print("Time to create the block_id array: {} s".format(t2-t1))

        # Ns, dumb, dump = np.shape(pol_symmetries)
        
        # # Now we can pull out the translations
        # trans_mask = CC.Methods.get_translations(pols, super_structure.get_masses_array())
        # self.symmetries = np.zeros( (Ns, self.n_modes, self.n_modes), dtype = TYPE_DP)
        # ptmp = pol_symmetries[:, :,  ~trans_mask] 
        # self.symmetries[:,:,:] = ptmp[:, ~trans_mask, :]

        # # Get the degeneracy
        # w = w[~trans_mask]
        # N_deg = np.ones(len(w), dtype = np.intc)
        # start_deg = -1
        # deg_space = [ [x] for x in range(self.n_modes)]
        # for i in range(1, len(w)):
        #     if np.abs(w[i-1] - w[i]) < __EPSILON__ :
        #         N_deg[i] = N_deg[i-1] + 1

        #         if start_deg == -1:
        #             start_deg = i - 1

        #         for j in range(start_deg, i):
        #             N_deg[j] = N_deg[i]
        #             deg_space[j].append(i)
        #             deg_space[i].append(j)
        #     else:
        #         start_deg = -1


        #self.N_degeneracy = N_deg
        #self.degenerate_space = deg_space

    def prepare_input_files(self, root_name = "tdscha", n_steps = 100, start_from_scratch = True, directory=".", run_symm = False):
        """
        PREPARE INPUT FILES
        ===================

        This method prepares the input files for the submission with the binary executable.
        This is usefull to prepare the input in a local computer and submit the calculation on a cluster,
        where it is easier to work.

        Parameters
        ----------
            root_name : string
                The title of the calculation.
            n_steps: int
                The number of steps to be performed in the lanczos calculation.
            start_from_scratch: bool
                If True the calculation is restarted from scratch.
            directory : string
                Path to the directory on which the input files will be saved
            run_symm : bool
                True if we use the Wigner representation

        
        This file will prepare inside the directory the following input files.
        (where XXX = root_name)

        XXX.json
        XXX.X.dat
        XXX.Y.dat
        XXX.syms.$i   ($i = 0, ..., N symmetries - 1)
        XXX.degs
        XXX.psi

        and the following optional files (in case the calculation should be restarted):

        XXX.Qbasis
        XXX.Pbasis


        The file XXX.json contains the generic information about the minimization,
        as well as all arrays not too big and that can be easily stored in a json file.

        All the other data contain 2d arrays or more sophisticated data.
        """

        Nsyms, _,_ = np.shape(self.symmetries[0])

        json_data = {"T" : self.T, 
                     "n_steps" : n_steps,
                     "ignore_v2" : self.ignore_harmonic,
                     "ignore_v3" : self.ignore_v3,
                     "ignore_v4" : self.ignore_v4,
                     "use_wigner" : self.use_wigner,
                     "run_sym": run_symm,
                     "data" : {
                         "n_configs" : int(self.N),
                         "n_modes" : int(self.n_modes),
                         "n_syms" : Nsyms,
                         "n_blocks" : len(self.symmetries),
                         "perturbation_modulus" : self.perturbation_modulus,
                         "reverse" : self.reverse_L,
                         "shift" : self.shift_value} }
        
        Parallel.barrier()

        if not start_from_scratch:
            raise NotImplementedError("Error, restarting from a previous calculation is not yet implemented.")

        if Parallel.am_i_the_master():
            root_fname = os.path.join(directory, root_name)

            if not os.path.exists(directory):
                os.makedirs(directory)

            # Writhe the json input file
            with open(root_fname + ".json", "w") as fp:
                json.dump(json_data, fp)

            # Save 1D arrays
            np.savetxt(root_fname + ".ndegs", self.N_degeneracy, fmt = "%d")
            np.savetxt(root_fname + ".blockid", self.sym_block_id, fmt = "%d")
            np.savetxt(root_fname + ".masses", self.m)
            np.savetxt(root_fname + ".freqs", self.w)
            np.savetxt(root_fname + ".rho", self.rho)
            
            # Save all the other data
            np.savetxt(root_fname + ".X.dat", self.X)
            np.savetxt(root_fname + ".Y.dat", self.Y)


            np.savetxt(root_fname + ".psi", self.psi)

            # Prepare the symmetry variables for the C code
            for i in range(len(self.symmetries)):
                np.savetxt(root_fname + ".block{:d}".format(i), self.degenerate_space[i], fmt = "%d")


                ns, b1, b2 = self.symmetries[i].shape
                with open(root_fname + ".symsb{:d}".format(i), "w") as fp:
                    for isym in range(ns):
                        for k1 in range(b1):
                            for k2 in range(b2):
                                fp.write(" {:22.16f}".format(self.symmetries[i][isym, k1, k2]))
                            fp.write("\n")
                        fp.write("\n")



    def load_from_input_files(self, root_name = "tdscha", directory="."):
        """
        Load the results from a calculation performed by the binary executable.
        
        NOTE: You must initialize the ensemble as did before calling the prepare_input_files method.
        Then execute the lanczos run with the tdscha-lanczos.x executable.
        Then load the results of the lanczos with this method.

        You must use the same keyword used in the prepare_input_files
        """

        if not os.path.exists(directory):
            raise IOError("Error, the specified directory '{}' does not exist".format(directory))

        abc_file = os.path.join(directory, "{}.abc".format(root_name))
        if not os.path.exists(abc_file):
            errmsg = """
Error, the file '{}' does not exist. 
       please, check if you correctly run the tdscha-lanczos.x executable.
""".format(abc_file)
            print(errmsg)
            raise IOError(errmsg)


        data_abc = np.loadtxt(abc_file, dtype= np.double)
        self.a_coeffs = data_abc[:,0]
        self.b_coeffs = data_abc[:, 1]
        self.c_coeffs = data_abc[:, 2]


        # Load the Pbasis and Qbasis only if the files exists.
        # These are very heavy files, and they are not needed for the spectral function
        # in the continued fraction representation.
        # (they are needed to restart a calculation)

        qbasis_file = os.path.join(directory, "{}.qbasis.out".format(root_name))
        pbasis_file = os.path.join(directory, "{}.pbasis.out".format(root_name))
        snorm_file = os.path.join(directory, "{}.snorm.out".format(root_name))

        if os.path.exists(qbasis_file):
            self.basis_Q = np.loadtxt(qbasis_file, dtype = np.double)
        else:
            warnmsg = """
File {} not found. basis_Q not loaded.
""".format(qbasis_file)
            warnings.warn(warnmsg)
        
        if os.path.exists(pbasis_file):
            self.basis_P = np.loadtxt(pbasis_file, dtype = np.double)
        else:
            warnmsg = """
File {} not found. basis_P not loaded.
""".format(pbasis_file)
            warnings.warn(warnmsg)

        if os.path.exists(snorm_file):
            self.s_norm = np.loadtxt(snorm_file, dtype = np.double).ravel()
        else:
            warnmsg = """
File {} not found. S norm not loaded.
""".format(snorm_file)
            warnings.warn(warnmsg)

        # Load the Json file with other general variables
        with open(os.path.join(directory, root_name + ".json"), "r") as fp:
            json_data = json.load(fp)

        self.perturbation_modulus = json_data["data"]["perturbation_modulus"]
        self.T = json_data["T"]
        self.ignore_harmonic = json_data["ignore_v2"]
        self.ignore_v3 = json_data["ignore_v3"]
        self.ignore_v4 = json_data["ignore_v4"]
        self.N = json_data["data"]["n_configs"]
        self.n_modes = json_data["data"]["n_modes"]
        self.reverse_L = json_data["data"]["reverse"]
        self.shift_value = json_data["data"]["shift"]




    def prepare_raman(self, pol_vec_in = np.array([1,0,0]), pol_vec_out = np.array([1,0,0]), mixed = False, pol_in_2 = None, pol_out_2 = None, unpolarized: int = None):
        """
        PREPARE LANCZOS FOR RAMAN SPECTRUM
        ==================================

        This subroutines prepare the perturbation for the Raman signal.

        The raman tensor is read from the dynamical matrix provided by the original ensemble.

        Parameters
        ----------
            pol_vec_in : ndarray (size =3)
                The polarization vector of the incoming light
            pol_vec_out : ndarray (size = 3)
                The polarization vector for the outcoming light
            unpolarized : int or None
                The perturbation for unpolarized raman (if different from None, overrides the behaviour
                of pol_vec_in and pol_vec_out). Indices goes from 0 to 6 (included).
                0 is alpha^2
                1 + 2 + 3 + 4 + 5 + 6 are beta^2
                alpha_0 = (xx + yy + zz)^2/9
                beta_1 = (xx -yy)^2 / 2
                beta_2 = (xx -zz)^2 / 2
                beta_3 = (yy -zz)^2 / 2
                beta_4 = 3xy^2
                beta_5 = 3xz^2
                beta_6 = 3yz^2

                The total unpolarized raman intensity is 45 alpha^2 + 7 beta^2
        """

        # Check if the raman tensor is present
        assert not self.dyn.raman_tensor is None, "Error, no Raman tensor found. Cannot initialize the Raman responce"

        # Get the raman vector (apply the ASR and contract the raman tensor with the polarization vectors)
        raman_v = self.dyn.GetRamanVector(pol_vec_in, pol_vec_out)
        
        if mixed:
            print('Prepare Raman')
            print('Adding other component of the Raman tensor')
            raman_v += self.dyn.GetRamanVector(pol_in_2, pol_out_2)

        # Get the raman vector in the supercelld
        n_supercell = np.prod(self.dyn.GetSupercell())

        if unpolarized is None:
            # Get the raman vector
            raman_v = self.dyn.GetRamanVector(pol_vec_in, pol_vec_out)

            # Get the raman vector in the supercelld
            new_raman_v = np.tile(raman_v.ravel(), n_supercell)

            # Convert in the polarization basis and store the intensity
            self.prepare_perturbation(new_raman_v, masses_exp=-1)
        else:
            px = np.array([1,0,0])
            py = np.array([0,1,0])
            pz = np.array([0,0,1])

            if unpolarized == 0:
                # Alpha
                raman_v = self.dyn.GetRamanVector(px, px)
                new_raman_v = np.tile(raman_v.ravel(), n_supercell) / 3
                self.prepare_perturbation(new_raman_v, masses_exp=-1)

                raman_v = self.dyn.GetRamanVector(py, py)
                new_raman_v = np.tile(raman_v.ravel(), n_supercell) / 3
                self.prepare_perturbation(new_raman_v, masses_exp=-1, add = True)

                raman_v = self.dyn.GetRamanVector(pz, pz)
                new_raman_v = np.tile(raman_v.ravel(), n_supercell) / 3
                self.prepare_perturbation(new_raman_v, masses_exp=-1, add = True)
            elif unpolarized == 1:
                # (xx -yy)^2 / 2
                raman_v = self.dyn.GetRamanVector(px, px)
                new_raman_v = np.tile(raman_v.ravel(), n_supercell) / np.sqrt(2)
                self.prepare_perturbation(new_raman_v, masses_exp=-1)

                raman_v = self.dyn.GetRamanVector(py, py)
                new_raman_v = - np.tile(raman_v.ravel(), n_supercell) / np.sqrt(2)
                self.prepare_perturbation(new_raman_v, masses_exp=-1, add = True)
            elif unpolarized == 2:
                # beta_2 = (xx -zz)^2 / 2
                raman_v = self.dyn.GetRamanVector(px, px)
                new_raman_v = np.tile(raman_v.ravel(), n_supercell) / np.sqrt(2)
                self.prepare_perturbation(new_raman_v, masses_exp=-1)

                raman_v = self.dyn.GetRamanVector(pz, pz)
                new_raman_v = - np.tile(raman_v.ravel(), n_supercell) / np.sqrt(2)
                self.prepare_perturbation(new_raman_v, masses_exp=-1, add = True)
            elif unpolarized == 3:
                # beta_2 = (yy -zz)^2 / 2
                raman_v = self.dyn.GetRamanVector(py, py)
                new_raman_v = np.tile(raman_v.ravel(), n_supercell) / np.sqrt(2)
                self.prepare_perturbation(new_raman_v, masses_exp=-1)

                raman_v = self.dyn.GetRamanVector(pz, pz)
                new_raman_v = - np.tile(raman_v.ravel(), n_supercell) / np.sqrt(2)
                self.prepare_perturbation(new_raman_v, masses_exp=-1, add = True)
            elif unpolarized == 4:
                # beta_2 = 3 xy^2
                raman_v = self.dyn.GetRamanVector(px, py)
                new_raman_v = np.tile(raman_v.ravel(), n_supercell) * np.sqrt(3)
                self.prepare_perturbation(new_raman_v, masses_exp=-1)
            elif unpolarized == 5:
                # beta_2 = 3 yz^2
                raman_v = self.dyn.GetRamanVector(py, pz)
                new_raman_v = np.tile(raman_v.ravel(), n_supercell) * np.sqrt(3)
                self.prepare_perturbation(new_raman_v, masses_exp=-1)
            elif unpolarized == 6:
                # beta_2 = 3 xz^2
                raman_v = self.dyn.GetRamanVector(px, pz)
                new_raman_v = np.tile(raman_v.ravel(), n_supercell) * np.sqrt(3)
                self.prepare_perturbation(new_raman_v, masses_exp=-1)
            else:
                raise ValueError("Error, unpolarized must be between [0, ... ,6] got invalid {}.".format(unpolarized))




        # Convert in the polarization basis and store the intensity
        self.prepare_perturbation(new_raman_v, masses_exp=-1)
        
    def get_prefactors_unpolarized_raman(self, index):
        """
        RETURNS THE PREFACTORS FOR COMPUTING THE UNPOLARIZED RAMAN
        ==========================================================
        
        It returns a dictionary with the prefactors
        
        The prefactors corresponds to the components of the unpolarized raman signal
        """
        labels = [i for i in range(7)]
        if not(index in labels):
            raise ValueError('{} should be in {}'.format(index, labels))
            
        dictionary = {'(xx+yy+zz)^2' : 45/9,\
                      '(xx-yy)^2'    : 7/2,\
                      '(xx-zz)^2'    : 7/2,\
                      '(yy-zz)^2'    : 7/2,\
                      '(xy)^2'       : 7*3,\
                      '(xz)^2'       : 7*3,\
                      '(yz)^2'       : 7*3}
        
        keys = list(dictionary.keys())
        
        
        return dictionary[keys[index]]
    
    def prepare_unpolarized_raman(self, index = 0, debug = False):
        """
        PREPARE UNPOLARIZED RAMAN SIGNAL
        ================================
        
        The raman tensor is read from the dynamical matrix provided by the original ensemble.
        
        The perturbations are prepared accordin to the formula (see https://doi.org/10.1021/jp5125266)
        
        ..math:
        
            I_unpol = 45/9 (xx + yy + zz)^2
                      + 7/2 [(xx-yy)^2 + (xx-zz)^2 + (yy-zz)^2]
                      + 7 * 3 [(xy)^2 + (yz)^2 + (xz)^2]
        """
        # Check if the raman tensor is present
        assert not self.dyn.raman_tensor is None, "Error, no Raman tensor found. Cannot initialize the Raman responce"
        
        labels = [i for i in range(7)]
        if not(index in labels):
            raise ValueError('{} should be in {}'.format(index, labels))
        
        epols = {'x' : np.array([1,0,0]),\
                 'y' : np.array([0,1,0]),\
                 'z' : np.array([0,0,1])}
        
        # (xx + yy + zz)^2
        if index == 0:
            raman_v  = self.dyn.GetRamanVector(epols['x'], epols['x'])
            raman_v += self.dyn.GetRamanVector(epols['y'], epols['y'])
            raman_v += self.dyn.GetRamanVector(epols['z'], epols['z'])
        # (xx - yy)^2    
        elif index == 1:
            raman_v  = self.dyn.GetRamanVector(epols['x'], epols['x'])
            raman_v -= self.dyn.GetRamanVector(epols['y'], epols['y'])
        # (xx - zz)^2       
        elif index == 2:
            raman_v  = self.dyn.GetRamanVector(epols['x'], epols['x'])
            raman_v -= self.dyn.GetRamanVector(epols['z'], epols['z'])
        # (yy - zz)^2   
        elif index == 3:
            raman_v  = self.dyn.GetRamanVector(epols['y'], epols['y'])
            raman_v -= self.dyn.GetRamanVector(epols['z'], epols['z'])
        # (xy)^2
        elif index == 4:
            raman_v = self.dyn.GetRamanVector(epols['x'], epols['y'])
        # (xz)^2
        elif index == 5:
            raman_v = self.dyn.GetRamanVector(epols['x'], epols['z'])
        # (yz)^2
        elif index == 6:
            raman_v = self.dyn.GetRamanVector(epols['y'], epols['z'])
            
        if debug:
            np.save('raman_v_{}'.format(index), raman_v)
            
        # Get the raman vector in the supercelld
        n_supercell = np.prod(self.dyn.GetSupercell())
        new_raman_v = np.tile(raman_v.ravel(), n_supercell)

        # Convert in the polarization basis and store the intensity
        self.prepare_perturbation(new_raman_v, masses_exp=-1)
        
        if debug:
            print('[NEW] Pertubation modulus with eq Raman tensors = {}'.format(self.perturbation_modulus))
        print()
                             
        return
    
    
    def prepare_unpolarized_raman_FT(self, index = 0, debug = False, eq_raman_tns = None, use_symm = True,\
                                     ens_av_raman = None, raman_tns_ens = None, add_2ph = True):
        """
        PREPARE UNPOLARIZED RAMAN SIGNAL CONSIDERING FLUCTUATIONS OF THE RAMAN TENSOR
        =============================================================================
        
        The raman tensor is read from the dynamical matrix provided by the original ensemble.
        
        The perturbations are prepared accordin to the formula (see https://doi.org/10.1021/jp5125266)
        
        ..math:
        
            I_unpol = 45/9 (xx + yy + zz)^2
                      + 7/2 [(xx-yy)^2 + (xx-zz)^2 + (yy-zz)^2]
                      + 7 * 3 [(xy)^2 + (yz)^2 + (xz)^2]
                      
        Parameters:
        -----------
            -index: the pol component of the unpolarized signal
            -debug: if true we save the second order Raman tensor
            -eq_raman_tns: np.array with shape (3, 3, 3 * N_at_uc), the equilibirum raman tensor
            -use_symm: bool, if True symmetries are enforced
            -ens_av_raman:  the ensemble on which we compute the averages of the Raman tensors
            -raman_tns_ens: np.array with shape (N_conf, 3, 3, 3 * N_at_sc), the raman tensors on the displaced configruations
        """
        # Check if the raman tensor is present
        assert not self.dyn.raman_tensor is None, "Error, no Raman tensor found. Cannot initialize the Raman responce"
        
        labels = [i for i in range(7)]
        if not(index in labels):
            raise ValueError('{} should be in {}'.format(index, labels))
        
        epols = {'x' : np.array([1,0,0]),\
                 'y' : np.array([0,1,0]),\
                 'z' : np.array([0,0,1])}
        
        # (xx + yy + zz)^2
        if index == 0:
            # raman_v  = self.dyn.GetRamanVector(epols['x'], epols['x'])
            # raman_v += self.dyn.GetRamanVector(epols['y'], epols['y'])
            # raman_v += self.dyn.GetRamanVector(epols['z'], epols['z'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   = epols['x'], pol_out   = epols['x'],\
                                             mixed = True,\
                                             pol_in_2 = epols['y'], pol_out_2 = epols['y'],\
                                             pol_in_3 = epols['z'], pol_out_3 = epols['z'],\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'xx_plus_yy_plus_zz')
        # (xx - yy)^2    
        elif index == 1:
            # raman_v  = self.dyn.GetRamanVector(epols['x'], epols['x'])
            # raman_v -= self.dyn.GetRamanVector(epols['y'], epols['y'])
            # NB we put just one minus sign because the component is (xx - yy)^2
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['x'],  pol_out  =  epols['x'],\
                                             mixed = True,\
                                             pol_in_2 = -epols['y'], pol_out_2 =  epols['y'],\
                                             pol_in_3 = np.zeros(3), pol_out_3 = np.zeros(3),\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'xx_minus_yy')
        # (xx - zz)^2       
        elif index == 2:
            # raman_v  = self.dyn.GetRamanVector(epols['x'], epols['x'])
            # raman_v -= self.dyn.GetRamanVector(epols['z'], epols['z'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['x'],  pol_out  =  epols['x'],\
                                             mixed = True,\
                                             pol_in_2 = -epols['z'], pol_out_2 =  epols['z'],\
                                             pol_in_3 = np.zeros(3), pol_out_3 = np.zeros(3),\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'xx_minus_zz')
        # (yy - zz)^2   
        elif index == 3:
            # raman_v  = self.dyn.GetRamanVector(epols['y'], epols['y'])
            # raman_v -= self.dyn.GetRamanVector(epols['z'], epols['z'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['y'],  pol_out  =  epols['y'],\
                                             mixed = True,\
                                             pol_in_2 = -epols['z'], pol_out_2 =  epols['z'],\
                                             pol_in_3 = np.zeros(3), pol_out_3 = np.zeros(3),\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'yy_minus_zz')
        # (xy)^2
        elif index == 4:
            # raman_v = self.dyn.GetRamanVector(epols['x'], epols['y'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['x'],  pol_out  =  epols['y'],\
                                             mixed = False,\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'xy_square')
        # (xz)^2
        elif index == 5:
            # raman_v = self.dyn.GetRamanVector(epols['x'], epols['z'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['x'],  pol_out  =  epols['z'],\
                                             mixed = False,\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'xz_square')
        # (yz)^2
        elif index == 6:
            # raman_v = self.dyn.GetRamanVector(epols['y'], epols['z'])
            self.prepare_anharmonic_raman_FT(raman = raman_tns_ens, raman_eq = eq_raman_tns,\
                                             pol_in   =  epols['y'],  pol_out  =  epols['z'],\
                                             mixed = False,\
                                             add_two_ph = add_2ph, symmetrize = use_symm,\
                                             ensemble = ens_av_raman,\
                                             save_raman_tensor2 = debug, file_raman_tensor2 = 'yz_square')
            
        return
       
   

    def prepare_anharmonic_raman_FT(self, raman = None, raman_eq = None,\
                                    pol_in = np.array([1.,0.,0.]), pol_out = np.array([1.,0.,0.]),\
                                    mixed = False, pol_in_2 = None, pol_out_2 = None,\
                                    pol_in_3 = None, pol_out_3 = None,\
                                    add_two_ph = False, symmetrize = False, ensemble = None,\
                                    save_raman_tensor2 = False, file_raman_tensor2 = None):
        """
        PREPARE THE PSI VECTOR FOR ANHARMONIC RAMAN SPECTRUM CALCULATION (NEW VERSION)
        ===========================================================================
        
        This works only with the Wigner representation if we add the two phonons effect. 
        Prepare the psi vector for RAMAN spectrum considering position-dependent raman tensors.
        
        Parameters:
        -----------
            -raman: nd.array (N_configs, E_comp, E_comp, 3 * N_at_sc),
                 the Raman tensor for all configurations.
                 Indices are: Number of configuration, electric field component,
                 electric field component, atomic coordinates in sc.
            rama_eq: nd.array, (E_comp, E_comp, 3 * N_at_uc), the effective charges at equilibrium.
                 Indices are: electric field component,
                 electric field component, atomic coordinate in uc.   
            -pol_in: nd.array, the polarization of in-out light. default is x
            -pol_out: nd.array, the polarization of in-out light. default is x
            -mixed: if True we can study the one and two phonon response to 
                    pol_in \cdto \Xi \cdot pol_in + pol_in_2 \cdto \Xi \cdot pol_in_2 + pol_in_3 \cdto \Xi \cdot pol_in_3
                    (\Xi is the Raman tensor)
            -pol_in_2:  nd.array, the polarization of in-out light. default is None
            -pol_out_2: nd.array, the polarization of in-out light. default is None
            -pol_in_3:  nd.array, the polarization of in-out light. default is None
            -pol_out_3: nd.array, the polarization of in-out light. default is None
            -add_two_ph: bool, if True two phonon processes are included in the calculation
            -symmetrize: bool, if True the first/second order Raman tensors are symmetrized
            -ensemble: a scha ensemble object for computing the averages
            -save_raman_tensor2: bool if True we save the second order Raman tensor
        """
        if not self.use_wigner and add_two_ph:
            raise NotImplementedError('The two phonon processes are implemented only in Wigner')
            
        if raman is None:
            raise ValueError('Must specify the raman tensors for all configurations!')
            
        if mixed:
            #Check that we have the other polarization vectors
            if (pol_in_2 is None) or (pol_out_2 is None):
                raise ValueError('Must specify pol_in_2 pol_out_2 if mixed = True!')
                
            if (pol_in_3 is None) or (pol_out_3 is None):
                raise ValueError('Must specify pol_in_3 pol_out_3 if mixed = True!')
                
            if len(pol_in_2) != 3 or len(pol_out_2) != 3:
                raise ValueError('pol_in_2 pol_out_2 must be array of len 3')
                
            if len(pol_in_3) != 3 or len(pol_out_3) != 3:
                raise ValueError('pol_in_3 pol_out_3 must be array of len 3')
                
        
        print()
        print('PREPARE THE RAMAN ANHARMONIC SPECTRUM CALCULATION')
        print('=================================================')
        print('Are we considering two ph effects? = {}'.format(add_two_ph))
        print('Are we using Wigner? = {}'.format(self.use_wigner))
        print('Are we symmetrizing the raman tensor? = {}'.format(symmetrize))
        print()
        if ensemble is not None:
            Nconf = ensemble.N
        else:
            Nconf = self.N
        
        required = 'N_conf - E_field - E_field - 3 * N_at_sc'
        assert raman.shape[0] == Nconf, 'The raman tensor in input have the wrong shape. The required is {}'.format(required)
        assert raman.shape[1] == 3, 'The raman tensor in input have the wrong shape. The required is {}'.format(required)
        assert raman.shape[2] == 3, 'The raman tensor in input have the wrong shape. The required is {}'.format(required)
        assert raman.shape[3] == self.nat * 3, 'The raman tensor in input have the wrong shape. The required is {}'.format(required)
        
        # alpha is the polarizability
        
        # Get the average of the raman tensor, np.array with shape = (3, 3, 3 * N_at_sc)
        d1alpha_dR_av = perturbations.get_d1alpha_dR_av(ensemble, raman, symmetrize = symmetrize)
        
        # Get the supercell dyn then set the raman tensor euqal to d1alpha_dR_av
        sc_dyn = self.dyn.GenerateSupercellDyn(self.dyn.GetSupercell())
        sc_dyn.raman_tensor = d1alpha_dR_av
        
        # Get the Raman vector np.array (3 * N_at_sc)
        raman_vector_sc = sc_dyn.GetRamanVector(pol_in, pol_out)
        
        if mixed:
            print('ONE PH SECTOR adding compoent pol_in_2 pol_out_2 of the Raman tensor')
            raman_vector_sc += sc_dyn.GetRamanVector(pol_in_2, pol_out_2)
            print('ONE PH SECTOR adding compoent pol_in_3 pol_out_3 of the Raman tensor')
            raman_vector_sc += sc_dyn.GetRamanVector(pol_in_3, pol_out_3)
            
        
        # Now rescale by the mass and go in polarizaiton basis
        self.prepare_perturbation(raman_vector_sc, masses_exp = -1)
        print('[NEW] Pertubation modulus with one ph effects only = {}'.format(self.perturbation_modulus))
        print()
        
        # NOW PREPARE THE SECOND RAMAN TENSOR
        if add_two_ph:
            if raman_eq is not None:
                print('[NEW] Getting the equilibirum RAMAN tensor...')
                print()
                n_supercell = np.prod(self.dyn.GetSupercell())
                # raman_eq is np.array with shape = (E_field, E_field, N_at_uc * 3)
                raman_eq_size = np.shape(raman_eq)
                MSG = """
                Error, raman tns of the wrong shape: {}
                """.format(raman_eq_size)
                assert len(raman_eq_size) == 3, MSG
                if not self.ignore_small_w:
                    assert raman_eq_size[2] * n_supercell == self.nat * 3 #self.n_modes + 3
                assert raman_eq_size[0] == raman_eq_size[1] == 3

                # Get the raman tensor in the supercell (E_field, E_filed, 3 * N_at_sc)
                raman_eq_gamma = np.zeros((3, 3, 3 * n_supercell * self.dyn.structure.N_atoms), dtype = type(raman_eq[0,0,0]))
                raman_eq_gamma = np.tile(raman_eq, n_supercell)
                
            print('[NEW] Getting the two phonon contribution in RAMAN...')

            # d2M_dR np.array with shape = (3 * N_atoms, 3 * N_atoms, Efield)
            if raman_eq is not None:
                print('[NEW] Subtracting the equilibirum RAMAN tensor...')
                # raman - raman_eq_gamma, np.array with shape = (N_configs, Efield, Efield, 3 * N_at_sc)
                # THE RESULT HAS shape = (Efield, Efield, 3 * N_at_sc, 3 * N_at_sc)
                d2alpha_dR = perturbations.get_d2alpha_dR_av(ensemble, raman - raman_eq_gamma, None, symmetrize = symmetrize)
            else:
                # THE RESULT HAS shape = (Efield, Efield, 3 * N_at_sc, 3 * N_at_sc)
                d2alpha_dR = perturbations.get_d2alpha_dR_av(ensemble, raman, None, symmetrize = symmetrize)
            
            print('[NEW] Divide by the masses')
            # Divide by the masses of the atoms in the supercell shape =  (Efield, Efield, 3 * N_at_sc, 3 * N_at_sc)
            d2alpha_dR = np.einsum('c, abcd, d -> abcd', np.sqrt(self.m)**-1, d2alpha_dR, np.sqrt(self.m)**-1)
            
            if save_raman_tensor2:
                print('[NEW] Saving the second-order SCHA Raman tensor')
                np.save('{}'.format(file_raman_tensor2), d2alpha_dR)
                return
            
            print('[NEW] Go in polarization basis')
            # Now go in polarization basis, np.array with shape = (E_field, E_field, n_modes, n_modes)
            # d2alpha_dR_muspace = np.einsum('cm, abcd, dn -> abmn', self.pols, d2alpha_dR, self.pols)
            # -> substitute
            tmp                = np.einsum('abcd, cm -> abmd', d2alpha_dR, self.pols)
            d2alpha_dR_muspace = np.einsum('abmd, dn -> abmn', tmp, self.pols)

            # Project along the direction of the filed, np.array with shape = (n_modes, n_modes)
            dXi_dR_muspace = np.einsum('abmn, a, b -> mn', d2alpha_dR_muspace, pol_in, pol_out)
            
            if mixed:
                print('TWO PH SECTOR adding component pol_in_2 pol_out_2 of the Raman tensor')
                dXi_dR_muspace += np.einsum('abmn, a, b -> mn', d2alpha_dR_muspace, pol_in_2, pol_out_2)
                print('TWO PH SECTOR adding component pol_in_3 pol_out_3 of the Raman tensor')
                dXi_dR_muspace += np.einsum('abmn, a, b -> mn', d2alpha_dR_muspace, pol_in_3, pol_out_3)
            
            # Symmetrize in mu space, np.array with shape = (n_modes, n_modes)
            dXi_dR_muspace = 0.5 * (dXi_dR_muspace + dXi_dR_muspace.T)

            # Get chi_minus and chi_plus tensors, np.array with shape = (n_modes, n_modes)
            chi_minus = self.get_chi_minus()
            chi_plus  = self.get_chi_plus()

            # Get the pertubations on a'^(1) b'^(1)
            pert_a = -np.einsum('nm, nm -> nm', np.sqrt(-0.5 * chi_minus), dXi_dR_muspace)
            pert_b = +np.einsum('nm, nm -> nm', np.sqrt(+0.5 * chi_plus) , dXi_dR_muspace)

            # Check if everything is symmetric
            assert np.all(np.abs(dXi_dR_muspace - dXi_dR_muspace.T) < 1e-10), "Second derivative of the polarizability is not symmetric in pol basis"
            assert np.all(np.abs(pert_a - pert_a.T) < 1e-10), "a'(1) pertubation is not symmetric in pol basis"
            assert np.all(np.abs(pert_b - pert_b.T) < 1e-10), "b'(1) pertubation is not symmetric in pol basis"

            # Now get the perturbation for a'^(1)
            current = self.n_modes
            for i in range(self.n_modes):
                self.psi[current : current + self.n_modes - i] = pert_a[i, i:]
                current = current + self.n_modes - i

            # Now get the pertrubation for b'^(1)
            for i in range(self.n_modes):
                self.psi[current : current + self.n_modes - i] = pert_b[i, i:]
                current = current + self.n_modes - i

            # Add the mask dot taking into account symmetric elements
            mask_dot = self.mask_dot_wigner()
            # OVERWRITE the pertubation modulus considering the two phonon sector
            self.perturbation_modulus = self.psi.dot(self.psi * mask_dot)

            print('[NEW] Perturbation modulus after adding two ph contributions RAMAN = {}'.format(self.perturbation_modulus))
            print()
    
        return
    
    
    
    def prepare_anharmonic_raman_FT_2ph(self, d2alpha_dR = None, pol_in = np.array([1.,0.,0.]), pol_out = np.array([1.,0.,0.]),\
                                    mixed = False, pol_in_2 = None, pol_out_2 = None):
        """
        PREPARE THE PSI VECTOR FOR RAMAN SPECTRUM CALCULATION (NEW VERSION) DIRECTLY FROM 2nd ORDER RAMAN TENSOR
        ========================================================================================================
        
        This function is useful if we want to interpolate the 2nd Raman tensor on a bigger supercell.
        
        This works only with the Wigner representation if we add the two phonons effect. 
        Prepare the psi vector for RAMAN spectrum considering position-dependent raman tensors.
        
        NOTE: we completely neglect the frist order Raman scattering!
        
        Parameters:
        -----------
            -d2alpha_dR: nd.array (E_comp, E_comp, 3 * N_at_sc, 3 * N_at_sc),
                 2nd order Raman tensor.
                 Indices are: Number of configuration, electric field component,
                 electric field component, atomic coordinates in sc.   
            -pol_in: nd.array, the polarization of in-out light. default is x
            -pol_out: nd.array, the polarization of in-out light. default is x
            -mixed: if True we can study the one and two phonon response to 
                    pol_in \cdot \Xi \cdot pol_out + pol_in_2 \cdot \Xi \cdot pol_out_2
                    (\Xi is the Raman tensor)
            -pol_in_2: nd.array, the polarization of in-out light. default is x
            -pol_out_2: nd.array, the polarization of in-out light. default is x
        """
        if not self.use_wigner:
            raise NotImplementedError('The two phonon processes are implemented only in Wigner')
            
        if d2alpha_dR is None:
            raise ValueError('Must specify the 2nd order Raman tensor!')
            
        exp_shape = (3, 3, self.nat * 3, self.nat * 3)
        if d2alpha_dR.shape != exp_shape:
            raise ValueError('The shape of the 2nd order Raman tensor is not correct, expected {}'.format(exp_shape))
            
        if mixed:
            if (pol_in_2 is None) or (pol_out_2 is None):
                raise ValueError('Must specify pol_in_2 pol_out_2 if mixed = True!')
                
            if len(pol_in_2) != 3 or len(pol_out_2) != 3:
                raise ValueError('pol_in_2 pol_out_2 must be array of len 3')
                
        
        print()
        print('PREPARE THE RAMAN ANHARMONIC SPECTRUM CALCULATION FROM 2nd ORDER RAMAN TENSOR')
        print('=============================================================================')
        # print('Are we considering two ph effects? = {}'.format(add_two_ph))
        print('Are we using Wigner? = {}'.format(self.use_wigner))
        # print('Are we symmetrizing the raman tensor? = {}'.format(symmetrize))
        print()
            
        print('TWO PH Going in polarization basis')
        # Now go in polarization basis, np.array with shape = (E_field, E_field, n_modes, n_modes)
        # d2alpha_dR_muspace = np.einsum('cm, abcd, dn -> abmn', self.pols, d2alpha_dR, self.pols)
        # -> substitute
        tmp                = np.einsum('abcd, cm -> abmd', d2alpha_dR, self.pols)
        d2alpha_dR_muspace = np.einsum('abmd, dn -> abmn', tmp, self.pols)
        # print(d2alpha_dR_muspace.shape)
        
        print('TWO PH Selecting the polarizations')
        # Project along the direction of the filed, np.array with shape = (n_modes, n_modes)
        dXi_dR_muspace = np.einsum('abmn, a, b -> mn', d2alpha_dR_muspace, pol_in, pol_out)
        # print(dXi_dR_muspace.shape)
        
        if mixed:
            print('TWO PH SECTOR adding component pol_in_2 pol_out_2 of the Raman tensor')
            dXi_dR_muspace += np.einsum('abmn, a, b -> mn', d2alpha_dR_muspace, pol_in_2, pol_out_2)

        # Symmetrize in mu space, np.array with shape = (n_modes, n_modes)
        dXi_dR_muspace = 0.5 * (dXi_dR_muspace + dXi_dR_muspace.T)

        # Get chi_minus and chi_plus tensors, np.array with shape = (n_modes, n_modes)
        chi_minus = self.get_chi_minus()
        chi_plus  = self.get_chi_plus()

        # Get the pertubations on a'^(1) b'^(1)
        pert_a = -np.einsum('nm, nm -> nm', np.sqrt(-0.5 * chi_minus), dXi_dR_muspace)
        pert_b = +np.einsum('nm, nm -> nm', np.sqrt(+0.5 * chi_plus) , dXi_dR_muspace)

        # Check if everything is symmetric
        assert np.all(np.abs(dXi_dR_muspace - dXi_dR_muspace.T) < 1e-10), "Second derivative of the polarizability is not symmetric in pol basis"
        assert np.all(np.abs(pert_a - pert_a.T) < 1e-10), "a'(1) pertubation is not symmetric in pol basis"
        assert np.all(np.abs(pert_b - pert_b.T) < 1e-10), "b'(1) pertubation is not symmetric in pol basis"
        
        print('[NEW] Perturbation modulus = {}'.format(self.perturbation_modulus))
        print()

        # Now get the perturbation for a'^(1)
        current = self.n_modes
        for i in range(self.n_modes):
            self.psi[current : current + self.n_modes - i] = pert_a[i, i:]
            current = current + self.n_modes - i

        # Now get the pertrubation for b'^(1)
        for i in range(self.n_modes):
            self.psi[current : current + self.n_modes - i] = pert_b[i, i:]
            current = current + self.n_modes - i

        # Add the mask dot taking into account symmetric elements
        mask_dot = self.mask_dot_wigner()
        # OVERWRITE the pertubation modulus considering the two phonon sector
        self.perturbation_modulus = self.psi.dot(self.psi * mask_dot)

        print('[NEW] Perturbation modulus adding two ph contributions RAMAN = {}'.format(self.perturbation_modulus))
        print()
    
        return
    
    
    
    def prepare_ir(self, effective_charges = None, pol_vec = np.array([1,0,0])):
        """
        PREPARE LANCZOS FOR INFRARED SPECTRUM COMPUTATION
        =================================================

        In this subroutine we prepare the lanczos algorithm for the computation of the
        infrared spectrum signal.

        Parameters
        ----------
            effective_charges : ndarray(size = (n_atoms, 3, 3), dtype = np.double)
                The effective charges. Indices are: Number of atoms in the unit cell,
                electric field component, atomic coordinate. If None, the effective charges
                contained in the dynamical matrix will be considered.
            pol_vec : ndarray(size = 3)
                The polarization vector of the light.
        """

        ec = self.dyn.effective_charges
        if not effective_charges is None:
            ec = effective_charges
        
        n_supercell = np.prod(self.dyn.GetSupercell())

        # Check the effective charges
        assert not ec is None, "Error, no effective charge found. Cannot initialize IR responce"

        ec_size = np.shape(ec)
        MSG = """
        Error, effective charges of the wrong shape: {}
        """.format(ec_size)
        assert len(ec_size) == 3, MSG
        if not self.ignore_small_w:
            assert ec_size[0] * ec_size[2] * n_supercell == self.n_modes + 3
        assert ec_size[1] == ec_size[2] == 3

        # shape = (N_at_uc, 3)
        z_eff = np.einsum("abc, b", ec, pol_vec)

        # Get the gamma effective charge
        new_zeff = np.tile(z_eff.ravel(), n_supercell)

        self.prepare_perturbation(new_zeff, masses_exp = -1)
    
    
    
    def prepare_anharmonic_ir_FT(self, ec = None, ec_eq = None, pol_vec_light = np.array([1.,0.,0.]), add_two_ph = False, symmetrize = False, ensemble = None):
        """
        PREPARE THE PSI VECTOR FOR ANHARMONIC IR SPECTRUM CALCULATION (NEW VERSION)
        ===========================================================================
        
        This works only with the Wigner representation if we add the two phonons effect. 
        Prepare the psi vector for IR spectrum considering position-dependent effective charges.
        
        The one phonon scetor is symmetrized by default
        
        Parameters:
        -----------
            -effective_charges: nd.array (N_configs, N_atoms_sc, E_comp, cart_comp),
                 the effective charges for all configurations.
                 Indices are: Number of configuration, number of atoms in the super cell,
                 electric field component, atomic coordinate.
            -effective_charges_eq: nd.array, (N_atoms_uc, E_comp, cart_comp), the effective charges at equilibrium.
                 Indices are: number of atoms in the unit cell,
                 electric field component, atomic coordinate.   
            -pol_vec_light: nd.array, the polarization of in-out light. default is x
            -add_two_ph: bool, if True two phonon processes are included in the calculation
            -symmetrize: bool, if True the first/second order effective charges are symmetrized
            -ensemble: a scha ensemble object for computing the averages
        """
        if not self.use_wigner and add_two_ph:
            raise NotImplementedError('The two phonon processes are implemented only in Wigner')
            
        if ec is None:
            raise ValueError('Must specify the effective charges for all configurations!')
            
         
        print()
        print('PREPARE THE IR ANHARMONIC SPECTRUM CALCULATION')
        print('==============================================')
        print('Are we considering two ph effects? = {}'.format(add_two_ph))
        print('Are we using Wigner? = {}'.format(self.use_wigner))
        print('Are we symmetrizing the effective charges? = {}'.format(symmetrize))
        print()
        
        required = 'N_conf N_at_sc E_field cart'
        assert ec.shape[0] == ensemble.N, 'The effective charges in input have the wrong shape. The required is {}'.format(required)
        assert ec.shape[1] == self.nat, 'The effective charges in input have the wrong shape. The required is {}'.format(required)
        assert ec.shape[2] == ec.shape[3] == 3, 'The effective charges in input have the wrong shape. The required is {}'.format(required)
            
        # Get the average of the dipole moment, np.array with shape = (3 * N_at_sc, 3)
        d1M_dR_av = perturbations.get_d1M_dR_av(ensemble, ec, symmetrize = symmetrize)
        
        # Project along the direction of light polarization, (3 * N_at_sc)
        Z = np.einsum("ab, b -> a", d1M_dR_av, pol_vec_light)
        
        # Now rescale by the mass and go in polarizaiton basis
        self.prepare_perturbation(Z.ravel(), masses_exp = -1)
        print('Pertubation modulus with one ph effects only = {}'.format(self.perturbation_modulus))
        print()
        
        # NOW PREPARE THE SECOND ORDER DIPOLE MOMENT
        if add_two_ph:
            if ec_eq is not None:
                print('[NEW] Getting the equilibirum effective charges...')
                print()
                n_supercell = np.prod(self.dyn.GetSupercell())
                # ec_eq is np.array with shape = (N_at_uc, E_field, cart)
                ec_eq_size = np.shape(ec_eq)
                MSG = """
                Error, effective charges of the wrong shape: {}
                """.format(ec_eq_size)
                assert len(ec_eq_size) == 3, MSG
                if not self.ignore_small_w:
                    assert ec_eq_size[0] * ec_eq_size[2] * n_supercell == self.n_modes + 3
                assert ec_eq_size[1] == ec_eq_size[2] == 3

                # Get the eq effective charges in the supercell (N_at_sc, E_field, 3)
                ec_eq_gamma = np.zeros((n_supercell * self.dyn.structure.N_atoms, 3, 3), dtype = type(ec_eq[0]))
                ec_eq_gamma = np.tile(ec_eq, (n_supercell,1,1))
                
            print('[NEW] Getting the two phonon contribution...')

            # d2M_dR np.array with shape = (3 * N_atoms, 3 * N_atoms, Efield)
            if ec_eq is not None:
                print('[NEW] Subtracting the equilibirum effective charges...')
                # ec - ec_eq_gamma, np.array with shape = (N_configs, N_at_sc, Efield, cart)
                d2M_dR = perturbations.get_d2M_dR_av(ensemble, ec - ec_eq_gamma, None, symmetrize = symmetrize)
            else:
                d2M_dR = perturbations.get_d2M_dR_av(ensemble, ec, None, symmetrize = symmetrize)

            # Divide by the masses of the atoms in the supercell
            d2M_dR = np.einsum('a, abc, b -> abc', np.sqrt(self.m)**-1, d2M_dR, np.sqrt(self.m)**-1)
            
            # Now go in polarization basis, np.array with shape = (n_modes, n_modes, E_filed)
            d2M_dR_muspace = np.einsum('am, abc, bn -> mnc', self.pols, d2M_dR, self.pols)

            # Project along the direction of the filed, np.array with shape = (n_modes, n_modes)
            dZ_dR_muspace = np.einsum('mnc, c -> mn', d2M_dR_muspace, pol_vec_light)
            
            # Symmetrize in mu space, np.array with shape = (n_modes, n_modes)
            dZ_dR_muspace = 0.5 * (dZ_dR_muspace + dZ_dR_muspace.T)

            # Get chi_minus and chi_plus tensors, np.array with shape = (n_modes, n_modes)
            chi_minus = self.get_chi_minus()
            chi_plus  = self.get_chi_plus()

            # Get the pertubations on a'^(1) b'^(1)
            pert_a = -np.einsum('nm, nm -> nm', np.sqrt(-0.5 * chi_minus), dZ_dR_muspace)
            pert_b = +np.einsum('nm, nm -> nm', np.sqrt(+0.5 * chi_plus) , dZ_dR_muspace)

            # Check if everything is symmetric
            assert np.all(np.abs(dZ_dR_muspace - dZ_dR_muspace.T) < 1e-10), "Second derivative of the dipole is not symmetric in pol basis"
            assert np.all(np.abs(pert_a - pert_a.T) < 1e-10), "a'(1) pertubation is not symmetric in pol basis"
            assert np.all(np.abs(pert_b - pert_b.T) < 1e-10), "b'(1) pertubation is not symmetric in pol basis"

            # Now get the perturbation for a'^(1)
            current = self.n_modes
            for i in range(self.n_modes):
                self.psi[current : current + self.n_modes - i] = pert_a[i, i:]
                current = current + self.n_modes - i

            # Now get the pertrubation for b'^(1)
            for i in range(self.n_modes):
                self.psi[current : current + self.n_modes - i] = pert_b[i, i:]
                current = current + self.n_modes - i

            # Add the mask dot taking into account symmetric elements
            mask_dot = self.mask_dot_wigner()
            # OVERWRITE the pertubation modulus considering the two phonon sector
            self.perturbation_modulus = self.psi.dot(self.psi * mask_dot)

            print('[NEW] Perturbation modulus after adding two ph contributions = {}'.format(self.perturbation_modulus))
            print()
    
        return
    
    
    
        
    def prepare_anharmonic_ir(self, ec = None, ec_eq = None, pol_vec_light = np.array([1.,0.,0.]), add_two_ph = False):
        """
        PREPARE THE PSI VECTOR FOR ANHARMONIC IR SPECTRUM CALCULATION
        =============================================================
        
        This works only with the Wigner representation if we add the two phonons effect. 
        Prepare the psi vector for IR spectrum considering position-dependent effective charges.
        
        Parameters:
        -----------
            -effective_charges: nd.array (N_configs, N_atoms_sc, E_comp, cart_comp),
                 the effective charges for all configurations.
                 Indices are: Number of configuration, number of atoms in the super cell,
                 electric field component, atomic coordinate.
            -effective_charges_eq: nd.array, the effective charges at equilibrium.
                 Indices are: number of atoms in the unit cell,
                 electric field component, atomic coordinate.   
            -pol_vec_light: nd.array, the polarization of in-out light. default is x
            -add_two_ph: bool, if True two phonon processes are included in the calculation
            -symm_eff_charges: bool, if True the effective charges are symmetrized
            -ensemble: a scha ensemble object to compute the effective charges
        """
        if not self.use_wigner and add_two_ph:
            raise NotImplementedError('The two phonon processes are implemented only in Wigner')
            
        if ec is None:
            raise ValueError('Must specify the effective charges for all configurations!')
            
        print()
        print('PREPARE THE IR ANHARMONIC SPECTRUM CALCULATION')
        print('==============================================')
        print('Are we considering two ph effects? = {}'.format(add_two_ph))
        print('Are we using Wigner? = {}'.format(self.use_wigner))
        print()
    
        # The effective charges for each configuration (N_configs, N_at_sc, E_field_comp, 3)
        eff = np.zeros((self.N, self.nat, 3, 3))

        assert ec.shape == eff.shape, 'The effective charges in input have the wrong shape. The required is {}'.format(eff.shape)
        
        # Get the effective charges
        eff = ec.copy()
        
        # FIRST DERIVATIVE OF THE DIPOLE
        # Project along the direction of light polarization, (N_configs, N_at_sc, 3)
        z_eff = np.einsum("iabc, b -> iac", eff, pol_vec_light)

        # FIRST DERIVATIVE OF THE DIPOLE
        # Average of effective charges on the ensemble (N_at_sc, 3)
        d1_M = np.einsum('i, iab -> ab', self.rho, z_eff) /np.sum(self.rho)

        # Now rescale by the mass and go in polarizaiton basis
        self.prepare_perturbation(d1_M.ravel(), masses_exp = -1)
        print('Pertubation modulus with one ph effects only = {}'.format(self.perturbation_modulus))
        print()
        
        # NOW PREPARE THE SECOND ORDER DIPOLE MOMENT
        if add_two_ph:
            if ec_eq is not None:
                print('Subtracting the equilibirum effective charges...')
                print()
                n_supercell = np.prod(self.dyn.GetSupercell())
                ec_eq_size = np.shape(ec_eq)
                MSG = """
                Error, effective charges of the wrong shape: {}
                """.format(ec_eq_size)
                assert len(ec_eq_size) == 3, MSG
                if not self.ignore_small_w:
                    assert ec_eq_size[0] * ec_eq_size[2] * n_supercell == self.n_modes + 3
                assert ec_eq_size[1] == ec_eq_size[2] == 3

                # Eq effective charges, (N_at_uc, 3)
                z_eff_eq = np.einsum("abc, b -> ac", ec_eq, pol_vec_light)
                # Eq effective charges at gamma, (N_at_sc, 3)
                z_eff_eq_gamma = np.tile(z_eff_eq.ravel(), n_supercell).reshape((self.nat, 3))

                # This should reduce the noise when computing the 2ph vertex
                z_eff -= z_eff_eq_gamma
                
            print('[OLD] Getting the two phonon contribution...')
            
            # Polarization vectors over mass, shape = (N_at_sc, n_modes)
            pols_mass = np.einsum('a, am -> am', np.sqrt(self.m)**-1, self.pols)

            # The mass rescaled projected effective charges in polarization basis, shape = (N_configs, n_modes)
            z_pols_mass = np.einsum('am, ia -> im ', pols_mass, z_eff.ravel().reshape((self.N, self.nat * 3)))

            # Eigenvalues of Upsilon mass rescaled, shape = (n_modes)
            xi2_inv = f_ups(self.w, self.T)

            # The mass rescaled displacements in polarization basis divided by xi2, shape = (N_configs, n_modes)
            u_xi2 = np.einsum('im, m -> im', self.X, xi2_inv)

            # Add the effective charges, shape = (N_configs, n_modes, n_modes)
            u_xi2_Z = np.einsum('in , im -> inm', u_xi2, z_pols_mass)

            # Get the reweighted average of the second derivative, shape = (n_modes, n_modes)
            d2_M = np.einsum('i, inm -> nm', self.rho, u_xi2_Z) /np.sum(self.rho)
            d2_M = 0.5 * (d2_M + d2_M.T)

            # Get chi_minus and chi_plus tensors
            chi_minus = self.get_chi_minus()
            chi_plus  = self.get_chi_plus()

            # Get the pertubations on a'^(1) b'^(1)
            pert_a = -np.einsum('nm, nm -> nm', np.sqrt(-0.5 * chi_minus), d2_M)
            pert_b = +np.einsum('nm, nm -> nm', np.sqrt(+0.5 * chi_plus) , d2_M)

            # Check if everything is symmetric
            assert np.all(np.abs(d2_M - d2_M.T) < 1e-10), "Second derivative of the dipole is not symmetric in pol basis"
            assert np.all(np.abs(pert_a - pert_a.T) < 1e-10), "a'(1) pertubation is not symmetric in pol basis"
            assert np.all(np.abs(pert_b - pert_b.T) < 1e-10), "b'(1) pertubation is not symmetric in pol basis"

            # Now get the perturbation for a'^(1)
            current = self.n_modes
            for i in range(self.n_modes):
                self.psi[current : current + self.n_modes - i] = pert_a[i, i:]
                current = current + self.n_modes - i

            # Now get the pertrubation for b'^(1)
            for i in range(self.n_modes):
                self.psi[current : current + self.n_modes - i] = pert_b[i, i:]
                current = current + self.n_modes - i

            # Add the mask dot taking into account symmetric elements
            mask_dot = self.mask_dot_wigner()
            # OVERWRITE the pertubation modulus considering the two phonon sector
            self.perturbation_modulus = self.psi.dot(self.psi * mask_dot)

            print('[OLD] Perturbation modulus after adding two ph contributions = {}'.format(self.perturbation_modulus))
            print()
            
        return
   
        
        
    def prepare_perturbation(self, vector, masses_exp = 1, add = False):
        r"""
        This function prepares the calculation for the Green function

        <v| G |v>

        Where |v> is the vector passed as input. If you want to compute the
        raman, for istance, it can be the vector of the Raman intensities.

        The vector can be obtained contracting it with the polarization vectors.
        The contraction can be on the numerator or on the denumerator, depending on the
        observable.
        
        NOTE: This function prepares the pertubation ONLY in the R sector
        Both IR and Raman has masses_exp = -1

        .. math ::

            v_\mu = \sum_a v_a e_\mu^a \cdot \sqrt{m_a} 

            v_\mu = \sum_a v_a \frac{e_\mu^a}{  \sqrt{m_a}} 

        Parameters
        ----------
            vector: ndarray( size = (3*natoms))
                The vector of the perturbation for the computation of the green function
            masses_exp : float
                The vector is multiplied by the square root of the masses raised to masses_exp.
                If you want to multiply each component by the square root of the masses use 1,
                if you want to divide by the quare root use -1, use 0 if you do not want to use the
                masses.
            add : bool
                If true, the perturbation is added on the top of the one already setup.
                Calling add does not cause a reset of the Lanczos
        """
        if not add:
            self.reset()
            self.psi = np.zeros(self.psi.shape, dtype = TYPE_DP)

        # Convert the vector in the polarization space
        m_on = np.sqrt(self.m) ** masses_exp
        print("SHAPE:", m_on.shape, vector.shape, self.pols.shape)
        new_v = np.einsum("a, a, ab->b", m_on, vector, self.pols)
        self.psi[:self.n_modes] += new_v

        # THIS IS OK IN THE WIGNER REPRESENTATION BECAUSE
        # THE PERTUBATION ENTERS ONLY IN THE R SECTOR
        self.perturbation_modulus = new_v.dot(new_v)

        if self.symmetrize:
            self.symmetrize_psi()

            
            
    def prepare_mode(self, index):
        """
        Prepare the perturbation on a single phonon mode.
        This is usefull to get the single mode contribution to the overall spectral function.

        Parameters
        ----------
            index : int
                The index of the mode in the supercell. Starting from 0 (lowest frequency, excluding acoustic modes at Gamma) 
        """
        self.reset()

        self.psi[:] = 0
        self.psi[index] = 1 
        self.perturbation_modulus = 1
        
        
    
    def prepare_two_ph(self, a, b):
        """
        Prepare the psi vector for a two phonon response.
        Available only in Winger.

        Parameters:
        ----------
            a, b: int indices of the modes (acustic are excluded).
        """
        if not self.use_wigner:
            raise NotImplementedError('The two phonon response is available only in Wigner')
            
        if a > self.n_modes or b > self.n_modes:
            raise ValueError('The a-b indices must be smaller than {}'.format(self.n_modes))
            
        print()
        print('PREPARE THE TWO PHONON PERTUBATION')
        print('Indices selected a = {} b = {}'.format(a,b))
        print()
            
        self.reset()
        self.psi[:] = 0
        
        # Get chi minus and chi plus, shape = (n_modes, n_modes)
        chi_plus  = self.get_chi_plus()
        chi_minus = self.get_chi_minus()
        
        # The matrix for the second derivatives
        mat_modes = np.zeros((self.n_modes, self.n_modes))
        
        mat_modes[a,b] = 0.5
        mat_modes[b,a] += 0.5
        
        # Get the pertbations on a' b'
        pert_a = -np.einsum('nm, nm -> nm', np.sqrt(-0.5 * chi_minus), mat_modes)
        pert_b = +np.einsum('nm, nm -> nm', np.sqrt(+0.5 * chi_plus) , mat_modes)
        
        # Now get the perturbation for a'^(1)
        current = self.n_modes
        for i in range(self.n_modes):
            self.psi[current : current + self.n_modes - i] = pert_a[i, i:]
            current = current + self.n_modes - i

        # Now get the pertrubation for b'^(1)
        for i in range(self.n_modes):
            self.psi[current : current + self.n_modes - i] = pert_b[i, i:]
            current = current + self.n_modes - i

        # Add the mask dot taking into account symmetric elements
        mask_dot = self.mask_dot_wigner()
        
        # Update the pertubation modulus
        self.perturbation_modulus = self.psi.dot(self.psi * mask_dot)
        
        return

        


    def get_vector_dyn_from_psi(self):
        """
        This function returns a standard vector and the dynamical matrix in cartesian coordinates
        
        This can be used to symmetrize the vector.

        The vector is returned in [Bohr] and the force constant matrix is returned in [Ry/bohr^2]
        """


        vector = self.psi[:self.n_modes]
        dyn = self.psi[self.n_modes:].reshape((self.n_modes, self.n_modes))

        w_a = np.tile(self.w, (self.n_modes, 1))
        w_b = np.tile(self.w, (self.n_modes, 1)).T 

        dyn *= ( 2*(w_a + w_b) * np.sqrt(w_a*w_b*(w_a + w_b)))

        # Get back the real vectors
        real_v = np.einsum("b, ab->a",  vector, self.pols)  /np.sqrt(self.m)
        real_dyn = np.einsum("ab, ca, db-> cd", dyn, self.pols, self.pols)
        real_dyn *= np.outer(np.sqrt(self.m), np.sqrt(self.m))

        return real_v, real_dyn 
    
    def set_psi_from_vector_dyn(self, vector, dyn):
        """
        Set the psi vector from a given vector of positions [bohr] and a force constant matrix [Ry/bohr^2].
        Used to reset the psi after the symmetrization.
        """

        new_v = np.einsum("a, ab->b",  np.sqrt(self.m) * vector, self.pols)
        
        new_dyn = dyn / np.outer(np.sqrt(self.m), np.sqrt(self.m))
        new_dyn = np.einsum("ab, ai, bj-> ij", new_dyn, self.pols, self.pols) 
        
        w_a = np.tile(self.w, (self.n_modes, 1))
        w_b = np.tile(self.w, (self.n_modes, 1)).T 

        new_dyn /= ( 2*(w_a + w_b) * np.sqrt(w_a*w_b*(w_a + w_b)))

        self.psi[:self.n_modes] = new_v
        self.psi[self.n_modes:] = new_dyn.ravel()

    def symmetrize_psi(self):
        """
        Symmetrize the psi vector.
        """
        
        # First of all, get the vector and the dyn
        vector, dyn = self.get_vector_dyn_from_psi()

        print ("Vector before symmetries:")
        print (vector)

        # Symmetrize the vector
        self.qe_sym.SetupQPoint()
        new_v = np.zeros( (self.nat, 3), dtype = np.float64, order = "F")
        new_v[:,:] = vector.reshape((self.nat, 3))
        self.qe_sym.SymmetrizeVector(new_v)
        vector = new_v.ravel()

        print ("Vector after symmetries:")
        print (vector)
               
        # Symmetrize the dynamical matrix
        dyn_q = CC.Phonons.GetDynQFromFCSupercell(dyn, np.array(self.dyn.q_tot), self.uci_structure, self.super_structure)
        self.qe_sym.SymmetrizeFCQ(dyn_q, self.dyn.q_stars, asr = "custom")
        dyn = CC.Phonons.GetSupercellFCFromDyn(dyn_q, np.array(self.dyn.q_tot), self.uci_structure, self.super_structure)

        # Push everithing back into the psi
        self.set_psi_from_vector_dyn(vector, dyn)

    def set_max_frequency(self, freq):
        """
        SETUP THE REVERSE LANCZOS
        =========================

        This function prepares the Lanczos algorithm in order to find the lowest eigenvalues
        You should provide the maximum frequencies of the standard spectrum.
        Then the Lanczos is initialized in order to solve the problem

        (Ia - L) x = b

        where a is sqrt(2*freq) so that should match the maximum energy. 
        Since Lanczos is very good in converging the biggest (in magnitude) eigenvectors, this
        procedure should accelerate the convergence of the low energy spectrum.

        NOTE: This method should be executed BEFORE the Lanczos run.

        Parameters
        ----------
            freq : float
               The frequencies (in Ry) of the maximum eigenvalue.
        """

        self.shift_value = 4*freq*freq
        self.reverse_L = True

        print("Shift value:", self.shift_value)

    def apply_L1(self):
        """
        APPLY THE L1
        ============

        This is the first part of the application, it involves only harmonic propagation.

        Results
        -------
            out_vect : ndarray(shape(self.psi))
                It returns the application of the harmonic part of the L matrix
        """

        out_vect = np.zeros(np.shape(self.psi), dtype = TYPE_DP)

        # Get the harmonic responce function
        out_vect[:self.n_modes] = (self.psi[:self.n_modes] * self.w) * self.w

        #print("freqsL1: ", self.w)
        #print("out 0:", out_vect[0])
        # Get the harmonic responce on the propagator
        w_a = np.tile(self.w, (self.n_modes, 1))
        w_b = np.tile(self.w, (self.n_modes, 1)).T 

        new_out = (w_a + w_b)**2
        out_vect[self.n_modes:] = new_out.ravel() * self.psi[self.n_modes:]

        #print("out 0 (just end):", out_vect[0])
        return out_vect

    
    
    def apply_L1_FT(self, transpose = False):
        r"""
        APPLY THE L1 AT FINITE TEMPERATURE
        ==================================

        This is the first part of the application, it involves only HARMONIC propagation.
        
        NORMAL: this method applies -L_harm: 
        :: math .
            \begin{bmatrix}
            -Z'' &   0  &  0 \\
            0    &  -X  & -Y \\
            0    &  -X' & -Y'
            \end{bmatrix}
        on the follwoing vector:
        :: math .
            \begin{bmatrix}
            \mathcal{R}^{(1)} \\
            \tilde{\Upsilon}^{(1)} \\
            \Re \tilde{A}^{(1)}.
            \end{bmatrix}
            
        WIGNER: this method applies +L_harm: 
        :: math .
            \begin{bmatrix}
            - \omega^2_\alpha & 0 & 0 \\
            0 &  -\omega^-_{\alpha\beta}^2  & 0 \\ 
            0 & 0 &  -\omega^+_{\alpha\beta}^2 
            \end{bmatrix}
        on the following vector:
        :: math .
            \begin{bmatrix}
            \tilde{\mathcal{R}}^{(1)}_\alpha\\ 
            \tilde{a}'^{(1)}_{\alpha\beta}\\ 
            \tilde{b}'^{(1)}_{\alpha\beta}.
            \end{bmatrix}
        
        If transpose = True it applies the transpose.
            
        Results
        -------
            -out_vect : ndarray(shape(self.psi))
                It returns the application of the harmonic part of the L matrix
        """
        # Prepare the free propagator on the positions
        out_vect = np.zeros(np.shape(self.psi), dtype = TYPE_DP)

        if self.ignore_harmonic:
            return out_vect 

        # The elements where w_a and w_b are exchanged are dependent
        # So we must avoid including them
        i_a = np.tile(np.arange(self.n_modes), (self.n_modes,1)).ravel()
        i_b = np.tile(np.arange(self.n_modes), (self.n_modes,1)).T.ravel()

        new_i_a = np.array([i_a[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        new_i_b = np.array([i_b[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        
        w_a = self.w[new_i_a]
        w_b = self.w[new_i_b]

        N_w2 = len(w_a)

        ##############################################
        # Get the harmonic responce function on R^(1)#
        ##############################################
        # Apply the diagonal free propagation 
        if not self.use_wigner:
#             print('[NORMAL]: harmonic R(1)')
            out_vect[:self.n_modes] = (self.psi[:self.n_modes] * self.w) * self.w
        else:
            # Use Wigner
#             print('[WIGNER]: harmonic R(1)')
            out_vect[:self.n_modes] = -(self.psi[:self.n_modes] * self.w) * self.w

        # Get the BE occupation number
        n_a = np.zeros(np.shape(w_a), dtype = TYPE_DP)
        n_b = np.zeros(np.shape(w_a), dtype = TYPE_DP)
        if self.T > 0:
            n_a = 1 / (np.exp( w_a / np.double(self.T / __RyToK__)) - 1)
            n_b = 1 / (np.exp( w_b / np.double(self.T / __RyToK__)) - 1)


        # Apply the non interacting X operator
        # Where R^(1) ends and Upsilon^(1)-a'^(1) starts
        start_Y = self.n_modes
        # Where Upsilon^(1)-a'^(1) ends and ReA^(1)-b'^(1) starts
        start_A = self.n_modes + N_w2
        
        #########################################################################
        # The R^(1) perturbation ends at start_Y
        # The Upsilon^(1)/a'^{(1)} perturbation start at start_Y
        # The ReA^(1)/b'^{(1)} perturbation starts at start_Y + 0.5*N_modes*(N_modes + 1)
        #################################################################################

        #print("start_Y: {} | start_A: {} | end_A: {} | len_psi: {}".format(start_Y, start_A, start_A + N_w2, len(self.psi)))

        ERR_MSG ="""
ERROR,
The initial vector for the Lanczos algorithm has a wrong dimension. 
This may be caused by the Lanczos initialized at the wrong temperature.
"""
        assert len(self.psi) == start_A + N_w2, ERR_MSG
        
        ############################################################
        # Get the harmonic responce function on Upsilon^(1)-a'^(1) #
        ############################################################
        
        if not self.use_wigner:
#             print('[NORMAL]: harmonic Y(1)')
            # Apply the diagonal free propagation on Y
            X_ab_NI = -w_a**2 - w_b**2 - (2*w_a *w_b) /((2*n_a + 1) * (2*n_b + 1))
            out_vect[start_Y: start_A] = - X_ab_NI * self.psi[start_Y: start_A]

            # Apply the off diagonal free propagation Y-ReA
            Y_ab_NI = - (8 * w_a * w_b) / ( (2*n_a + 1) * (2*n_b + 1))
            if not transpose:
                out_vect[start_Y : start_A] += - Y_ab_NI * self.psi[start_A: ]
            else:
                out_vect[start_A:] += - Y_ab_NI * self.psi[start_Y : start_A]

            #L_operator[start_Y : start_A, start_A:] = - np.diag(Y_ab_NI) * extra_count
            #L_operator[start_Y + np.arange(self.n_modes**2), start_A + exchange_frequencies] -=  Y_ab_NI / 2
        else:
#             print('[WIGNER]: harmonic a(1)')
            # Apply the diagonal free propagation in WIGNER on a'^{(1)}
            a_harm = -(w_a**2 + w_b**2 - 2. * w_a * w_b)
            out_vect[start_Y: start_A] = +a_harm * self.psi[start_Y: start_A]
            

        ########################################################
        # Get the harmonic responce function on ReA^(1)-b'^(1) #
        ########################################################
        
        if not self.use_wigner:
#             print('[NORMAL]: harmonic ReA(1)')
            # Apply the off diagonal free propagation ReA-Y
            X1_ab_NI = - (2*n_a*n_b + n_a + n_b) * (2*n_a*n_b + n_a + n_b + 1)*(2 * w_a * w_b) / ( (2*n_a + 1) * (2*n_b + 1))

            # Apply the off-diagonal free propagation
            if not transpose:
                out_vect[start_A:] += - X1_ab_NI * self.psi[start_Y: start_A]
            else:
                out_vect[start_Y: start_A] += - X1_ab_NI * self.psi[start_A:]
            #L_operator[start_A:, start_Y : start_A] = - np.diag(X1_ab_NI) / 1 * extra_count
            #L_operator[start_A + np.arange(self.n_modes**2), start_Y + exchange_frequencies] -= X1_ab_NI / 2

            # Apply the diagonal free propagation ReA
            Y1_ab_NI = - w_a**2 - w_b**2 + (2*w_a *w_b) /( (2*n_a + 1) * (2*n_b + 1))
            out_vect[start_A:] += - Y1_ab_NI * self.psi[start_A:]
            #L_operator[start_A:, start_A:] = -np.diag(Y1_ab_NI) / 1 * extra_count
            #L_operator[start_A + np.arange(self.n_modes**2),  start_A + exchange_frequencies] -= Y1_ab_NI / 2
        else:
#             print('[WIGNER]: harmonic b(1)')
            # Apply the diagonal free propagation in WIGNER on b'^{(1)}
            b_harm = -(w_a**2 + w_b**2 + 2. * w_a * w_b)
            out_vect[start_A:] = +b_harm * self.psi[start_A:]

        return out_vect


    
    
    
    def apply_L1_inverse_FT(self, psi, transpose = False):
        """
        APPLY THE INVERSE L1 AT FINITE TEMPERATURE
        ==========================================

        This method allows for preconditioning, as L1 is a diagonal application

        Results
        -------
            out_vect : ndarray(shape(self.psi))
                It returns the application of the harmonic part of the inverse L matrix
        """


        # Prepare the free propagator on the positions
        out_vect = np.zeros(np.shape(self.psi), dtype = TYPE_DP)

        if self.ignore_harmonic:
            return out_vect 

        # The elements where w_a and w_b are exchanged are dependent
        # So we must avoid including them
        i_a = np.tile(np.arange(self.n_modes), (self.n_modes,1)).ravel()
        i_b = np.tile(np.arange(self.n_modes), (self.n_modes,1)).T.ravel()

        new_i_a = np.array([i_a[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        new_i_b = np.array([i_b[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        
        w_a = self.w[new_i_a]
        w_b = self.w[new_i_b]

        N_w2 = len(w_a)

        # Get the harmonic responce function
        out_vect[:self.n_modes] = (psi[:self.n_modes] / self.w) / self.w


        n_a = np.zeros(np.shape(w_a), dtype = TYPE_DP)
        n_b = np.zeros(np.shape(w_a), dtype = TYPE_DP)

        not_populated_mask_a = 0.01 * w_a *__RyToK__ > self.T
        not_populated_mask_b = 0.01 * w_a *__RyToK__ > self.T

        if self.T > 0:
            n_a = 1 / (np.exp( w_a / np.double(self.T /__RyToK__)) - 1)
            n_b = 1 / (np.exp( w_b / np.double(self.T / __RyToK__)) - 1)
            n_a[not_populated_mask_a] = 0
            n_b[not_populated_mask_b] = 0


        # Apply the non interacting X operator
        start_Y = self.n_modes
        start_A = self.n_modes + N_w2


        ERR_MSG ="""
ERROR,
The initial vector for the Lanczos algorithm has a wrong dimension. 
This may be caused by the Lanczos initialized at the wrong temperature.
"""
        assert len(psi) == start_A + N_w2, ERR_MSG

        # Get the free two-phonon propagator
        X_ab_NI = -w_a**2 - w_b**2 - (2*w_a *w_b) /( (2*n_a + 1) * (2*n_b + 1))
        Y_ab_NI = - (8 * w_a * w_b) / ( (2*n_a + 1) * (2*n_b + 1))
        X1_ab_NI = - (2*n_a*n_b + n_a + n_b) * (2*n_a*n_b + n_a + n_b + 1)*(2 * w_a * w_b) / ( (2*n_a + 1) * (2*n_b + 1))
        Y1_ab_NI = - w_a**2 - w_b**2 + (2*w_a *w_b) /( (2*n_a + 1) * (2*n_b + 1))

        # Invert the propagator
        den = X_ab_NI * Y1_ab_NI - X1_ab_NI * Y_ab_NI

        # Regularize (avoid non invertibility when w_a = w_b and kbT << w)
        den_mask = den < __EPSILON__ 
        den[den_mask] = np.inf

        X_new = -Y1_ab_NI / den
        Y_new = Y_ab_NI / den
        X1_new = X1_ab_NI / den 
        Y1_new = - X_ab_NI / den

        X_new[den_mask] = - 1 / X_ab_NI[den_mask]

        # If T > 0, then also ReA could be inverted (only if w_a and w_b are thermally populated)
        if self.T > __EPSILON__:
            new_mask = (w_a == w_b) & (n_a > __EPSILON__)
            Y1_new[new_mask] = -1 / Y1_ab_NI[new_mask]

        out_vect[start_Y: start_A] = X_new * psi[start_Y: start_A]
        if not transpose:
            out_vect[start_Y : start_A] += Y_new * psi[start_A: ]
        else:
            out_vect[start_A:] += Y_new * psi[start_Y : start_A]

        #L_operator[start_Y : start_A, start_A:] = - np.diag(Y_ab_NI) * extra_count
        #L_operator[start_Y + np.arange(self.n_modes**2), start_A + exchange_frequencies] -=  Y_ab_NI / 2


        if not transpose:
            out_vect[start_A:] += X1_new * psi[start_Y: start_A]
        else:
            out_vect[start_Y: start_A] += X1_new * psi[start_A:]
        #L_operator[start_A:, start_Y : start_A] = - np.diag(X1_ab_NI) / 1 * extra_count
        #L_operator[start_A + np.arange(self.n_modes**2), start_Y + exchange_frequencies] -= X1_ab_NI / 2

        out_vect[start_A:] += Y1_new * psi[start_A:]
        #L_operator[start_A:, start_A:] = -np.diag(Y1_ab_NI) / 1 * extra_count
        #L_operator[start_A + np.arange(self.n_modes**2),  start_A + exchange_frequencies] -= Y1_ab_NI / 2


        return out_vect

    def apply_L1_static(self, psi, inverse = False, power = 1):
        """
        Apply the harmonic part of the L matrix for computing the static case (at any temperature).

        The inverse keyword, if True, compute the L^-1. If power is different from one, then multiply it to a specific power.

        """

        expected_dim = self.n_modes +  (self.n_modes * (self.n_modes + 1)) // 2
        ERRMSG = """
Error, for the static calculation the vector must be of dimension {}, got {}
""".format(expected_dim, len(psi))

        assert len(psi) == expected_dim, ERRMSG

        out_vect = np.zeros(psi.shape, dtype = TYPE_DP)

        if inverse:
            power *= -1
        
        # Here we apply the D2
        out_vect[:self.n_modes] = psi[:self.n_modes] * (self.w **(2 * power))
        #out_vect[:self.n_modes] = (psi[:self.n_modes] / self.w) / self.w 


        # Now we apply the inverse of the W matrix
        # The elements where w_a and w_b are exchanged are dependent
        # So we must avoid including them
        i_a = np.tile(np.arange(self.n_modes), (self.n_modes,1)).ravel()
        i_b = np.tile(np.arange(self.n_modes), (self.n_modes,1)).T.ravel()

        new_i_a = np.array([i_a[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        new_i_b = np.array([i_b[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        
        w_a = self.w[new_i_a]
        w_b = self.w[new_i_b]


        n_a = np.zeros(np.shape(w_a), dtype = TYPE_DP)
        n_b = np.zeros(np.shape(w_a), dtype = TYPE_DP)

        not_populated_mask_a = 0.01 * w_a *__RyToK__ > self.T
        not_populated_mask_b = 0.01 * w_a *__RyToK__ > self.T

        diff_n_ab = np.zeros( np.shape(w_a), dtype = TYPE_DP)
        w_ab_equal = np.abs(w_a - w_b) < 1e-8
        w_equal = w_a[w_ab_equal]
        n_equal = n_a[w_ab_equal]

        if self.T > 0:
            beta = np.double(__RyToK__ / self.T)
            n_a = 1 / (np.exp( w_a * beta) - 1)
            n_b = 1 / (np.exp( w_b * beta) - 1)
            n_a[not_populated_mask_a] = 0
            n_b[not_populated_mask_b] = 0

            diff_n_ab[:] = (n_a - n_b) / (w_a - w_b)
            diff_n_ab[w_ab_equal] = - beta * np.exp(w_equal * beta) * n_equal**2
            
        Lambda =  (n_a + n_b + 1) / (w_a + w_b) - diff_n_ab
        Lambda /= 4 * w_a * w_b

        out_vect[self.n_modes:] =  psi[self.n_modes:] / Lambda**power

        return out_vect
    
    
    
    
    
    def get_chi_minus(self):
        r"""
        Get the chi^- equilibrium tensor in the Wigner formalism.
        
        :: math .
            \tilde{\chi}^{-}_{\mu\nu} = \frac{\hbar\left[\omega_\alpha - \omega_\beta\right]\left[n_\alpha - n_\beta\right]}{2\omega_\alpha\omega_\beta}
            
        Results:
        -------
            -chi_minus: chi minus tensor, np.array with shape = (n_modes, n_modes)
        
        """
        # Prepare the result
        chi_minus = np.zeros((self.n_modes, self.n_modes), dtype = np.double)
        
        # Create the matrix with freqeuncies
        w = np.tile(self.w, (self.n_modes,1))
        
        # Create the Bose-Eninstein occupation number matrix
        n = np.zeros((self.n_modes, self.n_modes), dtype = np.double)
        
        if self.T > __EPSILON__:
            n = 1.0 / (np.exp(w * 157887.32400374097 /self.T) - 1.0)
        
        chi_minus = (w - w.T) * (n - n.T) /(2. * w * w.T)
        
        return chi_minus
    
    
    def get_chi_plus(self):
        r"""
        Get the chi^+ equilibrium tensor in the Wigner formalism.
        
        :: math .
            \tilde{\chi}^{+}_{\mu\nu} = \frac{\hbar\left[\omega_\alpha + \omega_\beta\right]\left[1 + n_\alpha + n_\beta\right]}{2\omega_\alpha\omega_\beta}
            
        Results:
        -------
            -chi_plus: chi plus tensor, np.array with shape = (n_modes, n_modes)
        
        """
        # Prepare the result
        chi_plus = np.zeros((self.n_modes, self.n_modes), dtype = np.double)
        
        # Create the matrix with freqeuncies
        w = np.tile(self.w, (self.n_modes,1))
        
        # Create the Bose-Eninstein occupation number matrix
        n = np.zeros((self.n_modes, self.n_modes), dtype = np.double)
        
        if self.T > __EPSILON__:
            n = 1.0 / (np.exp(w * 157887.32400374097 /self.T) - 1.0)
        
        chi_plus = (w + w.T) * (1 + n + n.T) /(2. * w * w.T)
        
        return chi_plus
    
    
    def get_a1_b1_wigner(self, get_a1 = True):
        r"""
        Get the the pertrubation on the a' matrix times sqrt(-0.5X^-) or on b' matrix
        from the alpha and beta pertrubation.
        
        This function is to check the inverse of the change of variables.
        
        :: math .
            \sqrt{-\frac{1}{2}\tilde{\chi}^-_{\mu\nu}} \tilde{a}'^{(1)}_{\mu\nu} = X_{\mu\nu} \cdot
            \left[\frac{+1}{2}\left(\frac{\hbar^2}{\omega_{\mu} \omega_{\nu}}\tilde{\alpha}^{(1)}_{\mu\nu} 
            + \tilde{\beta}^{(1)}_{\mu\nu}\right)\right];\\
            
        :: math .
           \tilde{b}'^{(1)}_{\mu\nu} = \frac{X_{\mu\nu}}{\sqrt{\frac{1}{2}\tilde{\chi}^+_{\mu\nu}}} 
           \left[\frac{-1}{2}
           \left(\frac{\hbar^2}{\omega_{\alpha} \omega_{\beta}}\tilde{ \alpha}_{\mu\nu} 
            - \tilde{ \beta}_{\mu\nu}\right)\right].;\\
            
        Parameters:
        -----------
            -get_a1: bool. If true we return the a' perturbation rescaled or the b' pertrubation
        
        Retruns:
        -------
            -a1: np.array with shape = n_modes * (n_modes + 1)/2
        """ 
        len_ = (self.n_modes * (self.n_modes + 1)) // 2
        
        # Get the beta1-alpha1 pertrubation as matrices, np.shape = (n_modes, n_modes)
        alpha_mat_1 = self.get_alpha1_beta1_wigner(get_alpha = True)
        beta_mat_1  = self.get_alpha1_beta1_wigner(get_alpha = False)
        
        # Now get alpha1 and beta1 as vectors
        alpha1 = np.zeros(len_, dtype = np.double)
        beta1  = np.zeros(len_, dtype = np.double)
        
        start = 0
        next = self.n_modes
        for i in range(self.n_modes):
            # Get the sum of the rescaled tensor
            alpha1[start : next] = alpha_mat_1[i, i:]
            beta1[start : next]  = beta_mat_1[i, i:]
            start = next 
            next = start + self.n_modes - i - 1 
           
        # Get the independent indeces
        # Avoid the exchange of w_a w_b
        i_a = np.tile(np.arange(self.n_modes), (self.n_modes,1)).ravel()
        i_b = np.tile(np.arange(self.n_modes), (self.n_modes,1)).T.ravel()

        new_i_a = np.array([i_a[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        new_i_b = np.array([i_b[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        
        # Get the independent indices
        w_a = self.w[new_i_a]
        w_b = self.w[new_i_b]

        n_a = np.zeros(np.shape(len(w_a)), dtype = TYPE_DP)
        n_b = np.zeros(np.shape(len(w_a)), dtype = TYPE_DP)
        
        if self.T > 0:
            n_a = 1 / (np.exp( w_a / np.double(self.T / __RyToK__)) - 1)
            n_b = 1 / (np.exp( w_b / np.double(self.T / __RyToK__)) - 1)
            
        # Get all the quantities to make the change of variables
        X = ((1 + 2 * n_a) * (1 + 2 * n_b) /8)
        w_a_b = (w_a * w_b)
        chi_minus = ((w_a - w_b) * (n_a - n_b)) /(2 * w_a * w_b)
        chi_plus  = ((w_a + w_b) * (1 + n_a + n_b)) /(2 * w_a * w_b)
        
        if get_a1:
            a1 = np.zeros(len_, dtype = np.double)
            a1 = 0.5 * X * (alpha1 / w_a_b + beta1)
            
            return a1
        else:
            b1 = -0.5 * X * (alpha1 /w_a_b - beta1)
            b1 /= np.sqrt(0.5 * chi_plus)

            return b1

        

    def get_alpha1_beta1_wigner(self, get_alpha = True):
        r"""
        Get the perturbation on the alpha/Upsilon or beta matrix from the psi vector in the Wigner formalism.
        
        Recall that alpha and beta are the starting free parameters of the Gaussian Wigner distribution.
        
        N.B.: This is the function that compute Upsilon^(1) matrix!
        
        :: math .
            \tilde{\Upsilon}^{(1)}_{\mu\nu} = \tilde{\alpha}^{(1)}_{\mu\nu} =
            \frac{\omega_\mu \omega_\nu}{\hbar^2 X_{\mu\nu}}
            \left(
            +\sqrt{-\frac{1}{2}\tilde{\chi}^-_{\mu\nu}} a'^{(1)}_{\mu\nu} 
            -\sqrt{+\frac{1}{2}\tilde{\chi}^+_{\mu\nu}} b'^{(1)}_{\mu\nu}
            \right) 
            
        :: math .
            \tilde{\beta}^{(1)}_{\mu\nu} = 
            \frac{1}{X_{\mu\nu}}
            \left(
             \sqrt{-\frac{1}{2}\tilde{\chi}^-_{\mu\nu}} a'^{(1)}_{\mu\nu} 
            + \sqrt{\frac{1}{2}\tilde{\chi}^+_{\mu\nu}} b'^{(1)}_{\mu\nu}\right) 
            
        Parameters:
        -----------
            -get_alpha: bool, if True alpha1 is returned. If False beta1 is returned.
            
        Returns:
        --------
            -alpha1 or beta1: array with shape = (n_modes, n_modes)
        """
        # Where the arrays start
        start_a = self.n_modes
        start_b = self.n_modes +  (self.n_modes * (self.n_modes + 1)) // 2

        # GET THE PERTURBED PARAMETERS a'^(1) and b'^(1)
        a_all = self.psi[start_a : start_b]
        b_all = self.psi[start_b :]
        
        # Get the independent indeces
        # Avoid the exchange of w_a w_b
        i_a = np.tile(np.arange(self.n_modes), (self.n_modes,1)).ravel()
        i_b = np.tile(np.arange(self.n_modes), (self.n_modes,1)).T.ravel()

        # Get the independent indeces
        new_i_a = np.array([i_a[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        new_i_b = np.array([i_b[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        
        # Get the independent freqeuncies
        w_a = self.w[new_i_a]
        w_b = self.w[new_i_b]

        n_a = np.zeros(np.shape(len(w_a)), dtype = TYPE_DP)
        n_b = np.zeros(np.shape(len(w_a)), dtype = TYPE_DP)
        
        if self.T > 0:
            n_a = 1 / (np.exp( w_a / np.double(self.T / __RyToK__)) - 1)
            n_b = 1 / (np.exp( w_b / np.double(self.T / __RyToK__)) - 1)
        
        # Get all the quantities to make the change of variables
        X         = ((1 + 2 * n_a) * (1 + 2 * n_b) /8)
        w2_on_X   = (w_a * w_b) / X
        chi_minus = ((w_a - w_b) * (n_a - n_b)) /(2 * w_a * w_b)
        chi_plus  = ((w_a + w_b) * (1 + n_a + n_b)) /(2 * w_a * w_b)
        
        if get_alpha:
            # Now rescale a'^(1) 
            new_a =  w2_on_X * np.sqrt(- 0.5 * chi_minus) * a_all
            # Now rescale b'^(1)
            new_b =  w2_on_X * np.sqrt(+ 0.5 * chi_plus)  * b_all

            # Prepare the result
            # This is Y(1)
            alpha1 = np.zeros((self.n_modes, self.n_modes), dtype = np.double)

            # Start filling
            start = 0
            next = self.n_modes
            for i in range(self.n_modes):
                # Get the difference of the rescaled tensor
                alpha1[i, i:] = new_a[start : next] - new_b[start : next]
                start = next 
                next = start + self.n_modes - i - 1 

                # Fill symmetric
                alpha1[i, :i] = alpha1[:i, i]
                
            return alpha1      
        else:
            # Now rescale a'^(1) 
            new_a =  (np.sqrt(- 0.5 * chi_minus) /X) * a_all
            # Now rescale b'^(1)
            new_b =  (np.sqrt(+ 0.5 * chi_plus)  /X) * b_all

            # Prepare the result
            beta1 = np.zeros( (self.n_modes, self.n_modes), dtype = np.double)

            # Start filling
            start = 0
            next = self.n_modes
            for i in range(self.n_modes):
                # Get the sum of the rescaled tensor
                beta1[i, i:] = new_a[start : next] + new_b[start : next]
                start = next 
                next = start + self.n_modes - i - 1 

                # Fill symmetric
                beta1[i, :i] = beta1[:i, i]

            return beta1
    
    
    
   
    def get_Y1(self, half_off_diagonal = False):
        """
        Get the perturbation on the Y matrix from the psi vector.
        This is used in the standard code.
        """
        start_Y = self.n_modes
        start_A = self.n_modes +  (self.n_modes * (self.n_modes + 1)) // 2

        Y_all = self.psi[start_Y : start_A]

        Y1 = np.zeros( (self.n_modes, self.n_modes), dtype = np.double)
        
        start = 0
        next = self.n_modes
        for i in range(self.n_modes):
            Y1[i, i:] = Y_all[start : next]
            start = next 
            next = start + self.n_modes - i - 1 

            # Fill symmetric
            Y1[i, :i] = Y1[:i, i]

        # Normalize each term outside the diagonal
        if half_off_diagonal:
            norm_mask = np.ones((self.n_modes, self.n_modes), dtype = np.double) / 2
            np.fill_diagonal(norm_mask, 1)

            Y1 *= norm_mask


        return  Y1


    def get_ReA1(self, half_off_diagonal = False):
        """
        Get the perturbation on the ReA matrix from the psi vector.

        If half_the_off_diagonal is true, divide by 2 the off-diagonal elements
        """
        start_A = self.n_modes +  (self.n_modes * (self.n_modes + 1)) // 2

        ReA_all = self.psi[start_A:]

        ReA1 = np.zeros( (self.n_modes, self.n_modes), dtype = np.double)
        start = 0
        next = self.n_modes
        for i in range(self.n_modes):
            ReA1[i, i:] = ReA_all[start : next]
            start = next 
            next = start + self.n_modes - i - 1 

            # Fill symmetric
            ReA1[i, :i] = ReA1[:i, i]

        # Normalize each term outside the diagonal
        if half_off_diagonal:
            norm_mask = np.ones((self.n_modes, self.n_modes), dtype = np.double) / 2
            np.fill_diagonal(norm_mask, 1)

            ReA1 *= norm_mask

        return  ReA1


    def apply_anharmonic_FT(self, transpose = False, test_weights = True, use_old_version = False):
        r"""
        APPLY ANHARMONIC EVOLUTION
        ==========================

        This term involves the anharmonic evolution:
        This calculates self-consistently the anharmonic evolution from the vector.
        
        NORMAL: this method applies -L_anharm (see Eq. (K4) in the Appendix of the Monacelli PRB): 
        :: math .
            \begin{bmatrix}
             0   &  -X''  &  0 \\
            -Z   &  -X    & 0 \\
            -Z'  &  -X'   & 0
            \end{bmatrix}
        on the follwoing vector:
        :: math .
            \begin{bmatrix}
            \mathcal{R}^{(1)} \\
            \tilde{\Upsilon}^{(1)} \\
            \Re \tilde{A}^{(1)}.
            \end{bmatrix}
        
        WIGNER: this method applies +L_anh:
        :: math.
            \mathcal{L}_{anh} 
            \begin{bmatrix}
            \tilde{\mathcal{R}}^{(1)}_\mu\\ \\
            \tilde{a}'^{(1)}_{\mu\nu}\\ \\
            \tilde{b}'^{(1)}_{\mu\nu}
            \end{bmatrix} 
            = 
            \begin{bmatrix}
            -\left\langle \frac{\partial \mathbb{V}}{\partial \tilde{Q}_\alpha}\right\rangle_{(1)}\\ \\
            +\sqrt{-\frac{1}{2}\tilde{\chi}_{\alpha\beta}^{-}}
            \left\langle \frac{\partial^2 \mathbb{V}}{\partial \tilde{Q}_\alpha \partial \tilde{Q}_\beta}\right\rangle_{(1)}\\ \\
            -\sqrt{\frac{1}{2}\tilde{\chi}_{\alpha\beta}^{+}}
            \left\langle \frac{\partial^2 \mathbb{V}}{\partial \tilde{Q}_\alpha \partial \tilde{Q}_\beta}\right\rangle_{(1)}
            \end{bmatrix}
        
        Parameters
        ----------
            -transpose : bool
                If True, the transpose of L is computed.
            -test_weights : bool
                If True, the weights are tested against those computed with finite differences
                It is time consuming, activate only for debugging
            -use_old_version: bool
                If true, it employes an old version of the subroutine that does not satisfy the permutation symmetry.
                Use this option only for testing purpouses.
                
        Results:
        -------
            -final_psi: np.array with shape = n_modes + n_modes * (n_modes + 1)
        """
        #print("Starting with psi:", self.psi)
        
        # Get the perturbation R^(1)
        R1 = self.psi[: self.n_modes]
        
        if not self.use_wigner:
#             print('[NORMAL]: get Y(1)')
            # Get the perturbation Upsilon^(1)
            Y1 = self.get_Y1(half_off_diagonal = transpose)
        else:
#             print('[WIGNER]: get Y(1)')
            # Use the Wigner equations
            # Upsilon^(1) = alpha^(1)
            Y1 = self.get_alpha1_beta1_wigner(get_alpha = True)

        # Weights to perform the pertrubed average
        weights = np.zeros(self.N, dtype = np.double)

        if not self.use_wigner:
#             print('[NORMAL]: get static egivals')
            # The standard code
            # Create the multiplicative matrices for the rest of the anharmonicity
            n_mu = 0
            if self.T > __EPSILON__:
                n_mu = 1.0 /(np.exp(self.w * 157887.32400374097 / self.T) - 1.0)
            # Eigenvalues of Upsilon and ReA at equilibrium
            Y_w   = 2 * self.w / (2 * n_mu + 1)
            ReA_w = 2 * self.w * n_mu * (n_mu + 1) / (2*n_mu + 1)

            # Check if we must compute the transpose
            if transpose:
                ReA1 = self.get_ReA1(half_off_diagonal = transpose)

                # The equation is
                # Y^(1)_new = 2 Ya Yb^2 Y^(1) + 2 Yb Ya^2 Y^(1)
                coeff_Y = np.einsum("a, b, b -> ab", Y_w, Y_w, Y_w)
                coeff_Y += np.einsum("a, a, b -> ab", Y_w, Y_w, Y_w)
                coeff_Y *= 2

                coeff_RA = np.einsum("a, b, b -> ab", Y_w, ReA_w, Y_w)
                coeff_RA += np.einsum("a, a, b -> ab", Y_w, ReA_w, Y_w)
                coeff_RA *= 2

                # Get the new perturbation
                Y1_new = Y1 * coeff_Y + ReA1 * coeff_RA

                # Override the old perturbation
                Y1 = Y1_new    
        else:
#             print('[WIGNER]: get static egivals')
            # We are using Wigner equations
            # Get the chi +/- array (n_modes, n_modes)
            chi_minus = self.get_chi_minus()
            chi_plus  = self.get_chi_plus()
            
#             print('chi_minus = ')
#             print(chi_minus)
#             print('chi plus = ')
#             print(chi_plus)


        #print("X:", self.X)
        #print("w:", self.w)
        #print ("R1:", R1)
        #print("Y1:", Y1)
        #print("T:", self.T)

        # Compute the perturbed average of BO potential in the polarization basis
        f_pert_av   = np.zeros(self.n_modes, dtype = np.double)
        d2v_pert_av = np.zeros((self.n_modes, self.n_modes), dtype = np.double, order = "C")

        # Check if you need to compute the fourth order
        apply_d4 = 1
        if self.ignore_v4:
            print('Removing D4')
            apply_d4 = 0
            
        # NEW: we can remove the D3 effect
        if self.ignore_v3:
            print('Removing D3')
            R1[:] = 0.

        # Prepare the symmetry variables for the C code
        # deg_space_new = np.zeros(np.sum(self.N_degeneracy), dtype = np.intc)
        # i = 0
        # i_mode = 0
        # j_mode = 0
        # #print("Mapping degeneracies:", np.sum(n_degeneracies))
        # while i_mode < self.n_modes:
        #     #print("cross_modes: ({}, {}) | deg_i = {}".format(i_mode, j_mode, n_degeneracies[i_mode]))
        #     deg_space_new[i] = self.degenerate_space[i_mode][j_mode]
        #     j_mode += 1
        #     i += 1
        #     if j_mode == self.N_degeneracy[i_mode]:
        #         i_mode += 1
        #         j_mode = 0


        # Compute the perturbed averages (the time consuming part is HERE !!!)
        #print("Entering in get pert...")
        n_syms, _, _ = np.shape(self.symmetries[0])
        #print("DEG:")
        #print(self.degenerate_space)
        
        # OLD Implementation still working
        if self.mode in (MODE_FAST_MPI, MODE_FAST_SERIAL): 
            # Get the pertrubed averages
            sscha_HP_odd.GetPerturbAverageSym(self.X, self.Y, self.w, self.rho, R1, Y1, self.T, apply_d4, n_syms,
                                              self.symmetries, self.N_degeneracy, self.degenerate_space ,self.sym_block_id, 
                                              f_pert_av, d2v_pert_av)
        
        elif self.mode == MODE_FAST_JULIA:
            if not __JULIA_EXT__:
                raise ImportError("Error while importing julia. Try with python-jl after pip install julia.")
                
            if self.sym_julia is None:
                MSG = "Error, the initialization must be called AFTER you change mode to JULIA."
                raise ValueError(MSG)
                
                
            # Prepare the parallelization function
            def get_f_proc(start_end):
                start = int(start_end[0])
                end   = int(start_end[1])
                #Parallel.all_print("Processor {} is doing:".format(Parallel.get_rank()), start_end)
                f_pert_av = julia.Main.get_perturb_f_averages_sym(self.X.T, self.Y.T, self.w, self.rho, R1, Y1, np.float64(self.T), bool(apply_d4),\
                                                                  self.sym_julia, self.N_degeneracy, self.deg_julia ,self.sym_block_id, start, end)
                return f_pert_av
            def get_d2v_proc(start_end):
                start = int(start_end[0])
                end   = int(start_end[1])
                d2v_dr2 = julia.Main.get_perturb_d2v_averages_sym(self.X.T, self.Y.T, self.w, self.rho, R1, Y1, np.float64(self.T), bool(apply_d4),\
                                                                  self.sym_julia, self.N_degeneracy, self.deg_julia ,self.sym_block_id, start, end)
                return d2v_dr2
            
            # Divide the configurations and symmetries on different processors (Here we get the range of work for each process)
            n_total = self.n_syms * self.N 
            n_processors = Parallel.GetNProc()
            count = n_total // n_processors
            remainer = n_total % n_processors
            
            # Assign which configurations should be computed by each processor
            indices = []
            for rank in range(n_processors):

                if rank < remainer:
                    start = np.int64(rank * (count + 1))
                    stop = np.int64(start + count + 1) 
                else:
                    start = np.int64(rank * count + remainer) 
                    stop = np.int64(start + count) 

                indices.append([start + 1, stop])

                
            # Execute the get_f_d2v_proc on each processor in parallel.
            f_pert_av   = Parallel.GoParallel(get_f_proc, indices, "+")
            d2v_pert_av = Parallel.GoParallel(get_d2v_proc, indices, "+")

        else:
            raise ValueError("Error, mode running {} not implemented.".format(self.mode))


            
              
        # OLD PART OF THE CODE
        #print("D2V:")
        #np.set_printoptions(threshold = 10000)
        #print(d2v_pert_av[:10, :10])#print("Out get pert")

#         print("R1 = {}".format(R1))
#         print("Y1 = {}".format(Y1))
#         print("<f> pert = {}".format(f_pert_av))
#         print("<d2v/dr^2> pert = {}".format(d2v_pert_av))
#         print("<d2v/dr^2>T pert = {}".format(d2v_pert_av.T))
#         print("<d2v/dr^2>T - <d2v/dr^2>  pert = {}".format(d2v_pert_av.T - d2v_pert_av))
#         print()

        # Compute the average with the OLD VERSION
        if use_old_version:
            # Get the weights of the perturbation (psi vector)
            sscha_HP_odd.GetWeights(self.X, self.w, R1, Y1, self.T, weights)

            if test_weights:
                other_weights = get_weights_finite_differences(self.X, self.w, self.T, R1, Y1)

                # There is a constant factor that should come from the normalization
                # But this does not depend on the configuration
                shift = weights - other_weights 

                # Remove the constant shift coming from renormalization
                # 1/2 Tr (Y^-1 * Y^{(1)})
                shift -= np.mean(shift) 

                # Compare the weights
                disp = np.max(np.abs(shift))
                dispersion = np.std(weights)

                if disp / dispersion >= 1e-3:
                    print("Perturbation:")
                    print(self.psi)

                    print("Weights with C:")
                    print(weights)

                    print()
                    print("Weights by finite differences:")
                    print(other_weights)

                    print()
                    print("Shifts:")
                    print(weights - other_weights)
                    
                    print()
                    print("Discrepancies (max = {}):".format(disp))
                    i_value = np.argmax(np.abs(shift))
                    print(shift)
                    print("I value of max: {}".format(i_value))

                    print("")
                    print("sigma = {}".format(dispersion))
                    print("Weights[{}] = {}".format(i_value, weights[i_value]))
                    print("OtherWeights[{}] = {}".format(i_value, other_weights[i_value]))


                assert disp / dispersion < 1e-3, "Error, the weights computed with the C did not pass the test"

            #print("Weights:", weights)

            # Get the averages on the perturbed ensemble
            w_is = np.tile(self.rho, (self.n_modes, 1)).T
            w_1 = np.tile(weights, (self.n_modes, 1)).T

            #print("rho shape:", np.shape(self.rho))
            #print("Shape w_is:", np.shape(w_is))

            # The force average
            avg_numbers = self.Y * w_is *  w_1 #np.einsum("ia, i, i -> ia", self.Y, w_is, w_1)
            #print("Shape Y:", np.shape(avg_numbers))
            f_pert_av = np.sum(avg_numbers, axis = 0) / self.N_eff


            #print("Shape F:", np.shape(f_pert_av))
            sscha_HP_odd.Get_D2DR2_PertV(self.X, self.Y, self.w, self.rho, weights, self.T, d2v_pert_av)


            #print("<f> pert = {}".format(f_pert_av))
            #print("<d2v/dr^2> pert = {}".format(d2v_pert_av))
            #print()
        
        # END OF THE OLD VERSION

        
        # Get the final vector
        final_psi = np.zeros(self.psi.shape, dtype = np.double)
        
        # Now get the perturbation for R^(1) (same in Wigner)
        final_psi[:self.n_modes] =  f_pert_av

        if not self.use_wigner:
#             print('[NORMAL]: anharmonic Y(1) ReA(1)')
            # Propagation for Upsilon^(1) and ReA^(1)
            if not transpose:
                # Get the perturbation D2 * Upsilon + Upsilon * D2
                pert_Y  = np.einsum("ab, a ->ab", d2v_pert_av, Y_w)
                pert_Y += np.einsum("ab, b ->ab", d2v_pert_av, Y_w)

                # Get the perturbation D2 * Re A +  Re A * D2
                pert_RA  = np.einsum("ab, a ->ab", d2v_pert_av, ReA_w)
                pert_RA += np.einsum("ab, b -> ab", d2v_pert_av, ReA_w)
            else:
                Y_inv = 1 / Y_w
                pert_Y = 0.5 * np.einsum("a, ab, b -> ab", Y_inv, d2v_pert_av, Y_inv)
                pert_RA = np.zeros(pert_Y.shape, dtype = np.double)

                # Now double the off diagonal values of pert_Y and pert_RA
                # This is to take into account the symmetric storage of psi
                sym_mask = np.ones(pert_Y.shape) * 2 
                np.fill_diagonal(sym_mask, 1) 
                pert_Y  *= sym_mask 
                pert_RA *= sym_mask
        else:
#             print("[WIGNER]: anharmonic a(1)' b(1)'")
            # We are using Wigner equations
            # Propagation for a'^(1)
            pert_Y  = np.einsum('ab, ab -> ab', +np.sqrt(-0.5 * chi_minus), d2v_pert_av)
            # Propagation for b'^(1)
            pert_RA = np.einsum('ab, ab -> ab', -np.sqrt(+0.5 * chi_plus),  d2v_pert_av)
        
            
#         print('pert_R = ')
#         print(f_pert_av)
#         print('pert_RA = ')
#         print(pert_RA)
            
        #####################
        # Update the vector #
        #####################
        
        # Note: the code deals with symmetric matrices in the following way.
        # Given a matrix you take the row on the right from the (1,1) element,
        # then you take the row on the right from the (2,2) element,
        # then you proceed with the (3,3) element.
        
        # Now get the perturbation for Upsilon^(1)/a'^(1)
        current = self.n_modes
        for i in range(self.n_modes):
            final_psi[current : current + self.n_modes - i] = pert_Y[i, i:]
            current = current + self.n_modes - i
            
#         print('final psi = ')
#         print(final_psi)

        # Now process the pertrubation of ReA^(1)/b'^(1)
        for i in range(self.n_modes):
            final_psi[current : current + self.n_modes - i] = pert_RA[i, i:]
            current = current + self.n_modes - i
            
#         print('final psi = ')
#         print(final_psi)

        # print("First element of pert_Y:", pert_Y[0,0])
        # print("Y_w = ", Y_w)
        # print("All pert Y:")
        # print(pert_Y)

        # print("Final psi:")
        # print(final_psi[self.n_modes: self.n_modes + 10])

        #print("Output:", final_psi)
        if not self.use_wigner:
#             print('[NORMAL]: anharmonic final')
            return -final_psi
        else:
#             print('[WIGNER]: anharmonic final')
            return +final_psi
        
        
        
        

    def apply_anharmonic_static(self):
        """
        Compute the anharmonic part of the L matrix for the static Hessian calculation.
        """ 

        Y1 = self.get_Y1()
        R1 = self.psi[: self.n_modes]


        # Create the Y matrix
        n_mu = 0
        if self.T > __EPSILON__:
            n_mu = 1.0 / ( np.exp(self.w * np.double(157887.32400374097) / self.T) - 1.0)
        Y_w = 2 * self.w / (2 * n_mu + 1)

        # prepare the modified of Y1
        Y1 = -2 * np.einsum("ab, a, b -> ab", Y1, Y_w, Y_w)

        # Compute the average SSCHA force and potential
        f_pert_av = np.zeros(self.n_modes, dtype = np.double)
        d2v_pert_av = np.zeros((self.n_modes, self.n_modes), dtype = np.double, order = "C")

        # Check if you need to compute the fourth order
        apply_d4 = 1
        if self.ignore_v4:
            apply_d4 = 0

        # Prepare the symmetry variables for the C code
        deg_space_new = np.zeros(np.sum(self.N_degeneracy), dtype = np.intc)
        i = 0
        i_mode = 0
        j_mode = 0
        #print("Mapping degeneracies:", np.sum(n_degeneracies))
        while i_mode < self.n_modes:
            #print("cross_modes: ({}, {}) | deg_i = {}".format(i_mode, j_mode, n_degeneracies[i_mode]))
            deg_space_new[i] = self.degenerate_space[i_mode][j_mode]
            j_mode += 1
            i += 1
            if j_mode == self.N_degeneracy[i_mode]:
                i_mode += 1
                j_mode = 0


        # Compute the perturbed averages (the time consuming part is HERE)
        #print("Entering in get pert...")
        sscha_HP_odd.GetPerturbAverageSym(self.X, self.Y, self.w, self.rho, R1, Y1, self.T, apply_d4, 
                                          self.symmetries, self.N_degeneracy, deg_space_new, 
                                          f_pert_av, d2v_pert_av)

        # Get the final vector
        final_psi = np.zeros(self.psi.shape, dtype = np.double)
        final_psi[:self.n_modes] =  - f_pert_av

        # Now get the perturbation on the vector
        current = self.n_modes
        for i in range(self.n_modes):
            final_psi[current : current + self.n_modes - i] = d2v_pert_av[i, i:]
            current = current + self.n_modes - i
        

        return final_psi


    def apply_L2(self):
        """
        APPLY THE L2
        ============

        L2 is the part of the L operators that mixes the two spaces.
        It involves the phi3 matrix.
        """

        if self.ignore_v3:
            return np.zeros(np.shape(self.psi), dtype = TYPE_DP)


        w_a = np.tile(self.w, (self.n_modes, 1)).ravel()
        w_b = np.tile(self.w, (self.n_modes, 1)).T.ravel()

        vector = self.psi[:self.n_modes]
        dyn = self.psi[self.n_modes:]
        new_dyn = -dyn * np.sqrt( (w_a + w_b)/(w_a*w_b)) / 2

        # Here the time consuming part
        if self.mode == 0:
            # DEBUGGING PYTHON VERSION (SLOW)
            out_v = SlowApplyD3ToDyn(self.X, self.Y, self.rho, self.w, self.T, new_dyn)
            out_d = SlowApplyD3ToVector(self.X, self.Y, self.rho, self.w, self.T, vector)
        elif self.mode >= 1:
            out_v = FastApplyD3ToDyn(self.X, self.Y, self.rho, self.w, self.T, new_dyn, self.symmetries,
                                     self.N_degeneracy, self.degenerate_space, self.mode)
            out_d = FastApplyD3ToVector(self.X, self.Y, self.rho, self.w, self.T, vector, self.symmetries,
                                        self.N_degeneracy, self.degenerate_space, self.mode)
        else:
            print("Error, mode %d not recognized." % self.mode)
            raise ValueError("Mode not recognized %d" % self.mode)
            
        out_d *= -np.sqrt( (w_a + w_b)/(w_a*w_b)) / 2

        out_vect = np.zeros(np.shape(self.psi), dtype = TYPE_DP)
        
        out_vect[:self.n_modes] = out_v
        out_vect[self.n_modes:] = out_d
        return out_vect

    def apply_L2_FT(self, transpose = False):
        """
        Apply the full matrix at finite temperature.
        """ 

        if self.ignore_v3:
            return np.zeros(self.psi.shape, dtype = TYPE_DP)

        return FastD3_FT(self.X, self.Y, self.rho, self.w, self.T, self.psi, self.symmetries, self.N_degeneracy, self.degenerate_space, self.mode, transpose)

    def apply_L3(self):
        """
        APPLY THE L3
        ============

        This is the last part of the L matrix, it puts in communication 
        the dyn part of psi with herselfs.
        """

        w_a = np.tile(self.w, (self.n_modes, 1)).ravel()
        w_b = np.tile(self.w, (self.n_modes, 1)).T.ravel()

        simple_output = np.zeros(np.shape(self.psi), dtype = TYPE_DP)

        #simple_output[self.n_modes:] = self.psi[self.n_modes:] * (w_a + w_b)**2

        if self.ignore_v4:
            return simple_output

        # Apply the D4
        
        dyn = self.psi[self.n_modes:] * np.sqrt((w_a + w_b) / (w_a * w_b)) / 2
        


        # Here the time consuming part [The most of all]!!!
        if self.mode == 0:
            # A very slow implementation
            # Use it just for debugging
            out_dyn = SlowApplyD4ToDyn(self.X, self.Y, self.rho, self.w, self.T, dyn)
        elif self.mode >= 1:
            # The fast C implementation
            #print ("Inside v4 MPI, this will take a while")
            out_dyn = FastApplyD4ToDyn(self.X, self.Y, self.rho, self.w, self.T, dyn,
                                       self.symmetries, self.N_degeneracy, self.degenerate_space, self.mode)       
            
        out_dyn *= np.sqrt((w_a + w_b) / (w_a * w_b)) / 2

        output = np.zeros(np.shape(self.psi), dtype = TYPE_DP)
        output[self.n_modes:] = out_dyn

        output += simple_output

        return output


    def apply_L3_FT(self, transpose = False):
        """
        APPLY THE L3
        ============

        This is the last part of the L matrix, it puts in communication 
        the dyn part of psi with herselfs.
        """

        simple_output = np.zeros(np.shape(self.psi), dtype = TYPE_DP)

        #simple_output[self.n_modes:] = self.psi[self.n_modes:] * (w_a + w_b)**2

        if not self.ignore_v4:
            simple_output[:] = FastD4_FT(self.X, self.Y, self.rho, self.w, self.T, self.psi, self.symmetries, self.N_degeneracy, self.degenerate_space, self.mode, transpose)


        return simple_output

    def apply_full_L(self, target = None, force_t_0 = False, force_FT = True, transpose = False, fast_lanczos = True):
        """
        APPLY THE L 
        ===========

        This function applies the L operator to the specified target vector.
        The target vector is first copied into the local psi, and the computed.
        NOTE: This function will overwrite the current psi with the specified
        target.

        Parameters
        ----------
            target : ndarray ( size = shape(self.psi)), optional
                The garget vector to which you want to apply the
                full L matrix
            force_t_0 : bool
                If False (default) the temperature is looked to chose if use the T = 0 or the finite temperature.
                If True it is forced the T=0 method (This will lead to wrong results at finite temperature).
            force_FT : bool
                If True the finite temperature method is forced even if T = 0.
                The results should be good, use it for testing.
                NOTE: only one between force_t_0 and force_FT should be true
            fast_lanczos : bool
                If true this method applies the L2 and L3 using the self-consistent way.
                This is much quicker, but needs to be tested
            transpose : bool
                Default is False, if it is true we apply the transpose
            fast_lanczos : bool
                See force_t_0 for details.
                
        Returns:
        -------
            -self.psi: the updated vector

        """
        if force_t_0 and force_FT:
            raise ValueError("Error, only one between force_t_0 and force_FT can be True")

        # Setup the target vector to the self.psi
        if not target is None:
            self.psi = target 

        # Check the initialization
        if not self.initialized:
            raise ValueError("Error, this class must be initialized before lunching the computation:\n Use the .init()")

        #if self.symmetrize:
        #    self.symmetrize_psi()

        #print("Psi before:")
        #print(self.psi[:self.n_modes])
            
            
        # Apply the whole L step by step to self.psi
        t1 = timer()
        if (force_t_0 or self.T < __EPSILON__) and not force_FT:
            # Harmonic evolution
            output = self.apply_L1()
        else:
            #HARMONIC evolution at finite temperature
            output = self.apply_L1_FT(transpose)
        t2 = timer()

        # Apply the quick_lanczos
        t4 = timer()
        if fast_lanczos and (not self.ignore_v3):
            print('Applying the anharmonic part of L')
            # AN-HARMONIC evolution finite temperature
            output += self.apply_anharmonic_FT(transpose)
            t3 = timer()
            t4 = t3
        # else:
        #     if (force_t_0 or self.T < __EPSILON__) and not force_FT:
        #         output += self.apply_L2()
        #     else:
        #         output += self.apply_L2_FT(transpose)
        #     t3 = timer()
        #     if (force_t_0 or self.T < __EPSILON__) and not force_FT:
        #         output += self.apply_L3()
        #     else:
        #         output += self.apply_L3_FT(transpose)
        #     t4 = timer()

        if self.verbose:
            print("Time to apply the full L: {}".format(t4 - t1))
        #print("Time to apply L2: {}".format(t3-t2))
        #print("Time to apply L3: {}".format(t4-t3))

        # Apply the shift reverse
        #print ("Output before:")
        #print (output[:self.n_modes])
        if self.reverse_L:
            output *= -1
        output += self.shift_value * self.psi
        #print ("Output after:")
        #print (output[:self.n_modes])

        # Now return the output
        #print ("out just before return:", output[0])
        self.psi = output
        #if self.symmetrize:
        #    self.symmetrize_psi()
        
        return self.psi

    def save_abc(self, file):
        """
        Save only the a, b, and c coefficients from the lanczos.
        In this way the calculation cannot be restarted.
        """

        total_len = len(self.c_coeffs)

        abc = np.zeros( (total_len, 3), dtype = np.double)
        abc[:,0] = self.a_coeffs[:total_len]
        abc[:,1] = self.b_coeffs
        abc[:,2] = self.c_coeffs

        np.savetxt(file, abc, header = "a; b; c")

    def load_abc(self, file):
        """
        Load only the a, b, and c coefficients from the ".abc" file
        """

        abc = np.loadtxt(file)
        self.a_coeffs = abc[:,0]
        self.b_coeffs = abc[:,1]
        self.c_coeffs = abc[:,2]

    def save_status(self, file):
        """
        Save the current data in npz compressed format, in order to reanalyze easily the result (or restart the Lanczos)
        later.

        Parameters
        ----------
            file : string
                Path to where you want to save the data. It must be an npz binary format. The extension
                will be added if it does not match the npz one
        """
        # Force all the processes to be here
        Parallel.barrier()

        # Add the correct extension
        if not ".npz" in file.lower():
            file += ".npz"
        
        # Save all the data
        if Parallel.am_i_the_master():
            np.savez_compressed(file, 
                                T = self.T,
                                nat = self.nat,
                                m = self.m,
                                w = self.w,
                                pols = self.pols,
                                n_modes = self.n_modes,
                                ignore_v3 = self.ignore_v3,
                                ignore_v4 = self.ignore_v4,
                                N = self.N,
                                rho = self.rho,
                                X = self.X,
                                Y = self.Y,
                                psi = self.psi,
                                a_coeffs = self.a_coeffs,
                                b_coeffs = self.b_coeffs,
                                c_coeffs = self.c_coeffs,
                                basis_Q = self.basis_Q,
                                basis_P = self.basis_P,
                                s_norm = self.s_norm,
                                krilov_basis = self.krilov_basis,
                                arnoldi_matrix = self.arnoldi_matrix,
                                reverse = self.reverse_L,
                                shift = self.shift_value,
                                perturbation_modulus = self.perturbation_modulus,
                                use_wigner = self.use_wigner,
                                ignore_small_w = self.ignore_small_w,
                                sym_julia = self.sym_julia,
                                deg_julia = self.deg_julia,
                                n_syms = self.n_syms)

            
    def load_status(self, file, is_file_instance = False):
        """
        Load a previously saved status from the speficied npz file.
        The file must be saved with save_status.

        If is_file_instance is True, the file is assumed to be already loaded with open(...)
        """

        # Force all the process to be here
        Parallel.barrier()

        if not is_file_instance:
            # Add the correct extension
            if not ".npz" in file.lower():
                file += ".npz"

            # Check if the provided file exists
            if not os.path.exists(file):
                print ("Error while loading %s file.\n" % file)
                raise IOError("Error while loading %s" % file)

        data = {}
        # Read the data only with the master
        if Parallel.am_i_the_master():
            
            # Fix the allow pickle error in numpy >= 1.14.4
            try:
                data = np.load(file, allow_pickle = True)
            except ValueError:
                print("Error in pickling the data")
                raise
            except:
                print("Error, while loading with allow_pickle (numpy version < 1.14.4?)")
                print("       numpy version = {}".format(np.__version__))
                print("Trying without pickling...")
                data = np.load(file) 

        # Now bcast to everyone
        data = Parallel.broadcast(dict(data))

        self.T = np.double(data["T"])
        self.nat = np.intc(data["nat"])
        self.m = data["m"]
        self.w = data["w"]
        self.pols = data["pols"]
        self.n_modes = np.intc(data["n_modes"])
        self.ignore_v3 = bool(data["ignore_v3"])
        self.ignore_v4 = bool(data["ignore_v4"])
        self.N = np.intc(data["N"])
        self.rho = data["rho"]
        self.X = data["X"]
        self.Y = data["Y"]
        self.psi = data["psi"]
        self.a_coeffs = data["a_coeffs"]
        self.b_coeffs = data["b_coeffs"]
        if "c_coeffs" in data:
            self.c_coeffs = data["c_coeffs"]
        # Make them as lists
        self.a_coeffs = list(self.a_coeffs)
        self.b_coeffs = list(self.b_coeffs)
        self.c_coeffs = list(self.c_coeffs)
        self.krilov_basis = data["krilov_basis"]
        self.arnoldi_matrix = data["arnoldi_matrix"]
        
        try:
            self.sym_julia = data["sym_julia"]
            self.deg_julia = data["deg_julia"]
            self.n_syms = data["n_syms"]
        except:
            print('ATTENTION THE JULIA VARIABLES  WERE NOT LOADED')
            self.sym_julia = None
            self.deg_julia = None
            self.n_syms = 1
            

        self.basis_Q = data["basis_Q"]
        self.basis_P = data["basis_P"]
        self.s_norm = data["s_norm"]
        
        self.use_wigner = data["use_wigner"]
        try:
            self.ignore_small_w = data["ignore_small_w"]
        except:
            self.ignore_small_w = False #data["ignore_small_w"]

        if "reverse" in data.keys():
            self.reverse_L = data["reverse"]
            self.shift_value = data["shift"]

        if "symmetries" in data.keys():
            self.symmetries = data["symmetries"]
            self.N_degeneracy = data["N_degeneracy"]
            self.initialized = data["initialized"]
            self.degenerate_space = data["degenerate_space"]
        
        if "perturbation_modulus" in data.keys():
            self.perturbation_modulus = data["perturbation_modulus"]

        
        # Prepare the L as a linear operator (Prepare the possibility to transpose the matrix)
        def L_transp(psi):
            return self.apply_full_L(psi, transpose= True)
        self.L_linop = scipy.sparse.linalg.LinearOperator(shape = (len(self.psi), len(self.psi)),\
                                                          matvec = self.apply_full_L, rmatvec = L_transp, dtype = TYPE_DP)

        # Define the preconditioner
        def M_transp(psi):
            return self.apply_L1_inverse_FT(psi, transpose = True)
        self.M_linop = scipy.sparse.linalg.LinearOperator(shape = (len(self.psi), len(self.psi)),\
                                                          matvec = self.apply_L1_inverse_FT, rmatvec = M_transp, dtype = TYPE_DP)


    def run_biconjugate_gradient(self, verbose = True, tol = 5e-4, maxiter = 1000, save_g = None, save_each = 1, use_preconditioning = True, algorithm = "bicgstab"):
        """
        STATIC RESPONSE
        ===============

        Get the static response inverting the green function.
        This is performed by exploiting the bi-conjugate gradient algorithm.

        Parameters
        ----------
            verbose : bool
                If true, print the status of the algorithm in standard output
            tol : float
                Tollerance of the biconjugate gradient algorithm 
            maxiter: int
                The maximum number of iterations for the biconjugate gradient.
            save_g: string
                Path to the file on which save the green function. 
                It is saved after a number of steps specified by save_each.
            save_each: int
                Determines after how many steps to save the green function.
            use_preconditioning: bool
                If true, uses the preconditioning to solve the gradient.
                The precondition is obtained by inverting analytically only the Harmonic propagation of the
                L matrix
            algorithm: string
                The algorithm used to invert the L matrix. One between:
                   - bicg : Conjugate Gradient (Default)
                   - bicgstab : Stabilized Conjugate Gradient.
                   - cg-minimize: Conjugate Gradient with preconditioned minimization.
                         This algorithm minimizes the auxiliary function
                         .. math::

                            f(x) = \frac 12 r H^-1 r

                        where :math:`r = Lx - b` and :math:`H = L^\dagger L`. 
                        The hessian :math:`H` is guessed neglecting interaction (as for the perfectly harmonic case).
                It will invoke the corresponding scipy subroutine

        Results
        -------
            G_inv: ndarray(size = (n_modes, n_modes))
                This is the mass-rescaled free energy Hessian.
                Its eigenvalues are the static frequencies, that determine the structure stability.
        """

        G_one_phonon = np.zeros( (self.n_modes, self.n_modes), dtype = np.double)

        if verbose:
            print()
            print("====================")
            print("BICONJUGATE GRADIENT")
            print("====================")
            print()
            print("We compute the static response with the")
            print("Biconjugate gradient algorithm.")
            print()

        
        # Check if the symmetries has been initialized
        if not self.initialized:
            self.prepare_symmetrization()

        j = np.zeros(1, dtype = np.intc)
        x_old = self.psi.copy()
        for i in range(self.n_modes):
            if verbose:
                # Print the status
                print()
                print()
                print("NEW STEP")
                print("--------")
                print()
                print("i = {} / {}".format(i + 1, self.n_modes))
                print()
            
            # Setup the known vector
            self.psi = np.zeros(self.psi.shape, dtype = type(self.psi[0]))
            self.psi[i] = 1

            x_old[:] = self.psi
            j[0] = 0
            def callback(xk, x_old = x_old, j = j):
                if np.isnan(np.sum(xk)):
                    raise ValueError("Error, NaN value found during the Biconjugate Gradient.") 
                if verbose:
                    disp = sum( (xk - x_old)**2)
                    print("BCG STEP {} | solution changed by {} (tol = {})".format(j[0], disp, tol))
                    j[0] += 1
                    x_old[:] = xk
                
            # Prepare the preconditioning
            M_prec = None
            x0 = self.M_linop.matvec(self.psi)
            if use_preconditioning:
                M_prec = self.M_linop
                #x0 = M_prec.matvec(self.psi)

            # Run the biconjugate gradient
            t1 = time.time()
            if algorithm.lower() == "bicgstab":
                res, info = scipy.sparse.linalg.bicgstab(self.L_linop, self.psi, x0 = x0, tol = tol, maxiter = maxiter, callback=callback, M = M_prec)
            elif algorithm.lower() == "bicg":
                res, info = scipy.sparse.linalg.bicg(self.L_linop, self.psi, x0 = x0, tol = tol, maxiter = maxiter, callback=callback, M = M_prec)
            elif algorithm.lower() == "cg-minimize":
                # This algorithm minimizes f(x) = 1/2  (Lx - b) H^-1 (Lx - b)
                # where H is the matrix H = L^T L (so it is positive definite). 
                # We pick the H inverse as the inverse of the SSCHA harmonic solution.  
                # To find x and compute x = A^-1 b

                # Here we define the function that returns f(x) and its gradient
                def func(x, b):
                    # Apply
                    r = self.L_linop.matvec(x) 
                    r -= b 

                    # Apply the precondition H^-1 = (L^t L)^-1 => M M^t
                    if use_preconditioning:
                        Hinv_r = self.M_linop.rmatvec(r)
                        Hinv_r = self.M_linop.matvec(Hinv_r)
                    else:
                        Hinv_r = r

                    # Now we get the gradient
                    gradient = self.L_linop.rmatvec(Hinv_r) 
                    
                    # We get the function
                    f = 0.5 * np.sum(r * Hinv_r)

                    if verbose:
                        print("Evaluated function: value = {} | norm gradient = {}".format(f, np.sum(gradient**2)))
                        print()

                    return f, gradient

                
                psi_vector = self.psi.copy() 

                # Setup the minimization parameters
                options = {"gtol" : tol, "maxiter" : maxiter, "disp" : verbose, "norm" : 2}
                
                # Start the minimization
                results = scipy.optimize.minimize(func, x0, args = (psi_vector), method = "bfgs", jac = True, options=options)

                # Get the number of iterations
                j[0] = results.nit

                # Check the success
                if results.success:
                    info = 0
                else:
                    info = 1

                if verbose:
                    print("Minimization terminated after {} evaluations.".format(results.nfev))

                # Get the result
                res = results.x.copy()
            else:
                raise ValueError("""
Error, algorithm type '{}' in subroutine run_biconjugate_gradient not implemented.
       the only supported algorithms are ['bicgstab', 'bicg']
""".format(algorithm))
            t2 = time.time()

            if  verbose:
                print()
                print("Time to solve the linear system: {} s".format(t2 - t1))
                print()

            # Check if the minimization converged
            assert info >= 0, "Error on input or breakdown of biconjugate gradient algorithm (info = {})".format(info)

            if info > 0:
                print("The biconjugate gradient (step {}) algorithm did not converge after {} iterations.".format(i+1, maxiter))
                print("Try to either reduce the tollerance or increase the number of iteriations")
                print()
            else:
                print("The biconjugate gradient converged after {} iterations.".format(j[0]))
            

            G_one_phonon[i, :] = res[:self.n_modes]
            if i % save_each == 0:
                if save_g is not None:
                    np.save(save_g, G_one_phonon)
            
        
        if verbose:
            print()
            print(" ================================== ")
            print(" THE BICONJUGATE GRADIENT CONVERGED ")
            print(" ================================== ")
            print()
            print()

            
        if save_g is not None:
            np.save(save_g, G_one_phonon)

        # Check the hermitianeity
        disp = np.max(np.abs(G_one_phonon - G_one_phonon.T))
        assert disp < 1e-4, "Error, the resulting one-phonon Green function is not Hermitian."

        # Force hermitianity
        G = 0.5 * (G_one_phonon + G_one_phonon.T)
        
        # Invert the green function to get the Hessian Matrix (mass-rescaled)
        G_inv = np.linalg.inv(G) 

        return G_inv


    def run_hessian_calculation(self, verbose = True, eigen_shift = 1, tol = 5e-4, max_iters = 1000, save_g = None, save_each = 1, use_preconditioning = True, algorithm = "cg"):
        r"""
        STATIC RESPONSE
        ===============

        Get the static suscieptibility with the conjugate gradient algorithm. 
        Then the hessian matrix is obtained inverting the static suscieptibility.

        Parameters
        ----------
            verbose : bool
                If true, print the status of the algorithm in standard output
            eigen_shift : float
                The eigenvalue problem is shifted by :math:`\lambda` as

                .. math ::

                    L' \rightarrow L\left[ P + (1 - P) \lambda\right]

                This does not change the final solution as we are interested in

                .. math ::

                    P L^{-1} P = P {L'}^{-1}P

                So, if the matrix is not positive definite, we can exploit this force int positive definite.

            tol : float
                Tollerance of the biconjugate gradient algorithm 
            maxiter: int
                The maximum number of iterations for the biconjugate gradient.
            save_g: string
                Path to the file on which save the green function. 
                It is saved after a number of steps specified by save_each.
            save_each: int
                Determines after how many steps to save the green function.
            use_preconditioning: bool
                If true, uses the preconditioning to solve the gradient.
                The precondition is obtained by inverting analytically only the Harmonic propagation of the
                L matrix
            algorithm: string
                The algorithm used to invert the L matrix. One between:
                   - cg : Conjugate Gradient (Default)
                It will invoke the corresponding scipy subroutine

        Results
        -------
            G_inv: ndarray(size = (n_modes, n_modes))
                This is the mass-rescaled free energy Hessian.
                Its eigenvalues are the static frequencies, that determine the structure stability.
        """

        G_one_phonon = np.zeros( (self.n_modes, self.n_modes), dtype = np.double)

        if verbose:
            print()
            print("=====================")
            print(" HESSIAN CALCULATION ")
            print("=====================")
            print()
            print("We compute the static response with the")
            print("Conjugate gradient algorithm.")
            print()

        
        # Check if the symmetries has been initialized
        if not self.initialized:
            self.prepare_symmetrization()

        # Prepare the psi vector
        # And the L operator for the efficient calculation of the static response
        self.psi = np.zeros( self.n_modes + self.n_modes*(self.n_modes + 1) // 2, dtype = TYPE_DP)

        def apply_static_L(psi):
            self.psi[:] = psi.copy()

            # Apply the shift
            self.psi[self.n_modes :] *= eigen_shift

            ret = self.apply_L1_static(self.psi)
            ret += self.apply_anharmonic_static()

            ret[self.n_modes :] *= eigen_shift
            return ret

        L_operator = scipy.sparse.linalg.LinearOperator(shape = (len(self.psi), len(self.psi)), matvec = apply_static_L, dtype = TYPE_DP)

        j = np.zeros(1, dtype = np.intc)
        x_old = self.psi.copy()
        for i in range(self.n_modes):
            if verbose:
                # Print the status
                print()
                print()
                print("NEW STEP")
                print("--------")
                print()
                print("i = {} / {}".format(i + 1, self.n_modes))
                print()
            
            # Setup the known vector
            self.psi = np.zeros(self.psi.shape, dtype = type(self.psi[0]))
            self.psi[i] = 1

            x_old[:] = self.psi
            j[0] = 0
            def callback(xk, x_old = x_old, j = j):
                if np.isnan(np.sum(xk)):
                    raise ValueError("Error, NaN value found during the Conjugate Gradient.") 
                if verbose:
                    disp = sum( (xk - x_old)**2)
                    print("CG STEP {} | solution changed by {} (tol = {})".format(j[0], disp, tol))
                    j[0] += 1
                    x_old[:] = xk
                
            # Prepare the preconditioning

            M_prec = None

            def prec_half(x):
                return self.apply_L1_static(x, inverse = True, power = 0.5)

            if use_preconditioning:
                M_prec = scipy.sparse.linalg.LinearOperator(L_operator.shape, matvec = prec_half)
            
            x0 = self.psi.copy()# / self.w[i] / self.w[i]

            # Run the biconjugate gradient
            t1 = time.time()
            if algorithm.lower() == "cg":
                if not use_preconditioning:
                    res = Tools.minimum_residual_algorithm(L_operator, self.psi.copy(), x0 = x0, precond = None, max_iters = max_iters)
                else:
                    res = Tools.minimum_residual_algorithm_precond(L_operator, self.psi.copy(), M_prec, max_iters = max_iters)
                info = 0
                #res, info = scipy.sparse.linalg.cg(L_operator, self.psi, x0 = x0)#, tol = tol, maxiter = maxiter, callback=callback, M = M_prec)
            #elif algorithm.lower() == "bicg":
            #    res, info = scipy.sparse.linalg.bicg(self.L_linop, self.psi, x0 = x0, tol = tol, maxiter = maxiter, callback=callback, M = M_prec)
            elif algorithm.lower() == "minimize" or "minimize-quad":
                # This algorithm minimizes f(x) = 1/2  (Lx - b) H^-1 (Lx - b)
                # where H is the matrix H = L^T L (so it is positive definite). 
                # We pick the H inverse as the inverse of the SSCHA harmonic solution.  
                # To find x and compute x = A^-1 b

                # Here we define the function that returns f(x) and its gradient
                def func_quad(x, b):
                    # Apply
                    Lx = L_operator.matvec(x) 

                    # Apply the precondition H^-1 = (L^t L)^-1 => M M^t
                    # if use_preconditioning:
                    #     Hinv_r = self.M_linop.rmatvec(r)
                    #     Hinv_r = self.M_linop.matvec(Hinv_r)
                    # else:
                    #     Hinv_r = r

                    # Now we get the gradient
                    r = Lx - b

                    gradient = L_operator.matvec(r)
                    
                    # We get the function
                    f = 0.5 * np.sum(r**2)

                    if verbose:
                        print("Evaluated function: value = {} | norm gradient = {}".format(f, np.sum(gradient**2)))
                        print()

                    return f, gradient

                def func_standard(x, b):
                    # Apply
                    Lx = L_operator.matvec(x) 

                    # Apply the precondition H^-1 = (L^t L)^-1 => M M^t
                    # if use_preconditioning:
                    #     Hinv_r = self.M_linop.rmatvec(r)
                    #     Hinv_r = self.M_linop.matvec(Hinv_r)
                    # else:
                    #     Hinv_r = r

                    # Now we get the gradient
                    r = Lx - b

                    gradient = r
                    
                    # We get the function
                    f = 0.5 * x.dot(Lx) - b.dot(x) 

                    if verbose:
                        print("Evaluated function: value = {} | norm gradient = {}".format(f, np.sum(gradient**2)))
                        print()

                    return f, gradient
                
                psi_vector = self.psi.copy() 

                # Setup the minimization parameters
                options = {"gtol" : tol, "maxiter" : max_iters, "disp" : verbose, "norm" : 2}
                
                if algorithm.lower() == "minimize":
                    results = scipy.optimize.minimize(func_standard, x0, args = (psi_vector), method = "bfgs", jac = True, options=options)
                elif algorithm.lower() == "minimize-quad":
                    results = scipy.optimize.minimize(func_quad, x0, args = (psi_vector), method = "bfgs", jac = True, options=options)
                    

                # Start the minimization

                # Get the number of iterations
                j[0] = results.nit

                # Check the success
                if results.success:
                    info = 0
                else:
                    info = 1

                if verbose:
                    print("Minimization terminated after {} evaluations.".format(results.nfev))

                # Get the result
                res = results.x.copy()
            else:
                raise ValueError("""
Error, algorithm type '{}' in subroutine run_biconjugate_gradient not implemented.
       the only supported algorithms are ['bicgstab', 'bicg']
""".format(algorithm))
            t2 = time.time()

            if  verbose:
                print()
                print("Time to solve the linear system: {} s".format(t2 - t1))
                print()

            # Check if the minimization converged
            assert info >= 0, "Error on input or breakdown of biconjugate gradient algorithm (info = {})".format(info)

            if info > 0:
                print("The biconjugate gradient (step {}) algorithm did not converge after {} iterations.".format(i+1, maxiter))
                print("Try to either reduce the tollerance or increase the number of iteriations")
                print()
            else:
                print("The biconjugate gradient converged after {} iterations.".format(j[0]))
                print("res = {}".format(res))
            

            G_one_phonon[i, :] = res[:self.n_modes]
            if i % save_each == 0:
                if save_g is not None:
                    np.save(save_g, G_one_phonon)
            
        
        if verbose:
            print()
            print(" ================================ ")
            print(" THE CONJUGATE GRADIENT CONVERGED ")
            print(" ================================ ")
            print()
            print()

            
        if save_g is not None:
            np.save(save_g, G_one_phonon)

        # Check the hermitianeity
        disp = np.max(np.abs(G_one_phonon - G_one_phonon.T))
        assert disp < 1e-4, "Error, the resulting one-phonon Green function is not Hermitian."

        # Force hermitianity
        G = 0.5 * (G_one_phonon + G_one_phonon.T)
        
        # Invert the green function to get the Hessian Matrix (mass-rescaled)
        G_inv = np.linalg.inv(G) 

        return G_inv



    def run_conjugate_gradient(self, eigval = 0, n_iters = 100, thr = 1e-5, verbose = True, guess_x = None):
        r"""
        RUN THE CONJUGATE GRADIENT (WORK METHOD)
        ========================================

        The conjugate gradient is a very fast algorithm 
        that allows to compute the (in principle) exact green function. 

        Given the initial vector in the psi direction, it will edit it to obtain:
        
        .. math ::

            \left|x\right> = \left(\mathcal{L} - I\lambda\right)^{-1} \left| {\psi} \right> 

        At the end of the algorithm, the self.psi variable will contain the :math:`\left|x\right>`
        vector.

        Parameters
        ----------
            eigval : float
                The value of the :math:`\lambda` for the inversion problem.
            n_iters : int
                The number of iterations
            thr : float
                The threshold between two iteration after which the algorithm is considered 
                to be converged.
            verbose : bool
                If true print the status of the iteration.
                
        """

        if guess_x is None:
            x = self.psi.copy()
        else:
            x = guess_x.copy()
        A_x = self.apply_full_L()

        r = x - A_x
        p = r.copy()

        mod_r = np.sqrt(r.dot(r))

        if verbose:
            print("   CG step %d : residual = %.4e | threshold = %.4e" % (0,mod_r, thr))

        if mod_r < thr:
            return x

        for i in range(n_iters):
            self.psi = p
            A_p = self.apply_full_L()
            alpha = r.dot(r) / p.dot(A_p)
            x += alpha * p 

            # Update
            r_new = r - alpha * A_p 
            beta = r_new.dot(r_new) / r.dot(r)

            r = r_new
            p = r + beta * p 


            # Check the new iteration
            mod_r = np.sqrt(r.dot(r))
            if verbose:
                print("   CG step %d : residual = %.4e | threshold = %.4e" % (i+1,mod_r, thr))

            if mod_r < thr:
                return x

        print("WARNING: CG ended before the convergence was achieved.") 
        return x

    def get_statical_responce_from_scratch(self, n_iters = 100, thr = 1e-5, verbose = True, sub_block = None, sub_space = None):
        """
        GET STATIC RESPONCE
        ===================

        This algorithm performs the CG minimization to obtain the static self-energy.

        Parameters
        ----------
            n_iters : int
                The number of maximum iteration for a single CG step.
            thr : float
                The threshold for the convergence of the CG algorithm.
            verbose : bool
                If true (default) prints the info during the minimization
            sub_block : list
                A list of indices that identifies the modes id that that you want to select.
                The algorithm is performed only in the reduced space of (N_modes x N_modes)
                given by the length of this list. 
                In this way you will neglect mode interaction, but you can save a lot of time.
                Leave as None if you want the whole space.
            sub_space : ndarray(size = (N_dim, 3*n_at))
                Compute the self-energy only in the subspace given. Leave it as none
                if you do not want to use this option

        Results
        -------
            fc_matrix : ndarray (size=(3*nat, 3*nat))
                The static self-energy.
        """

        n_dim_space = self.n_modes


        PLinvP = np.zeros((n_dim_space, n_dim_space), dtype = TYPE_DP, order = "C")

        if (not sub_block is None) and (not sub_space is None):
            raise ValueError("Error, you cannot specify both sub_block and sub_space.")

        if not sub_block is None:
            n_dim_space = len(sub_block)
        elif not sub_space is None:
            n_dim_space = len(sub_space)
            
        # initialize the algorithm
        for i in range(n_dim_space):
            # Setup the vector
            self.psi = np.zeros((self.n_modes + 1)*self.n_modes, dtype = TYPE_DP)
            guess = self.psi.copy()

            # Create the subbasis
            if not sub_block is None:
                self.psi[sub_block[i]] = 1
                guess[sub_block[i]] = self.w[sub_block[i]] ** 2
            elif not sub_space is None:
                self.psi[:self.n_modes] = sub_block[i].dot(self.pols)
                guess = None
            else:
                self.psi[i] = 1
                guess[i] = self.w[i] ** 2

            if verbose:
                print("")
                print("==== NEW STATIC COMPUTATION ====")
                print("Iteration: %d out of %d" % (i+1, n_dim_space))


            new_v = self.run_conjugate_gradient(n_iters = n_iters, thr = thr, verbose = verbose, guess_x = guess)
            
            if not sub_space is None:
                v_out = self.pols.dot(new_v[:self.n_modes])
                PLinvP[i, :] = np.array(sub_space).dot(v_out)
            else:
                PLinvP[i, :] = new_v[:self.n_modes]

        

        # Invert the P L^-1 P 
        D = np.linalg.inv(PLinvP)
        # Transform to a force constant matrix in cartesian coordinates

        if not sub_space is None:
            fc_matrix = np.einsum("ab, ai, bj->ij", D, np.array(sub_space), np.array(sub_space))
        else:
            fc_matrix = np.einsum("ab, ia, jb->ij", D, self.pols, self.pols)
        fc_matrix *= np.sqrt(np.outer(self.m, self.m))

        return fc_matrix




    def run(self, n_iter, save_dir = ".", verbose = True):
        """
        RUN LANCZOS ITERATIONS
        ======================

        This method performs the Lanczos algorithm to find
        the sequence of a and b coefficients that are the tridiagonal representation 
        of the L matrix to be inverted.

        Parameters
        ----------
            n_iter : int
                The number of iterations to be performed in the Lanczos algorithm.
            save_dir : string
                The directory in which you want to store the results step by step,
                in order to do a preliminar analysis or restart the calculation later.
            verbose : bool
                If true all the info during the minimization will be printed on output.
        """

        raise ValueError("Error, this run funciton has been deprecated, use run_FT instead.")

        # Check if the symmetries has been initialized
        if not self.initialized:
            self.prepare_symmetrization()

        # Get the current step
        i_step = len(self.a_coeffs)

        if verbose:
            header = """
<=====================================>
|                                     |
|          LANCZOS ALGORITHM          |
|                                     |
<=====================================>

Starting the algorithm. It may take a while.
Starting from step %d
""" % i_step
            print(header)

            OPTIONS = """
Should I ignore the third order effect? {}
Should I ignore the fourth order effect? {}
Max number of iterations: {}
""".format(self.ignore_v3, self.ignore_v4, n_iter)
            print(OPTIONS)


        # If this is the current step initialize the algorithm
        if i_step == 0:
            self.krilov_basis = []
            first_vector = self.psi / np.sqrt(self.psi.dot(self.psi))
            self.krilov_basis.append(first_vector)
        else:
            # Convert everything in a list
            self.krilov_basis = list(self.krilov_basis)
            self.a_coeffs = list(self.a_coeffs)
            self.b_coeffs = list(self.b_coeffs)
            self.arnoldi_matrix = list(self.arnoldi_matrix)

            if len(self.krilov_basis) != i_step + 1:
                print("Krilov dim: %d, number of steps perfomed: %d" % (len(self.krilov_basis), i_step))
                print("Error, the krilov basis dimension should be 1 more than the number of steps")
                raise ValueError("Error the starting krilov basis does not matches the matrix, Look stdout.")

        self.psi = self.krilov_basis[-1]

        for i in range(i_step, i_step+n_iter):
            if verbose:
                step_txt = """
 ===== NEW STEP %d =====

 """ % i
                print(step_txt)
                print("Length of the coefficiets: a = {}, b = {}".format(len(self.a_coeffs), len(self.b_coeffs)))
                print()

            # Apply the matrix L
            t1 = time.time()
            #self.psi = self.apply_full_L()
            self.psi = self.L_linop.dot(self.psi)
            t2 = time.time()

            if verbose:
                print("Time to apply the full L: %d s" % (t2 -t1))

            # Get the coefficients for the Lanczos/Arnoldi matrix
            t1 = time.time()
            arnoldi_row = []
            new_vect = self.psi.copy()


            # Lets repeat twice the orthogonalization
            converged = False
            for k_orth in range(N_REP_ORTH):
                for j in range(len(self.krilov_basis)):
                    coeff = new_vect.dot(self.krilov_basis[j])
                    if k_orth == 0:
                        arnoldi_row.append(self.psi.dot(self.krilov_basis[j]))

                    # Gram Schmidt
                    new_vect -= coeff * self.krilov_basis[j]
            
                # Add the new vector to the Krilov Basis
                norm = np.sqrt(new_vect.dot(new_vect))

                if verbose:
                    print("Vector norm after GS number {}: {:16.8e}".format(k_orth+1, norm))

                # Check the normalization (If zero the algorithm converged)
                if norm < __EPSILON__:
                    converged = True
                    if verbose:
                        print("Obtained a linear dependent vector.")
                        print("The algorithm converged.")
                    break
                
                new_vect /= norm 

            if not converged:
                self.krilov_basis.append(new_vect)
                self.psi = new_vect
            t2 = time.time()

            # Add the coefficients to the variables
            self.a_coeffs.append(arnoldi_row[-1])
            if len(arnoldi_row) > 1:
                self.b_coeffs.append(arnoldi_row[-2])
            self.arnoldi_matrix.append(arnoldi_row)

            if verbose:
                print("Time to perform the Gram-Schmidt and retrive the coefficients: %d s" % (t2-t1))
                print()
                print("a_%d = %.8e" % (i, self.a_coeffs[-1]))
                if i > 0:
                    print("b_%d = %.8e" % (i, self.b_coeffs[-1]))
                print()
            
            # Save the step
            if not save_dir is None:
                self.save_status("%s/LANCZOS_STEP%d" % (save_dir, i))
        
                if verbose:
                    print("Status saved into '%s/LANCZOS_STEP%d'" % (save_dir, i))
            
            if verbose:
                print("Lanczos step %d ultimated." % i)
            

            if converged:
                return


    def build_lanczos_matrix_from_coeffs(self, use_arnoldi=False):
        """
        BUILD THE LANCZOS MATRIX
        ========================

        This method builds the Lanczos matrix from the coefficients. 
        To execute this method correctly you must have already completed the Lanczos algorithm (method run)

        Parameters
        ----------
            use_arnoldi: bool
                If true the full matrix is computed, using all the coefficients from the
                Arnoldi iteration.
        """

        N_size = len(self.a_coeffs)
        matrix = np.zeros((N_size, N_size), dtype = TYPE_DP)
        if not use_arnoldi:
            for i in range(N_size):
                matrix[i,i] = self.a_coeffs[i]
                if i>= 1:
                    # Use the non-symmetric Lanczos if also c_coeffs are present
                    c_coeff = self.b_coeffs[i-1]
                    if len(self.c_coeffs) == len(self.b_coeffs):
                        c_coeff = self.c_coeffs[i-1]
                    matrix[i-1,i] = c_coeff
                    matrix[i,i-1] = self.b_coeffs[i-1]
        else:
            # Check if there are c_coeffs, in this way arnoldi matrix is not computed
            assert len(self.b_coeffs) > len(self.c_coeffs), "Error, cannot Arnoldi with non-symmetric Lanczos not implemented"
            for i in range(N_size):
                matrix[:i+1, i] = self.arnoldi_matrix[i]
                matrix[i, :i+1] = self.arnoldi_matrix[i]


        sign = 1
        if self.reverse_L:
            sign = -1

        matrix =  sign*matrix - sign* np.eye(N_size) * self.shift_value
                    
        return matrix


    def get_green_function_Lenmann(self, w_array, smearing, v_a, v_b, use_arnoldi = False):
        """
        GET GREEN FUNCTION
        ==================

        Compute the green function using the Lemman representation.

        Parameters
        ----------
            w_array : ndarray
                The list of frequencies in RY for which you want to compute the
                dynamical green function.
            smearing : float
                The smearing in RY to take a non zero imaginary part.
            v_a : ndarray(size = 3*self.nat)
                The perturbation operator (on atomic positions)
            v_b : ndarray(size = 3*self.nat)
                The probed responce operator (on atomic positions)
            use_arnoldi: bool
                If true the full arnoldi matrix is used to extract eigenvalues and 
                eigenvectors. Otherwise the tridiagonal Lanczos matrix is used.
                The first one prevents the loss of orthogonality problem.
        """

        # Get the Lanczos matrix
        matrix = self.build_lanczos_matrix_from_coeffs(use_arnoldi)

        assert len(self.c_coeffs) < len(self.b_coeffs), "Lenmann cannot be used with non-symmetric Lanczos"

        # Convert the vectors in the polarization basis
        new_va = np.einsum("a, a, ab->b", 1/np.sqrt(self.m), v_a, self.pols)
        new_vb = np.einsum("a, a, ab->b", 1/np.sqrt(self.m), v_b, self.pols)

        # Dyagonalize the Lanczos matrix
        eigvals, eigvects = np.linalg.eigh(matrix)

        kb = np.array(self.krilov_basis)
        kb = kb[:-1,:]
        #print (np.shape(eigvects), np.shape(kb))
        # Convert in krilov space
        new_eigv = np.einsum("ab, ac->cb", eigvects, kb)


        Na, Nb = np.shape(matrix)
        if Na != Nb:
            raise ValueError("Error, the Lanczos matrix must be square, dim (%d,%d)" % (Na, Nb))
        
        gf = np.zeros(len(w_array), dtype = np.complex128)

        for j in range(Na):
            eig_v = new_eigv[:self.n_modes, j]
            matrix_element = eig_v.dot(new_va) * new_vb.dot(eig_v)
            gf[:] += matrix_element / (eigvals[j]  - w_array**2 + 2j*w_array*smearing)

        return gf

    def get_static_odd_fc(self, use_arnoldi = False):
        """
        GET STATIC FORCE CONSTANT
        =========================

        Get the static force constant matrix

        Parameters
        ----------
            use_arnoldi: bool
                If true the full arnoldi matrix is used, otherwise the Lanczos tridiagonal
                matrix is used.
        """

        # Get the Lanczos matrix
        matrix = self.build_lanczos_matrix_from_coeffs(use_arnoldi)

        # Dyagonalize the Lanczos matrix
        eigvals, eigvects = np.linalg.eigh(matrix)

        Nk = len(self.krilov_basis)

        kb = np.array(self.krilov_basis)
        
        # Lanczos did not converged, discard the last vector
        if Nk > len(eigvals):
            kb = kb[:-1,:]

        #print (np.shape(eigvects), np.shape(kb))
        new_eigv = np.einsum("ab, ac->cb", eigvects, kb)

        Na, Nb = np.shape(matrix)
        if Na != Nb:
            raise ValueError("Error, the Lanczos matrix must be square, dim (%d,%d)" % (Na, Nb))
        

        fc_matrix = np.zeros( (3*self.nat, 3*self.nat), dtype = TYPE_DP)

        # Get the dynamical matrix in the polarization basis
        D = np.einsum("ai, bi, i->ab", new_eigv[:self.n_modes,:], new_eigv[:self.n_modes, :], eigvals)

        # Convert it in the standard basis
        fc_matrix = np.einsum("ab, ia, jb->ij", D, self.pols, self.pols)

        # for i in range(3*self.nat):
        #     # Define the vector
        #     v = np.zeros(3*self.nat, dtype = TYPE_DP)
        #     v[i] = 1

        #     # Convert the vectors in the polarization basis
        #     new_v = np.einsum("a, a, ab->b", np.sqrt(self.m), v, self.pols)
        #     # Convert in the krilov space 
        #     mat_coeff = np.einsum("a, ab", new_v, new_eigv[:self.n_modes, :])
        #     new_w = np.einsum("a, ba, a", mat_coeff, new_eigv[:self.n_modes,:], eigvals)

        #     #v_kb = np.einsum("ab, b", kb[:, :self.n_modes], new_v)
        #     # Apply the L matrix
        #     #w_kb = matrix.dot(v_kb)
        #     # Convert back in the polarization space
        #     #new_w = np.einsum("ab, a", kb[:, :self.n_modes], w_kb)
        #     # Convert back in real spaceDoes anyone know if there is a windows binary or a source code to run QE with GPU enhancement on windows. 
        #     w = np.einsum("a, b, ab ->a", 1/np.sqrt(self.m), new_w, self.pols)

        #     fc_matrix[i, :] = w
            

        # This is the dynamical matrix now we can multiply by the masses
        fc_matrix *= np.sqrt(np.outer(self.m, self.m))

        return fc_matrix


    def get_all_green_functions(self, N_steps = 100, mode_mixing = True, save_step_dir = None, verbose = True):
        """
        GET ALL THE GREEN FUNCTIONS
        ===========================

        This will compute a set of lanczos coefficients for each element of the odd matrix.
        a_n and b_n.
        We will run lanczos for all the elements and all the crosses.
        In this way we have the whole evolution with frequency of the matrix.

        NOTE: This can be a very intensive computation.

        Parameters
        ----------
            N_steps : int
                The number of Lanczos iteration for each green function
            mode_mixing : bool
                If True also non diagonal elements are computed, otherwise the 
                SSCHA eigenvector are supposed to be conserved, and only diagonal
                green functions are considered.
                If False the computation is much less expensive (a factor nat_sc),
                but it is approximated.
            save_step_dir : string
                If not None, the path to the directory in which you want to save 
                each step. So even if stopped the calculation can restart.
            verbose : bool
                If true print all the progress to standard output

        Results
        -------
            a_ns : ndarray( (n_modes, n_modes, N_steps))
                The a coefficients for each element in the mode x mode space
            b_ns : ndarray( (n_modes, n_modes, N_steps-1))
                The b_n coefficients for each mode in the space.
        """

        # Time the function
        t_start = time.time()

        # Check if the save directory exists
        # Otherwise we create it
        if not save_step_dir is None:
            if not os.path.exists(save_step_dir):
                makedirs(save_step_dir)

        # Load all the data
        a_ns = np.zeros( (self.n_modes, self.n_modes, N_steps), dtype = np.double)
        b_ns = np.zeros( (self.n_modes, self.n_modes, N_steps-1), dtype = np.double)

        # Incompatible with shift for now
        self.shift_value = 0

        # Compute the diagonal parts
        for i in range(self.n_modes):
            if verbose:
                print("\n")
                print("  ==========================  ")
                print("  |                        |  ")
                print("  |   DIAGONAL ELEMENTS    |  ")
                print("  |       STEP {:5d}       |  ".format(i))
                print("  |                        |  ")
                print("  ==========================  ")
                print()
            
            # Setup the Lanczos
            self.reset()

            # Prepare the perturbation
            self.psi[:] = 0
            self.psi[i] = 1

            # Run the Lanczos perturbation
            self.run(N_steps, save_dir = save_step_dir, verbose = verbose)

            if verbose:
                print()
                print("   ---- > LANCZOS RUN COMPLEATED < ----   ")
                print()

            # Save the status
            if save_step_dir:
                self.save_status("full_lanczos_diagonal_{}".format(i))
            
            # Fill the a_n and b_n
            a_tmp = np.zeros(N_steps, dtype = np.double)
            a_tmp[:len(self.a_coeffs)] = self.a_coeffs
            b_tmp = np.zeros(N_steps-1, dtype = np.double)
            b_tmp[:len(self.b_coeffs)] = self.b_coeffs
            a_ns[i, i, :] = a_tmp
            b_ns[i, i, :] = b_tmp
    
        # If we must compute the mode mixing
        if mode_mixing:
            for i in range(self.n_modes):
                for j in range(i+1, self.n_modes):
                    # TODO: Neglect (i,j) forbidden by symmetries

                    if verbose:
                        print("\n")
                        print("  ============================  ")
                        print("  |                          |  ")
                        print("  |   NON DIAGONAL ELEMENT   |  ")
                        print("  |    STEP ({:5d},{:5d})    |  ".format(i, j))
                        print("  |                          |  ")
                        print("  ============================  ")
                        print()
                    
                    # Setup the Lanczos
                    self.reset()

                    # Prepare the perturbation
                    self.psi[:] = 0
                    self.psi[i] = 1
                    self.psi[j] = 1

                    # Run the Lanczos perturbation
                    self.run(N_steps, save_dir = save_step_dir, verbose = verbose)

                    if verbose:
                        print()
                        print("   ---- > LANCZOS RUN COMPLEATED < ----   ")
                        print()

                    # Save the status
                    if save_step_dir:
                        self.save_status("full_lanczos_off_diagonal_{}_{}".format(i, j))
                    
                    # Fill the a_n and b_n
                    a_tmp = np.zeros(N_steps, dtype = np.double)
                    a_tmp[:len(self.a_coeffs)] = self.a_coeffs
                    b_tmp = np.zeros(N_steps-1, dtype = np.double)
                    b_tmp[:len(self.b_coeffs)] = self.b_coeffs
                    a_ns[i, j, :] = a_tmp
                    b_ns[i, j, :] = b_tmp
                    a_ns[j, i, :] = a_tmp
                    b_ns[j, i, :] = b_tmp

        t_end = time.time()

        total_time = t_end - t_start
        minutes = int(total_time / 60)
        hours = int(minutes / 60)
        minutes -= hours * 60
        seconds = int(total_time - hours*3600 - minutes * 60)

        if verbose:
            print()
            print()
            print("     ======================     ")
            print("     |                    |     ")
            print("     |        DONE        |      ")
            print("     |   In {:3d}:{:02d}:{:02d}s  |     ".format(hours, minutes, seconds))
            print("     ======================     ")
            print()
            print()
            
        return a_ns, b_ns



    def get_spectral_function_from_Lenmann(self, w_array, smearing, use_arnoldi=True):
        """
        GET SPECTRAL FUNCTION
        =====================

        This method computes the spectral function in the supercell
        using the Lenmann representation.

        Parameters
        ----------
            w_array : ndarray
                The list of frequencies for which you want to compute the
                dynamical green function.
            smearing : float
                The smearing to take a non zero imaginary part.
            use_arnoldi: bool
                If true the full arnoldi matrix is used to extract eigenvalues and 
                eigenvectors. Otherwise the tridiagonal Lanczos matrix is used.
                The first one prevents the loss of orthogonality problem.
        """
        # Get the Lanczos matrix
        matrix = self.build_lanczos_matrix_from_coeffs(use_arnoldi)

        # Dyagonalize the Lanczos matrix
        eigvals, eigvects = np.linalg.eigh(matrix)

        Na, Nb = np.shape(matrix)
        if Na != Nb:
            raise ValueError("Error, the Lanczos matrix must be square, dim (%d,%d)" % (Na, Nb))
        
        spectral = np.zeros(len(w_array), dtype = np.complex128)


        kb = np.array(self.krilov_basis)
        if np.shape(kb)[0] > Na:
            kb = kb[:-1,:]
        print ("Shape check: eigvects = {}, kb = {}".format( np.shape(eigvects), np.shape(kb)))
        new_eigv = np.einsum("ab, ac->cb", eigvects, kb)
        # TODO: Update for Lanczos biconjugate

        for j in range(Na):
            eig_v = new_eigv[:self.n_modes, j]
            matrix_element = np.conj(eig_v).dot(eig_v)
            spectral[:] += matrix_element / (eigvals[j]  - w_array**2 +2j*w_array*smearing)

        return -np.imag(spectral)


    def get_green_function_continued_fraction(self, w_array : np.ndarray[np.float64], use_terminator : bool = True, last_average: int = 1, smearing : np.float64 = 0):
        r"""
        CONTINUED FRACTION GREEN FUNCTION
        =================================

        In this way the continued fraction for the green function is used.
        This should converge faster than the Lenmann representation, and
        has the advantage of adding the possibility to add a terminator.
        This avoids to define a smearing.
        
        NORMAL: we invert:
        :: math .
            <p|(-\mathcal{L} - \omega^2)^{-1}|q>,
        
        WIGNER: we invert:
        :: math .
            <p|(\mathcal{L}_w + \omega^2)^{-1}|q>
            
        So in the continued fraction we have -/+ in front of the freqeuncy depending on the formalism used.

        Parameters
        ----------
            w_array : ndarray
                The list of frequencies in RY in which you want to compute the green function
            use_terminator : bool
                If true (default) a standard terminator is used.
            last_average : int
                How many a and b coefficients are averaged to evaluate the terminator?
            smearing : float
                The smearing parameter in RY. If none
        """
        n_iters = len(self.a_coeffs)

        gf = np.zeros(np.shape(w_array), dtype = np.complex128)
        
        sign = 1
        if self.reverse_L:
            sign = -1
            
        INFO = """
GREEN FUNCTION FROM CONTINUED FRACTION
Am I using Wigner? {}
Should I use the terminator? {}
Perturbation modulus = {}
Sign = {}""".format(self.use_wigner, use_terminator, self.perturbation_modulus, sign)
        
        print()
        print(INFO)

        # Get the terminator
        if use_terminator:
            # Get the last coeffs averaging
            a_av = np.mean(self.a_coeffs[-last_average:])
            b_av = np.mean(self.b_coeffs[-last_average:])
            c_av = b_av
            # Non-symmetric Lanczos
            if len(self.c_coeffs) == len(self.b_coeffs):
                c_av = np.mean(self.c_coeffs[-last_average:])

            a = a_av * sign - sign* self.shift_value
            b = b_av * sign
            c = c_av * sign
            
            if not self.use_wigner:
                gf[:] = (a - w_array**2 - np.sqrt( (a - w_array**2)**2 - 4*b*c + 0j))/(2*b*c)
            else:
                # Wigner
                gf[:] = (a + w_array**2 + np.sqrt( (a + w_array**2)**2 - 4*b*c + 0j))/(2*b*c)        
        else:
            # If we do not use the Terminator we get the last fraction
            a = self.a_coeffs[-1] * sign - sign* self.shift_value
            if not self.use_wigner:
                gf[:] = 1/ (a - w_array**2 + 2j*w_array*smearing)
            else:
                # Wigner
                gf[:] = 1/ (a + w_array**2 + 2j*w_array*smearing)

        # Continued fraction
        for i in range(n_iters-2, -1, -1):
            # Start getting the continued fraction from the last coeff
            a = self.a_coeffs[i] * sign - sign * self.shift_value
            b = self.b_coeffs[i] * sign
            c = b
            if len(self.c_coeffs) == len(self.b_coeffs): 
                c = self.c_coeffs[i] * sign
               
            if not self.use_wigner:
                gf = 1. / (a - w_array**2  + 2j*w_array*smearing - b * c * gf)
            else:
                # In Wigner we invert L + omega^2
                gf = 1. / (a + w_array**2  + 2j*w_array*smearing - b * c * gf)
         
        if not self.use_wigner:
            return gf * self.perturbation_modulus
        else:
            # Wigner
            return (-np.real(gf) + 1j * np.imag(gf)) * self.perturbation_modulus


        
           
    def get_full_L_debug_wigner(self, verbose = False, debug_d3 = None, symmetrize = True, overwrite_L_operator = True):
        """
        GET THE FULL L OPERATOR FOR DEBUG WIGNER
        ========================================
        Use this method to test if the change of variables that defines the symmetric Wigner representation works.
        
        Note: make sure that self.use_wigner is Fasle so when we compute the Green function
        we get the correct sign in front of omega in the contnued fraction.
        
        DO NOT USE FOR PRODUCTION: IT IS DANGEROUS

        Results
        -------
           L_op : ndarray(size = (nmodes * (2*nmodes + 1)), dtype = TYPE_DP)
              The full L operator.
        """
        if self.use_wigner:
            raise ErrorValue('Please make sure that use_wigner is False')
            
        # Avoid the dependent freqeuncies
        i_a = np.tile(np.arange(self.n_modes), (self.n_modes,1)).ravel()
        i_b = np.tile(np.arange(self.n_modes), (self.n_modes,1)).T.ravel()

        new_i_a = np.array([i_a[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        new_i_b = np.array([i_b[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        
        w_a = self.w[new_i_a]
        w_b = self.w[new_i_b]
        
        # N_w2 is the number of independent indeces
        N_w2 = len(w_a)

        if verbose:
            print()
            print('Getting the DEBUG WIGNER L operator')
        # Prepare the operator
        L_operator = np.zeros(shape = (self.n_modes + 2*N_w2, self.n_modes + 2*N_w2), dtype = TYPE_DP)
            

        n_a = np.zeros(np.shape(w_a), dtype = TYPE_DP)
        n_b = np.zeros(np.shape(w_a), dtype = TYPE_DP)
        if self.T > 0:
            n_a = 1 / (np.exp(w_a * 157887.32400374097/self.T) - 1)
            n_b = 1 / (np.exp(w_b * 157887.32400374097/self.T) - 1)

        if verbose:
            print("BE occ number", n_a[:self.n_modes])

        # Apply the non interacting X operator
        start_Y = self.n_modes
        start_A = self.n_modes + N_w2

        if not self.ignore_harmonic:
            if verbose:
                print('DEBUG WIGNER harmonic R(1) a(1) b(1)')
            # Harmonic evolution for R -> R sector
            L_operator[:self.n_modes, :self.n_modes] = +np.diag(self.w**2)
            
            # Harmonic evolution for a -> a sector
            a_a = w_a**2 + w_b**2 - 2 * w_a * w_b
            L_operator[start_Y: start_A, start_Y: start_A] = +np.diag(a_a)

            # Harmonic evolution for b -> b sector
            b_b = w_a**2 + w_b**2 + 2 * w_a * w_b
            L_operator[start_A:, start_A:] = +np.diag(b_b)
                

        # We ADDED all the non interacting (harmonic) propagators both for debug Wigner
        # In Wigner the Harmonic approach is working

        # Compute the d3 operator
        if debug_d3 is None:
            N_eff = np.sum(self.rho)
            Y_weighted = np.einsum("ia, i -> ia", self.Y, self.rho)
        if not self.ignore_v3:
            if verbose:
                print("Computing d3...")
            if not debug_d3 is None:
                d3 = debug_d3
            else:
                X_ups = np.einsum("ia, a -> ia", self.X, f_ups(self.w, self.T))

                d3_noperm = np.einsum("ia, ib, ic -> abc", X_ups, X_ups, Y_weighted)
                d3_noperm /= -N_eff 

                # Apply the permuatations
                d3 = d3_noperm.copy()
                d3 += np.einsum("abc->acb", d3_noperm)
                d3 += np.einsum("abc->bac", d3_noperm)
                d3 += np.einsum("abc->bca", d3_noperm)
                d3 += np.einsum("abc->cab", d3_noperm)
                d3 += np.einsum("abc->cba", d3_noperm)
                d3 /= 6

                if verbose:
                    np.save("d3_modes_nosym.npy", d3)

                # Perform the standard symmetrization
                if symmetrize:
                    # TODO: fix the symmetrize_d3_muspace
                    raise NotImplementedError('The symmetrizaiton on d3 is not implemented')
                    d3 = symmetrize_d3_muspace(d3, self.symmetries)

                    if verbose:
                        np.save("d3_modes_sym.npy", d3)
                        np.save("symmetries_modes.npy", self.symmetries)
                

            # Reshape the d3
            d3_small_space = np.zeros((N_w2, self.n_modes), dtype = np.double)
            # Get the d3 in the small space by getting the independent terms
            d3_small_space[:,:] = d3[new_i_a, new_i_b, :]

            if verbose:
                print("D3 of the following elements:")
                print(new_i_a)
                print(new_i_b)
                print("D3 small space")
                print(d3_small_space)
                print('D3 complete')
                print(d3)
        
            if verbose:
                print('DEBUG WIGNER getting the D3 contribution')
            # Chi for the independent indeces
            L2_minus = - (2 * (n_a - n_b) * (w_a - w_b)) /(w_a * w_b * (2 * n_a + 1) *  (2 * n_b + 1)) 
            L2_plus  = + (2 * (1 + n_a + n_b) * (w_a + w_b)) /(w_a * w_b * (2 * n_a + 1) *  (2 * n_b + 1)) 
            # X for the independent indices
            X = ((2 * n_a + 1) *  (2 * n_b + 1))/8

            # The shape of these tensors is (N_w2, n_modes) considering double counting
            extra_count_w = np.ones(N_w2, dtype = np.intc)
            extra_count_w[new_i_a != new_i_b] = 2
            d3_X  = np.einsum('ab, a -> ab', d3_small_space, X * extra_count_w)

            # The interacion between a rank 1 tensor a rank 2 tensor DOES require
            # to take into account double counting

            # The coeff betwee R(1) and a(1)
            L_operator[:start_Y, start_Y: start_A] = -d3_X.T
            # The coeff betwee R(1) and b(1)
            L_operator[:start_Y, start_A:] = +d3_X.T

            # The interaction between a rank 2 tensor a rank 1 tensor DOES NOT require
            # to take into account double counting

            # The shape of these tensors is (N_w2, n_modes)
            L2_minus_d3 = np.einsum('ab, a -> ab', d3_small_space, L2_minus)
            L2_plus_d3  = np.einsum('ab, a -> ab', d3_small_space, L2_plus)

            # The coeff betwee a(1) and R(1)
            L_operator[start_Y: start_A, :start_Y] = -L2_minus_d3
            # The coeff betwee b(1) and R(1)
            L_operator[start_A:, :start_Y] = +L2_plus_d3
             
   
        if not self.ignore_v4:
#             raise NotImplementedError('The symmetrizaiton on d4 is not implemented for debug Wigner')
            # Get the full D4 tensor in the polarization basis
            # it should be symmetric under permutations of the indices
            d4 =  np.einsum("ia, ib, ic, id -> abcd", X_ups, X_ups, X_ups, Y_weighted)
            d4 += np.einsum("ia, ib, ic, id -> abcd", X_ups, X_ups, Y_weighted, X_ups)
            d4 += np.einsum("ia, ib, ic, id -> abcd", X_ups, Y_weighted, X_ups, X_ups)
            d4 += np.einsum("ia, ib, ic, id -> abcd", Y_weighted, X_ups, X_ups, X_ups)
            d4 /= - 4 * N_eff

            if verbose:
                np.save("d4_modes_nosym.npy", d4)
            if symmetrize:
                raise NotImplementedError('The symmetrizaiton on d4 is not implemented')

            # Get the independent first two indep indices
            d4_small_space1 = np.zeros((N_w2, self.n_modes, self.n_modes), dtype = np.double)
            d4_small_space1[:,:,:] = d4[new_i_a, new_i_b, :, :]

            # Get the independent second two indep indices
            d4_small_space = np.zeros((N_w2, N_w2), dtype = np.double)
            d4_small_space[:,:] = d4_small_space1[:, new_i_a, new_i_b]
            
            if verbose:
                print("D4 of the following elements:")
                print(new_i_a)
                print(new_i_b)
                print("D4 in the reduced space")
                print(d4_small_space)
                print("D4 complete")
                print(d4)
                
            # Add the matrix elements
            L2_minus_d4_X = np.einsum('a, ab, b -> ab', L2_minus, d4_small_space, X * extra_count_w)
            L2_plus_d4_X  = np.einsum('a, ab, b -> ab', L2_plus, d4_small_space, X * extra_count_w)
            
            # Interaction of a(1)-a(1)
            L_operator[start_Y:start_A, start_Y:start_A] += L2_minus_d4_X
            
            # Interaction of b(1)-b(1)
            L_operator[start_A:, start_A:] += L2_plus_d4_X
            
            # Interaction of a(1)-b(1)
            L_operator[start_Y:start_A, start_A:] += -L2_minus_d4_X
            
            # Interaction of b(1)-a(1)
            L_operator[start_A:, start_Y:start_A] += -L2_plus_d4_X
            
                
        if verbose:
            print("L DEBUG WIGNER superoperator computed.")
            np.savez_compressed("L_super_analytical_debug_wigner.npz", L_operator)
        
        if overwrite_L_operator:
            if verbose:
                print('Overwriting the L operator with DEBUG WIGNER..')
            def matvec(x):
                return L_operator.dot(x)
            def rmatvec(x):
                return x.dot(L_operator)

            self.L_linop = scipy.sparse.linalg.LinearOperator(L_operator.shape, matvec = matvec, rmatvec = rmatvec)
        
        return L_operator

    def get_static_frequency(self, smearing: np.float64 = 0) -> np.float64:
        r"""
        GET THE STATIC FREQUENCY
        ========================

        The static frequency of a specific perturbation can be obtained as the limit of the 
        dynamical green function for w -> 0. 

        .. math ::

            \omega = \sqrt{\frac{1}{\Re G(\omega \rightarrow 0 + i\eta)}} 


        where :math:`\eta` is the smearing for the static frequency calculation.
        This frequency is the diagonal element of the free energy Hessian matrix acros the chosen perturbation.

        If :math:`\omega` is imaginary, a negative value is returned.

        Parameters
        ----------
            - smearing : float
                The smearing in Ry of the calculation

        Results
        -------
            - frequency : float
                The frequency of the perturbation :math:`\omega`
        """



        gf = self.get_green_function_continued_fraction(np.array([0]), False, smearing = smearing)

        w2 = 1 / np.real(gf)
        return np.float64(np.sqrt(np.abs(w2)) * np.sign(w2))

    def get_static_frequency(self, smearing: np.float64 = 0) -> np.float64:
        r"""
        GET THE STATIC FREQUENCY
        ========================

        The static frequency of a specific perturbation can be obtained as the limit of the 
        dynamical green function for w -> 0. 

        .. math ::

            \omega = \sqrt{\frac{1}{\Re G(\omega \rightarrow 0 + i\eta)}} 


        where :math:`\eta` is the smearing for the static frequency calculation.
        This frequency is the diagonal element of the free energy Hessian matrix acros the chosen perturbation.

        If :math:`\omega` is imaginary, a negative value is returned.

        Parameters
        ----------
            - smearing : float
                The smearing in Ry of the calculation

        Results
        -------
            - frequency : float
                The frequency of the perturbation :math:`\omega`
        """



        gf = self.get_green_function_continued_fraction(np.array([0]), False, smearing = smearing)

        w2 = 1 / np.real(gf)
        return np.float64(np.sqrt(np.abs(w2)) * np.sign(w2))

    
    
    
    def get_full_L_operator(self, verbose = False, only_pert = False, symmetrize = True, debug_d3 = None, overwrite_L_operator = True):
        """
        GET THE FULL L OPERATOR
        =======================
        
        Use this method to test everithing. I returns the full L operator as a matrix.
        It is very memory consuming, but it should be fast and practical for small systems.
        

        Results
        -------
           L_op : ndarray(size = (nmodes * (nmodes + 1)), dtype = TYPE_DP)
              The full L operator.
        """
        # The L operator
        L_operator = np.zeros( shape = (self.n_modes + self.n_modes * self.n_modes, self.n_modes + self.n_modes * self.n_modes), dtype = TYPE_DP)

        # Fill the first part with the standard dynamical matrix
        if not only_pert:
            L_operator[:self.n_modes, :self.n_modes] = np.diag(self.w**2)

        
        w_a = np.tile(self.w, (self.n_modes,1)).ravel()
        w_b = np.tile(self.w, (self.n_modes,1)).T.ravel()
        chi_beta = -.5 * np.sqrt(w_a + w_b)/(np.sqrt(w_a)*np.sqrt(w_b))


        B_mat = (w_a + w_b)**2
        if not only_pert:
            L_operator[self.n_modes:, self.n_modes:] = np.diag(B_mat)
        

        # Compute the d3 operator
#         new_X = np.einsum("ia,a->ai", self.X, f_ups(self.w, self.T))
        if debug_d3 is None:
            N_eff = np.sum(self.rho)
            Y_weighted = np.einsum("ia, i->ia", self.Y, self.rho)
            #new_Y = np.einsum("ia,i->ai", self.Y, self.rho)

        if not self.ignore_v3:
            if not debug_d3 is None:
                d3 = debug_d3
            else:
                if verbose:
                    print("Computing d3...")
                d3_noperm = np.einsum("ia,ib,ic->abc", self.X, self.X, Y_weighted)
                d3_noperm /= -N_eff 

                # Apply the permuatations
                d3 = d3_noperm.copy()
                d3 += np.einsum("abc->acb", d3_noperm)
                d3 += np.einsum("abc->bac", d3_noperm)
                d3 += np.einsum("abc->bca", d3_noperm)
                d3 += np.einsum("abc->cab", d3_noperm)
                d3 += np.einsum("abc->cba", d3_noperm)
                d3 /= 6

                if verbose:
                    np.save("d3_modes_nosym.npy", d3)
                    
                if symmetrize:
                    # Perform the standard symmetrization
                    d3 = symmetrize_d3_muspace(d3, self.symmetries)

                if verbose:
                    np.save("d3_modes_sym.npy", d3)
                    np.save("symmetries_modes.npy", self.symmetries)
            

            # Reshape the d3
            d3_reshaped = d3.reshape((self.n_modes, self.n_modes * self.n_modes))

            new_mat = np.einsum("ab,b->ab", d3_reshaped, chi_beta)

            L_operator[:self.n_modes, self.n_modes:] = new_mat
            L_operator[self.n_modes:, :self.n_modes] = new_mat.T
            
        if not self.ignore_v4:
            if verbose:
                print("Computing d4...")
            d4 =  np.einsum("ai,bi,ci,di", new_X, new_X, new_X, new_Y)
            d4 += np.einsum("ai,bi,ci,di", new_X, new_X, new_Y, new_X)
            d4 += np.einsum("ai,bi,ci,di", new_X, new_Y, new_X, new_X)
            d4 += np.einsum("ai,bi,ci,di", new_Y, new_X, new_X, new_X)
            d4 /= - 4 * N_eff

            if verbose:
                np.save("d4_modes_nosym.npy", d4)

            # Reshape the d4
            d4_reshaped = d4.reshape((self.n_modes * self.n_modes, self.n_modes * self.n_modes))

            new_mat = np.einsum("ab,a,b->ab", d4_reshaped, chi_beta, chi_beta)

            L_operator[self.n_modes:, self.n_modes:] += new_mat

        if verbose:
            print("L superoperator computed.")
        
        if overwrite_L_operator:
            print('overwriting the L operator')
            self.L_linop = L_operator
        
        return L_operator


    def get_full_L_operator_FT(self, verbose = False, debug_d3 = None, symmetrize = True, overwrite_L_operator = True):
        """
        GET THE FULL L OPERATOR (FINITE TEMPERATURE)
        ============================================

        Get the the full matrix L for the biconjugate Lanczos algorithm.
        Use this method to test everithing. I returns the full L operator as a matrix.
        It is very memory consuming, but it should be fast for small systems.

        Maybe we need to drop the exchange between a,b because they are symmetric by definition.
        The double counting for the exchange of a,b is now fixed.
        
        Now if self.use_harmonic == True we can compute the Wigner Lanczos matrix

        Results
        -------
           L_op : ndarray(size = (nmodes * (2*nmodes + 1)), dtype = TYPE_DP)
              The full L operator.
        """
        # The elements where w_a and w_b are exchanged are dependent
        # So we must avoid including them
        i_a = np.tile(np.arange(self.n_modes), (self.n_modes,1)).ravel()
        i_b = np.tile(np.arange(self.n_modes), (self.n_modes,1)).T.ravel()

        new_i_a = np.array([i_a[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        new_i_b = np.array([i_b[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        
        w_a = self.w[new_i_a]
        w_b = self.w[new_i_b]
        
        # N_w2 is the number of independent indeces
        N_w2 = len(w_a)

        if verbose:
            print()
            print('Getting the analytical L operator')
            print('Am I using Wigner? {}'.format(self.use_wigner))
        # Prepare the operator
        L_operator = np.zeros(shape = (self.n_modes + 2*N_w2, self.n_modes + 2*N_w2), dtype = TYPE_DP)

        # Set the Z'' harmonic
        if not self.ignore_harmonic:
            if not self.use_wigner:
                if verbose:
                    print('STANDARD harmonic on R(1)')
                L_operator[:self.n_modes, :self.n_modes] = np.diag(self.w**2)
            else:
                if verbose:
                    print('WIGNER harmonic on R(1)')
                L_operator[:self.n_modes, :self.n_modes] = -np.diag(self.w**2)

        #w_a = np.tile(self.w, (self.n_modes,1)).ravel()
        #w_b = np.tile(self.w, (self.n_modes,1)).T.ravel()

        n_a = np.zeros(np.shape(w_a), dtype = TYPE_DP)
        n_b = np.zeros(np.shape(w_a), dtype = TYPE_DP)
        if self.T > 0:
            n_a = 1 / (np.exp(w_a * 157887.32400374097/self.T) - 1)
            n_b = 1 / (np.exp(w_b * 157887.32400374097/self.T) - 1)

        if verbose:
            print("BE occ number", n_a[:self.n_modes])

        # Apply the non interacting X operator
        start_Y = self.n_modes
        start_A = self.n_modes + N_w2

        # Get the operator that exchanges the frequencies
        # For each index i (a,b), exchange_frequencies[i] is the index that correspond to (b,a)
        #exchange_frequencies = np.array([ (i // self.n_modes) + self.n_modes * (i % self.n_modes) for i in np.arange(self.n_modes**2)])
        #xx = np.tile(np.arange(self.n_modes), (self.n_modes, 1)).T.ravel()
        #yy = np.tile(np.arange(self.n_modes), (self.n_modes, 1)).ravel()
        #all_modes = np.arange(self.n_modes**2)
        #exchange_frequencies = xx + yy
        if not self.ignore_harmonic:
            if not self.use_wigner:
                if verbose:
                    print('STANDARD harmonic Y(1) ReA(1)')    
                # NOTE the double counting is NOT required for harmonic propagation
                # just check the harmonic matrix element
                extra_count = np.ones(N_w2, dtype = np.intc)
                extra_count[new_i_a == new_i_b] = 1.
                
                # Harmonic evolution for Y -> Y sector
                X_ab_NI = -w_a**2 - w_b**2 - (2*w_a *w_b) /((2*n_a + 1) * (2*n_b + 1))
                L_operator[start_Y: start_A, start_Y:start_A] = - np.diag(X_ab_NI)  * extra_count
                #L_operator[start_Y + np.arange(self.n_modes**2) , start_Y + exchange_frequencies] -= X_ab_NI / 2

                # Harmonic evolution for Y -> ReA sector
                Y_ab_NI = - (8 * w_a * w_b) / ((2*n_a + 1) * (2*n_b + 1))
                L_operator[start_Y : start_A, start_A:] = - np.diag(Y_ab_NI) * extra_count
                #L_operator[start_Y + np.arange(self.n_modes**2), start_A + exchange_frequencies] -=  Y_ab_NI / 2

                
                # Harmonic evolution for ReA -> Y sector
                X1_ab_NI = - (2*n_a*n_b + n_a + n_b) * (2*n_a*n_b + n_a + n_b + 1)*(2 * w_a * w_b) / ( (2*n_a + 1) * (2*n_b + 1))
                L_operator[start_A:, start_Y : start_A] = - np.diag(X1_ab_NI) / 1 * extra_count
                #L_operator[start_A + np.arange(self.n_modes**2), start_Y + exchange_frequencies] -= X1_ab_NI / 2

                # Harmonic evolution for ReA -> ReA sector
                Y1_ab_NI = - w_a**2 - w_b**2 + (2*w_a *w_b) /( (2*n_a + 1) * (2*n_b + 1))
                L_operator[start_A:, start_A:] = -np.diag(Y1_ab_NI) / 1 * extra_count
                #L_operator[start_A + np.arange(self.n_modes**2),  start_A + exchange_frequencies] -= Y1_ab_NI / 2
            else:
                # The harmonic application in Wigner is diagonal
                # NOTE the double counting is NOT required here
                # just check the harmonic matrix element
                if verbose:
                    print("WIGNER harmonic a'(1) b'(1)")
                # Diagonal harmonic propagation in Winger for a(1)
                a_NI  = -(w_a**2 + w_b**2 - 2 * w_a * w_b)
                L_operator[start_Y: start_A, start_Y: start_A] = + np.diag(a_NI)  

                # Diagonal harmonic propagation in Winger for b(1)
                b_NI = -(w_a**2 + w_b**2 + 2. * w_a * w_b)
                L_operator[start_A:, start_A:] = + np.diag(b_NI)
                

        # We ADDED all the non interacting (harmonic) propagators both for standard and Wigner
        # In WIgner the Harmonic approach is working

        # Compute the d3 operator
        
        #new_X = np.einsum("ia,a->ai", self.X, f_ups(self.w, self.T))
        if debug_d3 is None:
            N_eff = np.sum(self.rho)
            Y_weighted = np.einsum("ia, i -> ia", self.Y, self.rho)
        #new_Y = np.einsum("ia,i->ai", self.Y, self.rho)

        if not self.ignore_v3:
            if verbose:
                print("Computing d3...")
            if not debug_d3 is None:
                d3 = debug_d3
            else:
                X_ups = np.einsum("ia, a -> ia", self.X, f_ups(self.w, self.T))

                d3_noperm = np.einsum("ia, ib, ic -> abc", X_ups, X_ups, Y_weighted)
                d3_noperm /= -N_eff 

                # Apply the permuatations
                d3 = d3_noperm.copy()
                d3 += np.einsum("abc->acb", d3_noperm)
                d3 += np.einsum("abc->bac", d3_noperm)
                d3 += np.einsum("abc->bca", d3_noperm)
                d3 += np.einsum("abc->cab", d3_noperm)
                d3 += np.einsum("abc->cba", d3_noperm)
                d3 /= 6

                if verbose:
                    np.save("d3_modes_nosym.npy", d3)

                # Perform the standard symmetrization
                if symmetrize:
                    # TODO: fix the symmetrize_d3_muspace
                    raise NotImplementedError('The symmetrizaiton on d3 is not implemented')
                    d3 = symmetrize_d3_muspace(d3, self.symmetries)

                    if verbose:
                        np.save("d3_modes_sym.npy", d3)
                        np.save("symmetries_modes.npy", self.symmetries)
                

            # Reshape the d3
            d3_small_space = np.zeros((N_w2, self.n_modes), dtype = np.double)
            # Get the d3 in the small space by getting the independent terms
            d3_small_space[:,:] = d3[new_i_a, new_i_b, :]

            if verbose:
                print("D3 of the following elements:")
                print(new_i_a)
                print(new_i_b)
                print("D3 small space")
                print(d3_small_space)
                print('D3 complete')
                print(d3)

            #d3_reshaped = d3.reshape((self.n_modes* self.n_modes, self.n_modes))
            #d3_reshaped1 = d3.reshape((self.n_modes, self.n_modes* self.n_modes))
            
            if not self.use_wigner:
                if verbose:
                    print('STANDARD getting the D3 contribution')
                # Get the Z coeff between Y with R
                Z_coeff = 2 * ((2*n_a + 1)*w_b + (2*n_b + 1)*w_a) / ((2*n_a + 1) * (2*n_b + 1))
                Z_coeff = np.einsum("ab, a -> ab", d3_small_space, Z_coeff)
                L_operator[start_Y: start_A, :start_Y] = -Z_coeff

                # Get the Z' coeff between ReA and R
                Z1_coeff = 2 *((2*n_a + 1)*w_b*n_b*(n_b + 1) + (2*n_b + 1)*w_a*n_a*(n_a+1)) / ((2*n_a + 1) * (2*n_b + 1))
                Z1_coeff = np.einsum("ab, a -> ab", d3_small_space, Z1_coeff)
                L_operator[start_A:, :start_Y] = - Z1_coeff

                # Get the X'' coeff between R and Y with double counting
                extra_count = np.ones(N_w2, dtype = np.intc)
                extra_count[new_i_a != new_i_b] = 2
                X2_coeff = (2*n_b + 1) * (2*n_a +1) / (8*w_a *w_b)
                X2_coeff = np.einsum("ab, a -> ba", d3_small_space, X2_coeff * extra_count)
                L_operator[:start_Y, start_Y: start_A] = -X2_coeff
                
                # The coeff between R and ReA is zero.
            else:
                if verbose:
                    print('WIGNER getting the D3 contribution')
                # Chi for the independent indeces
                chi_minus = ((n_a - n_b) * (w_a - w_b) /(2 * w_a * w_b)) 
                chi_plus  = ((1 + n_a + n_b) * (w_a + w_b) /(2 * w_a * w_b))
                
                # The shape of these tensors is (N_w2, n_modes) considering double counting
                extra_count_w = np.ones(N_w2, dtype = np.intc)
                extra_count_w[new_i_a != new_i_b] = 2
                d3_chi_plus  = np.einsum('ab, a -> ab', d3_small_space, np.sqrt(+0.5 * chi_plus)  * extra_count_w)
                d3_chi_minus = np.einsum('ab, a -> ab', d3_small_space, np.sqrt(-0.5 * chi_minus) * extra_count_w)
                
                # The interacion between a rank 1 tensor a rank 2 tensor DOES require
                # to take into account double counting
                
                # The coeff betwee R'(1) and a'(1)
                L_operator[:start_Y, start_Y: start_A] = +d3_chi_minus.T
                # The coeff betwee R'(1) and b'(1)
                L_operator[:start_Y, start_A:] = -d3_chi_plus.T
                
                # The interacion between a rank 2 tensor a rank 1 tensor DOES NOT require
                # to take into account double counting
                
                # The shape of these tensors is (N_w2, n_modes)
                chi_plus_d3  = np.einsum('ab, a -> ab', d3_small_space, np.sqrt(+0.5 * chi_plus))
                chi_minus_d3 = np.einsum('ab, a -> ab', d3_small_space, np.sqrt(-0.5 * chi_minus))
                
                # The coeff betwee a'(1) and R'(1)
                L_operator[start_Y: start_A, :start_Y] = +chi_minus_d3
                # The coeff betwee b'(1) and R'(1)
                L_operator[start_A:, :start_Y] = -chi_plus_d3 
             
   
        if not self.ignore_v4:
            # Get the D4 tensor in the polarization basis
            # it should be symmetric under permutations of the indices
            d4 =  np.einsum("ia, ib, ic, id -> abcd", X_ups, X_ups, X_ups, Y_weighted)
            d4 += np.einsum("ia, ib, ic, id -> abcd", X_ups, X_ups, Y_weighted, X_ups)
            d4 += np.einsum("ia, ib, ic, id -> abcd", X_ups, Y_weighted, X_ups, X_ups)
            d4 += np.einsum("ia, ib, ic, id -> abcd", Y_weighted, X_ups, X_ups, X_ups)
            d4 /= - 4 * N_eff

            if verbose:
                np.save("d4_modes_nosym.npy", d4)

            # TODO: add the standard symmetrization
            if symmetrize:
                raise NotImplementedError('The symmetrizaiton on d4 is not implemented')

            # Get the independent first two indep indices
            d4_small_space1 = np.zeros((N_w2, self.n_modes, self.n_modes), dtype = np.double)
            d4_small_space1[:,:,:] = d4[new_i_a, new_i_b, :, :]

            # Get the independent second two indep indices
            d4_small_space = np.zeros((N_w2, N_w2), dtype = np.double)
            d4_small_space[:,:] = d4_small_space1[:, new_i_a, new_i_b]
            
            if verbose:
                print("D4 of the following elements:")
                print(new_i_a)
                print(new_i_b)
                print("D4 in the reduced space")
                print(d4_small_space)
                print("D4 complete")
                print(d4)
                
            if not self.use_wigner:
                if verbose:
                    print('STANDARD getting the D4 contribution')
                # When two tensors of rank 2 interact we need a factor of 2 overall
                extra_count = np.ones(N_w2, dtype = np.intc)
                extra_count[new_i_a != new_i_b] = 2
                    
                # Anharmonic interaction of Y(1) with Y(1)
                X_coeff_left = -((2 * w_a * n_b + 2 * w_b * n_a + w_a + w_b)) /(4 * (2*n_a + 1) * (2*n_b + 1))
                X_coeff_right = ((2 * n_a + 1) * (2 * n_b + 1)) /(w_a * w_b)
                X_coeff = np.einsum('a, ab, b -> ab', X_coeff_left, d4_small_space, X_coeff_right * extra_count)
                L_operator[start_Y:start_A, start_Y:start_A] += -X_coeff
                
                # Anharmonic interaction of ReA(1) with Y(1)
                X1_coeff_left = -(w_a * n_a * (n_a + 1) * (2 * n_b + 1) + w_b * n_b * (n_b + 1) * (2 * n_a + 1)) /(4 * (2*n_a + 1) * (2*n_b + 1))
                X1_coeff_right = ((2 * n_a + 1) * (2 * n_b + 1))/(w_a * w_b)
                X1_coeff = np.einsum('a, ab, b -> ab', X1_coeff_left, d4_small_space, X1_coeff_right * extra_count)
                L_operator[start_A:, start_Y:start_A] += -X1_coeff
            else:
                if verbose:
                    print('WIGNER getting the D4 contribution')
                # When two tensors of rank 2 interact we need a factor of 2 overall
                extra_count_w = np.ones(N_w2, dtype = np.intc)
                extra_count_w[new_i_a != new_i_b] = 2
                
                chi_plus  = ((1 + n_a + n_b) * (w_a + w_b) /(2 * w_a * w_b))
                chi_minus  = ((n_a - n_b) * (w_a - w_b) /(2 * w_a * w_b))
                
                plus_D4_plus   = np.einsum('a, ab, b -> ab', np.sqrt(+0.5 * chi_plus),  d4_small_space, np.sqrt(+0.5 * chi_plus) * extra_count_w)
                minus_D4_minus = np.einsum('a, ab, b -> ab', np.sqrt(-0.5 * chi_minus), d4_small_space, np.sqrt(-0.5 * chi_minus) * extra_count_w)
                minus_D4_plus  = np.einsum('a, ab, b -> ab', np.sqrt(-0.5 * chi_minus), d4_small_space, np.sqrt(+0.5 * chi_plus) * extra_count_w)
                plus_D4_minus  = np.einsum('a, ab, b -> ab', np.sqrt(+0.5 * chi_plus) , d4_small_space, np.sqrt(-0.5 * chi_minus) * extra_count_w)
                
                # Anharmonic interaction of a'(1) and a'(1)
                L_operator[start_Y: start_A, start_Y: start_A] += -minus_D4_minus
            
                # Anharmonic interaction of b'(1) and b'(1)
                L_operator[start_A:, start_A:] += -plus_D4_plus
                
                # Anharmonic interaction of a'(1) and b'(1)
                L_operator[start_Y: start_A, start_A :] += +minus_D4_plus
                
                # Anharmonic interaction of b'(1) and a'(1)
                L_operator[start_A :, start_Y: start_A] += +plus_D4_minus
                
        if verbose:
            print("L superoperator computed.")
            if not self.use_wigner:
                np.savez_compressed("L_super_analytical_standard.npz", L_operator)
            else:
                np.savez_compressed("L_super_analytical_wigner.npz", L_operator)
        
        if overwrite_L_operator:
            if verbose:
                print('Overwriting the L operator..')
                print('Am I using Wigner? {}'.format(self.use_wigner))
            def matvec(x):
                return L_operator.dot(x)
            def rmatvec(x):
                return x.dot(L_operator)

            self.L_linop = scipy.sparse.linalg.LinearOperator(L_operator.shape, matvec = matvec, rmatvec = rmatvec)
        
        return L_operator
    
    
    
    
    def mask_dot_wigner(self, debug = False):
        """
        Builds a mask in order to do a symmetric Lanczos.
        
        The Lanczos is symmetric in the basis where we store all the matrices, so
        when we run a Lanczos in all the scalar product we need to take into account
        a double counting for the dependent indeces
        
        Returns:
        --------
            -double_mask: nd.array with size = n_modes + n_modes * (n_modes + 1)
        """
        # Prepare the result
        double_mask = np.ones((self.n_modes + self.n_modes * (self.n_modes + 1)))
        
        # Where a'(1) and b'(1) starts
        start_a = self.n_modes
        start_b = self.n_modes + (self.n_modes * (self.n_modes + 1))//2
        
        # Get the indep indices
        i_a = np.tile(np.arange(self.n_modes), (self.n_modes,1)).ravel()
        i_b = np.tile(np.arange(self.n_modes), (self.n_modes,1)).T.ravel()

        # Avoid the exchange of indices
        new_i_a = np.array([i_a[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
        new_i_b = np.array([i_b[i] for i in range(len(i_a)) if i_a[i] >= i_b[i]])
           
        if debug:
            print()
            print("start_a'(1) = ", start_a)
            print("start_b'(1) = ", start_b)
            print('new_i_b')
            print(new_i_b)
            print('new_i_a')
            print(new_i_a)
        
        # Where we have dep indices insert a 2 for double ocunting
        double_mask[start_a: start_b][new_i_b < new_i_a] = 2
        double_mask[start_b:][new_i_b < new_i_a] = 2
        
        if debug:
            print('mask dot prod for R(1) = ')
            print(double_mask[:start_a])
            print("mask dot prod for a'(1) = ")
            print(double_mask[start_a: start_b])
            print("mask dot prod for b'(1) = ")
            print(double_mask[start_b:])
            print()
        
        return double_mask


            
    def run_FT(self, n_iter, save_dir = None, save_each = 5, verbose = True, n_rep_orth = 0, n_ortho = 10, flush_output = True, debug = False, prefix = "LANCZOS", run_simm = False, optimized = False):
        """
        RUN LANCZOS ITERATIONS FOR FINITE TEMPERATURE
        =============================================

        This method performs the biconjugate Lanczos algorithm to find
        the sequence of a and b and c coefficients that are the tridiagonal representation 
        of the L matrix to be inverted.
        
        NOTE: when we use the Wigner formalism the Lanczos matrix is symmetric in the vector space where all the elements
        of the tensors are considered (also those that are related by symmetry). Since the application of L
        is done in the reduced space where we discart these elements we have to take into
        account this by multiplying by two the off diagonal components in the scalar products.

        Parameters
        ----------
            n_iter : int
                The number of iterations to be performed in the Lanczos algorithm.
            save_dir : string
                The directory in which you want to store the results step by step,
                in order to do a preliminar analysis or restart the calculation later.
                If None (default), the steps are not saved.
            save_each : int
                If save dir is not None, the results are saved each N step, with N the value of save_each argument.
            verbose : bool
                If true all the info during the minimization will be printed on output.
            n_rep_orth : int
                The number of times in which the GS orthonormalization is repeated.
                The higher, the lower the precision of the Lanczos step, the lower, the higher
                the probability of finding ghost states
            n_ortho : int
                The number of vectors to be considered for the GS biorthogonalization. (if None, all are considered)
            flush_output : bool
                If true it flushes the output at each step. 
                This is usefull to avoid ending without any output if a calculation is killed before it ends normally.
                However, it could slow down things a bit on clusters.
            debug : bool
                If true prints a lot of more info about the Lanczos
                as the gram-shmidth procdeure and checks on the coefficients. 
                This is usefull to spot an error or the appeareance of ghost states due to numerical inaccuracy.
            run_simm : bool
                If true the biconjugate Lanczos is transformed in a simple Lanczos with corrections in the scalar product
            optimized : bool
                If True we pop the vectors P and Q that we do not use during the Lanczos
        """

        self.verbose = verbose
        # Check if the symmetries has been initialized
        if not self.initialized:
            if verbose:
                print('Not initialized. Now we symmetrize\n')
            self.prepare_symmetrization()

        # Check if the psi vector is prepared
        ERROR_MSG = """
Error, you must initialize a perturbation to start the Lanczos.
Use prepare_raman/ir or prepare_perturbation before calling the run method.
"""
        if self.psi is None:
            print(ERROR_MSG)
            raise ValueError(ERROR_MSG)

        psi_norm = np.sum(self.psi**2)
        if np.isnan(psi_norm) or psi_norm == 0:
            print(ERROR_MSG)
            raise ValueError(ERROR_MSG)

        # If save_dir does not exist, create it
        if Parallel.am_i_the_master():
            if save_dir is not None:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
         
        # run_simm is allowed only if we use the wigner representation
        if run_simm and not self.use_wigner:
            raise NotImplementedError('The symmetric Lanczos works only with Wigner. Set use_wigner to True and make sure that you are not using the analytic wigner L!')
            
            
        # Getting the mask product for the Wigner implementation
        if run_simm:
            if verbose:
                print('Running the standard Lanczos algorithm with Wigner')
                print('Getting the mask dot product')
                print()
            mask_dot = self.mask_dot_wigner(debug)


        
        # Get the current step
        i_step = len(self.a_coeffs)

        if verbose:
            header = """
<=====================================>
|                                     |
|          LANCZOS ALGORITHM          |
|                                     |
<=====================================>

Starting the algorithm. It may take a while.
Starting from step %d
""" % i_step
            print(header)

            OPTIONS = """
Should I ignore the third order effect? {}
Should I ignore the fourth order effect? {}
Should I use the Wigner formalism? {}
Should I use a standard Lanczos? {}
Max number of iterations: {}
""".format(self.ignore_v3, self.ignore_v4, self.use_wigner, run_simm, n_iter)
            print(OPTIONS)
        

        # If this is the current step initialize the algorithm
        if i_step == 0:
            self.basis_Q = []
            self.basis_P = []
            self.s_norm = []
            # Normalize the first vector in the Standard or Wigner representation
            if not run_simm:
                first_vector = self.psi / np.sqrt(self.psi.dot(self.psi))
            else:
                first_vector = self.psi / np.sqrt(self.psi.dot(self.psi * mask_dot))
            self.basis_Q.append(first_vector)
            self.basis_P.append(first_vector)
            self.s_norm.append(1)
        else:
            print('Restarting the Lanczos')
            print('There is no control on the len of basis_Q')
            # Convert everything in a list
            self.basis_Q = list(self.basis_Q)
            self.basis_P = list(self.basis_P)
            self.s_norm  = list(self.s_norm)
            self.a_coeffs = list(self.a_coeffs)
            self.b_coeffs = list(self.b_coeffs)
            self.c_coeffs = list(self.c_coeffs)
            #self.arnoldi_matrix = list(self.arnoldi_matrix)

            
            # if len(self.basis_Q) != i_step + 1:
            #     print("Krilov dim: %d, number of steps perfomed: %d" % (len(self.basis_Q), i_step))
            #     print("Error, the Krilov basis dimension should be 1 more than the number of steps")
            #     raise ValueError("Error the starting krilov basis does not matches the matrix, Look stdout.")

        assert len(self.basis_Q) == len(self.basis_P), "Something wrong when restoring the Lanczos."
        assert len(self.s_norm) == len(self.basis_P), "Something wrong when restoring the Lanczos."
        assert len(self.b_coeffs) == len(self.c_coeffs), "Something wrong when restoring the Lanczos. len b = {} len c = {}".format(len(self.b_coeffs), len(self.c_coeffs))


        # Select the two vectors for the biconjugate Lanczos iterations
        psi_q = self.basis_Q[-1]
        psi_p = self.basis_P[-1]

        if debug:
            print("Q basis:", self.basis_Q)
            print("P basis:", self.basis_P)
            print("S norm:", self.s_norm)
            print("SHAPE PSI Q, P :", psi_q.shape, psi_p.shape)

        # Convergence flag
        next_converged = False
        
        # Here starts the Lanczos
        for i in range(i_step, i_step + n_iter):
            if verbose:
                step_txt = """
 ===== NEW STEP %d =====

 """ % (i + 1)
                print(step_txt)
                print("Length of the coefficiets: a = {}, b = {}".format(len(self.a_coeffs), len(self.b_coeffs)))
                print()

                if flush_output:
                    sys.stdout.flush()

            # Application of L
            t1 = time.time()
            if not self.use_wigner:
                if verbose:
                    print("Running the BICONJUGATE Lanczos with standard representation!\n")
                    print()
                L_q = self.L_linop.matvec(psi_q)
                # psi_p is normalized (this must be considered when computing c coeff) 
                p_L = self.L_linop.rmatvec(psi_p) 
            else:
                if verbose:
                    print("The Wigner representation is used!\n")
                    print()
                # Get the application on psi_q
                L_q = self.L_linop.matvec(psi_q)
                if run_simm:
                    # This is done because we are running the symmetric Lanczos q=p
                    if verbose:
                        print()
                        print("Running the SYMMETRIC Lanczos with Wigner!\n")
                    p_L = np.copy(L_q)
                else:
                    # This should be done only with the analytical Wigner Matrix only for testing
                    # This should be done in the case q != p
                    if verbose:
                        print()
                        print("Running the BICONJUGATE Lanczos with Wigner analytic!\n")
                    p_L = self.L_linop.rmatvec(psi_p)   
            t2 = time.time()
            # End of L application

            if debug:
                if not run_simm:
                    print("Modulus of L_q: {}".format(np.sqrt(L_q.dot(L_q))))
                    print("Modulus of p_L: {}".format(np.sqrt(p_L.dot(p_L))))
                else:
                    print("Modulus of L_q: {}".format(np.sqrt(L_q.dot(L_q * mask_dot))))
                    print("Modulus of p_L: {}".format(np.sqrt(p_L.dot(p_L * mask_dot))))

            # Get the normalization of p_k (with respect to s_k)
            c_old = 1
            if len(self.c_coeffs) > 0:
                c_old = self.c_coeffs[-1]
            p_norm = self.s_norm[-1] / c_old
            if debug:
                print("p_norm: {}".format(p_norm))

            # Get the a coefficient
            if not run_simm:
                a_coeff = psi_p.dot(L_q) * p_norm
            else:
                a_coeff = psi_p.dot(L_q * mask_dot) * p_norm

            # Check if something whent wrong
            if np.isnan(a_coeff):
                ERR_MSG = """
Invalid value encountered during the Lanczos.
Check if you have correctly initialized the algorithm.
This may happen if the SCHA matrix has imaginary or zero frequencies,
or if the acoustic sum rule is not satisfied.
"""
                raise ValueError(ERR_MSG)    

            # Get the two residual vectors
            rk = L_q - a_coeff * psi_q
            # If this is not the first step
            if len(self.basis_Q) > 1:
                rk -= self.c_coeffs[-1] * self.basis_Q[-2]

            sk = p_L - a_coeff * psi_p 
            old_p_norm = 0
            # If this is not the first step
            if len(self.basis_P) > 1:
                # Get the multiplication factor to rescale the old p to the normalization of the new one.
                if len(self.c_coeffs) < 2:
                    old_p_norm = self.s_norm[-2]
                else:
                    old_p_norm = self.s_norm[-2] / self.c_coeffs[-2] 
                    # C is smaller than s_norm as it does not contain the first vector
                    # But this does not matter as we are counting from the end of the array

                # TODO: Check whether it better to use this or the default norms to update sk
                sk -= self.b_coeffs[-1] * self.basis_P[-2] * (old_p_norm / p_norm)

            # Get the normalization of sk 
            if not run_simm:
                s_norm = np.sqrt(sk.dot(sk))
            else:
                s_norm = np.sqrt(sk.dot(sk * mask_dot))
               
            # This normalization regularizes the Lanczos
            sk_tilde = sk / s_norm 
            # Add the p normalization of L^t p that was divided from the s_k
            s_norm *= p_norm 
            
            # Get the b and c coeffs
            if not run_simm:
                b_coeff = np.sqrt(rk.dot(rk))
                c_coeff = (sk_tilde.dot(rk / b_coeff)) * s_norm 
            else:
                b_coeff = np.sqrt(rk.dot(rk * mask_dot))
                c_coeff = (sk_tilde.dot((rk / b_coeff) * mask_dot)) * s_norm 

            
            if debug:
                print("new_p_norm: {}".format(s_norm / c_coeff))
                print("old_p_norm: {}".format(old_p_norm))

                print("Modulus of rk: {}".format(b_coeff))
                if not run_simm:
                    print("Modulus of sk: {}".format(np.sqrt(sk.dot(sk))))
                else:
                    print("Modulus of sk: {}".format(np.sqrt(sk.dot(sk * mask_dot))))

                if verbose:
                    print("Direct computation resulted in:")
                    print("     |  a = {}".format(a_coeff))
                    print("     |  b = {}".format(b_coeff))
                    print("     |  c = {}".format(c_coeff))
                    if run_simm:
                        print("     |  |b-c| = {}".format(np.abs(b_coeff - c_coeff)))

            # Check the convergence
            self.a_coeffs.append(a_coeff)
            if np.abs(b_coeff) < __EPSILON__ or next_converged:
                if verbose:
                    print("Converged (b coefficient is {}, |b| < {})".format(b_coeff, __EPSILON__))
                converged = True
                break 
            if np.abs(c_coeff) < __EPSILON__ or next_converged:
                if verbose:
                    print("Converged (c coefficient is {}, |c| < {})".format(c_coeff, __EPSILON__))
                converged = True
                break


            # Get the vectors for the next iteration
            psi_q = rk / b_coeff

            # psi_p is the normalized p vector, the sk_tilde one
            psi_p = sk_tilde.copy()


            # AFTER THIS p_norm refers to the norm of P in the previous step as psi_p has been updated
            if debug:
                if not run_simm:
                    print("1) Check c = ", psi_q.dot(p_L) * p_norm)
                    print("2) Check b = ", psi_p.dot(L_q) * s_norm / c_coeff)
                else:
                    print("1) Check c = ", psi_q.dot(p_L * mask_dot) * p_norm)
                    print("2) Check b = ", psi_p.dot(L_q * mask_dot) * s_norm / c_coeff)

            if debug:
                # Check the tridiagonality
                print("Tridiagonal matrix: (lenp: {}, lens: {})".format(len(self.basis_P), len(self.s_norm)))
                for k in range(len(self.basis_P)):
                    if k >= 1:
                        pp_norm = self.s_norm[k] / self.c_coeffs[k-1]
                    else:
                        pp_norm = self.s_norm[k]

                    if not run_simm:
                        print("p_{:d} L q_{:d} = {} | p_{:d} norm = {}".format(k, len(self.basis_P)-1, pp_norm * self.basis_P[k].dot(L_q), k, pp_norm))
                    else:
                        print("p_{:d} L q_{:d} = {} | p_{:d} norm = {}".format(k, len(self.basis_P)-1, pp_norm * self.basis_P[k].dot(L_q * mask_dot), k, pp_norm))
                        
                pp_norm = s_norm / c_coeff
                if not run_simm:
                    print("p_{:d} L q_{:d} = {} | p_{:d} norm = {}".format(len(self.basis_P), len(self.basis_P)-1, pp_norm * psi_p.dot(L_q), k+1, pp_norm))
                else:
                    print("p_{:d} L q_{:d} = {} | p_{:d} norm = {}".format(len(self.basis_P), len(self.basis_P)-1, pp_norm * psi_p.dot(L_q * mask_dot), k+1, pp_norm))


                # Check the tridiagonality
                print()
                print("Transposed:".format(len(self.basis_P), len(self.s_norm)))
                if not run_simm:
                    for k in range(len(self.basis_Q)):
                        print("q_{:d} L^T p_{:d} = {} | p_{:d} norm = {}".format(k, len(self.basis_P)-1, p_norm* self.basis_Q[k].dot(p_L), k, p_norm))
                    print("q_{:d} L^T p_{:d} = {} | p_{:d} norm = {}".format(len(self.basis_P), len(self.basis_P)-1, p_norm* psi_q.dot(p_L), k+1, p_norm))
                else:
                    for k in range(len(self.basis_Q)):
                        print("q_{:d} L^T p_{:d} = {} | p_{:d} norm = {}".format(k, len(self.basis_P)-1, p_norm* self.basis_Q[k].dot(p_L * mask_dot), k, p_norm))
                    print("q_{:d} L^T p_{:d} = {} | p_{:d} norm = {}".format(len(self.basis_P), len(self.basis_P)-1, p_norm* psi_q.dot(p_L * mask_dot), k+1, p_norm))
                    

            t1 = time.time()


            # Lets repeat twice the orthogonalization
            converged = False
            new_q = psi_q.copy()
            new_p = psi_p.copy()

            if debug:
                if not run_simm:
                    norm_q = np.sqrt(new_q.dot(new_q))
                    norm_p = np.sqrt(new_p.dot(new_p))
                    print("Norm of q = {} and p = {} BEFORE Gram-Schmidt".format(norm_q, norm_p))
                    print("current p dot q = {} (should be 1)".format(new_q.dot(new_p) * s_norm / c_coeff))
                else:
                    norm_q = np.sqrt(new_q.dot(new_q * mask_dot))
                    norm_p = np.sqrt(new_p.dot(new_p * mask_dot))
                    print("Norm of q = {} and p = {} before Gram-Schmidt".format(norm_q, norm_p))
                    print("current p dot q = {} (should be 1)".format(new_q.dot(new_p * mask_dot) * s_norm / c_coeff))
                    

                # Check the Gram-Schmidt
                print("GS orthogonality check: (should all be zeros)")
                print("step) Q dot old Ps  | P dot old Qs")
                for k in range(len(self.basis_P)):
                    if k >= 1:
                        pp_norm = self.s_norm[k] / self.c_coeffs[k-1]
                    else:
                        pp_norm = self.s_norm[k]
                     
                    if not run_simm:
                        q_dot_pold = self.basis_P[k].dot(new_q) * pp_norm
                        p_dot_qold = self.basis_Q[k].dot(new_p) * pp_norm
                    else:
                        q_dot_pold = self.basis_P[k].dot(new_q * mask_dot) * pp_norm
                        p_dot_qold = self.basis_Q[k].dot(new_p * mask_dot) * pp_norm
                    print("{:4d}) {:16.8e} | {:16.8e}".format(k, q_dot_pold, p_dot_qold))

            # Start the Gram Schmidt procedure        
            for k_orth in range(n_rep_orth):
                ortho_q = 0
                ortho_p = 0

                # The starting vector
                start = 0
                # n_ortho says how many vectors we include in the GS
                if n_ortho is not None:
                    start = len(self.basis_P) - n_ortho
                    if start < 0:
                        start = 0

                for j in range(start, len(self.basis_P)):
                    if not run_simm:
                        coeff1 = self.basis_P[j].dot(new_q)
                        coeff2 = self.basis_Q[j].dot(new_p)
                    else:
                        coeff1 = self.basis_P[j].dot(new_q * mask_dot)
                        coeff2 = self.basis_Q[j].dot(new_p * mask_dot)

                    # Gram Schmidt
                    new_q -= coeff1 * self.basis_P[j]
                    new_p -= coeff2 * self.basis_Q[j]

                    #print("REP {} COEFF {}: scalar: {}".format(k_orth+1, j, coeff1))
                    
                    ortho_q += np.abs(coeff1)
                    ortho_p += np.abs(ortho_p)

                # Add the new vector to the Krilov Basis
                if not run_simm:
                    normq = np.sqrt(new_q.dot(new_q))
                else:
                    normq = np.sqrt(new_q.dot(new_q * mask_dot))
                if verbose:
                    print("Vector norm (q) after GS number {}: {:16.8e}".format(k_orth+1, normq))

                # Check the normalization (If zero the algorithm converged)
                if normq < __EPSILON__:
                    next_converged = True
                    if verbose:
                        print("Obtained a linear dependent Q vector.")
                        print("The algorithm converged.")
                    
                new_q /= normq

                # Normalize the p vector
                if not run_simm:
                    normp = new_p.dot(new_p)
                else:
                    normp = new_p.dot(new_p * mask_dot)
                if verbose:
                    print("Vector norm (p biconjugate) after GS number {}: {:16.8e}".format(k_orth, normp))

                # Check the normalization (If zero the algorithm converged)
                if np.abs(normp) < __EPSILON__:
                    next_converged = True
                    if verbose:
                        print("Obtained a linear dependent P vector.")
                        print("The algorithm converged.")
            
                new_p /= normp

                # Now we need to update s_norm to enforce p dot q = 1
                if not run_simm:
                    s_norm = c_coeff / new_p.dot(new_q)
                else:
                    s_norm = c_coeff / new_p.dot(new_q * mask_dot)

                # We have a correctly satisfied orthogonality condition
                if ortho_p < __EPSILON__ and ortho_q < __EPSILON__:
                    break


            if not converged:
                # Add the new q and p vectors
                self.basis_Q.append(new_q)
                self.basis_P.append(new_p)
                psi_q = new_q.copy()
                psi_p = new_p.copy()

                # Add the new coefficients to the Arnoldi matrix
                self.b_coeffs.append(b_coeff)
                self.c_coeffs.append(c_coeff)
                self.s_norm.append(s_norm)
                
                # Pop the elements we do not use to optimize the use of memory
                if optimized:
                    print('Optimize RAM consumption')
                    if len(self.basis_P) > 3:
                        print('P basis popping the elements we do not use')
                        self.basis_P.pop(-4)

                    if len(self.basis_Q) > 3:
                        print('Q basis popping the elements we do not use')
                        self.basis_Q.pop(-4)
                        
                    if len(self.s_norm) > 3:
                        print('s norm popping the elements we do not use')
                        self.s_norm.pop(-4)

            t2 = time.time()


            if verbose:
                print("Time to perform the Gram-Schmidt and retrive the coefficients: %d s" % (t2-t1))
                print()
                print("a_%d = %.8e" % (i, self.a_coeffs[-1]))
                print("b_%d = %.8e" % (i, self.b_coeffs[-1]))
                print("c_%d = %.8e" % (i, self.c_coeffs[-1]))
                if run_simm:
                    print("|b_%d - c_%d| = %.8e" % (i, i, np.abs(self.b_coeffs[-1] - self.c_coeffs[-1])))
                print()
            
            # Save the step
            if not save_dir is None:
                if (i + 1) % save_each == 0:
                    self.save_status("%s/%s_STEP%d" % (save_dir, prefix, i+1))
        
                    if verbose:
                        print("Status saved into '%s/%s_STEP%d'" % (save_dir, prefix, i+1))
                
            if verbose:
                print("Lanczos step %d ultimated." % (i +1))
            

        if converged:
            print("   last a coeff = {}".format(a_coeff))    
            

            

        
    
    

    def run_full_diag(self, number, discard_dyn = True, n_iter = 100):
        r"""
        FULL LANCZOS DIAGONALIZATION
        ============================

        This function runs the standard Lanczos iteration progress.
        It returns the eigenvalues and eigenvectors of the L operator.
        These can be used for computing the spectral function, and the full
        green function as:

        .. math ::

            G_{ab}(\omega) = \sum_{\alpha} \frac{\left<a | \lambda_\alpha\right>\left<\lambda_\alpha|b\right>}{\lambda_\alpha - \omega^2 + i\eta}

        where :math:`\lambda` are eigenvalues and vectors returned by this method, while :math:`\eta` is a
        smearing parameter chosen by the user. 
        Remember the eigenvectors are defined in the polarization basis and they comprend also the dynamical matrix degrees of freedom.
        Since in most application you want to discard the dynamical matrices, you can select discard_din = True.

        The used Lanczos algorithm is the one by ARPACK, as implemented in scipy.sparse module

        Parameters
        ----------
            number = int
                The number of the n highest eigenvalues to be found
            discard_dyn : bool, optional
                If True the dynamical matrix component of the output eigenvectors will be discarded.
            n_iter : int, optional
                The maximum number of Lanczos iterations. Usually must be much higher than the
                number of states you want to describe.
        """

        # Perform the lanczos operation
        eigvals, eigvects = scipy.sparse.linalg.eigsh(self.L_linop, k = number, v0 = self.psi, ncv= n_iter)

        self.eigvals = eigvals
        self.eigvects = eigvects

        # Check if the dynamical part must be discarded
        if discard_dyn:
            eigvects = eigvects[:self.n_modes, :]
    

        return eigvals, eigvects

    # def GetSupercellSpectralFunctionFromEig(self, w_array, smearing):
    #     r"""
    #     GET SPECTRAL FUNCTION
    #     =====================

    #     Get the spectral function from the eigenvalues and eigenvectors.
    #     The method run_full_diag must already be runned.

    #     This method returns the spectral function in the supercell.
    #     The spectral function is computed as:

    #     .. math ::

    #         G_{ab}(\omega) = \sum_{\alpha} \frac{\left<a | \lambda_\alpha\right>\left<\lambda_\alpha|b\right>}{\lambda_\alpha - \omega^2 + i\eta}

    #     where :math:`\lambda` are eigenvalues and vectors returned by this method, while :math:`\eta` is a
    #     smearing parameter chosen by the user.

    #     Parameters
    #     ----------
    #         w_array : ndarray
    #             The frequencies to which you want to compute the spectral function.
    #         smearing : float
    #             The smearing of the spectral function.

    #     Returns
    #     -------
    #         s(w) : ndarray
    #             The -ImG(w), the opposite of the imaginary part of the Green function. 
    #     """

    #     # Exclude dynamical
    #     eigvects = self.eigvects[:self.n_modes, :]

    #     N_w = len(w_array)
    #     N_alpha = len(self.eigvals)

    #     # Transform the vectors back in cartesian coordinates
    #     new_vects = np.einsum("ab, ca, c->cb", eigvects, self.pols, 1 / np.sqrt(self.m))

    #     spectral_weight = np.einsum("ab, ab -> b", new_vects, np.conj(new_vects))
    #     spectral_function = np.zeros(N_w, dtype = np.complex128)

    #     l_alpha = np.tile(self.eigvals, (N_w, 1))
    #     p_w = np.tile(spectral_weight, (N_w, 1))
    #     _w_ = np.tile(w_array, (N_alpha, 1)).T 

    #     big_mat = p_w / (l_alpha - _w_**2 + 1j*smearing)
    #     spectral_function[:] = np.sum(big_mat, axis = 1)

    #     return - np.imag(spectral_function)


    # def GetFullSelfEnergy(self):
    #     r"""
    #     GET SELF ENERGY 
    #     ===============

    #     Get the self-energy matrix from the eigenvalues and eigenvectors.
    #     The method run_full_diag must already be runned.

    #     This method returns the self energy in the supercell.
    #     It is computed as

    #     .. math ::

    #         \Pi_{ab} = \sum_{\alpha} \lambda_\alpha\left<a | \lambda_\alpha\right>\left<\lambda_\alpha|b\right>

    #     where :math:`\lambda` are eigenvalues and vectors returned by this method.
    #     The matrix is in real (cartesian) space.

    #     Returns
    #     -------
    #         s(w) : ndarray
    #             The -ImG(w), the opposite of the imaginary part of the Green function. 
    #     """

    #     # Exclude dynamical
    #     eigvects = self.eigvects[:self.n_modes, :]


    #     # Transform the vectors back in cartesian coordinates
    #     new_vects = np.einsum("ab, ca, c->cb", eigvects, self.pols, 1 / np.sqrt(self.m))

    #     self_energy = np.einsum("ab, cb, b", new_vects, np.conj(new_vects), self.eigvals)

    #     return self_energy




def SlowApplyD3ToDyn(X, Y, rho, w, T, input_dyn):
    """
    Apply the D3 vector.

    This is a testing function. It is slow, as it is a pure python implementation.
    """

    new_X = np.einsum("ab,b->ab", X, f_ups(w, T))

    
    n_rand, n_modes = np.shape(X)
    N_eff = np.sum(rho)

    v_out = np.zeros(n_modes, dtype = TYPE_DP)
    for a in range(n_modes):
        for b in range(n_modes):
            for c in range(n_modes):
                # Prepare the D3 calculation
                in_av = new_X[:, a] * new_X[:, b] * Y[:, c]
                in_av +=  new_X[:, a] * new_X[:, c] * Y[:, b]
                in_av +=  new_X[:, c] * new_X[:, b] * Y[:, a]
                in_av *= rho

                # Apply D3
                v_out[a] += - np.sum(in_av) * input_dyn[n_modes*b + c] / (3*N_eff)
    
    return v_out

def FastApplyD3ToDyn(X, Y, rho, w, T, input_dyn,  symmetries, n_degeneracies, degenerate_space, mode = 1, transpose = False):
    """
    Apply the D3 to dyn
    ======================

    This is a wrapper to the fast C function.


    For details on the mode, look at the parameters list

    Parameters
    ----------
       X : ndarray(size = (n_modes, n_configs), dtype = np.double / np.float32)
           The X array (displacement in mode basis). Note that the dtype should match the mode
       Y : ndarray(size = (n_modes, n_configs))
           The Y array (forces in mode basis).
       rho : ndarray(size = n_configs)
           The weights of the configurations
       w : ndarray(size = n_modes)
           The list of frequencies
       T : float
           The temperature
       input_dyn : ndarray (size = n_modes*n_modes)
           The vector of the input dynamical matrix
       mode : int
           The mode for the execution:
              1) CPU : OpenMP parallelization
       symmetries : ndarray( size =(n_sym, n_modes, n_modes), dtype = np.double)
           The symmetries in the polarization basis.
       n_degeneracies : ndarray( size = n_modes, dtype = np.intc)
           The number of degenerate eigenvalues for each mode
       degenerate_space : list of lists
           The list of modes in the eigen subspace in which that mode belongs to.


    Results
    -------
       output_vector : ndarray (size = n_modes)
           The result of the calculation
    """

    n_modes = len(w)

    #transp = 0
    #if transpose:
    #    transp = 1

    output_vector = np.zeros(n_modes, dtype = TYPE_DP)
    #print( "Apply to dyn, nmodes:", n_modes, "shape:", np.shape(output_vector))
    
    deg_space_new = np.zeros(np.sum(n_degeneracies), dtype = np.intc)
    i = 0
    i_mode = 0
    j_mode = 0
    #print("len1 = ", len(deg_space_new), "len2 = ", n_modes)
    #print("Mapping degeneracies:", np.sum(n_degeneracies))
    while i_mode < n_modes:
        #print("i= ", i_mode, "Ndeg:", n_degeneracies[i_mode], "j = ", j_mode, "len = ", len(degenerate_space[i_mode]))
        #print("new_i = ", i, "tot = ", np.sum(n_degeneracies))
        #print("cross_modes: ({}, {}) | deg_imu = {} | i = {}".format(i_mode, j_mode, n_degeneracies[i_mode], i))

        deg_space_new[i] = degenerate_space[i_mode][j_mode]
        j_mode += 1
        i+=1

        if j_mode == n_degeneracies[i_mode]:
            i_mode += 1
            j_mode = 0
    

    sscha_HP_odd.ApplyV3ToDyn(X, Y, rho, w, T, input_dyn, output_vector, mode, symmetries, n_degeneracies, deg_space_new)
    return output_vector


def FastApplyD3ToVector(X, Y, rho, w, T, input_vector, symmetries, n_degeneracies, degenerate_space, mode = 1):
    """
    Apply the D3 to vector
    ======================

    This is a wrapper to the fast C function.


    For details on the mode, look at the parameters list

    Parameters
    ----------
       X : ndarray(size = (n_modes, n_configs), dtype = np.double / np.float32)
           The X array (displacement in mode basis). Note that the dtype should match the mode
       Y : ndarray(size = (n_modes, n_configs))
           The Y array (forces in mode basis).
       rho : ndarray(size = n_configs)
           The weights of the configurations
       w : ndarray(size = n_modes)
           The list of frequencies
       T : float
           The temperature
       input_vector : ndarray (size = n_modes)
           The input dynamical matrix
       mode : int
           The mode for the execution:
              1) CPU : OpenMP parallelization
       symmetries : ndarray( size =(n_sym, n_modes, n_modes), dtype = np.double)
           The symmetries in the polarization basis.
       n_degeneracies : ndarray( size = n_modes, dtype = np.intc)
           The number of degenerate eigenvalues for each mode
       degenerate_space : list of lists
           The list of modes in the eigen subspace in which that mode belongs to.

    Results
    -------
       output_dyn : ndarray (size = n_modes*n_modes)
           The result of the calculation
    """
    n_modes = len(w)
    output_dyn = np.zeros(n_modes*n_modes, dtype = TYPE_DP)
    #print( "Apply to vector, nmodes:", n_modes, "shape:", np.shape(output_dyn))

    deg_space_new = np.zeros(np.sum(n_degeneracies), dtype = np.intc)
    i = 0
    i_mode = 0
    j_mode = 0
    #print("Mapping degeneracies:", np.sum(n_degeneracies))
    while i_mode < n_modes:
        #print("cross_modes: ({}, {}) | deg_i = {}".format(i_mode, j_mode, n_degeneracies[i_mode]))
        deg_space_new[i] = degenerate_space[i_mode][j_mode]
        j_mode += 1
        i += 1
        if j_mode == n_degeneracies[i_mode]:
            i_mode += 1
            j_mode = 0
    
    sscha_HP_odd.ApplyV3ToVector(X, Y, rho, w, T, input_vector, output_dyn, mode, symmetries, n_degeneracies, deg_space_new)
    return output_dyn


def FastD3_FT(X, Y, rho, w, T, input_psi, symmetries, n_degeneracies, degenerate_space, mode = 1, transpose = False):
    """
    Apply the D3 to vector
    ======================

    This is a wrapper to the fast C function.


    For details on the mode, look at the parameters list

    Parameters
    ----------
       X : ndarray(size = (n_modes, n_configs), dtype = np.double / np.float32)
           The X array (displacement in mode basis). Note that the dtype should match the mode
       Y : ndarray(size = (n_modes, n_configs))
           The Y array (forces in mode basis).
       rho : ndarray(size = n_configs)
           The weights of the configurations
       w : ndarray(size = n_modes)
           The list of frequencies
       T : float
           The temperature
       input_psi : ndarray
           The input density matrix
       mode : int
           The mode for the execution:
              1) CPU : OpenMP parallelization
       symmetries : ndarray( size =(n_sym, n_modes, n_modes), dtype = np.double)
           The symmetries in the polarization basis.
       n_degeneracies : ndarray( size = n_modes, dtype = np.intc)
           The number of degenerate eigenvalues for each mode
       degenerate_space : list of lists
           The list of modes in the eigen subspace in which that mode belongs to.

    Results
    -------
       output_psi : ndarray 
           The output density matrix
    """
    n_modes = len(w)


    transp = 0
    if transpose:
        transp = 1

    total_length = len(input_psi)

    output_psi = np.zeros(total_length, dtype = TYPE_DP)
    #print( "Apply to vector, nmodes:", n_modes, "shape:", np.shape(output_dyn))

    # Get the start and end_A
    start_A = ((n_modes + 1) * n_modes) // 2 + n_modes 
    end_A = n_modes + (n_modes + 1) * n_modes


    deg_space_new = np.zeros(np.sum(n_degeneracies), dtype = np.intc)
    i = 0
    i_mode = 0
    j_mode = 0
    #print("Mapping degeneracies:", np.sum(n_degeneracies))
    while i_mode < n_modes:
        #print("cross_modes: ({}, {}) | deg_i = {}".format(i_mode, j_mode, n_degeneracies[i_mode]))
        deg_space_new[i] = degenerate_space[i_mode][j_mode]
        j_mode += 1
        i += 1
        if j_mode == n_degeneracies[i_mode]:
            i_mode += 1
            j_mode = 0
    
    print ("Degenerate space: ")
    print (deg_space_new)
    
    sscha_HP_odd.ApplyV3_FT(X, Y, rho, w, T, input_psi, output_psi, mode, symmetries, n_degeneracies, deg_space_new, start_A, end_A, transp)
    return output_psi



def FastD4_FT(X, Y, rho, w, T, input_psi, symmetries, n_degeneracies, degenerate_space, mode = 1):
    """
    Apply the D4 to vector
    ======================

    This is a wrapper to the fast C function.


    For details on the mode, look at the parameters list

    Parameters
    ----------
       X : ndarray(size = (n_modes, n_configs), dtype = np.double / np.float32)
           The X array (displacement in mode basis). Note that the dtype should match the mode
       Y : ndarray(size = (n_modes, n_configs))
           The Y array (forces in mode basis).
       rho : ndarray(size = n_configs)
           The weights of the configurations
       w : ndarray(size = n_modes)
           The list of frequencies
       T : float
           The temperature
       input_psi : ndarray
           The input density matrix
       mode : int
           The mode for the execution:
              1) CPU : OpenMP parallelization
       symmetries : ndarray( size =(n_sym, n_modes, n_modes), dtype = np.double)
           The symmetries in the polarization basis.
       n_degeneracies : ndarray( size = n_modes, dtype = np.intc)
           The number of degenerate eigenvalues for each mode
       degenerate_space : list of lists
           The list of modes in the eigen subspace in which that mode belongs to.

    Results
    -------
       output_psi : ndarray 
           The output density matrix
    """
    n_modes = len(w)

    total_length = len(input_psi)

    output_psi = np.zeros(total_length, dtype = TYPE_DP)
    #print( "Apply to vector, nmodes:", n_modes, "shape:", np.shape(output_dyn))

    # Get the start and end_A
    start_A = ((n_modes + 1) * n_modes) // 2 + n_modes 
    end_A = n_modes + (n_modes + 1) * n_modes


    deg_space_new = np.zeros(np.sum(n_degeneracies), dtype = np.intc)
    i = 0
    i_mode = 0
    j_mode = 0
    #print("Mapping degeneracies:", np.sum(n_degeneracies))
    # Preparing the symmetry variables for the fast calculation
    while i_mode < n_modes:
        #print("cross_modes: ({}, {}) | deg_i = {}".format(i_mode, j_mode, n_degeneracies[i_mode]))
        deg_space_new[i] = degenerate_space[i_mode][j_mode]
        j_mode += 1
        i += 1
        if j_mode == n_degeneracies[i_mode]:
            i_mode += 1
            j_mode = 0
    
    sscha_HP_odd.ApplyV4_FT(X, Y, rho, w, T, input_psi, output_psi, mode, symmetries, n_degeneracies, deg_space_new, start_A, end_A)
    return output_psi

    

def SlowApplyD3ToVector(X, Y, rho, w, T, input_vector):
    """
    Apply the D3 vector.

    This is a testing function. It is slow, as it is a pure python implementation.
    """

    new_X = np.einsum("ab,b->ab", X, f_ups(w, T))
    
    n_rand, n_modes = np.shape(X)
    N_eff = np.sum(rho)

    v_out = np.zeros(n_modes*n_modes, dtype = TYPE_DP)
    for a in range(n_modes):
        for b in range(n_modes):
            for c in range(n_modes):
                # Prepare the D3 calculation
                in_av = new_X[:, a] * new_X[:, b] * Y[:, c]
                in_av +=  new_X[:, a] * new_X[:, c] * Y[:, b]
                in_av +=  new_X[:, c] * new_X[:, b] * Y[:, a]
                in_av *= rho

                # Apply D3
                v_out[a*n_modes + b] += - np.sum(in_av) * input_vector[c] / (3*N_eff)
    
    return v_out




def SlowApplyD4ToDyn(X, Y, rho, w, T, input_dyn):
    """
    Apply the D4 matrix.

    This is a testing function. It is slow, as it is a pure python implementation.
    """


    new_X = np.einsum("ab,b->ab", X, f_ups(w, T))

    
    n_rand, n_modes = np.shape(X)
    N_eff = np.sum(rho)

    v_out = np.zeros(n_modes*n_modes, dtype = TYPE_DP)
    for a in range(n_modes):
        for b in range(n_modes):
            for c in range(n_modes):
                for d in range(n_modes):
                    # Prepare the D3 calculation
                    in_av =  new_X[:, a] * new_X[:, b] * new_X[:, c] * Y[:, d]
                    in_av += new_X[:, a] * new_X[:, b] * Y[:, c] * new_X[:, d]
                    in_av += new_X[:, a] * Y[:, b] * new_X[:, c] * new_X[:, d]
                    in_av += Y[:, a] * new_X[:, b] * new_X[:, c] * new_X[:, d]

                    in_av *= rho

                    # Apply D3
                    v_out[a*n_modes + b] += - np.sum(in_av) * input_dyn[n_modes*c + d] / (4*N_eff)
    
    return v_out


def FastApplyD4ToDyn(X, Y, rho, w, T, input_dyn, symmetries, n_degeneracies, degenerate_space, mode = 1):
    """
    Apply the D3 to vector
    ======================

    This is a wrapper to the fast C function.

    Remember to use the correct dtype value:
    if mode == GPU:
       dtype = np.float32
    if mode == CPU:
       dtype = np.float64
get_
    For details on the mode, look at the parameters list

    Parameters
    ----------
       X : ndarray(size = (n_modes, n_configs), dtype = np.double / np.float32)
           The X array (displacement in mode basis). Note that the dtype should match the mode
       Y : ndarray(size = (n_modes, n_configs))
           The Y array (forces in mode basis).
       rho : ndarray(size = n_configs)
           The weights of the configurations
       w : ndarray(size = n_modes)
           The list of frequencies
       T : float
           The temperature
       input_dyn : ndarray (size = n_modes*n_modes)
           The input dynamical matrix
       symmetries : ndarray( size =(n_sym, n_modes, n_modes), dtype = np.double)
           The symmetries in the polarization basis.
       n_degeneracies : ndarray( size = n_modes, dtype = np.intc)
           The number of degenerate eigenvalues for each mode
       degenerate_space : list of lists
           The list of modes in the eigen subspace in which that mode belongs to.
       mode : int
           The mode for the execution:
              1) CPU : OpenMP parallelization

    Results
    -------
       output_dyn : ndarray (size = n_modes*n_modes)
           The result of the calculation
    """
    n_modes = len(w)
    output_dyn = np.zeros(n_modes*n_modes, dtype = TYPE_DP)


    deg_space_new = np.zeros(np.sum(n_degeneracies), dtype = np.intc)
    i = 0
    i_mode = 0
    j_mode = 0
    #print("Mapping degeneracies:", np.sum(n_degeneracies))
    while i_mode < n_modes:
        #print("cross_modes: ({}, {}) | deg_i = {}".format(i_mode, j_mode, n_degeneracies[i_mode]))
        deg_space_new[i] = degenerate_space[i_mode][j_mode]
        j_mode += 1
        i += 1
        if j_mode == n_degeneracies[i_mode]:
            i_mode += 1
            j_mode = 0


    
    #print( "Apply to vector, nmodes:", n_modes, "shape:", np.shape(output_dyn))
    sscha_HP_odd.ApplyV4ToDyn(X, Y, rho, w, T, input_dyn, output_dyn, mode, symmetries, n_degeneracies, deg_space_new)
    return output_dyn





# Here some functions to analyze the data that comes out by a Lanczos
def GetFreeEnergyCurvatureFromContinuedFraction(a_ns, b_ns, pols_sc, masses, mode_mixing = True,\
    use_terminator = True, last_average = 5, smearing = 0):
    """
    GET THE FREE ENERGY CURVATURE FROM MANY LANCZOS
    ===============================================

    This function computes the free energy curvature from the result
    of a full Lanczos computation between all possible perturbations.

    Parameters
    ----------
        a_ns : ndarray(size = (n_modes, n_modes, N_steps))
            The a_n coefficients for each Lanczos perturbation
        b_ns : ndarray(size = (n_modes, n_modes, N_steps-1))
            The b_n coefficients for each Lanczos perturbation
        pols_sc : ndarray(size = (3*nat_sc, n_modes))
            The polarization vectors in the supercell
        masses : ndarray(size = (3*nat_sc))
            The mass associated to each component of pols_sc
        use_terminator : bool
            If true the infinite volume interpolation is performed trought the
            terminator trick
        last_average : int
            Used in combination with the terminator, average the last 'last_average'
            coefficients and replicate them.
        smearing : float
            The smearing for the green function calculation. 
            Usually not needed for this kind of calculation.

    Results
    -------
        odd_fc : ndarray( (3*nat_sc, 3*nat_sc))
            The free energy curvature in the supercell

    """

    n_modes = np.shape(pols_sc)[1]
    nat_sc = int(np.shape(pols_sc)[0] / 3)
    N_steps = np.shape(a_ns)[2]

    assert N_steps -1 == np.shape(b_ns)[2], "Error, an and bn has an incompatible size:\n a_n = {}, b_n = {}".format(np.shape(a_ns), np.shape(b_ns))
    
    
    mat_pol = np.zeros( (n_modes, n_modes), dtype = np.double)
    for i in range(n_modes):

        # Get the number of steps
        n_steps = np.arange(N_steps-1)[b_ns[i, i, :] == 0]
        if len(n_steps) == 0:
            n_steps = N_steps
        else:
            n_steps = n_steps[0] + 1

        
        # Create the Lanczos class
        lanc = Lanczos(None)
        lanc.a_coeffs = a_ns[i, i, :n_steps]
        lanc.b_coeffs = b_ns[i, i, :n_steps - 1]
        lanc.perturbation_modulus = 1


        print("Computing ({},{}) ... n_steps = {}".format(i, i, n_steps))

        # get the green function from continued fraction
        gf = lanc.get_green_function_continued_fraction(np.array([0]), use_terminator = use_terminator, \
            smearing = smearing, last_average = last_average)[0]
        
        mat_pol[i,i] = np.real(gf)

    # If there is the mode-mixing compute also the off-diagonal terms
    if mode_mixing:
        for i in range(n_modes):
            for j in range(i+1, n_modes):
                # Get the number of steps
                n_steps = np.arange(N_steps-1)[b_ns[i, j, :] == 0]
                if len(n_steps) == 0:
                    n_steps = N_steps
                else:
                    n_steps = n_steps[0] + 1


                # Create the Lanczos class)
                lanc = Lanczos(None)
                lanc.a_coeffs = a_ns[i, j, :n_steps]
                lanc.b_coeffs = b_ns[i, j, :n_steps-1]
                lanc.perturbation_modulus = 2

                print("Computing ({},{}) ..., n_steps = {}".format(i, j, n_steps))

                # get the green function from continued fraction
                gf = lanc.get_green_function_continued_fraction(np.array([0]), use_terminator = use_terminator, \
                    smearing = smearing, last_average = last_average)[0]
                
                # Lanczos can compute only diagonal green functions
                # Therefore we need to trick it to get the off-diagonal elements
                # <1|L|2> = 1/2*( <1+2|L|1+2> - <1|L|1>  - <2|L|2>)
                mat_pol[i,j] = (np.real(gf) - mat_pol[i,i] - mat_pol[j,j]) / 2
                mat_pol[j,i] = (np.real(gf) - mat_pol[i,i] - mat_pol[j,j]) / 2

    # The green function is the inverse of the free energy curvature
    np.savetxt("gf_mat.dat", mat_pol)
    fc_pols = np.linalg.inv(mat_pol)
    np.savetxt("fc_pols.dat", fc_pols)

    # Get back into real space
    epols_m = np.einsum("ab, a->ab", pols_sc, np.sqrt(masses)) 
    fc_odd = np.einsum("ab, ca, da ->cd", fc_pols, epols_m, epols_m)

    return fc_odd


def symmetrize_d3_muspace(d3, symmetries):
    """
    SYMMETRIZE D3 IN MODE SPACE
    ===========================

    This function symmetrizes the d3 in the mu space.
    It is quite fast.

    Parameters
    ----------
        d3 : ndarray(n_modes, n_modes, n_modes)
            The d3 tensor to be symmetrized
        symmetries : ndarray(N_sym, n_modes, n_modes)
            The full symmetry matrix

    Results
    -------
        new_d3 : ndarray(n_modes, n_modes, n_modes)
            The d3 tensor symmetrized
    """

    print("Symmetrizing d3: SHAPE SYMMETRY:", symmetries.shape)

    new_d3 = np.zeros(np.shape(d3), dtype = np.double)

    N_sym, nmode, dumb = np.shape(symmetries)

    for i in range(N_sym):
        symmat = symmetries[i, :, :]
        print("SYM {}:".format(i+1))
        print(symmetries[i,:,:])

        ap = np.einsum("abc, lc ->abl", d3, symmat)
        ap = np.einsum("abc, lb ->alc", ap, symmat)
        ap = np.einsum("abc, la ->lbc", ap, symmat)
        #ap = np.einsum("abc, aa, bb, cc->abc", d3, symmat, symmat, symmat)

        new_d3 += ap 
    
    new_d3 /= N_sym
    return new_d3


def get_weights_finite_differences(u_tilde, w, T, R1, Y1):
    """
    Computes the weights of the configurations using a finite difference
    approach.
    This is time consuiming, use it for testing purpouses.

    Parameters
    ----------
        u_tilde : ndarray(size = (N_random, n_modes))
            The displacement in the polarization space (mass rescaled)
        w : ndarray(n_modes)
            the SCHA frequencies
        T : float
            Temperature
        R1 : ndarray(size = n_modes)
            The perturbation on the centroid positions
        Y1 : ndarray(size = (n_modes, n_modes), symmetric)
            The perturbation on the Y matrix

    Returns
    -------
        weights : ndarray(size = N_random)
            The weights that correspond to this perturbation
    """
    n_conf, n_modes = u_tilde.shape

    # get the Y matrix
    Y_mu = 2 * w 

    if T > __EPSILON__:
        n = 1. / ( np.exp(w * 157887.32400374097 / T) - 1)
        Y_mu /= (2 * n + 1)

    Y = np.diag(Y_mu) 

    lambda_small = 1e-9

    R1_norm = np.sum(R1**2)
    Y1_norm = np.sum(Y1**2)

    norm = np.sqrt(R1_norm + Y1_norm)

    print("Normalization: {}".format(norm))
    
    R1_direction = R1 / norm
    Y1_direction = Y1 / norm



    #print("DISP R:", R1)
    #print("DISP Y:", Y1)

    new_Y = Y + Y1_direction * lambda_small
    new_u_tilde = u_tilde - np.tile(R1_direction * lambda_small, (n_conf, 1))

    # Get the weights before and after the perturbation
    w_old = np.zeros(n_conf, dtype = np.double) 
    w_new = np.zeros(n_conf, dtype = np.double) 

    for i in range(n_conf):
        w_old[i] = np.exp(-.5 * u_tilde[i, :].dot(Y.dot(u_tilde[i, :])))
        w_new[i] = np.exp(-.5 * new_u_tilde[i, :].dot(new_Y.dot(new_u_tilde[i, :])))
        
    w_old *= np.sqrt(np.linalg.det(Y / (2 * np.pi))) 
    w_new *= np.sqrt(np.linalg.det(new_Y / (2 * np.pi)))

    # Test normalization
    #print("Normalization old:", np.sum(w_old) / n_conf)
    #print("Normalization new:", np.sum(w_new) / n_conf)

    xc = np.sum(u_tilde) / n_conf
    #print("Avg:", xc)
    Y_num = np.sum( (u_tilde - xc)**2) / n_conf
    #print("Y from ens:", Y_num, " (from w = {})".format(np.linalg.inv(Y)))

    # Get the derivative with respect to the parameter
    weights = (w_new/w_old - 1) / lambda_small * norm 

    return weights


    """
    Return the full L matrix from the Lanczos utilities, by exploiting the linear operator.
    This is very usefull for testing purpouses.

    NOTE: The memory required to store the full matrix may diverge.

    If static is true, instead of the Lanczos matrix, the symmetric one ad-hoc for the static case is employed.
    """

    L_op = lanczos.L_linop
    if static == True:
        lanczos.psi = np.zeros(lanczos.n_modes + lanczos.n_modes * (lanczos.n_modes + 1) // 2, dtype = TYPE_DP)

        def apply_static(v):
            lanczos.psi[:] = v
            out = np.zeros(v.shape, dtype = TYPE_DP) 
            if compute_harm:
                out[:] = lanczos.apply_L1_static(v) 
            if compute_anharm:
                out += lanczos.apply_anharmonic_static()
            return out 

        npsi = len(lanczos.psi)

        L_op =  scipy.sparse.linalg.LinearOperator( shape = (npsi, npsi), dtype = TYPE_DP, matvec = apply_static, rmatvec = apply_static)

    n_iters = len(lanczos.psi)

    v = np.zeros(lanczos.psi.shape, dtype = np.double)

    L_matrix = np.zeros((n_iters, n_iters), dtype = np.double)

    for i in range(n_iters):
        print("Step {} out of {}".format(i+1, n_iters))

        v[:] = 0.0
        v[i] = 1.0

        if transpose:
            L_matrix[:, i] = L_op.rmatvec(v)
        else:
            L_matrix[:, i] = L_op.matvec(v)

    return L_matrix


def get_full_L_matrix(lanczos, transpose = False, static = False, compute_anharm = True, compute_harm = True):
    """
    Return the full L matrix from the Lanczos utilities, by exploiting the linear operator.
    This is very usefull for testing purpouses.
    NOTE: The memory required to store the full matrix may diverge.
    If static is true, instead of the Lanczos matrix, the symmetric one ad-hoc for the static case is employed.
    """
    print()
    print('Getting the full L matrix')
    print('Are we using the Wigner representation = {}'.format(lanczos.use_wigner))
    print()

    L_op = lanczos.L_linop
    
    if static == True:
        lanczos.psi = np.zeros(lanczos.n_modes + lanczos.n_modes * (lanczos.n_modes + 1) // 2, dtype = TYPE_DP)

        def apply_static(v):
            lanczos.psi = v
            out = np.zeros(v.shape, dtype = TYPE_DP) 
            if compute_harm:
                out[:] = lanczos.apply_L1_static(v) 
            if compute_anharm:
                out += lanczos.apply_anharmonic_static()
            return out 

        npsi = len(lanczos.psi)

        L_op =  scipy.sparse.linalg.LinearOperator( shape = (npsi, npsi), dtype = TYPE_DP, matvec = apply_static, rmatvec = apply_static)

    n_iters = len(lanczos.psi)

    v = np.zeros(lanczos.psi.shape, dtype = np.double)

    L_matrix = np.zeros((n_iters, n_iters), dtype = np.double)

    # In this way we get the columns of L
    for i in range(n_iters):
        print("Step {} out of {}".format(i+1, n_iters))

        v[:] = 0.0
        v[i] = 1.0

        if transpose:
            L_matrix[:, i] = L_op.rmatvec(v)
        else:
            L_matrix[:, i] = L_op.matvec(v)
            
        print('The colum i = {}'.format(i+1))
        print(L_matrix[:,i])

    return L_matrix




def min_stdes(func, args, x0, step = 1e-2, n_iters = 100):
    """
    A simple steepest descend algorithm with fixed step. Used for testing purpouses
    """
    
    x = x0.copy()
    for i in range(n_iters):
        f, grad = func(x, args)

        x -= grad * step

        print("F: {} | G: {}".format( f, np.sqrt(np.sum(grad**2))))
    return x



