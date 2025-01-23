from __future__ import print_function
from __future__ import division

import tdscha.DynamicalLanczos

import sys, os
import time
import warnings, difflib
import numpy as np
import json

from timeit import default_timer as timer

# Import the scipy sparse modules
import scipy, scipy.sparse.linalg

import cellconstructor as CC
import cellconstructor.Phonons
import cellconstructor.symmetries

import sscha.Ensemble as Ensemble
import tdscha.Tools as Tools
import sscha_HP_odd

# Override the print function to print in parallel only from the master
import cellconstructor.Settings as Parallel
from sscha.Parallel import pprint as print




class StaticHessian(object):
    def __init__(self, ensemble = None, verbose = False, lanczos_input={}):
        """
        STATIC HESSIAN
        ==============

        This class is for the advanced computation of the static hessian matrix.
        This exploit the inversion of the auxiliary systems, which allows for including the
        fourth order contribution exploiting sparse linear algebra to speedup the calculation.

        You can either initialize directly the object passing the ensemble with the configurations,
        or call the init function after the object has been defined.

        You can pass all arguments expected for initializing the DynamicalLanczos.Lanczos object
        as a dictionary called lanczos_input
        (for example the modes to be excluded, or the running mode).

        Check the DynamicalLanczos.Lanczos docs for more details
        """


        # The minimization variables
        self.vector = None
        self.lanczos_input = lanczos_input
        self.lanczos = tdscha.DynamicalLanczos.Lanczos(ensemble, **lanczos_input)
        self.verbose = False
        self.prefix = "hessian_calculation"
        self.preconitioned = True

        if ensemble is not None:
            self.init(ensemble, verbose)

        # Setup the attribute control
        # Every attribute introduce after this line will raise an exception
        self.__total_attributes__ = [item for item in self.__dict__.keys()]
        self.fixed_attributes = True # This must be the last attribute to be setted


    def __setattr__(self, name, value):
        """
        This method is used to set an attribute.
        It will raise an exception if the attribute does not exists (with a suggestion of similar entries)
        """
        
        if "fixed_attributes" in self.__dict__:
            if name in self.__total_attributes__:
                super(StaticHessian, self).__setattr__(name, value)
            elif self.fixed_attributes:
                similar_objects = str( difflib.get_close_matches(name, self.__total_attributes__))
                ERROR_MSG = """
        Error, the attribute '{}' is not a member of '{}'.
        Suggested similar attributes: {} ?
        """.format(name, type(self).__name__,  similar_objects)

                raise AttributeError(ERROR_MSG)
        else:
            super(StaticHessian, self).__setattr__(name, value)

    def load_status(self, fname):
        """
        Load the current vector from a previous calculation
        """
        self.lanczos.load_status(fname + ".npz")

        n_modes = self.lanczos.n_modes
        lenv = (self.lanczos.n_modes * (self.lanczos.n_modes + 1)) // 2
        n_g = (n_modes * (n_modes + 1)) // 2
        n_w = (n_modes * (n_modes**2 + 3*n_modes + 2)) // 6
        self.vector = np.zeros( n_g + n_w, dtype = tdscha.DynamicalLanczos.TYPE_DP)

        self.vector[:] = np.loadtxt(fname)

        # Load all the rest from the json file
        with open(fname + ".json", "r") as fp:
            data = json.load(fp)
            for x in data:
                self.__setattr__(x, data[x])



    def save_status(self, fname):
        """
        Save the current vector to a file.
        """
        np.savetxt(fname, self.vector)
        self.lanczos.save_status(fname)

        # Save all the other info in a json file
        selfdict = vars(self)
        new_dict = {}
        IGNORE = ["lanczos", "vector", "fixed_attributes", "__total_attributes__"]

        for x in selfdict:
            if not x in IGNORE:
                new_dict[x] = selfdict[x]

            
        with open(fname + ".json", "w") as fp:
            json.dump(new_dict, fp)


    def init(self, ensemble = None, verbose = True):
        """
        Initialize the StaticHessian with a given ensemble

        Parameters
        ----------
            ensemble : sscha.Ensemble.Ensemble
                The object that contains the configurations
            preconditioner : bool
                If true, the initial vector is set up for the preconditioned minimization.
                Be carefull to use it together with a preconditioned algorithm in the run method,
                otherwise you will start from a very far point from convergence.
            verbose : bool
                If true prints the memory occupied for the calculation
        """

        if ensemble is not None:
            self.lanczos = tdscha.DynamicalLanczos.Lanczos(ensemble, **self.lanczos_input)
            self.lanczos.init()

            n_modes = self.lanczos.n_modes
            lenv = (self.lanczos.n_modes * (self.lanczos.n_modes + 1)) // 2
            n_g = (n_modes * (n_modes + 1)) // 2
            n_w = (n_modes * (n_modes**2 + 3*n_modes + 2)) // 6


            self.vector = np.zeros( n_g + n_w, dtype = tdscha.DynamicalLanczos.TYPE_DP)
            
            # Initialize vector with the initial guess (the SSCHA matrix)
            counter = 0
            for i in range(n_modes):
                if not self.preconitioned:
                    self.vector[counter] = 1 / self.lanczos.w[i]**2
                else:
                    self.vector[counter] = 1 / self.lanczos.w[i]

                counter += n_modes - i
        else:
            if self.lanczos.N == 0:
                raise ValueError("Error, you must specify an ensemble or initialize the system loading a lanczos algorithm.")

        self.verbose = verbose


        if verbose:
            print("Memory of StaticHessian initialized.")
            # The seven comes from all the auxiliary varialbes necessary in the gradient computation and the CG
            print("     memory requested: {} Gb of RAM per process".format((self.vector.nbytes) * 3 / 1024**3))
            print("     (excluding memory occupied to store the ensemble)")
        

    def run(self, n_steps, save_dir = None, threshold = 1e-6, algorithm = "cg-prec", extra_options = {}):
        """
        RUN THE HESSIAN MATRIX CALCULATION
        ==================================

        This subroutine runs the algorithm that computes the hessian matrix.

        After this subroutine finished, the results are stored in the
        self.Ginv and selfW.W variables.
        You can retrieve the Hessian matrix as a CC.Phonons.Phonons object
        with the retrieve_hessian() subroutine.


        Parameters
        ----------
            n_steps : int
                The maximum number of steps after which the calculation is stopped.
            save_dir : string
                Path to the directory on which, after each step, the file are saved.
            threshold : float
                The threshold below which the residual are converged.
            algorithm : string
                The algorithm used for the inversion. 
                The algorithms are implemented in the sscha.Tools method, look at them for a more specific guide.
                    - cg : Conjugate Gradient
                        This is the 'minimal residual method'. Look at sscha.Tools.minimal_residual_method
                    - fom : Full orthogonalization Method
                        This is a method based on krilov subspace as Lanczos. It requires the dimension of the Krilov subspace
                        to be specified in extra_options (Krylov_dimension)
            extra_options : dict
                The extra arguments to be passed to the minimizer function, they depend on the algorithm.
        """
        # Prepare the saving directory
        if save_dir is not None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        # Prepare the A matrix
        lenv = len(self.vector)
        n_modes = self.lanczos.n_modes
        A = scipy.sparse.linalg.LinearOperator((lenv, lenv), matvec = self.apply_L)
        
        def prec_mult(x):
            return self.apply_L(x, preconditioner = True)
        A_precond_half = scipy.sparse.linalg.LinearOperator((lenv, lenv), matvec = prec_mult)

        #A_real = sscha.Tools.get_matrix_from_sparse_linop(A)
        #A_prec = sscha.Tools.get_matrix_from_sparse_linop(A_precond_half)
        #np.savetxt("A.dat", A_real) 
        #np.savetxt("A_precond.dat", A_prec)

        # Define the function to save the results
        def callback(x, iters):
            if save_dir is not None:
                xtilde = x
                np.savetxt(os.path.join(save_dir, "{}_step{:05d}.dat".format(self.prefix, iters)), xtilde)
            sys.stdout.flush()
        

        x0 = self.vector.copy()
        b = np.zeros( lenv, dtype = tdscha.DynamicalLanczos.TYPE_DP)

        # Initialize vector of the known terms
        counter = 0
        for i in range(n_modes):
            b[counter] = 1 
            counter += n_modes - i
        

        t1 = time.time()
        if algorithm.lower() == "cg":
            res = Tools.minimum_residual_algorithm(A, b, x0, max_iters = n_steps, conv_thr = threshold, callback = callback)
        elif algorithm.lower() == "cg-prec":
            res = Tools.minimum_residual_algorithm_precond(A, b, A_precond_half, x0 = x0, max_iters = n_steps, conv_thr = threshold, callback = callback)
        else:
            raise NotImplementedError("Error, algorithm '{}' not implemented.".format(algorithm))
        t2 = time.time()

        self.vector[:] = res

        if save_dir is not None:
            self.save_status( os.path.join(save_dir, self.prefix))


        if self.verbose:
            print()
            print(r"/======================================\\")
            print(r"|                                      |")
            print(r"|    HESSIAN CALCULATION CONVERGED     |")
            print(r"|                                      |")
            print(r"\\======================================/")
            print()
            print()
            totaltime = t2 -t1 
            hours = totaltime // 3600
            totaltime -= hours * 3600
            mins = totaltime // 60
            totaltime -= mins * 60
            print("Total time to converge: {} h {} m {:.2f} s".format(hours, mins, totaltime))
            print()
        
            



    # def _run_cg_separated(self, n_steps, save_dir = None, threshold = 1e-8):
    #     """
    #     RUN THE HESSIAN MATRIX CALCULATION
    #     ==================================

    #     NOTE: Deprecated, use run instead

    #     This subroutine runs the algorithm that computes the hessian matrix.

    #     After this subroutine finished, the result are stored in the
    #     self.Ginv and selfW.W variables.
    #     You can retrive the Hessian matrix as a CC.Phonons.Phonons object
    #     with the retrive_hessian() subroutine.

    #     The algorithm is a generalized conjugate gradient minimization
    #     as the minimum residual algorithm, to optimize also non positive definite hessians.

    #     Parameters
    #     ----------
    #         n_steps : int
    #             Number of steps to converge the calculation
    #         save_dir : string
    #             Path to the directory in which the results are saved.
    #             Each step the status of the algorithm is saved and can be restored.
    #             TODO
    #         thr : np.double
    #             Threshold for the convergence of the algorithm. 
    #             If the gradient is lower than this threshold, the algorithm is 
    #             converged.
    #     """
    #     raise DeprecationWarning("This function has been deprecated, use run instead!")

    #     # Check if the saving directory does not exists, create it
    #     if save_dir is not None:
    #         if not os.path.exists(save_dir):
    #             if self.verbose:
    #                 print("Saving directory '{}' not found. I generate it.".format(save_dir))
    #             os.makedirs(save_dir)

    #     # Prepare all the variable for the minimization
    #     pG = np.zeros(self.Ginv.shape, dtype = tdscha.DynamicalLanczos.TYPE_DP)
    #     pG_bar = np.zeros(self.Ginv.shape, dtype = tdscha.DynamicalLanczos.TYPE_DP)

    #     pW = np.zeros(self.W.shape, dtype = tdscha.DynamicalLanczos.TYPE_DP)
    #     pW_bar = np.zeros(self.W.shape, dtype = tdscha.DynamicalLanczos.TYPE_DP)


    #     # Perform the first application
    #     rG, rW = self.get_gradient(self.Ginv, self.W)
    #     rG_bar, rW_bar = self.apply_L(rG, rW)

    #     # Setup the initial
    #     pG[:,:] = rG
    #     pG_bar[:,:] = rG_bar[:,:]

    #     pW[:,:] = rW
    #     pW_bar[:,:] = rW_bar

    #     while self.step < n_steps:
    #         if self.verbose:
    #             print("Hessian calculation step {} / {}".format(self.step + 1, n_steps))
            
    #         ApG = pG_bar
    #         ApW = pW_bar

    #         ApG_bar, ApW_bar = self.apply_L(pG_bar, pW_bar)

    #         rbar_dot_r = np.einsum("ab, ab -> a", rG_bar, rG) + np.einsum("ab, ab ->a", rW_bar, rW)

    #         alpha = rbar_dot_r
    #         alpha /= np.einsum("ab, ab ->a", pG_bar, ApG) + np.einsum("ab, ab ->a", pW_bar, ApW)

    #         # Update the solution
    #         self.Ginv[:,:] += np.einsum("a, ab ->ab", alpha, pG)
    #         self.W[:,:] += np.einsum("a, ab ->ab", alpha, pW)

    #         if save_dir is not None:
    #             # Save the status of the hessian matrix
    #             hessian = self.retrive_hessian()
    #             fname = os.path.join(save_dir, "hessian_step{:05d}_".format(self.step))
    #             hessian.save_qe(fname)

    #             # TODO:
    #             # Save another status file that can be used to restart the calculation
    #             # This involves saving also the W

    #         # Update r and r_bar
    #         rG[:,:] -= np.einsum("a, ab -> ab", alpha, ApG)
    #         rW[:,:] -= np.einsum("a, ab -> ab", alpha, ApW)

    #         rG_bar[:,:] -= np.einsum("a, ab -> ab", alpha, ApG_bar)
    #         rW_bar[:,:] -= np.einsum("a, ab -> ab", alpha, ApW_bar)

    #         rbar_dot_r_new = np.einsum("ab, ab -> a", rG_bar, rG) + np.einsum("ab, ab ->a", rW_bar, rW)
    #         beta = rbar_dot_r / rbar_dot_r_new

    #         # Update p and p_bar
    #         pG[:,:] = rG[:,:] + np.einsum("a, ab -> ab", beta, pG)
    #         pW[:,:] = rW[:,:] + np.einsum("a, ab -> ab", beta, pW)
    #         pG_bar[:,:] = rG_bar[:,:] + np.einsum("a, ab -> ab", beta, pG_bar)
    #         pW_bar[:,:] = rW_bar[:,:] + np.einsum("a, ab -> ab", beta, pW_bar)

    #         self.step += 1

    #         # Check the residual
    #         thr = np.max(np.abs(rG))
    #         if self.verbose:
    #             print("   residual = {} (The threshold is {})".format(thr, threshold))
    #         if thr < threshold:
    #             if self.verbose:
    #                 print()
    #                 print("CONVERGED!")
    #             break

    def run_no_mode_mixing(self, nsteps, save_dir = None, restart_from_file = False):
        """
        RUN THE HESSIAN CALCULATION NEGLECTING MODE MIXING
        ==================================================

        This algorithm computes the Hessian matrix by applying the static approximation on the diagonal
        elements of the dynamical Green function for each phonon mode of the auxiliary force constant matrix.

        This neglects mode mixing introduced by the "bubble" and higher-order phonon-phonon scattering diagrams,
        but contains anharmonicity non-perturbatively.

        TODO: exploit mode symmetries to avoid to repeat calculations for degenerate modes.

        Parameters
        ----------
            nsteps : int
                The number of steps for each Lanczos iterations to converge
            save_dir : string
                Path to the file on which the Lanczos data will be saved for restarting.
            restart_from_file: bool
                If True, we load the data from save_dir and continue from a previous calculation.
                NOT YET IMPLEMENTED

        Results
        -------
            hessian : CC.Phonons.Phonons()
                The free energy Hessian, without the mode-mixing approximation.
        """
        nmodes = self.lanczos.n_modes

        Gw = np.zeros( (nmodes, nmodes), dtype = np.double)


        if not save_dir == None:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

        for i in range(nmodes):
            run_lanczos = True
            # Avoid redundancies in simulating degenerate modes.
            if i > 0:
                if np.abs(self.lanczos.w[i-1] - self.lanczos.w[i]) < 1e-10:
                    if self.verbose:
                        print("SKIPPING MODE {} (Degeneracy)".format(i+1))
                    run_lanczos = False 

            if run_lanczos:
                self.lanczos.prepare_mode(i)

                if self.verbose:
                    print()
                    string = "RUNNING MODE {} out of {}".format(i + 1, nmodes)
                    print(string)
                    print("-"*len(string))
                    print()

                new_nsteps = nsteps

                if save_dir is not None:
                    final_lanc_filename = os.path.join(save_dir, self.prefix + "_M{:d}_conv".format(i + 1))

                    if  restart_from_file:
                        # Check if the lanczos needs to be loaded
                        if os.path.exists( final_lanc_filename ):
                            if self.verbose:
                                print("Loading from {}...".format(final_lanc_filename))
                            self.lanczos.load_status(final_lanc_filename)
                        else:
                            all_files = [x for x in os.listdir(save_dir) if x.startswith(self.prefix + "_M{:d}".format(i+1)) and x.endswith(".npz") and "STEP" in x]
                            if len(all_files) > 0:
                                count = [int(x.split(".")[-2].split("STEP")[-1]) for x in all_files]
                                index = np.argmax(count)
                                filename = os.path.join(save_dir, all_files[index])

                                if self.verbose:
                                    print("Loading from {}...".format(filename))
                                self.lanczos.load_status(filename)

                        # Get the number of steps
                        if len(self.lanczos.a_coeffs):
                            new_nsteps = nsteps - len(self.lanczos.a_coeffs)
                            

                if new_nsteps > 0:
                    self.lanczos.run_FT(new_nsteps, save_dir = save_dir, verbose = self.verbose, prefix = self.prefix + "_M{:d}".format(i+1))

                # Save the final status
                if save_dir is not None:
                    self.lanczos.save_status(final_lanc_filename)
                
                # END if run_lanczos

            # Get the static limit from the dynamical response funciton
            gf0 = self.lanczos.get_green_function_continued_fraction(np.array([0]), use_terminator = False, smearing = 0)[0]
            Gw[i, i] = gf0

        # Retrive the hessian
        W = np.zeros((nmodes, nmodes, nmodes), dtype = np.double)

        # TODO: Apply the correct preconditioning to the vector
        #       this would allow to use the no_mode_mixing as an ansatz for the full algorithm.
        self.preconitioned = False
        self.vector = self.get_vector(Gw, W)

        return self.retrieve_hessian()


    def retrieve_hessian(self, noq = False):
        """
        Return the Hessian matrix as a CC.Phonons.Phonons object.

        Note that you need to run the Hessian calculation (run method), otherwise this
        method returns the SSCHA dynamical matrix.

        If noq is true, it returns a numpy matrix in the supercell.
        This enable to get the Hessian even if the dynamical matrix of the Lanczos is not initialized,
        as it occurs when loading from a file.
        """

        new_vector = self.vector.copy()

        if self.preconitioned:
           new_vector = self.apply_L(self.vector, preconditioner = True) 

        Ginv = self.get_G_W(new_vector, ignore_W = True)

        G = np.linalg.inv(Ginv)
        dyn = self.lanczos.pols.dot(G.dot(self.lanczos.pols.T))

        dyn[:,:] *= np.outer(np.sqrt(self.lanczos.m), np.sqrt(self.lanczos.m))

        if noq:
            return dyn
        #CC.symmetries.CustomASR(dyn)

        # We copy the q points from the SSCHA dyn
        nq_tot = len(self.lanczos.dyn.q_tot)
        q_points = np.array(self.lanczos.dyn.q_tot)
        uc_structure = self.lanczos.dyn.structure.copy()
        ss_structure = self.lanczos.dyn.structure.generate_supercell(self.lanczos.dyn.GetSupercell())

        # Compute the dynamical matrix
        dynq = CC.Phonons.GetDynQFromFCSupercell(dyn, q_points, uc_structure, ss_structure)

        # Create the CellConstructor Object.
        hessian_matrix = CC.Phonons.Phonons(self.lanczos.dyn.structure, nqirr = nq_tot) #self.lanczos.dyn.Copy()
        for i in range(nq_tot):
            hessian_matrix.dynmats[i] = dynq[i, :, :]
            hessian_matrix.q_tot[i] = q_points[i, :]
        
        # Adjust the star according to the symmetries
        hessian_matrix.AdjustQStar()
            
        #CC.symmetries.CustomASR(hessian_matrix.dynmats[0])

        return hessian_matrix


    def apply_L(self, vect, preconditioner = False, inverse_preconditioner = False):
        """
        Apply the system matrix to the full array.
        """
        if self.verbose:
            t1 = time.time()

        if preconditioner and inverse_preconditioner:
            raise ValueError("Error, only one between preconditioner and inverse_preconditioner can be True")

        lenv = self.lanczos.n_modes
        lenv += (self.lanczos.n_modes * (self.lanczos.n_modes + 1)) // 2

        Ginv, W = self.get_G_W(vect)

        if self.verbose:
            t2 = time.time()
            print("Time to transform the vector in the G and W matrix: {} s".format(t2  - t1))

        ti = time.time()
        for i in range(self.lanczos.n_modes):
            
            vector = np.zeros(lenv, dtype = tdscha.DynamicalLanczos.TYPE_DP)
            vector[:self.lanczos.n_modes] = Ginv[i, :]
            vector[self.lanczos.n_modes:] = _from_matrix_to_symmetric(W[i, :, :])

            # Here the L application (TODO: Here eventual preconditioning)
            self.lanczos.psi = vector

            if not preconditioner and not inverse_preconditioner:
                outv = self.lanczos.apply_L1_static(vector)
                outv += self.lanczos.apply_anharmonic_static()
            else:
                if preconditioner:
                    power = 0.5
                if inverse_preconditioner:
                    power = -0.5
                outv = self.lanczos.apply_L1_static(vector, inverse = True, power = power)

            Ginv[i, :] = outv[:self.lanczos.n_modes]
            W[i, :, :] = _from_symmetric_to_matrix(outv[self.lanczos.n_modes:], self.lanczos.n_modes)

            if self.verbose:
                tnew = time.time()
                deltat = tnew - ti
                etatime = deltat * (self.lanczos.n_modes -i -1)
                print("Vector {} / {} done (time = {:.2} s | ETA = {:.2f} s)".format(i +1, self.lanczos.n_modes, deltat, etatime))
                ti = tnew

        if self.verbose:
            t3 = time.time()
            print("Total time to apply the L matrix: {} s".format(t3- t2))


        # Reget the final vector
        vect = self.get_vector(Ginv, W)

        if self.verbose:
            t4 = time.time()
            print("Total time to convert to 1D vector: {} s".format(t4-t3))
            sys.stdout.flush()

        return vect

    def get_G_W(self, vector, ignore_W = False):
        """
        From a vector of the status, return the G and W tensors.
        If ignore_W is True, only G is returned
        """

        n_modes = self.lanczos.n_modes
        # The first part of the vector describes the G
        G = np.zeros((n_modes, n_modes), dtype = np.double)

        counter = 0
        for i in range(n_modes):
            for j in range(i, n_modes):
                G[i, j] = vector[counter]
                G[j, i] = vector[counter]
                counter += 1

        if ignore_W:
            return G

        W = np.zeros((n_modes, n_modes,  n_modes), dtype = np.double)
        
        for i in range(n_modes):
            for j in range(i, n_modes):
                for k in range(j, n_modes):
                    W[i, j, k] = vector[counter]
                    W[i, k, j] = vector[counter]
                    W[j, i, k] = vector[counter]
                    W[j, k, i] = vector[counter]
                    W[k, i, j] = vector[counter]
                    W[k, j, i] = vector[counter]
                    counter += 1
        
        return G, W

    def get_vector(self, G, W):
        """
        Get the symmetric vector from G and W.
        This subroutine takes only the upper diagonal from the tensors.
        """
        n_modes = self.lanczos.n_modes

        n_g = (n_modes * (n_modes + 1)) // 2
        n_w = (n_modes * (n_modes**2 + 3*n_modes + 2)) // 6

        vector = np.zeros(n_g + n_w, dtype = np.double)

        counter = 0
        for i in range(n_modes):
            for j in range(i, n_modes):
                vector[counter] = G[i, j]
                counter += 1

        assert counter == n_g 

        for i in range(n_modes):
            for j in range(i, n_modes):
                for k in range(j, n_modes):
                    vector[counter] = W[i, j, k]
                    counter += 1
        
        assert counter == n_w + n_g

        return vector




def _from_symmetric_to_matrix(vector, n):
    """
    Takes as input a vector of size n * (n+1) / 2 and returns a vector (n,n) 
    where the originalvalues are interpreted as the upper triangle of a symmetric matrix
    """
    shape = list(vector.shape) + [n]
    shape[-2] = n 
    new_vector = np.zeros(shape, dtype = vector.dtype)

    counter = 0
    for x in range(n):
        for y in range(x, n):
            new_vector[..., x, y] = vector[..., counter]
            new_vector[..., y, x] = vector[..., counter]
            counter += 1
    return new_vector


def _from_matrix_to_symmetric(vector):
    """ 
    This perform the opposite operation as the previous function
    """
    n = vector.shape[-1]
    shape = list(vector.shape)[:-1]
    shape[-1] = (n * (n+1)) // 2
    new_vector = np.zeros(shape, dtype = vector.dtype)

    counter = 0
    for x in range(n):
        for y in range(x, n):
            new_vector[..., counter] = vector[..., x, y]
            counter += 1
    return new_vector




        
