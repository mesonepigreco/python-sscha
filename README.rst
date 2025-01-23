Introduction
============

What is TD-SCHA?
----------------

TD-SCHA is a theory to simulate quantum nuclear motion in materials with strong anharmonicity.
TD-SCHA stands for Time-Dependent Self-Consistent Harmonic Approximation, and it is the
dynamical extension of the SCHA theory, that can describe equilibrium properties of materials accounting for
both quantum and dynamical nuclear fluctuations. 
The tdscha python library allows to perform dynamical linear response calculation on top of the equilibrium results (computed with the python-sscha package).


Why would I need tdscha?
------------------------

**A Tool for Advanced Material Simulations**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Enhanced Simulation Capabilities*
   TD-SCHA is an essential tool for researchers and professionals in material science, particularly for simulating transport or thermal properties of materials, phase diagrams, and phonon-related properties.

*Integration with python-sscha*
   Seamless integration with python-sscha allows for the inclusion of both thermal and quantum phonon fluctuations in *ab initio* simulations.

**Leveraging the SSCHA Method**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Quantum and Thermal Fluctuations*
   The Stochastic Self-Consistent Harmonic Approximation (SSCHA) is a full-quantum method optimizing the nuclear wave-function or density matrix to minimize free energy, crucial for simulating highly anharmonic systems.

*Efficiency and Cost-Effectiveness*
   Despite its full quantum and thermal nature, the computational cost is comparable to classical molecular dynamics, enhanced by the algorithm's ability to exploit crystal symmetries.

**User-Friendly and Versatile**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*Python Library and Stand-alone Software*
   Available both as a Python library and stand-alone software, with input scripts sharing syntax with Quantum ESPRESSO.

*Broad Compatibility*
   Can couple with any *ab initio* engine and interacts through the Atomic Simulation Environment (ASE) with an interface for automatic job submission on remote clusters.

**Getting Started**
^^^^^^^^^^^^^^^^^^

*Easy to Use*
   User-friendly with short, readable input files and comprehensive tutorials. 
*Download and Explore*
   Download and install python-sscha, and start exploring the tutorials to enhance your material simulation projects.


How to install
==============

You need to have installed python-sscha and CellConstructor to work with tdscha. 
Please, read the installation guide on those package before proceeding further.
You may find all instructions on the `official website <www.sscha.eu>`_.   


To install from PiPy, simply type

.. code :: bash

   pip install tdscha

You can alternatively clone the repository from github and install the package with

.. code :: bash

   pip install .

To submit calculation in parallel, you need to have ``mpi4py`` and ``julia`` installed, or compile from source the MPI C version of the code.
This can be achieved **on a fresh installation** with:

.. code:: bash

   MPICC=mpicc python setup.py install

If you have not a fresh installation, remove the build directory before running the previous command.
With julia enabled, only mpi4py needs to be installed and properly configured to run in parallel.


Install with Intel FORTRAN compiler
-----------------------------------

The setup.py script works automatically with the GNU FORTRAN compiler. However, due to some differences in linking lapack,
to use the intel compiler you need to edit a bit the setup.py script.

In this case, you need to delete the lapack linking from the
setup.py and include the -mkl as linker option.
Note that you must force to use the same liker compiler as the one used for the compilation. 

Install with a specific compiler path
-------------------------------------

This can be achieved by specifying the environment variables on which setup.py relies:

1. CC (C compiler)
2. FC (Fortran compiler)
3. LDSHARED (linking)

If we want to use a custom compiler in /path/to/fcompiler we may run the setup as:

.. code-block:: console

   FC=/path/to/fcompiler LDSHARED=/path/to/fcompiler python setup.py install



Quick start
===========

Go to the `tutorials <https://sscha.eu/Tutorials/tutorial_05_ramanir/>`_
