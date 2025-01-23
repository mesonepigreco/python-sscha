#ifndef __LANCZOS__
#define __LANCZOS__
#include <iostream>
#include <fstream>
#include <stdlib.h>
#include <string.h>
#include <exception>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "LanczosFunctions.h"

#define RY_TO_K 157887.32400374097
using namespace std;

class Lanczos {
    int N, n_syms, n_modes, n_steps, n_blocks;
    int i_step;
    bool ignore_v2, ignore_v3, ignore_v4, reverse_L, restart;

    double T, shift_value;

    double *w, *nbose, *rho;
    int * N_degeneracy, *blocks_ids,  **good_deg_space;

    double *X, *Y, *psi,  **symmetries;


    double *a, *b, *c;
    double *Qbasis, *Pbasis, *snorm;

    string rootname;

    int n_psi;

    void update_nbose();

    // Apply the noninteracting propagation from the psi to the out_vect
    void apply_L1(double * out_vect, bool transpose);

    // Apply the anharmonic part of the L  interaction
    void apply_anharmonic(double * out_vect, bool transpose);

    void get_Y1(bool half_off_diagonal);
    void get_ReA1(bool half_off_diagonal);

    void apply_full_L(double * target, bool transpose, double * output);

    int get_sym_index(int, int);
    void get_indices_from_sym_index(int index, int &a, int &b);
public:
    // Constructors
    Lanczos(string rootname);
    ~Lanczos();

    // Load everything from the input files
    void setup_from_input(string rootname);

    

    // Run the lanczos algorithm
    void run();
};


#endif

