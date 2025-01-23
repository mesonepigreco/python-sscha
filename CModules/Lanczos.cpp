#include "Lanczos.hpp"
#include "Utils.hpp"
#include <cmath>
#include <chrono>

#define DEBUG_READ false
#define DEBUG_LANC false
#define EPSILON 1e-12

using namespace std;
namespace pt = boost::property_tree;

Lanczos::Lanczos(string root_name) {
    rootname = root_name;
    i_step = 0;
    setup_from_input(root_name);
}
Lanczos::~Lanczos() {
    free(nbose);
    free(rho);
    free(w);

    free(N_degeneracy);
    free(blocks_ids);

    // Free good_deg_space
    for (int i = 0; i < n_blocks; ++i){
        free (good_deg_space[i]);
        free(symmetries[i]);
    }
    free(good_deg_space);
    free(symmetries);

    free(X);
    free(Y);
    free(psi);

    free(Qbasis);
    free(Pbasis);
    free(snorm);
    free(a);
    free(b);
    free(c);
}


void Lanczos::setup_from_input(string rootname) {

    pt::ptree root;
    pt::read_json(rootname + ".json", root);

    
    
    // Fill the generic values
    T = root.get<double>("T");
    n_steps = root.get<int>("n_steps");
    ignore_v2 = root.get<bool>("ignore_v2");
    ignore_v3 = root.get<bool>("ignore_v3");
    ignore_v4 = root.get<bool>("ignore_v4");

    // Check the optional key restart (default false)
    int step_done = 0;
    restart = false;
    if (root.count("restart")) {
        restart = root.get<bool>("restart");
        step_done = root.get<int>("step_done");
    }

    if (step_done >= n_steps) {
        throw invalid_argument("Error, n_steps must be greather than step_done.\n");
    }

    // Now read the more complex values
    pt::ptree &data = root.get_child("data");
    N = data.get<int>("n_configs");
    n_modes = data.get<int>("n_modes");
    n_syms = data.get<int>("n_syms");
    n_blocks = data.get<int>("n_blocks");

    reverse_L = data.get<bool>("reverse");
    shift_value = data.get<double>("shift");
    
    n_psi = n_modes + n_modes * (n_modes + 1);

    if (DEBUG_READ && am_i_the_master()) {
        cout << "[DEBUG READ] N = " << N << endl;
        cout << "[DEBUG READ] n_modes = " << n_modes << endl;
        cout << "[DEBUG READ] n_syms = " << n_syms << endl;
        cout << "[DEBUG READ] reverse_L = " << reverse_L << endl;
        cout << "[DEBUG READ] shift_value = " << shift_value << endl;
        cout << "[DEBUG READ] n_psi = " << n_psi << endl;
    }


    // Allocate the memory for the data
    N_degeneracy = (int*) malloc(sizeof(int) * n_modes);
    w = (double*) malloc(sizeof(double) * n_modes);
    nbose = (double*) malloc(sizeof(double) * n_modes);
    rho = (double *) malloc(sizeof(double) * N);
    psi = (double *) malloc(sizeof(double) * n_psi);
    blocks_ids = (int*) malloc(sizeof(int) * n_modes);

    //Ups1 = (double*) calloc(sizeof(double), n_modes*n_modes);
    //ReA1 = (double*) calloc(sizeof(double), n_modes*n_modes);

    X = (double *) malloc(sizeof(double) * N * n_modes);
    Y = (double *) malloc(sizeof(double) * N * n_modes);

    a = (double*) calloc(sizeof(double), n_steps);
    b = (double*) calloc(sizeof(double), n_steps);
    c = (double*) calloc(sizeof(double), n_steps);



    Qbasis = (double*) calloc(sizeof(double), n_psi * (n_steps + 1));
    Pbasis = (double*) calloc(sizeof(double), n_psi * (n_steps + 1));
    snorm = (double*) calloc(sizeof(double), n_steps + 1);
    
    // Read the arrays from the files
    fstream file(rootname + ".ndegs");
    if (file.is_open()) {
        string line;
        for (int k = 0; k < n_blocks; ++k) {
            file >> N_degeneracy[k];
        }
    }
    file.close();

    file.open(rootname + ".freqs");
    if (file.is_open()) {
        string line;
        for (int k = 0; k < n_modes; ++k) {
            file >> w[k];
        }
    }
    file.close();


    file.open(rootname + ".rho");
    if (file.is_open()) {
        string line;
        int counter = 0;
        for (int k = 0; k < N; ++k) {
            file >> rho[k];
        }
    }
    file.close();

    file.open(rootname + ".psi");
    if (file.is_open()) {
        string line;
        for (int k = 0; k < n_psi; ++k) {
            file >> psi[k];
        }
    }
    file.close();


    // Now read the 2D arrays (TODO: read also previous steps)
    file.open(rootname + ".X.dat");
    if (file.is_open()) {
        string line;
        for (int k = 0; k < N*n_modes ; ++k) {
            file >> X[k];
        }
    }
    file.close();


    file.open(rootname + ".Y.dat");
    if (file.is_open()) {
        string line;
        for (int k = 0; k < N*n_modes ; ++k) {
            file >> Y[k];
        }
    }
    file.close();


    // Check if we need a restart
    // And load P and Q
    if (restart) {
        file.open(rootname + ".abc");
        if (file.is_open()) {
            string line;
            for (int k = 0; k < step_done ; ++k) {
                file >> a[k];
                file >> b[k];
                file >> c[k];
            }
        }
        file.close();

        file.open(rootname + ".qbasis.out");
        if (file.is_open()) {
            string line;
            for (int k = 0; k < n_psi * (step_done+1) ; ++k) {
                file >> Qbasis[k];
            }
        }
        file.close();

        file.open(rootname + ".pbasis.out");
        if (file.is_open()) {
            string line;
            for (int k = 0; k < n_psi * (step_done+1) ; ++k) {
                file >> Pbasis[k];
            }
        }
        file.close();

        file.open(rootname + ".snorm.out");
        if (file.is_open()) {
            string line;
            for (int k = 0; k < step_done + 1; ++k) {
                file >> snorm[k];
            }
        }
        file.close();

        i_step = step_done;
    }


    file.open(rootname + ".blockid");
    if (file.is_open()) {
        string line;
        for (int k = 0; k < n_modes; ++k) {
            file >> blocks_ids[k];
        }
    }
    file.close();


    // Get the length of the degenerate space
    good_deg_space = (int**) malloc(sizeof(int*) * n_blocks);
    symmetries = (double**) malloc(sizeof(double*) * n_blocks);
    for (int i = 0; i < n_blocks; ++i) {
        good_deg_space[i] = (int*) calloc(sizeof(int), N_degeneracy[i]);
        symmetries[i] = (double*) calloc(sizeof(double), n_syms * N_degeneracy[i] * N_degeneracy[i]);

        file.open(rootname + ".block" + to_string(i));
        if (file.is_open()) {
            for (int j = 0; j < N_degeneracy[i]; ++j) {
                file >> good_deg_space[i][j];
            }
        }
        file.close();

        file.open(rootname + ".symsb" + to_string(i));
        if (file.is_open()) {
            for (int j = 0; j < n_syms*N_degeneracy[i]*N_degeneracy[i]; ++j) {
                file >> symmetries[i][j];
            }
        }
        file.close();
    }

    if (DEBUG_LANC && am_i_the_master()) 
        cout << "PSI POINTER (LINE " << __LINE__ << "): " << psi << endl;

    // Update the Bose-Einstein statistics
    update_nbose();


    if (DEBUG_LANC && am_i_the_master()) 
        cout << "PSI POINTER (LINE " << __LINE__ << "): " << psi << endl;

}

void Lanczos::update_nbose() {
    for (int i = 0; i < n_modes; ++i) {
        nbose[i] = 0;

        if (T > 0) {
            nbose[i] = 1.0 / (exp( (w[i] * RY_TO_K) / T) - 1);
        }
    }
}


void Lanczos::apply_L1(double * out_vect, bool transpose) {
    // Clean the out_vect
    int i, j, k;
    int x, y;
    int N_w2;

    N_w2 = (n_modes * (n_modes + 1)) / 2;


    for (i = 0; i < n_psi; ++i) {
        out_vect[i] = 0;
    }

    // Apply the R - R propagation
    for (i = 0; i < n_modes; ++i) {
        out_vect[i] = (psi[i] * w[i]) *w[i];
    }

    int start_Y = n_modes;
    int start_A = n_modes + N_w2;

    double X_ni, Y_ni, X1_ni, Y1_ni;
    double den;
    for (i = 0; i < N_w2; ++i) {
        get_indices_from_sym_index(i, x, y);
        
        X_ni = -w[x]*w[x];
        X_ni -= w[y]*w[y];
        den = (2*nbose[x] + 1) * (2*nbose[y] + 1);
        X_ni -= (2*w[x]*w[y]) / den;
        
        out_vect[start_Y + i] = - X_ni * psi[start_Y + i];

        Y_ni = -(8*w[x]*w[y]) / den;

        if (transpose) 
            out_vect[start_A + i] += -Y_ni * psi[start_Y + i];
        else
            out_vect[start_Y + i] += -Y_ni * psi[start_A + i];
        
        X1_ni = -(2*nbose[x]*nbose[y] + nbose[x] + nbose[y]);
        X1_ni *= 2*nbose[x]*nbose[y] + nbose[x] + nbose[y] + 1;
        X1_ni *= 2*w[x]*w[y];
        X1_ni /= den;

        if (transpose)
            out_vect[start_Y + i] += -X1_ni * psi[start_A + i];
        else
            out_vect[start_A + i] += -X1_ni * psi[start_Y + i];
        
        Y1_ni = -w[x]*w[x] - w[y]*w[y] + (2*w[x]*w[y]) / den;

        if (i < 10  && DEBUG_LANC && am_i_the_master()) {
            cout << scientific << setprecision(8);
            cout << "X1[" << i << "] = " << X1_ni << endl;
            cout << "Y1[" << i << "] = " << Y1_ni << endl;
        }

        out_vect[start_A + i] += -Y1_ni * psi[start_A + i];
    }

    if (DEBUG_LANC && am_i_the_master()) {
        cout << scientific << setprecision(8) << endl;
        cout << "Harmonic interacting output:" << endl;
        cout << "From start_Y:" << endl;
        for (int i = 0; i < 10 ; ++i) {
            cout << out_vect[start_Y + i] << " " ;
        }
        cout << endl << "From start_A:" << endl;
        for (int i = 0; i < 10 ; ++i) {
            cout << out_vect[start_A + i] << " " ;
        }cout << endl;
        cout << "Old psi from start_Y:" << endl;
        for (int i = 0; i < 10 ; ++i) {
            cout << psi[start_Y + i] << " " ;
        }
        cout << endl << "Old psi start_A:" << endl;
        for (int i = 0; i < 10 ; ++i) {
            cout << psi[start_A + i] << " " ;
        }cout << endl;
    }
}


void Lanczos::apply_anharmonic(double * final_psi, bool transpose) {
    int N_w2 = (n_modes * (n_modes + 1)) / 2;
    int x, y;
    double * Y1 = (double*) calloc(sizeof(double), N_w2);
    double * R1 = (double*) calloc(sizeof(double), n_modes);

    int start_Y = n_modes;
    int start_A = n_modes + N_w2;

    double Y_wa, Y_wb, coeff_Y, ReA_w1, ReA_w2, coeff_RA;

    // Fill the R perturbation
    for (int i = 0; i < n_modes; ++i) {
        R1[i] = psi[i];
    }

    if (transpose) {
        for (int i = 0; i < N_w2; ++i) {
            get_indices_from_sym_index(i, x, y);

            Y_wa =  2 * w[x] / (2 * nbose[x] + 1);
            Y_wb =  2 * w[y] / (2 * nbose[y] + 1);
            coeff_Y = 2* (Y_wa*Y_wb*Y_wb + Y_wa*Y_wa*Y_wb);

            ReA_w1 = 2*w[x]*nbose[x] * (nbose[x] + 1) / (2*nbose[x] + 1);
            ReA_w2 = 2*w[y]*nbose[y] * (nbose[y] + 1) / (2*nbose[y] + 1);
            coeff_RA = 2*(Y_wa*ReA_w2*Y_wb + Y_wa*ReA_w1*Y_wb);

            Y1[i] = psi[start_Y + i] * coeff_Y;
            Y1[i] += psi[start_A + i] * coeff_RA;
        }
    } else {
        for (int i = 0; i <  N_w2; ++i)
            Y1[i] = psi[start_Y + i];
    }


    // Compute the sscha average force and potential
    double * f_pert_av = (double*) calloc(sizeof(double), n_modes);
    double * d2v_pert_av = (double*) calloc(sizeof(double), n_modes*n_modes);

    double * Y1_new = (double*) malloc(sizeof(double) * n_modes*n_modes);
    for (int i = 0; i < N_w2; ++i) {
        get_indices_from_sym_index(i, x, y);
        if (x != y && transpose) {
            Y1_new[n_modes*x + y] = Y1[i] / 2;
            Y1_new[n_modes*y + x] = Y1[i] / 2;
        } else {
            Y1_new[n_modes*x + y] = Y1[i] ;
            Y1_new[n_modes*y + x] = Y1[i] ;
        }
    }


    if (! ignore_v3) {
        get_f_average_from_Y_pert_sym_fast(X, Y, w, Y1_new, T, n_modes, N, rho, symmetries, n_syms, N_degeneracy, good_deg_space, blocks_ids, f_pert_av);
        get_d2v_dR2_from_R_pert_sym_fast(X, Y, w, R1, T, n_modes, N, rho, symmetries, n_syms, N_degeneracy, good_deg_space, blocks_ids, d2v_pert_av);
    }

    // cout << endl;
    // cout << "D2v[...10, ...10] = " << scientific << setprecision(3) << endl;
    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < 10; ++j) 
    //         cout << d2v_pert_av[n_modes * i + j] << " ";
    //     cout << endl;
    // }
    // cout << endl;


    if (! ignore_v4) {
        // This subroutine gives seg. fault if optimized with -O3
        get_d2v_dR2_from_Y_pert_sym_fast(X, Y, w, Y1_new, T, n_modes, N, rho, symmetries, n_syms, N_degeneracy, good_deg_space, blocks_ids, d2v_pert_av);
    }



    // Free memory 
    free(Y1);
    free(Y1_new);
    free(R1);


    for (int i = 0; i < n_modes; ++i) {
        final_psi[i] += -f_pert_av[i];
    }

    double pert_Y, pert_RA;

    for (int i = 0; i < N_w2; ++i) {
        get_indices_from_sym_index(i, x, y);

        Y_wa =  2 * w[x] / (2 * nbose[x] + 1);
        Y_wb =  2 * w[y] / (2 * nbose[y] + 1);

        ReA_w1 = 2*w[x]*nbose[x] * (nbose[x] + 1) / (2*nbose[x] + 1);
        ReA_w2 = 2*w[y]*nbose[y] * (nbose[y] + 1) / (2*nbose[y] + 1);

        if (transpose) {
            pert_Y = 0.5 *  d2v_pert_av[x*n_modes + y] / (Y_wa * Y_wb);
            pert_RA = 0;

            if (x != y) pert_Y *= 2;
        } else {
            pert_Y = d2v_pert_av[x*n_modes + y] * (Y_wa + Y_wb);
            pert_RA = d2v_pert_av[x*n_modes + y] * (ReA_w1 + ReA_w2);
        }

        final_psi[start_Y + i] += -pert_Y;
        final_psi[start_A + i] += -pert_RA;
        
    }

    if (DEBUG_LANC && am_i_the_master()) {
        cout << "Final psi [from 45...]:" << endl;
        for (int i = 0; i < 10; ++i) {
            cout << final_psi[start_Y + i] << " ";
        }cout << endl;
    }

    free(f_pert_av);
    free(d2v_pert_av);
}


void Lanczos::apply_full_L(double * target, bool transpose, double * output) {
    // Copy the target into psi
    if (!(target == NULL)) {
        for (int i = 0; i < n_psi; ++i) psi[i] = target[i];
    }

    // Delete the output
    for (int i = 0; i < n_psi; ++i) {
        output[i] = 0;
    }

    auto t1 = chrono::steady_clock::now();
    if (!ignore_v2)
        apply_L1(output, transpose);

    if ((!ignore_v3) || (!ignore_v4))
        apply_anharmonic(output, transpose);

    auto t2 = chrono::steady_clock::now();

    auto diff = t2- t1;

    if (am_i_the_master())
        cout << "Time to apply the L matrix: " <<  chrono::duration <double, milli> (diff).count() << " ms" << endl;



    // Reverse the output if requested
    if (reverse_L) {
        for (int i = 0; i < n_psi; ++i)
            output[i] *= -1;
    }
    if (shift_value != 0) {
        for (int i = 0; i < n_psi; ++i) 
            output[i] += shift_value;
    }

    // Copy the output on self.psi
    for (int i = 0; i < n_psi; ++i) 
        psi[i] = output[i];
}


void Lanczos::run() {
    // Run the lanczos algorithm
    double * psi_q, *psi_p;

    double * L_q = (double*) malloc(sizeof(double) * n_psi);
    double * p_L = (double*) malloc(sizeof(double) * n_psi);

    // Store the lenght of the variables
    int lena = i_step, lenb = i_step, lenc = i_step, lens = i_step + 1;
    int len_bq = i_step + 1, len_bp = i_step + 1;
    
    // Prepare the first vector computing the norm of psi
    double psi_norm = 0;
    for (int i = 0; i < n_psi; ++i) psi_norm += psi[i] * psi[i];
    psi_norm = sqrt(psi_norm);


    // Fill the first vector of P and Q basis

    psi_q = (double *) malloc(sizeof(double) * n_psi);
    psi_p = (double *) malloc(sizeof(double) * n_psi);
    double * sk_tilde = (double*) malloc(sizeof(double) * n_psi);

    if (i_step == 0) {
        for (int i = 0; i < n_psi; ++i) {
            Qbasis[i] = psi[i] / psi_norm;
            Pbasis[i] = psi[i] / psi_norm;
        }
        snorm[0] = 1;
    } 


    for (int j = 0; j < n_psi; ++j) {
        psi_q[j] = Qbasis[i_step * n_psi + j];
        psi_p[j] = Pbasis[i_step * n_psi + j];
    }

    double a_coeff, b_coeff, c_coeff;

    // Open the file for writing
    fstream file_abc;
    fstream file_qbasis;//(rootname + ".qbasis.out", fstream::out);
    fstream file_pbasis;
    fstream file_snorm;
    
    if (am_i_the_master()) {
        if (restart) {
            file_abc.open(rootname + ".abc", fstream::out | fstream::app);
            file_qbasis.open(rootname + ".qbasis.out", fstream::out | fstream::app);
            file_pbasis.open(rootname + ".pbasis.out", fstream::out | fstream::app);
            file_snorm.open(rootname + ".snorm.out", fstream::out | fstream::app);
        } else {
            file_abc.open(rootname + ".abc", fstream::out);
            file_qbasis.open(rootname + ".qbasis.out", fstream::out);
            file_pbasis.open(rootname + ".pbasis.out", fstream::out);
            file_snorm.open(rootname + ".snorm.out", fstream::out);
        }
    }

    // Setup the required precision after the restart
    if (file_qbasis.is_open()) 
        file_qbasis << scientific << setprecision(16);

    if (file_pbasis.is_open())
        file_pbasis << scientific << setprecision(16);

    if (file_snorm.is_open())
        file_snorm << scientific << setprecision(16);

    if (! restart) {
        if (file_qbasis.is_open()) {
            file_qbasis << scientific << setprecision(16);
            for (int j = 0; j < n_psi; ++j) {
                file_qbasis << psi_q[j] << " ";
            }
            file_qbasis << endl;

        } 
        if (file_pbasis.is_open()) { 
            file_pbasis << scientific << setprecision(16);
            for (int j = 0; j < n_psi; ++j) {
                file_pbasis << psi_p[j] << " ";
            }
            file_pbasis << endl;
        }

        if (file_snorm.is_open()) {
            file_snorm << scientific << setprecision(16) << snorm[0] << endl << flush;
        }
    }

    // Here the run
    bool next_converged = false;
    bool converged = false;


    if (DEBUG_LANC && am_i_the_master()) 
        cout << "PSI POINTER (LINE " << __LINE__ << "): " << psi << endl;

    for (int i = i_step; i < n_steps; ++i) {
        if (am_i_the_master()) {
            cout << endl;
            cout << "===== NEW STEP " << i + 1 <<" =====" <<endl << endl;
        }

        // Apply Lq and pL
        // This is the most time consuming part of the code
        apply_full_L(psi_q, false, L_q);
        apply_full_L(psi_p, true, p_L);

        if (DEBUG_LANC && am_i_the_master()) {
            cout << "L_q [from " << n_modes << "]" << endl; 
            for(int j = n_modes; j < n_modes + 10; ++j) {
                cout << scientific << setprecision(3) << L_q[j] << " ";
            }
            cout << endl;
        }
        if (DEBUG_LANC && am_i_the_master()) {
            cout << "p_L [from " << n_modes << "]" << endl; 
            for(int j = n_modes; j < n_modes + 10; ++j) {
                cout << scientific << setprecision(3) << p_L[j] << " ";
            }
            cout << endl;
        }

        double c_old = 1;
        if (lenc > 0) {
            c_old = c[lenc-1];
        }

        if (DEBUG_LANC && am_i_the_master()) {
            double L_qmod = 0, p_Lmod = 0;
            for (int j = 0; j < n_psi; ++j) {
                L_qmod += L_q[j] * L_q[j];
                p_Lmod += p_L[j] * p_L[j];
            }
            L_qmod = sqrt(L_qmod);
            p_Lmod = sqrt(p_Lmod);

            // cout << scientific << setprecision(8);
            // cout << "Modulus of L_q: " << L_qmod << endl;
            // cout << "Modulus of p_L: " << p_Lmod << endl;

            // cout << "(len  BP: " << len_bp << " BQ: " << len_bq << " C: " << lenc << " )"<< endl;
        }



        double p_norm = snorm[lens-1] / c_old;
        double old_p_norm = 0;
        if (DEBUG_LANC && am_i_the_master()) 
            cout << "p_norm: " << setprecision(6) << p_norm << endl;

        a_coeff = 0;
        for (int j = 0; j < n_psi; ++j) {
            a_coeff += psi_p[j] * L_q[j];
        }
        a_coeff *= p_norm;

        // Get the residual
        for (int j = 0; j < n_psi; ++j) {
            L_q[j] -= a_coeff * psi_q[j];
            if (len_bq > 1) {
                L_q[j] -= c[lenc - 1] * Qbasis[ n_psi * (len_bq - 2) + j];
            }

            p_L[j] -= a_coeff * psi_p[j];
            if (len_bp > 1) {
                if (lenc < 2) 
                    old_p_norm = snorm[lens - 2];
                else
                    old_p_norm = snorm[lens - 2] / c[lenc - 2];
                
                p_L[j] -= b[lenb - 1] * Pbasis[n_psi * (len_bp - 2) + j] * (old_p_norm / p_norm);
            }
        }

        double s_norm_coeff = 0;
        b_coeff = 0;
        c_coeff = 0;
        for (int j = 0; j < n_psi; ++j) {
            b_coeff += L_q[j] * L_q[j];
            s_norm_coeff += p_L[j] * p_L[j];
        }
        s_norm_coeff = sqrt(s_norm_coeff);
        b_coeff = sqrt(b_coeff);
        if (DEBUG_LANC && am_i_the_master()) {
            cout << "Modulus of sk: " << s_norm_coeff << endl;
            cout << "Modulus of rk: " << b_coeff << endl;
            cout << "old_p_norm: " << old_p_norm << endl;

        }

        for (int j = 0; j < n_psi; ++j) {
            sk_tilde[j] = p_L[j] / s_norm_coeff;
        }
        s_norm_coeff *= p_norm;
        for (int j = 0; j < n_psi; ++j) {
            c_coeff += (sk_tilde[j] * (L_q[j] / b_coeff)) * s_norm_coeff;
        }

        if (DEBUG_LANC && am_i_the_master()) {
            cout << "New p norm: " << s_norm_coeff / c_coeff << endl;
        }

        // Append the lanczos
        a[lena++] = a_coeff;

        // Check if the algorithm converged
        if (abs(b_coeff) < EPSILON || next_converged) {
            converged = true;
            break;
        }
        if (abs(c_coeff) < EPSILON || next_converged) {
            converged = true;
            break;
        }

        // Fill the psi_q and psi_o vectors
        for (int j = 0; j < n_psi; ++j) {
            psi_q[j] = L_q[j] / b_coeff;
            psi_p[j] = sk_tilde[j];
        }

        converged = false;

        // Update the basis
        for(int j = 0; j < n_psi; ++j) {
            Qbasis[ n_psi * (len_bq) + j] = psi_q[j];
            Pbasis[ n_psi * len_bp + j] = psi_p[j];
        }
        len_bq++;
        len_bp++;

        // Update the coefficients
        b[lenb++] = b_coeff;
        c[lenc++] = c_coeff;
        snorm[lens++] = s_norm_coeff;


        // Write on output
        if (am_i_the_master()) {
            cout << "Lanczos coefficients:" << endl << endl;
            cout << "a_" << i << " = " << scientific << setprecision(16) << a_coeff << endl;
            cout << "b_" << i << " = " << b_coeff << endl;
            cout << "c_" << i << " = " << c_coeff << endl << endl << fixed;


            if (file_abc.is_open()) {
                file_abc << scientific << setprecision(16) << a_coeff << "\t" << b_coeff << "\t" << c_coeff << endl << flush;
            } else {
                cerr << "ERROR: FILE ABC not opened" << endl;
            }

            if (file_snorm.is_open()) {
                file_snorm << snorm[lens - 1] << endl << flush;
            }

            if(file_qbasis.is_open() && file_pbasis.is_open()) {
                for (int j = 0; j < n_psi; ++j) {
                    file_qbasis << psi_q[j] << " ";
                    file_pbasis << psi_p[j] << " ";
                }
                file_qbasis << endl << flush;
                file_pbasis << endl << flush;
            }
        }
    }

    if (converged && am_i_the_master()) {
        cout << "  Converged." << endl;
        cout << " The last a coefficient is " << scientific << setprecision(16) << a_coeff <<endl;
        file_abc << a_coeff << endl;
    }
    
    if (am_i_the_master()) {
        cout << "I'm done with the calculation!" << endl;
    }

    file_abc.close();
    free(L_q);
    free(p_L);
    free(psi_p);
    free(psi_q);
    free(sk_tilde);
}


int Lanczos::get_sym_index(int a, int b) {
    int ret = 0;

    if (b < a) return get_sym_index(b, a);

    for (int i = 0; i < a; ++i)
        ret += n_modes - i;
    ret += b - a;

    /* int ret = 0;

    if (b > a) return get_sym_index(b, a);

    for (int i = 0; i < a; ++i) {
        ret += i + 1;
    } 
    ret += b;*/
    return ret; 
}

void Lanczos::get_indices_from_sym_index(int index, int &a, int &b) {
    int i, j;
    int new_i = index;
    a = 0; b = 0;
    while (new_i >= n_modes - a) {
        new_i -= n_modes - a;
        a++;
    }
    b = new_i + a;
   /*  int i, j;
    int counter = 0;
    a = 0;
    b = 0;
    for (i= 0; i < index; ++i) {
        b++;
        if (b > a) {
            b = 0;
            a++;
        }
    } */
}
