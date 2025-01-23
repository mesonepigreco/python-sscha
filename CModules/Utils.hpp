#ifndef LANC_UTILS
#define LANC_UTILS
#include <mpi.h>
#include <iostream>
#include <fstream>

bool am_i_the_master(void);
bool is_file_exist(const char *fileName);
#endif