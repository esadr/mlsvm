#ifndef COMMON_FUNCS_H
#define COMMON_FUNCS_H

#include <petscmat.h>
#include <string>
#include <iostream>

class CommonFuncs{

public:
    void exp_matrix(Mat& A, std::string file_path, std::string file_name);
    void exp_matrix(Mat& A, std::string file_path, std::string file_name, std::string sender_func);
    void exp_vector(Vec& A, std::string file_path, std::string file_name);
    void exp_vector(Vec& A, std::string file_path, std::string file_name, std::string sender_func);

    double calc_euclidean_dist(const PetscInt ncols_A, const PetscInt ncols_B,
                                            const PetscInt *cols_A, const PetscInt *cols_B,
                                            const PetscScalar *vals_A, const PetscScalar *vals_B);

};

#endif //COMMON_FUNCS_H
