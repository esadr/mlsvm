#ifndef COMMON_FUNCS_H
#define COMMON_FUNCS_H

#include <petscmat.h>
#include <string>
#include <iostream>
#include <vector>

class CommonFuncs{

public:
    void exp_matrix(Mat& A, std::string file_path, std::string file_name);
    void exp_matrix(Mat& A, std::string file_path, std::string file_name, std::string sender_func);
    void exp_vector(Vec& A, std::string file_path, std::string file_name);
    void exp_vector(Vec& A, std::string file_path, std::string file_name, std::string sender_func);

    double calc_euclidean_dist(const PetscInt ncols_A, const PetscInt ncols_B,
                                            const PetscInt *cols_A, const PetscInt *cols_B,
                                            const PetscScalar *vals_A, const PetscScalar *vals_B);

    double calc_manhatan_dist(const PetscInt ncols_A, const PetscInt ncols_B,
                                            const PetscInt *cols_A, const PetscInt *cols_B,
                                            const PetscScalar *vals_A, const PetscScalar *vals_B);


    PetscScalar vec_vec_dot_product(const PetscInt ncols_A, const PetscInt *cols_A,
                                            const PetscScalar *vals_A, const std::vector<double>& v_B);


//    template <class T>
//    T sum_array(const std::vector<T>& vec_In){
//        T total=0;
//        for(unsigned i=0; i < vec_In.size(); ++i){
//            total += vec_In[i];
//        }
//        std::cout << "total:" << total << std::endl;
//        return total;
//    }

    template <typename T>
    T sum_vector(const std::vector<T>& vec_In);

    template <typename T>
    PetscScalar mean_vector(const std::vector<T>& vec_In);

    template <class T>
    PetscScalar STD_vector(T *arrayIn, unsigned int array_size, PetscScalar& mean);

    template <class T>
    void zscore_vector(T *arrayIn, unsigned int array_size);

    template <typename T>
    T sum_array(T *arrayIn, unsigned int array_size);

    template <typename T>
    PetscScalar mean_array(T *arrayIn, unsigned int array_size);

    template <typename T>
    PetscScalar STD_array(T *arrayIn, unsigned int array_size, PetscScalar& mean);

    template <typename T>
    void zscore_array(T *arrayIn, unsigned int array_size);
};

#endif //COMMON_FUNCS_H




























