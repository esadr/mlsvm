#ifndef COMMON_FUNCS_H
#define COMMON_FUNCS_H

#include <petscmat.h>
#include <string>
#include <iostream>
#include <vector>

class CommonFuncs{
public:
    void exp_matrix(Mat& A, std::string file_path,
                    std::string file_name);
    void exp_matrix(Mat& A, std::string file_path,
                    std::string file_name, std::string sender_func);
    void exp_vector(Vec& A, std::string file_path,
                    std::string file_name);
    void exp_vector(Vec& A, std::string file_path,
                    std::string file_name, std::string sender_func);

    double calc_euclidean_dist(const PetscInt ncols_A,
                               const PetscInt ncols_B,
                               const PetscInt *cols_A,
                               const PetscInt *cols_B,
                               const PetscScalar *vals_A,
                               const PetscScalar *vals_B);

    double calc_manhatan_dist(const PetscInt ncols_A,
                              const PetscInt ncols_B,
                              const PetscInt *cols_A,
                              const PetscInt *cols_B,
                              const PetscScalar *vals_A,
                              const PetscScalar *vals_B);

    void set_weight_type(int new_weight_type, double aux_param);
    double convert_distance_to_weight(double distance);

    PetscScalar vec_vec_dot_product(const PetscInt ncols_A,
                                    const PetscInt *cols_A,
                                    const PetscScalar *vals_A,
                                    const std::vector<double>& v_B);


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
    PetscScalar STD_vector(T *arrayIn,
                           unsigned int array_size,
                           PetscScalar& mean);

    template <class T>
    void zscore_vector(T *arrayIn,
                       unsigned int array_size);

    template <typename T>
    T sum_array(T *arrayIn, unsigned int array_size);

    template <typename T>
    PetscScalar mean_array(T *arrayIn,
                           unsigned int array_size);

    template <typename T>
    PetscScalar STD_array(T *arrayIn,
                          unsigned int array_size,
                          PetscScalar& mean);

    template <typename T>
    void zscore_array(T *arrayIn,
                      unsigned int array_size);

    Mat sample_data(Mat& m_data,
                    float sample_size_fraction,
                    std::string preferred_srand);

    std::string run_ext_command(const std::string ext_cmd);
    std::string run_ext_command_single_output(const std::string ext_cmd);

    void get_unique_random_id(int start_range, int end_range,
                              int number_random_id,
                              std::string preferred_srand,
                              std::vector<int>& v_rand_idx);

    /*
     * Add a label vector to the first column of data matrix
     * Purpose: separate test data needs to have the labels in the first column.
     */
    void addLabel2Data(Mat& m_data, Vec& v_label, Mat& m_label_data);

    /*
     * check if a directory exist, return True and otherwise return False
     * https://www.unix.com/programming/24159-how-check-if-directory-file-exist-using-c-c.html#neo-top
     */
    bool checkDirectoryExistance(std::string directory);

    /*
     * create a requested directory in case it is not exist
     *
     */
    bool createDirectory(std::string directory_full_path);

private:
    int weight_type=0;
    double weight_gamma=0;
};

#endif //COMMON_FUNCS_H
