#ifndef K_FOLD_H
#define K_FOLD_H

#include <string>
#include "petscmat.h"
#include <vector>

class k_fold{
private:
    Mat m_in_data_, m_min_data_, m_maj_data_;
    Vec v_in_label_;
    std::vector<PetscInt> min_shuffled_indices_;
    std::vector<PetscInt> maj_shuffled_indices_;
    PetscInt num_data_points_, num_min_points, num_maj_points;

public:


    /*
     * read the input data from file and load it into m_in_data_ matrix
     * read the input label from file and load it into v_in_label_ vector
     * the ds_path and ds_name are read from config_params
     * the _data.dat, _label.dat are used to postfix to the ds_name for data and label files
     * check the size of data and labels and return its result
     */
    void read_in_data(std::string input_train_data="", std::string input_train_label="");

    /*
     * Only devide the input data to 2 separate classes
     * I don't need the original data and labels after this
     * so, I release them and keep m_min_data_, m_maj_data_
     */
    void divide_data();

    /*
     * Get the size of the data from local variables (num_mXX_points) which are set during the divide_data
     * Create a random sequence of numbers for them and store them in local vectors (mXX_shuffled_indices_)
     * The seed for srand comes from the config params and it is recorded in output for debug
     */
    void shuffle_data(std::string preferred_srand="");

    /*
     * Get the random sequence from local vectors for both class
     * At each iteration, take i-th subset of each vector and sort it
     * Then get submatrix for that as test part for that class
     * Take the rest as training for that class
     * Combine both test parts with labels as one test matrix
     * Note: iteration should start from ZERO
     */
//    bool cross_validation(int current_iteration,int total_iterations, Mat& m_min_train_data, Mat& m_maj_train_data, Mat& m_test_data);
    // refer to reports
    void cross_validation(int current_iteration,int total_iterations,
                          std::string p_data_fname="",
                          std::string n_data_fname="",
                          std::string test_data_fname="");
    /*
     * for debugging a specific train and test data, we write the output of divide data to files
     * This make it possible for Flann to read the data and calculate the K-NN
     */
    void prepare_traindata_for_flann();

    /*
     * Not shuffle data, only partition data
     * At each iteration, take i-th subset of data as submatrix to produce the test part for that class
     * Take the rest as training for that class
     * Combine both test parts with labels as one test matrix
     * Note: iteration should start from ZERO
     */
    void cross_validation_simple(Mat& m_data_p, Mat& m_data_n, Vec& v_vol_p, Vec& v_vol_n,
                                 int current_iteration, int total_iterations,
                                 Mat& m_train_data_p, Mat& m_train_data_n, Mat& m_test_data,
                                 Vec& v_train_vol_p, Vec& v_train_vol_n);

    /*
     * I guess I should skip this in future, but for now it is very similar to what I have in the model selection
     * It gets 2 test matrices for both classes and add the labels to the first column of a new matrix
     * which contains both of them
     * Note: the dt_test_p, dt_test_n will destroy in the end of this function because they are not needed anymore
     */
    void combine_test_data(Mat& test_total, Mat& dt_test_p, Mat& dt_test_n);

    /*
     * I won't need this after I call Flann directly, but till then I need to export the
     * data for each class to petsc binary format and read them in another script
     * to calculate the flann for training parts and save them in petsc binary format
     * again and read them
     */
//    void write_output(char f_name[PETSC_MAX_PATH_LEN], Mat m_Out);    //write the output to file
    void write_output(std::string f_name, Mat m_Out);    //write the output to file
//    /*
//     * release the local matrix and vector for input data
//     */
//    ~kfold(){
//        MatDestroy(&m_in_data_);
//        VecDestroy(&v_in_label_);
//    }

};

#endif // K_FOLD_H


