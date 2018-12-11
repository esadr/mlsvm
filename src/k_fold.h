#ifndef K_FOLD_H
#define K_FOLD_H

#include <string>
#include "petscmat.h"
#include <vector>
#include <unordered_set>

class k_fold{
private:
    Mat m_in_data_, m_min_data_, m_maj_data_;
    Vec v_in_label_;
    std::vector<PetscInt> min_shuffled_indices_;
    std::vector<PetscInt> maj_shuffled_indices_;
    PetscInt num_data_points_, num_min_points, num_maj_points;

    void cross_validation_class(int curr_iter,int total_iter,
                Mat& m_full_data, Mat& m_train_data, Mat& m_test_data,
                PetscInt * arr_idx_train, PetscInt& train_size,
                std::unordered_set<PetscInt>& uset_test_indices,
                std::vector<PetscInt>& v_full_idx_to_train_idx,
                const std::string& info,
                const std::vector<PetscInt>& v_shuffled_indices,
                bool debug_status=false);

    void filter_NN(Mat& m_full_NN_indices, Mat& m_full_NN_dists,
                std::unordered_set<PetscInt>& uset_test_indices,
                PetscInt * arr_train_indices, PetscInt& train_size,
                std::vector<PetscInt>& v_full_idx_to_train_idx,
                Mat& m_filtered_NN_indices, Mat& m_filtered_NN_dists,
                const std::string& info,bool debug_status=false);

    void extractk_NN(Mat& m_full_NN_indices,
                Mat& m_full_NN_dists,
                Mat& m_filtered_NN_indices,
                Mat& m_filtered_NN_dists,
                const std::string& info);

public:


    /*
     * I guess I should skip this in future, but for now it is very similar to
     *      what I have in the model selection
     * It gets 2 test matrices for both classes and add the labels to the
     *      first column of a new matrix which contains both of them
     * Note: the dt_test_p, dt_test_n will destroy in the end of this function
     *      because they are not needed anymore
     */
    void combine_two_classes_in_one(Mat& test_total, Mat& dt_test_p,
                Mat& dt_test_n, bool destroy_input_matrices=true);


    /*
     * read the input data from file and load it into m_in_data_ matrix
     * read the input label from file and load it into v_in_label_ vector
     * the ds_path and ds_name are read from config_params
     * the _data.dat, _label.dat are used to postfix to the ds_name for data and label files
     * check the size of data and labels and return its result
     */
    void read_in_data(std::string input_train_data="", std::string input_train_label="");


    /*
     * read the input data from file and load it into m_min_data_ matrix and m_maj_data_
     * the ds_path and ds_name are read from config_params
     * the _data.dat, _label.dat are used to postfix to the ds_name for data and label files
     */
    void read_in_divided_data(Mat& m_min_data, Mat& m_maj_data);

    /*
     * read 2 matrices for each class, totally 4 matrices
     */
    void read_in_full_NN(Mat& m_min_NN_indices,Mat& m_min_NN_dists,
                         Mat& m_maj_NN_indices,Mat& m_maj_NN_dists);

    /*
     * Only devide the input data to 2 separate classes
     * I don't need the original data and labels after this
     * so, I release them and keep m_min_data_, m_maj_data_
     */
    void divide_data(bool export_file=false);

    /*
     * Get the size of the data from local variables (num_mXX_points) which are set during the divide_data/read_in_divided_data
     * initialize two local vectors (min_shuffled_indices_ , maj_shuffled_indices_)
     * this happens only once for many Exp and many iterations since the size of data in each class will not change
     */

    void initialize_vectors();


    /*
     * Get the size of the data from local variables (num_mXX_points) which are set during the divide_data
     * Create a random sequence of numbers for them and store them in local vectors (mXX_shuffled_indices_)
     * The seed for srand comes from the config params and it is recorded in output for debug
     */
    void shuffle_data(std::string preferred_srand="",bool debug_status=false);

    /*
     * Get the random sequence from local vectors for both class
     * At each iteration, take i-th subset of each vector and sort it
     * Then get submatrix for that as test part for that class
     * Take the rest as training for that class
     * Combine both test parts with labels as one test matrix
     * Note: iteration should start from ZERO
     */
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
    void cross_validation_simple(Mat& m_data_p, Mat& m_data_n,
                 Vec& v_vol_p, Vec& v_vol_n,
                 int current_iteration, int total_iterations,
                 Mat& m_train_data_p, Mat& m_train_data_n, Mat& m_test_data,
                 Vec& v_train_vol_p, Vec& v_train_vol_n);


    /*
     * I won't need this after I call Flann directly, but till then I need to export the
     * data for each class to petsc binary format and read them in another script
     * to calculate the flann for training parts and save them in petsc binary format
     * again and read them
     */
//    void write_output(char f_name[PETSC_MAX_PATH_LEN], Mat m_Out);    //write the output to file
    void write_output(std::string f_name, Mat m_Out, std::string desc="");    //write the output to file

    void prepare_data_for_iteration(int current_iteration,int total_iterations,
            Mat& m_min_full_data,Mat& m_min_train_data,
            Mat& m_min_full_NN_indices,Mat& m_min_full_NN_dists,
            Mat& m_min_WA,Vec& v_min_vol, Mat& m_maj_full_data,
            Mat& m_maj_train_data,Mat& m_maj_full_NN_indices,
            Mat& m_maj_full_NN_dists,Mat& m_maj_WA,Vec& v_maj_vol,
            bool debug_status=false);

    /*
     * Date : Aug 2018
     * Purpose: work with separated test data
     * Skip the cross validation
     *
     * Note:
     * 1- The full_NN with large number of neighbors still needs to be filtered
     *      apparently, if I pass the a empty set of test indices, all the
     *      points are returned with first desired number of neighbors using
     *      filter_NN function.
     *
     * 2- WA matrices should be created.
     * 3- make sure the test file is exist
     *
     */

    void prepare_data_using_separate_testdata(
            Mat& m_min_full_data, Mat& m_min_full_NN_indices,
            Mat& m_min_full_NN_dists, Mat& m_min_WA, Vec& v_min_vol,
            Mat& m_maj_full_data, Mat& m_maj_full_NN_indices,
            Mat& m_maj_full_NN_dists,Mat& m_maj_WA,Vec& v_maj_vol);


    Mat get_m_min_data(){ return m_min_data_ ;}
    Mat get_m_maj_data(){ return m_maj_data_ ;}
    void free_resources();


};

#endif // K_FOLD_H


