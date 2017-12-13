#ifndef LOADER_H
#define LOADER_H

#include "etimer.h"
//#include <petscviewerhdf5.h>    //for hdf5
#include <petscmat.h>
#include <petscvec.h>
#include "config_params.h"
#include "config_logs.h"
//#include <iostream>             //for cout (printing times)
#include <string>               //for file name

class Loader {

private :
//    std::string f_indices_name_, ds_indices_name_, f_dists_name_,  ds_dists_name_;
    const char * f_indices_name_;
    const char * ds_indices_name_;
    const char * f_dists_name_;
    const char * ds_dists_name_;
//    PetscInt size_= 0;
//    int weight_type=0;
//    double weight_gamma=0;

    /*
     * converts the distance between to points to edge weight
     * e.g. for Euclidean distance the smaller distance means larger weight which shows the strength of the connection
     */
    PetscScalar convert_distance_to_weight(PetscScalar distance);

public :
//    Loader(const std::string, const std::string ,const std::string , const std::string);
//    Loader(const char *, const char * );
    Loader(const std::string &f_indices_file_name, const std::string &f_dists_file_name);
    Loader(const char *, const char * ,const char * , const char *);
//    Loader(char *);
    Loader(){}   // added at 0819_1555 for loading the test data

    Mat load_WA_binary(const char * );
    Mat load_flann_binary();                //not active method of mlsvm_classifier


    void create_WA_matrix(Mat& m_NN_idx,Mat& m_NN_dis,Mat& m_WA,const std::string& info,bool debug_status=false);


    Mat load_norm_data_sep(const std::string f_name); //load normalized data for each class seperately

    Vec init_volume(PetscScalar, PetscInt num_elements);
//    Vec init_volume(PetscScalar val, int pref_size);    //just for test and debug //deprecated 012717-1713

    /*
     * I don't think I ever use load_train_data or load_test_data,
     * because it comes from the model_selection as I need it there,
     * but here I have another loader for reading a matrix from file
     */
    Mat load_test_data(const char * f_name, int& num_node, int& num_elem);
    Mat load_train_data(const char * f_name, int& num_node, int& num_elem);    //load data
    Mat read_input_matrix(const std::string f_name);  //only for debug and run sample
    Vec read_input_vector(const std::string f_name);  //only for debug and run sample
//    Vec convert_matrix_to_vector(Mat& m_input);
};



#endif // LOADER_H





/*
 * debug level 5 :
 * - count number of zero weight in the flann result
 *
 *
 */
