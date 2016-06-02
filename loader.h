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
    PetscInt size_= 0;
    int weight_type=0;
    double weight_gamma=0;
public :
//    Loader(const std::string, const std::string ,const std::string , const std::string);
//    Loader(const char *, const char * );
    Loader(const std::string f_indices_file_name, const std::string f_dists_file_name);
    Loader(const char *, const char * ,const char * , const char *);
//    Loader(char *);
    Loader(){}   // added at 0819_1555 for loading the test data

    Mat load_WA_binary(const char * );
    Mat load_flann_binary();            //current method (Jul 27,2015)



    void set_weight_type(int new_weight_type){
        this->weight_type = new_weight_type;
        if(this->weight_type == 2){ // for guassian distance, it reads the gamma from param.xml
            this->weight_gamma = Config_params::getInstance()->get_ld_weight_param();
        }
    }


    PetscScalar calc_distance(PetscScalar raw_weight);

//    Mat load_norm_data_sep(const char *); //load normalized data for each class seperately
    Mat load_norm_data_sep(const std::string f_name); //load normalized data for each class seperately

    Vec init_volume(PetscScalar);
    Vec init_volume(PetscScalar val, int pref_size);    //just for test and debug

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
