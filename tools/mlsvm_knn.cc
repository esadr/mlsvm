#include "../config_params.h"
#include <flann/flann.hpp>
//#include <flann/util/serialization.h>
#include "../loader.h"
#include "../common_funcs.h"


Config_params* Config_params::instance = NULL;

//using namespace flann;

void run_flann(Mat& m_data, Mat& m_indices, Mat& m_dists);

int main(int argc, char **argv){
    PetscInitialize(&argc, &argv, NULL, NULL);
    ETimer t_all;
    //read XML parameters
    Config_params::getInstance()->read_params("./params.xml", argc, argv, Config_params::flann); //@ 040317-1842
    /* ------------------------- Prepare FLANN Commands---------------------------- */
    std::string str_NN {std::to_string(Config_params::getInstance()->get_nn_number_of_neighbors())};
    std::string str_nn_distance_type {std::to_string(Config_params::getInstance()->get_nn_distance_type())};
//    std::string str_NN_params {" " + str_NN + " "+ str_nn_distance_type} ;

    Loader ld;
    if(Config_params::getInstance()->get_nn_number_of_classes() == 1){
        Mat m_data = ld.load_norm_data_sep(Config_params::getInstance()->get_single_norm_data_f_name());
        Mat m_indices, m_dists;
        run_flann(m_data, m_indices, m_dists);

        /* ------------------------- Run FLANN ---------------------------- */
//        std::cout << "[Main] sh_command for 1 class:" << sh_command << std::endl;
//        system(sh_command.c_str());
    }else{  // two classes
//        /* ------------------------- Divide Data ---------------------------- */
//        ETimer t_kf;
//        k_fold kf;
//        kf.read_in_data();
//        kf.divide_data(true);   //true: export the divided data into files
//        t_kf.stop_timer("[Main] reading and deviding data and writing to files in k-fold class");



//        std::string min_command = py_path +"/python ./scripts/flann.py "+ Config_params::getInstance()->get_p_norm_data_f_name()+ str_NN_params;
//        std::string maj_command = py_path +"/python ./scripts/flann.py "+ Config_params::getInstance()->get_n_norm_data_f_name()+ str_NN_params;
//        /* ------------------------- Run FLANN ---------------------------- */
//        ETimer t_flann_min;
//        std::cout << "[Main] python min_command:" << min_command << std::endl;
//        system(min_command.c_str());
//        t_flann_min.stop_timer("[Main] flann for minority class");

//        ETimer t_flann_maj;
//        std::cout << "[Main] maj_command:" << maj_command << std::endl;
//        system(maj_command.c_str());
//        t_flann_maj.stop_timer("[Main] flann for majority class");
    }
    t_all.stop_timer("[Main] whole saving flann");

    std::cout << "KNN is calculated and saved successfully!\n";
    PetscFinalize();
    return 0;
}


void run_flann(Mat& m_data, Mat& m_indices, Mat& m_dists){
    // - - - - load the data into flann matrix - - - -
    PetscInt num_row, num_col;
    PetscInt i, j, ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    MatGetSize(m_data, &num_row, &num_col);
    std::vector<float>vecData;
//    std::vector<std::string> labels;

    for(i =0; i <num_row; i++){
        int tmp_nnz_indices=0;
        std::vector<float> v_nn_vals;
        MatGetRow(m_data,i,&ncols,&cols,&vals);
        for (j=0; j<ncols; j++) {
            if(j < ncols && j==cols[tmp_nnz_indices]){
                v_nn_vals.push_back(vals[j]);
            }else{
                v_nn_vals.push_back(0);
            }
        }
        MatRestoreRow(m_data,i,&ncols,&cols,&vals);
        vecData.insert(vecData.end(), v_nn_vals.begin(), v_nn_vals.end());
    }
    flann::Matrix<float> data(vecData.data(),num_row,num_col);
    std::cout << "[KNN][RF] "<< num_row << " data points are loaded successfully!\n";
    // - - - - call flann - - - -


    //call flann
    flann::Index<flann::L2<float> > index_(data, flann::KDTreeIndexParams(16));
    index_.buildIndex();

    std::vector<std::vector<int> > indicies;
    std::vector<std::vector<float> > dists;

    flann::SearchParams params(128);
    params.cores = 0; //automatic core selection
    index_.knnSearch(data, indicies, dists, Config_params::getInstance()->get_nn_number_of_neighbors(),  params);
    std::cout << "Number of nodes in the created graph:" << indicies.size() << std::endl;

    //store the indices, dists to 2 separate matrices
//    for(unsigned row_idx =0; row_idx < num_row; row_idx++){
    for(unsigned int row_idx =0; row_idx < 10; row_idx++){
        for(unsigned int nn_idx = 0; nn_idx < indicies[row_idx].size(); nn_idx++){
            unsigned int node_idx = indicies[row_idx][nn_idx];
            double dist = dists[row_idx][nn_idx];
            std::cout << "row_idx:" << row_idx << ", nn_idx:" << nn_idx << ", node_idx:"<< node_idx<< ", dist:"<< dist << std::endl;
//            MatSetValue(m_indices,row_idx,j,,INSERT_VALUES);
        }
    }

    //end!
}
