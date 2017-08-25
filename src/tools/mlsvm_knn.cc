#include "../config_params.h"
#include <flann/flann.hpp>
//#include <flann/util/random.h>
#include "../loader.h"
#include "../common_funcs.h"
#include "../k_fold.h"


Config_params* Config_params::instance = NULL;

void run_flann(Mat& m_data, Mat& m_indices, Mat& m_dists);

int main(int argc, char **argv){
//    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscInitialize(NULL, NULL, NULL, NULL);
    ETimer t_all;
    //read XML parameters
    Config_params::getInstance()->read_params("./params.xml", argc, argv, Config_params::flann); //@ 040317-1842
    /* ------------------------- Prepare FLANN Commands---------------------------- */
    std::string str_NN {std::to_string(Config_params::getInstance()->get_nn_number_of_neighbors())};
//    std::string str_nn_distance_type {std::to_string(Config_params::getInstance()->get_nn_distance_type())};
    if(Config_params::getInstance()->get_nn_distance_type() != 1){
        std::cout << "[main] Only Euclidean distance is supported using this tool, for other distances please refer to user manual. Exit!" << std::endl;
        return 1;
    }

//    std::string str_NN_params {" " + str_NN + " "+ str_nn_distance_type} ;

    Loader ld;
    if(Config_params::getInstance()->get_nn_number_of_classes() == 1){
        Mat m_data = ld.load_norm_data_sep(Config_params::getInstance()->get_single_norm_data_f_name());
        Mat m_indices, m_dists;
        run_flann(m_data, m_indices, m_dists);

        //export to file
        std::string nn_indices_fname = Config_params::getInstance()->get_ds_path() +
                        Config_params::getInstance()->get_ds_name() + "_norm_data_indices.dat";
        std::string nn_dists_fname = Config_params::getInstance()->get_ds_path() +
                        Config_params::getInstance()->get_ds_name() + "_norm_data_dists.dat";

        CommonFuncs cf;
        cf.exp_matrix(m_indices, "", nn_indices_fname , "mlsvm_knn");
        cf.exp_matrix(m_dists, "", nn_dists_fname , "mlsvm_knn");
        MatDestroy(&m_indices);
        MatDestroy(&m_dists);
        std::cout << "KNN results for 1 class is saved successfully!" << std::endl;

    }else{  // two classes
        /* ------------------------- Divide Data ---------------------------- */
        ETimer t_kf;
        k_fold kf;
        kf.read_in_data();
        kf.divide_data(true);   //I need the data files, so I export them to files
        Mat m_min_data = kf.get_m_min_data();
        Mat m_maj_data = kf.get_m_maj_data();
        t_kf.stop_timer("[Main] reading and deviding data in k-fold class");

        //                                  ----- minority class -----
        Mat m_min_indices, m_min_dists;
        run_flann(m_min_data, m_min_indices, m_min_dists);

        //export to file
        std::string nn_min_indices_fname = Config_params::getInstance()->get_ds_path() +
                        Config_params::getInstance()->get_ds_name() + "_min_norm_data_indices.dat";
        std::string nn_min_dists_fname = Config_params::getInstance()->get_ds_path() +
                        Config_params::getInstance()->get_ds_name() + "_min_norm_data_dists.dat";

        CommonFuncs cf;
        cf.exp_matrix(m_min_indices, "", nn_min_indices_fname , "mlsvm_knn");
        cf.exp_matrix(m_min_dists, "", nn_min_dists_fname , "mlsvm_knn");
        MatDestroy(&m_min_indices);
        MatDestroy(&m_min_dists);

        //                                  ----- majority class -----
        Mat m_maj_indices, m_maj_dists;
        run_flann(m_maj_data, m_maj_indices, m_maj_dists);

        //export to file
        std::string nn_maj_indices_fname = Config_params::getInstance()->get_ds_path() +
                        Config_params::getInstance()->get_ds_name() + "_maj_norm_data_indices.dat";
        std::string nn_maj_dists_fname = Config_params::getInstance()->get_ds_path() +
                        Config_params::getInstance()->get_ds_name() + "_maj_norm_data_dists.dat";

        cf.exp_matrix(m_maj_indices, "", nn_maj_indices_fname , "mlsvm_knn");
        cf.exp_matrix(m_maj_dists, "", nn_maj_dists_fname , "mlsvm_knn");
        MatDestroy(&m_maj_indices);
        MatDestroy(&m_maj_dists);

        std::cout << "KNN results for 2 classes are saved successfully!" << std::endl;
        kf.free_resources();
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
    int num_nearest_neighbors = Config_params::getInstance()->get_nn_number_of_neighbors();
    MatGetSize(m_data, &num_row, &num_col);

    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row, num_nearest_neighbors,PETSC_NULL, &m_indices);
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row, num_nearest_neighbors,PETSC_NULL, &m_dists);
    std::cout << "[KNN][RF] num_nn:"<< num_nearest_neighbors << std::endl;

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
//    flann::seed_random(0);          //set the random seed to 1 for debug


    params.cores = 0; //automatic core selection
    index_.knnSearch(data, indicies, dists, num_nearest_neighbors,  params);
    std::cout << "Number of nodes in the created graph:" << indicies.size() << std::endl;
//    exit(1);
    //store the indices, dists to 2 separate matrices
    for(unsigned int row_idx =0; row_idx < num_row; row_idx++){
//        std::cout << "\nrow " << row_idx <<": ";
        for(j = 0; j < indicies[row_idx].size(); j++){
            unsigned int node_idx = indicies[row_idx][j];
            double dist = dists[row_idx][j];
//            std::cout << "(" << j << ", "<< node_idx << ")  ";
            MatSetValue(m_indices,row_idx,node_idx,1,INSERT_VALUES);
            MatSetValue(m_dists,row_idx,node_idx,dist,INSERT_VALUES);
        }
    }
    MatAssemblyBegin(m_indices,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_indices,MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(m_dists,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_dists,MAT_FINAL_ASSEMBLY);


    //end!
}
