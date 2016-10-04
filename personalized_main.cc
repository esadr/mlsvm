#include "etimer.h"
#include "loader.h"          //load the flann data into the WA matrix
#include "model_selection.h"
#include "config_params.h"
#include <vector>
#include "personalized.h"
#include "common_funcs.h"

Config_params* Config_params::instance = NULL;

#define Multiple_Runs_is_active 0           //1 means, it is in debug mode and only run once
#define Calculate_flann_is_needed 1         //1 means, no Flann, only works for single run


int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);


    Config_params::getInstance()->init_to_default();
    Config_params::getInstance()->read_params("./params.xml", argc, argv);
    Config_params::getInstance()->print_params();

    ETimer t_all;

    const char* env_p = std::getenv("PY_PATH");
    if(env_p == NULL ){
        printf("\n[Main] the PY_PATH is not defined. Exit(1)!\n\n");
        printf("Please define the PY_PATH to the python location. For more information, read the README file\n\n");
        exit(1);
    }

    std::string data_file_full_name {Config_params::getInstance()->get_ds_path()+
                        Config_params::getInstance()->get_ds_name()+ "_zsc_data.dat"};

    std::string input_train_label {Config_params::getInstance()->get_ds_path()+
                        Config_params::getInstance()->get_ds_name()+ "_label.dat"};

    std::string output_flann_fname  {Config_params::getInstance()->get_tmp_path()+
                        Config_params::getInstance()->get_ds_name()+ "_" +
                        Config_params::getInstance()->get_exp_info() + "_zsc_data_indices.dat"};

    //================== Call NN ===========================
    std::string str_NN {std::to_string(Config_params::getInstance()->get_nn_number())};
    std::string str_nn_distance_type {std::to_string(Config_params::getInstance()->get_nn_distance_type())};
#if Calculate_flann_is_needed == 1        // for debugging large datasets which takes long time
    std::string py_path= env_p;
    std::string command = py_path +"/python ./scripts/flann_personalized.py "+ data_file_full_name +" " +str_NN +
                        " "+str_nn_distance_type+ " " + output_flann_fname;
    std::cout << "[PerMain] command:" << command << std::endl;
    ETimer t_NN;
    system(command.c_str());
    t_NN.stop_timer("[PerMain] calculate the NN");
#endif

    Mat m_data, m_raw_NN, m_filtered_NN, m_stats;
    Vec v_lbl_in;
    Loader ld_;
    m_raw_NN = ld_.load_norm_data_sep(output_flann_fname);

    personalized pd;
    pd.filter_test_point(m_raw_NN, m_filtered_NN);  //m_raw_NN is destroyed inside the function

//    CommonFuncs cf1;         /// - - - - export information to debug the problem - - - -
//    std::string str_o_mfilteredNN_fname = "m_filteredNN.dat";
//    cf1.exp_matrix(m_filtered_NN, "./data",str_o_mfilteredNN_fname,"[PD]");

//    exit(1);
    //================== Loading normalized dataset ===========================
    PetscInt            num_row_, num_col_, i, ncols;
    const PetscInt      *cols;
    const PetscScalar   *vals;
    PetscScalar         *arr_labels;
    std::vector<int>    v_lbl_target;
    std::vector<int>    v_lbl_predict;


    m_data = ld_.load_norm_data_sep(data_file_full_name);
    v_lbl_in = ld_.read_input_vector(input_train_label);

    MatGetSize(m_data, &num_row_, &num_col_);
    v_lbl_target.reserve(num_row_);
    v_lbl_predict.reserve(num_row_);
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row_  ,4,4,PETSC_NULL, &m_stats); //+1 is for label
    // - - - - create a vector of labels - - - -
    PetscMalloc1(num_row_, &arr_labels);
    VecGetArray(v_lbl_in,&arr_labels);

    for(i = 0; i< num_row_ ;i++){
        v_lbl_target.push_back(arr_labels[i]);
    }
    VecRestoreArray(v_lbl_in,&arr_labels);
    PetscFree(arr_labels);

    printf("[ParMain] data and flann results are loaded successfully!\n");
//exit(1);
    //================== Personalized classifier for each point ===========================
    summary final_summary;

    //only for debug
//    num_row_ = 1;       // just for debug
    for(i =0; i < num_row_ ; i++){
        MatGetRow(m_filtered_NN,i,&ncols,&cols,&vals);
        //call the solver to train the neighbor points (indices in vals), expect the vals[0] which is the main point
        //predict the main point (vals[0]) with the final model
        //save the predicted label in vector of predicts

        if(Config_params::getInstance()->get_multi_level_status() == 1){        // Multi level
            if(Config_params::getInstance()->get_ms_status())
//                pd.uniform_design_index_base(i, m_data, v_lbl_target, 0, 0, 0 , vals, ncols, v_lbl_predict);
//            else
//                pd.Single_training_index_base(i, m_data, v_lbl_target, 0, 0, 0 , vals, ncols, v_lbl_predict);
                pd.multi_level_solver(i, m_data, m_filtered_NN, v_lbl_target, vals, ncols, v_lbl_predict);
        }else{                                                                  // Single level
            if(Config_params::getInstance()->get_ms_status())
                pd.uniform_design_index_base(i, m_data, v_lbl_target, 0, 0, 0 , vals, ncols, v_lbl_predict);
            else
                pd.Single_training_index_base(i, m_data, v_lbl_target, 0, 0, 0 , vals, ncols, v_lbl_predict);
        }

        pd.calc_statistics(i, m_stats, v_lbl_target, v_lbl_predict, vals, ncols);


/*
        Mat m_p_data, m_n_data;
        MatCreateSeqAIJ(PETSC_COMM_SELF,ncols,num_col_,num_col_, PETSC_NULL, &m_p_data);
        MatCreateSeqAIJ(PETSC_COMM_SELF,ncols,num_col_,num_col_, PETSC_NULL, &m_n_data);

        pd.get_separate_matrices(m_data, v_lbl_target,vals, ncols, m_p_data, m_n_data);
        CommonFuncs cf;         /// - - - - export information to debug the problem - - - -
        std::string str_output_p_data_fname = "m_p_data_"+ std::to_string(i) + ".dat";
        std::string str_output_n_data_fname = "m_n_data_"+ std::to_string(i) + ".dat";
        cf.exp_matrix(m_p_data, "./data",str_output_p_data_fname ,"[PD]");
        cf.exp_matrix(m_n_data, "./data",str_output_n_data_fname ,"[PD]");
*/
        //std::cout << "target:" << v_lbl_target[i] <<  ", predict:" << v_lbl_predict[i] << std::endl;

        MatRestoreRow(m_filtered_NN,i,&ncols,&cols,&vals);
    }
    MatAssemblyBegin(m_stats,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_stats,MAT_FINAL_ASSEMBLY);

//    CommonFuncs cf;         /// - - - - export information to debug the problem - - - -
//    std::string str_output_stats_fname = "m_stats_"+ str_NN + ".dat";
//    cf.exp_matrix(m_stats, "./data",str_output_stats_fname ,"[PD]");

    //evaluate all the predictions
    pd.evaluate_results(v_lbl_target, v_lbl_predict, final_summary);

    // free resources
    MatDestroy(&m_data);
    MatDestroy(&m_filtered_NN);
    VecDestroy(&v_lbl_in);

    t_all.stop_timer("[PD_main]whole training");

//    Config_params::getInstance()->print_final_results();
    printf("[ParMain]finished successfully!\n");
    PetscFinalize();
    return 0;
}










