//#include <iostream>
#include "etimer.h"
#include "k_fold.h"
#include "loader.h"          //load the flann data into the WA matrix
#include "main_recursion.h"
#include "coarsening.h"         //coarse the WA matrix
#include "model_selection.h"
#include "config_params.h"
#include "common_funcs.h"
//#include "ut_mr.h"

Config_params* Config_params::instance = NULL;

#define Multiple_Runs   1       //1 means, it is in debug mode and only run once
#define flann_is_needed 1       //1 means, Flann is required, 0 means, No flann and use the old files(only works for single run)
#define sep_test_data   0       //1 means, test data is passed as a seperate file, only works for single run

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    Config_params::getInstance()->read_params("./params.xml", argc, argv);  // read parameters
    switch(Config_params::getInstance()->get_main_function()){
    ///*********************************************************************
    ///*                              SVM                                  *
    ///*********************************************************************
    case 0: //Classification
    {
        Config_params::getInstance()->set_master_models_info();
        int num_repeat_exp_ = Config_params::getInstance()->getInstance()->get_main_num_repeat_exp();
        int num_kf_iter_ = Config_params::getInstance()->getInstance()->get_main_num_kf_iter();
        int total_iter_ = num_repeat_exp_ * num_kf_iter_;
        ETimer t_all;

        // load data files
        k_fold kf;                                          // read the input data again and shuffle it
        Mat m_min_full_data, m_maj_full_data;
        kf.read_in_divided_data(m_min_full_data, m_maj_full_data);      // read the data for both class
        kf.initialize_vectors();
        //#memory optimization (I can release the memory after I created the train and test data to save memory usage,
        //drawback is loading the data files in the beginning)
        // - - - - - load full indices matrics - - - - -
        Mat m_min_full_NN_indices,m_min_full_NN_dists,m_maj_full_NN_indices,m_maj_full_NN_dists;
        kf.read_in_full_NN(m_min_full_NN_indices,m_min_full_NN_dists,m_maj_full_NN_indices,m_maj_full_NN_dists);

    #if Multiple_Runs == 1          /// - - - - - - - - - - - - Multiple runs - - - - - - - - - - - -
        // - - - - - - Repeat the expriment for multiple times - - - - - -
        for(int r=0; r< num_repeat_exp_; r++){
            Config_params::getInstance()->set_main_current_exp_id(r);
            Config_params::getInstance()->update_srand_seed();   // new experiment seed is different from the earlier experiments
            kf.shuffle_data();              // creat vector of indices for 1 complete iteration including multiple v-cycle

            /// - - - - - - Run a whole cross validation (k-fold)) over the training data - - - - - -
            for (int i=0; i < num_kf_iter_; i++){
                Config_params::getInstance()->set_main_current_kf_id(i);
                ETimer t_iteration;
                printf("\n\n====Exp:%d, Iteration %d==== \n\n",r,i);
                Mat m_min_train_data,m_min_WA;
                Mat m_maj_train_data,m_maj_WA;
                Vec v_p_vol, v_n_vol;
                // set the name of testdata file
                std::string f_name_test_data = Config_params::getInstance()->get_tmp_path() +"kfold_test_data_exp_"+
                        std::to_string(r)+"_fold_"+std::to_string(i)+ "_exp_" + Config_params::getInstance()->get_exp_info();
                Config_params::getInstance()->set_test_ds_f_name(f_name_test_data);

                std::string f_name_p_train_data = Config_params::getInstance()->get_tmp_path() +"kfold_p_train_data_exp_"+
                        std::to_string(r)+"_fold_"+std::to_string(i)+ "_exp_" + Config_params::getInstance()->get_exp_info();
                Config_params::getInstance()->set_p_e_k_train_data_f_name(f_name_p_train_data);

                std::string f_name_n_train_data = Config_params::getInstance()->get_tmp_path() +"kfold_n_train_data_exp_"+
                        std::to_string(r)+"_fold_"+std::to_string(i)+ "_exp_" + Config_params::getInstance()->get_exp_info();
                Config_params::getInstance()->set_n_e_k_train_data_f_name(f_name_n_train_data);

                std::cout << "testdata file name is "   << Config_params::getInstance()->get_test_ds_f_name()
                                                        << Config_params::getInstance()->get_p_e_k_train_data_f_name()
                                                        << Config_params::getInstance()->get_n_e_k_train_data_f_name() << std::endl;

                kf.prepare_data_for_iteration(i,num_kf_iter_,m_min_full_data,m_min_train_data,m_min_full_NN_indices,m_min_full_NN_dists,m_min_WA,v_p_vol
                                              ,m_maj_full_data,m_maj_train_data,m_maj_full_NN_indices,m_maj_full_NN_dists,m_maj_WA,v_n_vol);
//                exit(1);
    #endif  // end of Multiple_Runs == 1

                // bypass the classification to export the cross-validation for comparison to other solvers
//                /*
            //====================== create validation data ===============================
                ETimer t_sample;
                Mat m_VD_p, m_VD_n;
                CommonFuncs cf;
                m_VD_p = cf.sample_data(m_min_train_data, Config_params::getInstance()->get_ms_VD_sample_size_fraction(), Config_params::getInstance()->get_cpp_srand_seed());
                m_VD_n = cf.sample_data(m_maj_train_data, Config_params::getInstance()->get_ms_VD_sample_size_fraction(), Config_params::getInstance()->get_cpp_srand_seed());

                t_sample.stop_timer("[Main] validation data is sampled from the finest training data");
            //====================== Multilevel Solver ===============================
                ETimer t_solver;
                Mat m_P_minority, m_P_majority;
                std::vector<ref_results> v_ref_results;
                MainRecursion multilevel_solver;        // the test data file name is initialized in the constructor of MR class
                multilevel_solver.main(m_min_train_data, m_P_minority, m_min_WA, v_p_vol, m_maj_train_data, m_P_majority, m_maj_WA,
                                       v_n_vol, m_VD_p, m_VD_n, 0, v_ref_results);
                Refinement rf;
                rf.add_best_model(v_ref_results);
                //free the matrices
                MatDestroy(&m_min_train_data);
                MatDestroy(&m_maj_train_data);
                MatDestroy(&m_min_WA);
                MatDestroy(&m_maj_WA);
                VecDestroy(&v_p_vol);
                VecDestroy(&v_n_vol);

                MatDestroy(&m_VD_p);    // release the validation data
                MatDestroy(&m_VD_n);
//                */

        #if Multiple_Runs == 1
//                t_solver.stop_timer("[Main] Total Solver at iteration ",std::to_string(i));

                t_iteration.stop_timer("[Main] Vcycle (including loading datasets) at iteration ",std::to_string(i));
                Config_params::getInstance()->print_final_results();

                //save the final metadata for models of the current k-fold   @072617-1157
                Config_params::getInstance()->update_master_models_info();

                std::string test_file = Config_params::getInstance()->get_test_ds_f_name();
                if( remove(test_file.c_str())==0)
                    std::cout << "[Main] test data file is removed successfully from " << test_file << std::endl;

            }   // end of for loop for "Whole cross fold"
        }   // end of for loop for "Repeat the experiment"
        #endif
        //destory the remaining objects to free the memory
        MatDestroy(&m_min_full_data);
        MatDestroy(&m_maj_full_data);
        MatDestroy(&m_min_full_NN_indices);
        MatDestroy(&m_min_full_NN_dists);
        MatDestroy(&m_maj_full_NN_indices);
        MatDestroy(&m_maj_full_NN_dists);

        std::cout << "EndofExperiment " ;
        Config_params::getInstance()->print_final_results();

        t_all.stop_timer("[Main] whole test including all iterations");
        printf("[Main] total number of iterations:%d \n",total_iter_);

        Config_params::getInstance()->export_models_metadata();


    //    Config_params::getInstance()->print_params();
        printf("[Main] MLSVM finished successfully!\n");
    }
    break;

    case 1:
        std::cout << "Regression has not added yet!" << std::endl;
        break;

    }
    PetscFinalize();
    return 0;
}
