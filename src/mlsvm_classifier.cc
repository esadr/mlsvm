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

int main(int argc, char **argv)
{
//    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscInitialize(NULL, NULL, NULL, NULL);
    paramsInst->read_params("./params.xml", argc, argv);  // read parameters
    // moved from read_params to here 083018
    // it caused problems with mlsvmClassifierSepTest
    paramsInst->set_inputs_file_names();
    paramsInst->setTestdataExist();
    switch(paramsInst->get_main_function()){
    ///*********************************************************************
    ///*                              SVM                                  *
    ///*********************************************************************
    case 0: //Classification
    {
        paramsInst->set_master_models_info();
        int num_repeat_exp_ = paramsInst->getInstance()->get_main_num_repeat_exp();
        int num_kf_iter_ = paramsInst->getInstance()->get_main_num_kf_iter();
        int total_iter_ = num_repeat_exp_ * num_kf_iter_;

        #if export_SVM_models == 1
            paramsInst->check_create_directory(paramsInst->get_model_path() );
        #endif

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

        // - - - - - - Repeat the expriment for multiple times - - - - - -
        for(int r=0; r< num_repeat_exp_; r++){
            paramsInst->set_main_current_exp_id(r);
            paramsInst->update_srand_seed();   // new experiment seed is different from the earlier experiments
            kf.shuffle_data();              // creat vector of indices for 1 complete iteration including multiple v-cycle

            /// - - - - - - Run a whole cross validation (k-fold)) over the training data - - - - - -
            for (int i=0; i < num_kf_iter_; i++){
                paramsInst->set_main_current_kf_id(i);
                ETimer t_iteration;
                printf("\n           ==================== Exp:%d, Iteration %d ==================== \n",r,i);
                Mat m_min_train_data,m_min_WA;
                Mat m_maj_train_data,m_maj_WA;
                Vec v_p_vol, v_n_vol;
                // r is the current experiment, i is the current iteration (k-fold id)
                paramsInst->set_current_iter_file_names(r, i);

                kf.prepare_data_for_iteration(i,num_kf_iter_,m_min_full_data,m_min_train_data,
                                              m_min_full_NN_indices,m_min_full_NN_dists,m_min_WA,v_p_vol
                                              ,m_maj_full_data,m_maj_train_data,m_maj_full_NN_indices,
                                              m_maj_full_NN_dists,m_maj_WA,v_n_vol);

            // bypass the classification to export the cross-validation training data
            //    (comparison with other solvers)
            #if dbl_exp_train_data == 0
            //====================== create validation data ===============================
                ETimer t_sample;
                Mat m_VD_p, m_VD_n;
                CommonFuncs cf;
                m_VD_p = cf.sample_data(m_min_train_data, paramsInst->get_ms_VD_sample_size_fraction(),
                                        paramsInst->get_cpp_srand_seed());
                m_VD_n = cf.sample_data(m_maj_train_data, paramsInst->get_ms_VD_sample_size_fraction(),
                                        paramsInst->get_cpp_srand_seed());

                t_sample.stop_timer("[MC] validation data is sampled from the finest training data");
            //====================== Multilevel Solver ===============================
                ETimer t_solver;
                Mat m_P_minority, m_P_majority;
                std::vector<ref_results> v_ref_results;
                paramsInst->set_timer_start_coarsening();
                // the test data file name is initialized in the constructor of MR class
                MainRecursion multilevel_solver;
                multilevel_solver.main(m_min_train_data, m_P_minority,
                                       m_min_WA, v_p_vol, m_maj_train_data, m_P_majority, m_maj_WA,
                                       v_n_vol, m_VD_p, m_VD_n, 0, v_ref_results);
                paramsInst->set_timer_end_refinement();
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
            #if timer_complexity_analysis == 1
                paramsInst->print_coarsening_refinement_times();
            #endif
                t_iteration.stop_timer("[MC] Vcycle (including loading datasets) at iteration ",
                                       std::to_string(i));
                //save the final metadata for models of the current k-fold   @072617-1157
                paramsInst->update_master_models_info();


//            std::cout << "[MC] EXIT\n" ;            exit(1);
            // 1 saves the test files for comparison or run standalone prediction
            #if save_test_files == 0
                std::string test_file = paramsInst->get_test_ds_f_name();
                std::string info_file = test_file + ".info";
                remove(info_file.c_str());
                if( remove(test_file.c_str())==0)
                    std::cout << "[MC] test data file is removed successfully from " << test_file << std::endl;
            #endif // end of save_test_file
            #endif //dbl_exp_train_data

            paramsInst->reportFinalModelInVCycle();
            }   // end of for loop for "Whole cross validation"
        }   // end of for loop for "Repeat the experiment"
        //destory the remaining objects (free the memory)
        MatDestroy(&m_min_full_data);
        MatDestroy(&m_maj_full_data);
        MatDestroy(&m_min_full_NN_indices);
        MatDestroy(&m_min_full_NN_dists);
        MatDestroy(&m_maj_full_NN_indices);
        MatDestroy(&m_maj_full_NN_dists);

    #if dbl_exp_train_data == 0
        std::cout << "[MC] End of all experiments\n" ;
        paramsInst->print_final_results();

        t_all.stop_timer("[MC] Whole test including all iterations");
        printf("[MC] Total number of iterations:%d \n",total_iter_);

    #if export_SVM_models == 1
        paramsInst->export_models_metadata();
    #endif //export_SVM_models      //in case of exporting the data, the models are not trained and the output is useless
    #endif //dbl_exp_train_data

    //    paramsInst->print_params();
        printf("[MC] MLSVM Classifier finished successfully!\n");
    }
    break;

    case 1:
        std::cout << "Regression has not added yet!" << std::endl;
        break;

    }
    PetscFinalize();
    return 0;
}
