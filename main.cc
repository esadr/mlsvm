#include <iostream>
#include "etimer.h"
#include "k_fold.h"
#include "loader.h"          //load the flann data into the WA matrix
#include "main_recursion.h"
#include "coarsening.h"         //coarse the WA matrix
#include "model_selection.h"
#include "config_params.h"
#include "ut_mr.h"

#include "common_funcs.h"       //only for debug

Config_params* Config_params::instance = NULL;

#define Multiple_Runs   1       //1 means, it is in debug mode and only run once
#define flann_is_needed 1       //1 means, Flann is required, 0 means, No flann and use the old files(only works for single run)
#define sep_test_data   0       //1 means, test data is passed as a seperate file, only works for single run

///*********************************************************************
///*                              Multi Level                          *
///*********************************************************************
int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    // read parameters
    Config_params::getInstance()->init_to_default();
    Config_params::getInstance()->read_params("./params.xml", argc, argv);
    Config_params::getInstance()->print_params();

    int num_repeat_exp_ = Config_params::getInstance()->getInstance()->get_main_num_repeat_exp();
    int num_kf_iter_ = Config_params::getInstance()->getInstance()->get_main_num_kf_iter();
    int total_iter_ = num_repeat_exp_ * num_kf_iter_;
    ETimer t_all;


#if Multiple_Runs == 1          /// - - - - - - - - - - - - Multiple runs - - - - - - - - - - - -
    // - - - - - - Repeat the expriment for multiple times - - - - - -

    for(int r=0; r< num_repeat_exp_; r++){
        if(r==5){   // only after 5th complete experiment
            Config_params::getInstance()->set_best_parameters(Gmean);
        }
        Config_params::getInstance()->update_srand_seed();   // new experiment seed is different from the earlier experiments
        ETimer t_kf_rdsh;
        k_fold kf;                                          // read the input data again and shuffle it
        kf.read_in_data();
        kf.divide_data();
        kf.shuffle_data();
        t_kf_rdsh.stop_timer("[Main] k_fold read, divide and shuffle");

        // - - - - - - Run a whole cross fold over the training data - - - - - -
        for (int i=0; i < num_kf_iter_; i++){
            ETimer t_iteration;
            printf("\n\n====Exp:%d, Iteration %d==== \n\n",r,i);
            kf.cross_validation(i,num_kf_iter_);            //writes the min and maj training and test data to 3 files
#else                           /// - - - - - - - - - - - - Single run - - - - - - - - - - - -
    if(flann_is_needed){            // flann is required
        int curr_iter = 0;
        int total_iter = num_kf_iter_;
        printf("\n\n===[debug] single run Iteration %d ==== \n\n",curr_iter);                    // $$debug
        k_fold kf;
        kf.read_in_data();
        kf.divide_data();
        if(!sep_test_data){         // No separate test data is passed
            kf.shuffle_data();
            kf.cross_validation(curr_iter,total_iter);      //writes the min and maj training and test data to 3 files for flann
            printf("Testdata is calculated as part of Cross Validation!\n");
        }else{                      //separate data is provided, so we don't need to create testdata from the input dataset
            kf.prepare_traindata_for_flann();               //writes only the min and maj training to 2 files for flann
            std::string test_file_name = Config_params::getInstance()->get_ds_path() + Config_params::getInstance()->get_ds_name() + "_test.dat";
            Config_params::getInstance()->set_test_ds_f_name(test_file_name);
            printf("Separate testdata is provided!\n");
            std::cout << "[Main] test file name is:" << Config_params::getInstance()->get_test_ds_f_name() << std::endl;
        }
    }else{
        printf("\n\n[Main]===[debug] single run without running Flann, using old files ==== \n\n");  // $$debug
    }
#endif  // end of Multiple_Runs == 1

//    #ifndef eDEBUG                  ///normal case which reads the parameter from OS
            const char* env_p = std::getenv("PY_PATH");
            if(env_p == NULL ){
                printf("\n[Main] the PY_PATH is not defined. Exit(1)!\n\n");
                printf("Please define the PY_PATH to the python location. For more information, read the README file\n\n");
                exit(1);
            }
//    #else                           // debug case which uses a hardcoded path
//         const char* env_p = "/home/esfp/anaconda/bin/";
//    #endif  // end of eDEBUG fot PY_PATH


         //================== Loading Flann data ===========================
            ETimer t_load;

            if(flann_is_needed){        // Run the flann for both classes
                std::string str_NN {std::to_string(Config_params::getInstance()->get_nn_number())};
                std::string str_nn_distance_type {std::to_string(Config_params::getInstance()->get_nn_distance_type())};
                std::string str_NN_params {" " + str_NN + " "+ str_nn_distance_type} ;

                std::string py_path= env_p;
                std::string min_command = py_path +"/python ./scripts/flann.py "+ Config_params::getInstance()->get_p_norm_data_f_name()+ str_NN_params;
                std::string maj_command = py_path +"/python ./scripts/flann.py "+ Config_params::getInstance()->get_n_norm_data_f_name()+ str_NN_params;

                ETimer t_flann_min;
                std::cout << "[Main] python min_command:" << min_command << std::endl;
                system(min_command.c_str());
                t_flann_min.stop_timer("[Main] flann for minority class");

                ETimer t_flann_maj;
                std::cout << "[Main] min_command:" << min_command << std::endl;
                system(maj_command.c_str());
                t_flann_maj.stop_timer("[Main] flann for majority class");
            }

            Mat p_WA, n_WA;
            Vec p_vol, n_vol;
            printf("\t\t====[Loading flann] start reading the minority class (P)==== \n");
            Loader p_loader(Config_params::getInstance()->get_p_indices_f_name(),Config_params::getInstance()->get_p_dist_f_name());
            p_loader.set_weight_type(Config_params::getInstance()->get_ld_weight_type());
            p_WA = p_loader.load_flann_binary();
            //by default the volume will initialize to 1
            p_vol = p_loader.init_volume(1);


            ETimer t_ld_flann;
            printf("\t\t====[Loading flann] start reading the majority class (N)==== \n");
            Loader n_loader(Config_params::getInstance()->get_n_indices_f_name(),Config_params::getInstance()->get_n_dist_f_name());
            n_loader.set_weight_type(Config_params::getInstance()->get_ld_weight_type());
            n_WA = n_loader.load_flann_binary();
            n_vol = n_loader.init_volume(1);
            t_ld_flann.stop_timer("[Main] load flann for both classes takes ");
        //================== Loading normalized dataset ===========================
            ETimer t_ld_data;
            Mat p_data, n_data;
            printf("\t\t====[Loading normalized data] Reading data ==== \n");

            p_data = p_loader.load_norm_data_sep(Config_params::getInstance()->get_p_norm_data_f_name());
            n_data = n_loader.load_norm_data_sep(Config_params::getInstance()->get_n_norm_data_f_name());

            t_ld_data.stop_timer("[Main] Reading the normalized data for both classes");


        //====================== Multilevel Solver ===============================
            ETimer t_solver;
            Mat m_P_minority, m_P_majority;
            std::vector<ref_results> v_ref_results;
            MainRecursion multilevel_solver;        // the test data file name is initialized in the constructor of MR class
            multilevel_solver.main(p_data, m_P_minority, p_WA,p_vol, n_data, m_P_majority,n_WA, n_vol,0,v_ref_results)  ;
            Refinement rf;
            rf.add_best_model(v_ref_results);
            //free the matrices
            MatDestroy(&p_WA);
            MatDestroy(&n_WA);
            VecDestroy(&p_vol);
            VecDestroy(&n_vol);

    #if Multiple_Runs == 1
            t_solver.stop_timer("[Main] Total Solver at iteration ",std::to_string(i));

            t_iteration.stop_timer("[Main] Vcycle (including loading datasets) at iteration ",std::to_string(i));

        }   // end of for loop for "Whole cross fold"
    }   // end of for loop for "Repeat the experiment"
    #endif

    Config_params::getInstance()->print_final_results();

    t_all.stop_timer("[Main] whole test including all iterations");
    printf("[Main] total number of iterations:%d \n",total_iter_);

//    Config_params::getInstance()->print_params();
    printf("[Main] MLSVM finished successfully!\n");

    PetscFinalize();
    return 0;
}
