#include <iostream>
#include "etimer.h"
#include "k_fold.h"
#include "loader.h"          //load the flann data into the WA matrix
//#include "main_recursion.h"
//#include "coarsening.h"         //coarse the WA matrix
#include "model_selection.h"
#include "config_params.h"
#include "ut_mr.h"

#include "common_funcs.h"       //only for debug

Config_params* Config_params::instance = NULL;



///*********************************************************************
///*                              Single Level                         *
///*********************************************************************
int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);


    Config_params::getInstance()->init_to_default();
    Config_params::getInstance()->read_params("./params.xml", argc, argv);
    Config_params::getInstance()->print_params();

    int num_repeat_exp_ = Config_params::getInstance()->getInstance()->get_main_num_repeat_exp();
    int num_kf_iter_ = Config_params::getInstance()->getInstance()->get_main_num_kf_iter();
    int total_iter_ = num_repeat_exp_ * num_kf_iter_;
    ETimer t_all;


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
            kf.cross_validation(i,num_kf_iter_);

            Vec p_vol, n_vol;
            Loader p_loader(Config_params::getInstance()->get_p_indices_f_name(),Config_params::getInstance()->get_p_dist_f_name());
            Loader n_loader(Config_params::getInstance()->get_n_indices_f_name(),Config_params::getInstance()->get_n_dist_f_name());

            ETimer t_ld_data;
            Mat p_data, n_data;
            printf("\t\t====[Loading normalized data] Reading data ==== \n");

            p_data = p_loader.load_norm_data_sep(Config_params::getInstance()->get_p_norm_data_f_name());
            n_data = n_loader.load_norm_data_sep(Config_params::getInstance()->get_n_norm_data_f_name());

            PetscInt num_row_p, num_row_n;
            MatGetSize(p_data, &num_row_p, NULL);
            MatGetSize(n_data, &num_row_n, NULL);

            p_vol = p_loader.init_volume(1,num_row_p);
            n_vol = n_loader.init_volume(1,num_row_n);
            t_ld_data.stop_timer("[Main] Reading the normalized data for both classes");

            ///====================== Single Solver ===============================
            ETimer t_solver;
            ModelSelection ms;

            solution sol_debug;
            double C=0, gamma=0;
            bool inherit_params = Config_params::getInstance()->get_best_params_status();
            if(inherit_params){
                C = Config_params::getInstance()->get_best_C();
                gamma = Config_params::getInstance()->get_best_gamma();
            }else{
                C = Config_params::getInstance()->get_svm_C();
                gamma = Config_params::getInstance()->get_svm_gamma();
            }
            printf("[Main]{SL} C:%g, G:%g \n", C, gamma);

            // Note: inherit_params is important and normally it should be false, because we don't know the best parameters
            std::cout << "[Main]{SL} inherit params is " << inherit_params << std::endl;
//            exit(1);
            ms.uniform_design(p_data, p_vol, n_data, n_vol, inherit_params, C, gamma,1,sol_debug);

            // free resources
            MatDestroy(&p_data);
            MatDestroy(&n_data);
            VecDestroy(&p_vol);
            VecDestroy(&n_vol);



            t_solver.stop_timer("[Main] Total Solver at iteration ",std::to_string(i));

            t_iteration.stop_timer("[Main] Vcycle (including loading datasets) at iteration ",std::to_string(i));

        }   // end of for loop for "Whole cross fold"
    }   // end of for loop for "Repeat the experiment"

    Config_params::getInstance()->print_final_results();

    t_all.stop_timer("[Main] all single level runs");
    printf("[Main] total number of iterations:%d \n",total_iter_);

//    Config_params::getInstance()->print_params();
    printf("[Main]{SL} SVM finished successfully!\n");

    PetscFinalize();
    return 0;
}







