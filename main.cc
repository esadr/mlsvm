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

#define debug_single_run 0      //1 means, it is in debug mode and only run once
#define debug_no_ML 0
#define test_coarsening 0       //1 means, it only test the coarsening, so you need to comment the rest of the code after it
#define ut_test_only 0

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);

#if ut_test_only == 1
    ut_MR utmr;             // - - - - start test specific part - - - -
//    utmr.test_calc_center();
//    utmr.test_calc_distances();
    utmr.test_ms_ud();
    PetscFinalize();
    return 1;                // - - - - end test specific part - - - -
#endif

    Config_params::getInstance()->init_to_default();
    Config_params::getInstance()->read_params("./params.xml", argc, argv);
    Config_params::getInstance()->print_params();

    int num_repeat_exp_ = Config_params::getInstance()->getInstance()->get_main_num_repeat_exp();
    int num_kf_iter_ = Config_params::getInstance()->getInstance()->get_main_num_kf_iter();
    int total_iter_ = num_repeat_exp_ * num_kf_iter_;
    ETimer t_all;


#if test_coarsening == 1
    /// - - - - test coarsening - - - - DEBUG only
    Mat p_WA, m_P, p_WA_c;
    Vec p_vol , p_vol_c;
    std::cout << "start loader\n";
    std::string f_ind = "../datasets/15point_index.dat";
    std::string f_dis = "../datasets/15point_dis.dat";
    Loader p_loader(f_ind, f_dis);
    p_loader.set_weight_type(Config_params::getInstance()->get_ld_weight_type());

    std::cout << "start loading flann\n";
    p_WA = p_loader.load_flann_binary();
    std::cout << "flann is loaded\n";
    p_vol = p_loader.read_input_vector("../datasets/15point_volume.dat");
    std::cout << "vol is loaded\n";

    Mat p_data, p_data_c;
    p_data = p_loader.load_norm_data_sep("../datasets/15point_data.dat");
    std::cout << "data is loaded\n";

    Coarsening cs;
    std::vector<NodeId> tmp;
    m_P = cs.calc_P(p_WA,p_vol,tmp);
    p_data_c = cs.calc_aggregate_data(m_P,p_data,p_vol,tmp);
    p_vol_c = cs.calc_coarse_volumes(m_P,p_vol);

    p_WA_c = cs.calc_WA_c(m_P, p_WA);
    cs.filter_weak_edges(p_WA_c,0.8);

    CommonFuncs cf;
    cf.exp_matrix(p_data_c,"./debug/","15point_aggregate_data.dat");
    cf.exp_vector(p_vol_c,"./debug/","15point_aggregate_vol.dat");
    exit(1) ;
#endif
#if test_coarsening ==0

    #if debug_single_run == 0
        // - - - - - - Repeat the expriment for multiple times - - - - - -
        for(int r=0; r< num_repeat_exp_; r++){
            if(r==5){   // only after 10th complete experiment
                Config_params::getInstance()->set_best_parameters(Gmean);
            }

            Config_params::getInstance()->update_srand_seed();   // new experiment seed is different from the earlier experiments
            k_fold kf;                                          // read the input data again and shuffle it
            // - - - - - - Run a whole cross fold over the training data - - - - - -
            for (int i=0; i < num_kf_iter_; i++){

                ETimer t_iteration;
                printf("\n\n====Exp:%d, Iteration %d==== \n\n",r,i);
                kf.cross_validation(i,num_kf_iter_);

    #endif

    #if debug_single_run == 1
                int curr_iter = 1;
                int total_iter = 5;
    //            Config_params::getInstance()->debug_only_set_srand_seed("1459985777991");
                k_fold kf;
                kf.cross_validation(curr_iter,total_iter);                                    // $$debug
                printf("\n\n===[debug] single run Iteration %d ==== \n\n",curr_iter);                    // $$debug
    #endif

            //================== Loading Flann data ===========================
                ETimer t_load;
    // #ifndef eDEBUG
                const char* env_p = std::getenv("PY_PATH");
                if(env_p == NULL ){
                    printf("\n[main] the PY_PATH is not defined. Exit(1)!\n\n");
                    printf("Please define the PY_PATH to the python location. For more information, read the README file\n\n");
                    exit(1);
                }
    // #endif
    // #ifdef eDEBUG
    //             const char* env_p = "/home/esfp/anaconda/bin/";
    // #endif




                std::string py_path= env_p;
                std::string min_command = py_path +"/python ../mlsvm/scripts/flann.py ./data/ kfold_min_train.dat";
                std::string maj_command = py_path +"/python ../mlsvm/scripts/flann.py ./data/ kfold_maj_train.dat";

//                int response=1; // logic: it works as if the script was failed and the while retry the script
//                int cnt_err=0;
                system(min_command.c_str());
                system(maj_command.c_str());
    //            while (response != 0){
    //                if(cnt_err){
    //                    printf("Error in running flann.py for minority class, happens: %d\n",cnt_err);
    //                }
    //                 response = system(min_command.c_str());
    //                cnt_err++;
    //                if(cnt_err >3){
    //                    if(response )    SETERRQ(PETSC_COMM_WORLD,1,"Error in running flann.py for minority class more than 3 time ");
    //                }
    //            }

    //            response=1;
    //            cnt_err=0;
    //            while (response != 0){
    //                if(cnt_err){
    //                    printf("Error in running flann.py for majority class, happens: %d\n",cnt_err);
    //                }
    //                 response = system(maj_command.c_str());
    //                cnt_err++;
    //                if(cnt_err >3){
    //                    if(response )    SETERRQ(PETSC_COMM_WORLD,1,"Error in running flann.py for majority class more than 3 time ");
    //                }
    //            }

                Mat p_WA, n_WA;
                Vec p_vol, n_vol;
                printf("\t\t====[Loading flann] start reading the minority class (P)==== \n");
                Loader p_loader(Config_params::getInstance()->get_p_indices_f_name(),Config_params::getInstance()->get_p_dist_f_name());
                p_loader.set_weight_type(Config_params::getInstance()->get_ld_weight_type());
                p_WA = p_loader.load_flann_binary();
                //by default the volume will initialize to 1
                p_vol = p_loader.init_volume(1);


                printf("\t\t====[Loading flann] start reading the majority class (N)==== \n");
                Loader n_loader(Config_params::getInstance()->get_n_indices_f_name(),Config_params::getInstance()->get_n_dist_f_name());
                n_loader.set_weight_type(Config_params::getInstance()->get_ld_weight_type());
                n_WA = n_loader.load_flann_binary();
                n_vol = n_loader.init_volume(1);
    //             t_load.stop_timer("[main] load flann for both classes takes ");
            //================== Loading normalized dataset ===========================
                ETimer t_ld_data;
                Mat p_data, n_data;
                printf("\t\t====[Loading normalized data] Reading data ==== \n");

                p_data = p_loader.load_norm_data_sep(Config_params::getInstance()->get_p_norm_data_f_name());
                n_data = n_loader.load_norm_data_sep(Config_params::getInstance()->get_n_norm_data_f_name());

                t_ld_data.stop_timer("[main] Reading the normalized data for both classes");


            // - - - - - - - - Debug No multilevel  - - - - - - - -
    #if debug_no_ML == 1
        ModelSelection ms;
        solution sol_debug;
        ms.set_untouch_file_name(Config_params::getInstance()->get_test_ds_f_name());
        double C = Config_params::getInstance()->get_svm_C();
        double gamma = Config_params::getInstance()->get_svm_gamma();

        sol_debug =  ms.UD(p_data, p_vol, n_data, n_vol, 1, C , gamma,0);

        // free resources
        MatDestroy(&p_data);
        MatDestroy(&n_data);
        VecDestroy(&p_vol);
        VecDestroy(&n_vol);
        MatDestroy(&p_WA);
        MatDestroy(&n_WA);

        t_all.stop_timer("[main] [debug]Single run without multilevel");
        PetscFinalize();
        return 0;
    #endif

            //====================== Multilevel Solver ===============================

                ETimer t_solver;
                Mat m_P_minority, m_P_majority;
                MainRecursion multilevel_solver;        // the test data file name is initialized in the constructor of MR class
                multilevel_solver.main(p_data, m_P_minority, p_WA,p_vol, n_data, m_P_majority,n_WA, n_vol,0)  ;
                //free the matrices
                MatDestroy(&p_WA);
                MatDestroy(&n_WA);
                VecDestroy(&p_vol);
                VecDestroy(&n_vol);

    #if debug_single_run == 0
                t_solver.stop_timer("[main] Total Solver at iteration ",std::to_string(i));

                t_iteration.stop_timer("[main] Vcycle (including loading datasets) at iteration ",std::to_string(i));

            }   // end of for loop for "Whole cross fold"
        }   // end of for loop for "Repeat the experiment"
    #endif

        Config_params::getInstance()->print_final_results();

        t_all.stop_timer("[main] whole test including all iterations");
        printf("[main] total number of iterations:%d \n",total_iter_);
#endif      //test_coarsening
    Config_params::getInstance()->print_params();
    PetscFinalize();
    printf("[main] MLSVM finished successfully!\n");
    return 0;
}






