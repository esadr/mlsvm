#include "etimer.h"
#include "k_fold.h"
#include "loader.h"          //load the flann data into the WA matrix
#include "main_recursion.h"
#include "coarsening.h"         //coarse the WA matrix
#include "model_selection.h"
#include "config_params.h"
#include "common_funcs.h"
#include <cassert>

Config_params* Config_params::instance = NULL;

int main(int argc, char **argv)
{
    PetscInitialize(NULL, NULL, NULL, NULL);
    Config_params::getInstance()->read_params("./params.xml",
                                              argc, argv);  // read parameters

    Config_params::getInstance()->set_master_models_info();
    Config_params::getInstance()->Config_params::set_fixed_file_names();
//    int num_repeat_exp_ = Config_params::getInstance()->getInstance()
//                                            ->get_main_num_repeat_exp();
//    int num_kf_iter_ = Config_params::getInstance()->getInstance()
//                                            ->get_main_num_kf_iter();
//    int total_iter_ = num_repeat_exp_ * num_kf_iter_;
    ETimer t_all;

    // load data files
    // read the input data again and shuffle it
    k_fold kf;
    Mat m_min_full_data, m_maj_full_data;
    // read the data for both class
    kf.read_in_divided_data(m_min_full_data, m_maj_full_data);

    kf.initialize_vectors();
    // - - - - - load full indices matrics - - - - -
    Mat m_min_full_NN_indices,m_min_full_NN_dists,m_maj_full_NN_indices,
            m_maj_full_NN_dists;
    kf.read_in_full_NN(m_min_full_NN_indices,m_min_full_NN_dists,
            m_maj_full_NN_indices,m_maj_full_NN_dists);
    int r=0;
    Config_params::getInstance()->set_main_current_exp_id(r);
    // new experiment seed is different from the earlier experiments
    Config_params::getInstance()->update_srand_seed();
    // creat vector of indices for 1 complete iteration including multiple v-cycle
//        kf.shuffle_data();

    /// - - - - - - Run a whole cross validation (k-fold)) over the training data - - - - - -
    int i=0;
    Config_params::getInstance()->set_main_current_kf_id(i);
    ETimer t_iteration;
    Mat m_min_train_data,m_min_WA;
    Mat m_maj_train_data,m_maj_WA;
    Vec v_p_vol, v_n_vol;
    // r is the current experiment, i is the current iteration (k-fold id)
//    Config_params::getInstance()->set_current_iter_file_names(r, i);

    kf.prepare_data_using_separate_testdata(
            m_min_full_data, m_min_full_NN_indices,
            m_min_full_NN_dists,m_min_WA,v_p_vol,
            m_maj_full_data, m_maj_full_NN_indices,
            m_maj_full_NN_dists,m_maj_WA,v_n_vol);
//    exit(1);
    //====================== create validation data ===========================
    ETimer t_sample;
    Mat m_VD_p, m_VD_n;
    CommonFuncs cf;
    m_VD_p = cf.sample_data(m_min_full_data,
                Config_params::getInstance()->get_ms_VD_sample_size_fraction(),
                Config_params::getInstance()->get_cpp_srand_seed());
//    exit(1);
    m_VD_n = cf.sample_data(m_maj_full_data,
                Config_params::getInstance()->get_ms_VD_sample_size_fraction(),
                Config_params::getInstance()->get_cpp_srand_seed());

    t_sample.stop_timer("[MC] validation data is sampled from the finest training data");
//    exit(1);

    //====================== check test data file ===============================
    assert(Config_params::getInstance()->get_test_ds_f_name() != "" &&
           "Test file is not set, please use --test_data parameter!");
    std::cout << "[main] test file is set to "
              << Config_params::getInstance()->get_test_ds_f_name()
              << std::endl;
    //====================== Multilevel Solver ===============================
    ETimer t_solver;
    Mat m_P_minority, m_P_majority;
    std::vector<ref_results> v_ref_results;
    Config_params::getInstance()->set_timer_start_coarsening();
    // the test data file name is initialized in the constructor of MR class
    MainRecursion multilevel_solver;
    multilevel_solver.main(m_min_full_data, m_P_minority, m_min_WA, v_p_vol,
                           m_maj_full_data, m_P_majority, m_maj_WA, v_n_vol,
                           m_VD_p, m_VD_n, 0, v_ref_results);

//    exit(1);

    Config_params::getInstance()->set_timer_end_refinement();
    Refinement rf;
    rf.add_best_model(v_ref_results);
    //free the matrices
    MatDestroy(&m_min_WA);
    MatDestroy(&m_maj_WA);
    VecDestroy(&v_p_vol);
    VecDestroy(&v_n_vol);

    MatDestroy(&m_VD_p);    // release the validation data
    MatDestroy(&m_VD_n);
    #if timer_complexity_analysis == 1
        Config_params::getInstance()->print_coarsening_refinement_times();
    #endif
    t_iteration.stop_timer("[MC] Vcycle (including loading datasets) at iteration ",
                           std::to_string(i));
    //save the final metadata for models of the current k-fold   @072617-1157
    Config_params::getInstance()->update_master_models_info();

    //destory the remaining objects (free the memory)
    MatDestroy(&m_min_full_data);
    MatDestroy(&m_maj_full_data);
    MatDestroy(&m_min_full_NN_indices);
    MatDestroy(&m_min_full_NN_dists);
    MatDestroy(&m_maj_full_NN_indices);
    MatDestroy(&m_maj_full_NN_dists);

#if dbl_exp_train_data == 0
    Config_params::getInstance()->print_final_results();
    t_all.stop_timer("[MC] Whole test including all iterations");

#if export_SVM_models == 1
    Config_params::getInstance()->export_models_metadata();
#endif //export_SVM_models      //in case of exporting the data, the models are not trained and the output is useless
#endif //dbl_exp_train_data

    printf("[MC] MLSVM Classifier finished successfully!\n");

    PetscFinalize();
    return 0;
}
