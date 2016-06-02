#ifndef CONFIG_LOGS_H
#define CONFIG_LOGS_H

/*
 * Log levels for dbl
 * 0: No stdout
 * 1: info
 * 3: debug the steps of code
 * 5: print small information about the variable
 * 7: print the matrices and vectors
 * 9: print the middle and temp variables inside for loops
 * 11:
 * 13:
 * 15:
 */

#define timer_print                 1      // 1: Active , 0: Disable
//Export the matrices or not
#define debug_export                1      // 1: Export , 0: Not Export

#define debug_level                 0
//---- kFold ----
#define dbl_kf_shuffle_data         0           // default 0
#define dbl_kf_cross_validation     0           // default 0
//---- Loader ----
#define dbl_LD_LWAB                 0
#define dbl_LD_LFB                  0           // load flann binary
#define dbl_LD_LNDS                 0           // 0 Default, 5 print size of data
//---- Coarsening ----
#define dbl_CO_calcP                1          // 1 normal with #edges
#define dbl_CO_vNorm                0           // calculate the normalized vector
#define dbl_CO_cInv                 0           // calculate the inverse vector
#define dbl_CO_CAD                  0           // 0 Default [Calculate Aggregate data]
#define dbl_CO_calc_WA_c            0
#define dbl_CO_calc_coarse_vol      0
#define dbl_CO_FWE                  3           // filter weak edges (3 prints # filtered edges)
//---- MainRecursion (Refinement) ----
#define debug_level_main_rec        0           // 0 Default
#define dbl_MR_refinement           1           // 1 Default
#define dbl_MR_calc_performance     0           // 0 Default
#define dbl_MR_ref_partition        3           // 1 Default, 3 report steps
#define dbl_MR_CGI                  0           // 0 Default
#define dbl_MR_find_groups          0           // 0 Default
#define dbl_MR_calc_center          0           // 0 Default
#define dbl_MR_calc_distances       0           // 0 Default
#define dbl_MR_get_parts            1           // 0 Default (Metis), 1 #parts, 5 report the metis response, 7 print parts
#define dbl_MR_single_part_matrix   0           // 0 Default
#define dbl_MR_CSC                  7           // 0 Default
#define dbl_MR_fSN                  1           // 1 Default find_SV_neighbors
//---- Model Selection ----
#define dbl_MS_UD                   1           // 1 Default
#define dbl_MS_combine_test_data    0
#define dbl_MS_ud_param_generator   3           // 3 shows new parameters
#define dbl_MS_cross_fold           0           // 0 Default
#define dbl_MS_PS                   3           // 0 Default
#define dbl_MS_read_problem         0           // 0 Default
#define dbl_MS_SB1                  5           // select_best_1st - print results of diff C,Gamma at 5
#define dbl_MS_test_predict         1           // 1 it should be for real runs
#define dbl_MS_set_weights          0           // 0 Default, 1 shows the weight in the WSVM
#define dbl_MS_malloc_weights       0           // 0 Default, 1 prints the labels for classes
#define dbl_MS_free_model_selection 0           // 0 Default, 1 prints which class called the free method

//----- Reports --------
#define report_MS_quiet_SVM         1           // 1 silent, 0 normal
#define report_MS_untouched_ds      1           // model selection
#define rpt_TD_only_l1              0           // report the Test data only at the finest level (to collect result for papers)

//----- Unit Test ------
#define ut_test_only                0           // 0 Default(unit test are disabled), 1 only do the unit test and exit with 1

#endif // CONFIG_LOGS_H




