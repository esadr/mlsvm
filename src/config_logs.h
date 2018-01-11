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
#define dbl_kf_shuffle_data         0           // Default 0
#define dbl_kf_cross_validation     0           // Default 0
#define dbl_KF_CVC                  0           // Default 0
#define dbl_KF_CVS                  0           // Default 0
#define dbl_KF_FN                   0           // Default 0
#define dbl_KF_CTD                  0           // Default 0
#define dbl_KF_CTC                  0           // Default 1 [combine_two_classes_in_one]   //release 0
#define dbl_KF_PDFI                 0           // Default 1 [prepare_data_for_iteration]   //release 0
#define dbl_KF_WOUT                 0           // Default 1 [write_output]                 //release 0
#define dbl_KF_rdd                  0           // Default 0 [read divided data]
#define dbl_KF_rfn                  0           // Default 0 [read full NN]
//---- Loader ----
#define dbl_LD_LWAB                 0
#define dbl_LD_LFB                  0           // 0 Default, load flann binary
#define dbl_LD_CWAM                 0           // 0 Default, create WA matrix using input NN matrices
#define dbl_LD_CD                   0           // 0 Default, 3 print raw and calculated distance
#define dbl_LD_LNDS                 0           // 0 Default, 5 print size of data
//---- Coarsening ----
#define dbl_CO_calcP                1           // 1 normal with #edges                 //release 0
#define dbl_CO_vNorm                0           // calculate the normalized vector
#define dbl_CO_cInv                 0           // calculate the inverse vector
#define dbl_CO_CAD                  0           // 0 Default [Calculate Aggregate data]
#define dbl_CO_calc_WA_c            0
#define dbl_CO_calc_coarse_vol      0
#define dbl_CO_FWE                  3           // filter weak edges (3 prints # filtered edges) //release 0
//---- MainRecursion (Refinement) ----
#define dbl_MR_main                 1           // 1 Default 3 level info //release 0
#define debug_level_main_rec        0           // 0 Default
#define dbl_MR_refinement           1           // 1 Default



#define dbl_MR_calc_performance     0           // 0 Default
//---- Refinement ----
#define dbl_RF_main                 1           // 1 Default
#define dbl_RF_INFO                 0           // 1 Default    //release 0
#define dbl_RF_main_with_partition  0           // 1 Default, 3 report steps        //release 0
#define dbl_RF_main_no_partition    0           // 1 Default                        //release 0
#define dbl_RF_FSN                  0           // 1 Default find_SV_neighbors      //release 0
#define dbl_RF_ABM                  0           // 0 Default, 3 report different levels results after sort, 5 report before and after sort

//---- Partitioning ----
#define dbl_PR_get_parts            5           // 0 Default (Metis), 1 #parts, 5 report the metis response, 7 print parts
#define dbl_PR_SEW                  0           // 0 Default, 9 prints the adjwgt
#define dbl_PR_single_part_matrix   0           // 0 Default
#define dbl_PR_CSC                  0           // 0 Default    //calc_single_center
#define dbl_PR_calc_center          1           // 0 Default
#define dbl_PR_calc_distances       0           // 0 Default
#define dbl_PR_find_groups          3           // 0 Default    5 prints final groups
#define dbl_PR_CAC                  0           // 0 Default    //calc_avg_center
#define dbl_PR_CCV                  0           // 0 Default    //calc_center_volume
#define dbl_PR_CGI                  0           // 0 Default
#define dbl_PR_CPM                  0           // 0 Default    //calc_performance_measure  5 prints final predicted label
//---- Model Selection ----
#define dbl_MS_UD                   1           // 1 Default
#define dbl_MS_UDSepVal             0           // 1 Default    //release 0
#define dbl_MS_UDIB                 1           // 1 Default
#define dbl_MS_combine_test_data    0
#define dbl_MS_ud_param_generator   0           // 0 Default, 3 shows new parameters
#define dbl_MS_cross_fold           0           // 0 Default
#define dbl_MS_PS                   0           // 0 Default
#define dbl_MS_SB1                  0           // 0 Default, 3 prints after sort, 5 prints prints before and after sort
#define dbl_MS_set_weights          0           // 0 Default, 1 shows the weight in the WSVM
#define dbl_MS_UDC                  1           // 1 Default

//---- Solver ----
#define dbl_SV_free_solver          0           // 0 Default, 1 prints which class called the free method
#define dbl_SV_malloc_weights       0           // 0 Default, 1 prints the labels for classes
#define dbl_SV_read_problem         0           // 0 Default, 1 prints the number of points in classes
#define dbl_SV_RPIB                 0           // 0 Default    //read_problem_index_base
#define dbl_SV_SWNP                 0           // 0 Default    //set_weights_num_points
#define dbl_SV_SWSV                 0           // 0 Default    //set_weights_sum_volume
#define dbl_SV_SWSVIB               0           // 0 Default    //set_weights_sum_volume_index_base
#define dbl_SV_PDSWSPIB             0           // 0 Default    //PD_set_weights_sum_num_point_IB
#define dbl_SV_TM                   0           // 0 Default    1 prints C, gamma
#define dbl_SV_PDTMIB               0           // 0 Default    1 prints C, gamma
#define dbl_SV_TM_report_time       0           // 0 Default    1 prints time of train_model
#define dbl_SV_test_predict         0           // 0 Default    1 prints the details        3 prints the test data summary
#define dbl_SV_TPIB                 0           // 0 Default    1 prints the details        //1 cause many nan in partitioning mode
#define dbl_SV_predict_label        0           // 0 Default    3 prints the both labels
#define dbl_SV_predict_VD           0           // 0 Default    3 prints the both labels
#define dbl_SV_predict_VD_rpt_time  0           // 0 Default    1 prints time
#define dbl_SV_predict_label1       0           // 0 Default    3 prints the both labels
#define dbl_SV_PSSM                 0           // 0 Default    3 prints number of support vectors
#define dbl_SV_PD_predict_a_label   0           // 0 Default    3 prints the both labels
//----- Reports --------
#define report_MS_quiet_SVM         1           // 1 silent, 0 normal
#define report_MS_untouched_ds      1           // model selection
#define rpt_TD_only_l1              0           // report the Test data only at the finest level (to collect result for papers)
//----- Unit Test ------
#define ut_test_only                0           // 0 Default(unit test are disabled), 1 only do the unit test and exit with 1



#define dbl_SV_PDTPIB               0           // 0 Default    1 prints the details
#define export_SVM_models           0           // 0 Default    1 Save the SVM models in svm_models folder
#define save_test_files             0           // 0 Default removes the test file, 	1 keeps them
#define dbl_exp_train_data          0           // 0 Default, 1 only export the data for comparison with other solvers
#define timer_complexity_analysis   0           // 0 Default, 1 only for reporting the detail of time for coarsening and refinement
#endif // CONFIG_LOGS_H




