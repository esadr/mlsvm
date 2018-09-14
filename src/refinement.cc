#include "refinement.h"
#include "partitioning.h"
#include <cmath>        // round (ceil)
#include "etimer.h"
#include "loader.h"
#include "common_funcs.h"
#include "solver.h"
#include "k_fold.h"
#include <cassert>

#include <thread>

struct selected_agg
{
    int index;
    double value;

    selected_agg(int col_index, double fraction_val) :
        index(col_index), value(fraction_val) {}

    bool operator > (const selected_agg sel_agg) const {
        return (value > sel_agg.value);
    }
};


solution Refinement::main(Mat& m_data_p, Mat& m_P_p, Vec& v_vol_p, Mat&m_WA_p
                          , Mat& m_data_n, Mat& m_P_n, Vec& v_vol_n, Mat&m_WA_n
                          , Mat& m_VD_p, Mat& m_VD_n
                          , solution& sol_coarser,int level
                          , std::vector<ref_results>& v_ref_results){
    paramsInst->set_main_current_level_id(level);
#if dbl_RF_main >=5
    PetscInt num_row_p_data =0, num_row_n_data =0;
    MatGetSize(m_data_p,&num_row_p_data,NULL);
    MatGetSize(m_data_n,&num_row_n_data,NULL);
    printf("[RF][main]{beginnig} num fine data p:%d, n:%d\n"
           ,num_row_p_data,num_row_n_data);


    PetscInt num_row_p_P =0, num_row_n_P =0, num_col_p_P =0, num_col_n_P =0;
    MatGetSize(m_P_p,&num_row_p_P, &num_col_p_P);
    MatGetSize(m_P_n,&num_row_n_P, &num_col_n_P);
    printf("[RF][main]{beginnig} Minority class P matrix dim [%d,%d]\n"
           ,num_row_p_P,num_col_p_P);
    printf("[RF][main]{beginnig} Majority class P matrix dim [%d,%d]\n"
           ,num_row_n_P,num_col_n_P);

    printf("[RF][main] sol_coarser.p_index.size():%lu, "
           "sol_coarser.n_index.size():%lu\n"
           , sol_coarser.p_index.size()
           , sol_coarser.n_index.size());
#endif
    // these are fix for all levels as the sum
    // of volumes for all the points are preserved
    PetscScalar sum_all_vol_p,sum_all_vol_n;
    VecSum(v_vol_p, &sum_all_vol_p);
    VecSum(v_vol_n, &sum_all_vol_n);

    Mat m_new_neigh_p, m_new_neigh_n;
    IS IS_neigh_p, IS_neigh_n;
    /// - - - - - - - get new points for finer level - - - - - - -
    find_SV_neighbors(m_data_p,m_P_p,sol_coarser.p_index, m_WA_p
                      , m_new_neigh_p,"Minority",IS_neigh_p);
    find_SV_neighbors(m_data_n,m_P_n,sol_coarser.n_index, m_WA_n
                      , m_new_neigh_n,"Majority",IS_neigh_n);


    // - - - - get the size of neighbors - - - -
    PetscInt num_neigh_row_p_ =0, num_neigh_row_n_ =0;
    MatGetSize(m_new_neigh_p,&num_neigh_row_p_,NULL);
    MatGetSize(m_new_neigh_n,&num_neigh_row_n_,NULL);
    printf("[RF][main] num new neighbor p:%d, n:%d\n"
                ,num_neigh_row_p_,num_neigh_row_n_);

//    if(num_neigh_row_p_ == 0 || num_neigh_row_n_ == 0){
//        printf("\n[Error]:[MR][refinement] Empty matrices for new neighbors,\n Exit!\n");
//        exit(1);
//    }
    assert (num_neigh_row_p_ && "[MR][refinement] Empty matrices for new neighbors");
    assert (num_neigh_row_n_ && "[MR][refinement] Empty matrices for new neighbors");

    solution sol_refine;
    summary summary_TD;
    /***********************************************************************************************/
    /*                                  Start Partitioning                                         */
    /***********************************************************************************************/

    /*  - The information for partitioning is related to points in the finer level(current)
     *      which are going to pass to the SVM to creat a model
     *  - The WA, Vol contains all the points which are not needed.
     *  - Only the information for the points which are neighbor to SV of coarser level are required
     *  m_neigh_WA: only specific rows and columns
     *  v_nei_vol: volumes for the points in the m_neigh_WA
     */
    if( (num_neigh_row_p_ + num_neigh_row_n_)  > paramsInst->get_pr_start_partitioning() ){
        // - - - - get the neighbors data points - - - -
        Mat m_neigh_WA_p, m_neigh_WA_n;
        Vec v_neigh_Vol_p, v_neigh_Vol_n;

        printf("[RF][main] Get the neighbors WA, Vol at level:%d\n",level);
        MatGetSubMatrix(m_WA_p, IS_neigh_p, IS_neigh_p,MAT_INITIAL_MATRIX,&m_neigh_WA_p);
        MatGetSubMatrix(m_WA_n, IS_neigh_n, IS_neigh_n,MAT_INITIAL_MATRIX,&m_neigh_WA_n);
        VecGetSubVector(v_vol_p, IS_neigh_p, &v_neigh_Vol_p);
        VecGetSubVector(v_vol_n, IS_neigh_n, &v_neigh_Vol_n);

        MatDestroy(&m_WA_p);    // free resources
        MatDestroy(&m_WA_n);

    #if dbl_RF_main >=5   //5
        PetscInt num_row_neigh_WA_p, num_row_neigh_WA_n, num_col_neigh_WA_p, num_col_neigh_WA_n;
        MatGetSize(m_neigh_WA_p,&num_row_neigh_WA_p, &num_col_neigh_WA_p);
        MatGetSize(m_neigh_WA_n,&num_row_neigh_WA_n, &num_col_neigh_WA_n);
        PetscPrintf(PETSC_COMM_WORLD,"[RF][main]{beginnig} Minority class neigh_WA dim is:[%dx%d]\n",num_row_neigh_WA_p,num_col_neigh_WA_p);
        PetscPrintf(PETSC_COMM_WORLD,"[RF][main]{beginnig} Majority class neigh_WA dim is:[%dx%d]\n",num_row_neigh_WA_n,num_col_neigh_WA_n);
    #endif

        // try different partitioning of same input data (1 ignores it), iter passes to get_parts as random seed to metis
        int num_iter_refinement = 1;        //#TODO only one iteration is used, multiple iterations didn't improve the results
        // - - - -  load the test data matrix - - - -
        Mat m_TD ;
        Loader test_loader;
        m_TD = test_loader.load_norm_data_sep(this->test_dataset_f_name);
        PetscInt num_row_TD;
        MatGetSize(m_TD, &num_row_TD, NULL);

        // get validation data size
        PetscInt            num_VD_p, num_VD_n, num_VD_both;
        MatGetSize(m_VD_p, &num_VD_p, NULL);
        MatGetSize(m_VD_n, &num_VD_n, NULL);

        k_fold kf;
        Mat m_VD_both;
        kf.combine_two_classes_in_one(m_VD_both, m_VD_p, m_VD_n,false);  //false make sure that the input matrices won't be destroyed inside the function
        num_VD_both = num_VD_p + num_VD_n;


        std::vector<Mat> v_mat_avg_centers(num_iter_refinement);
        std::vector<Mat> v_mat_all_predict_validation(num_iter_refinement);
        std::vector<Mat> v_mat_all_predict_TD(num_iter_refinement);
        std::unordered_set<PetscInt> uset_SV_index_p;
        std::unordered_set<PetscInt> uset_SV_index_n;
        uset_SV_index_p.reserve(2*num_neigh_row_p_);
        uset_SV_index_p.reserve(2*num_neigh_row_n_);

        // - - - - multiple iterations with different partitioning - - - -
        Partitioning pt;
        for(int iter=0; iter < num_iter_refinement; iter++){                                       // #performance remove this loop and update the functions signiture
            printf("[RF][main] + + + + Partitioning, level:%d, iter:%d + + + + \n",level,iter);

            // - - - - - - - - - - calc number of partitions - - - - - - - - - -  #1
            int partition_max_size = paramsInst->get_pr_partition_max_size();
            PetscInt num_vertex_p, num_vertex_n;
            MatGetSize(m_neigh_WA_p, &num_vertex_p,NULL);
            MatGetSize(m_neigh_WA_n, &num_vertex_n,NULL);
            int num_partitions_p , num_partitions_n;
            num_partitions_p = ceil( (float)num_vertex_p / (float)partition_max_size ); // ceil is used to make sure we get enough number of partitions
            num_partitions_n = ceil( (float)num_vertex_n / (float)partition_max_size );

            // - - - - - - - - - - partition data points, calc center- - - - - - - - - -  #2
            Mat m_parts_p, m_centers_p;
            std::vector<PetscScalar> v_sum_vol_parts_p(num_partitions_p);
            if(num_partitions_p < 2){             // if the number of minority points are not large enough, call single partition
                pt.single_part_matrix(m_new_neigh_p, m_parts_p);
                pt.calc_single_center(m_new_neigh_p, v_neigh_Vol_p, m_centers_p, v_sum_vol_parts_p);
            }else{
                std::vector<std::vector<PetscInt>> vv_parts_p(num_partitions_p);
                /* DEBUG
                int num_non_empty_part_p = pt.get_parts(m_neigh_WA_p,v_neigh_Vol_p, num_partitions_p, level, vv_parts_p, m_parts_p);   // get partitions (Metis)
                if(num_non_empty_part_p != num_partitions_p){
                    std::cout << "[RF][main] inside while for get_part P num_part_p:" << num_partitions_p <<
                                     ", num_non_empty_part_p:" << num_non_empty_part_p <<" Exit after export!" << std::endl;
                    CommonFuncs cf;         /// - - - - export information to debug the problem - - - -
                    cf.exp_vector(v_vol_p, "./data","whole_v_vol_p.dat","[RF]");
                    cf.exp_vector(v_neigh_Vol_p, "./data","v_neigh_vol_p.dat","[RF]");
                    cf.exp_matrix(m_neigh_WA_p, "./data","m_neigh_WA_p.dat","[RF]");
                    exit(1);
                }
                */
                pt.get_parts(m_neigh_WA_p,v_neigh_Vol_p, num_partitions_p, level, vv_parts_p, m_parts_p,iter);   // get partitions (Metis)
                pt.calc_center(m_new_neigh_p, v_neigh_Vol_p, vv_parts_p, m_parts_p, m_centers_p, v_sum_vol_parts_p);
            }

            Mat m_parts_n, m_centers_n;
            std::vector<PetscScalar> v_sum_vol_parts_n(num_partitions_n);
            if(num_partitions_n < 2){             // if the number of majority points are not large enough, call single partition
                pt.single_part_matrix(m_new_neigh_n, m_parts_n);
                pt.calc_single_center(m_new_neigh_n, v_neigh_Vol_n, m_centers_n, v_sum_vol_parts_n);
            }else{
                std::vector<std::vector<PetscInt>> vv_parts_n(num_partitions_n);
                int num_non_empty_part_n =  pt.get_parts(m_neigh_WA_n,v_neigh_Vol_n, num_partitions_n, level, vv_parts_n, m_parts_n,iter);
                if(num_non_empty_part_n != num_partitions_n){   // debug empty partitions
                    std::cout << "[RF][main] inside while for get_part N num_part_n:" << num_partitions_n <<
                                     ", num_non_empty_part_n:" << num_non_empty_part_n <<" Exit after export!" << std::endl;
                    CommonFuncs cf;         /// - - - - export information to debug the problem - - - -
                    cf.exp_vector(v_vol_n, "./data","whole_v_vol_n.dat","[RF]");
                    cf.exp_vector(v_neigh_Vol_n, "./data","v_neigh_vol_n.dat","[RF]");
                    cf.exp_matrix(m_neigh_WA_n, "./data","m_neigh_WA_n.dat","[RF]");
                    exit(1);
                }
                pt.calc_center(m_new_neigh_n, v_neigh_Vol_n, vv_parts_n, m_parts_n, m_centers_n, v_sum_vol_parts_n);
            }


            // - - - - - - - - - - calc distances - - - - - - - - - -  #3
            PetscInt            num_part_p, num_part_n, num_features;
            MatGetSize(m_centers_p, &num_part_p, &num_features);
            MatGetSize(m_centers_n, &num_part_n, NULL);

            Mat m_dist;
            pt.calc_distances(num_part_p, num_part_n, m_centers_p, m_centers_n, m_dist);

            // - - - - - - - - - - find group - - - - - - - - - -  #4
            std::vector<std::pair<int, int> > v_groups;
            pt.find_groups(num_part_p, num_part_n, m_dist, v_groups );

            // - - - - - - - - - - calc_avg_center - - - - - - - - - -  #5
            pt.calc_avg_center(m_centers_p, m_centers_n, v_groups, v_sum_vol_parts_p, v_sum_vol_parts_n, v_mat_avg_centers[iter]);
            //#performance #bottleneck (the matrix is column major order) @080417-2343
            MatCreateSeqDense(PETSC_COMM_SELF, num_part_p+num_part_n, num_row_TD,NULL, &v_mat_all_predict_TD[iter]);



//            MatCreateSeqDense(PETSC_COMM_SELF, num_part_p+num_part_n, num_VD_p+num_VD_n ,NULL, &v_mat_all_predict_validation[iter]);
            //#performance #bottleneck (the matrix is column major order) @080417-2343
            MatCreateSeqDense(PETSC_COMM_SELF, num_part_p+num_part_n, num_VD_both, NULL, &v_mat_all_predict_validation[iter]);


            ETimer t_all_parts_training;
            for(int i = 0; i < v_groups.size() ; i++ ){
                std::vector<PetscInt> v_p_index;
                std::vector<PetscInt> v_n_index;
                pt.create_group_index(i, v_groups, m_parts_p, m_parts_n, v_p_index, v_n_index);
                // - - - - - - Train & predict Model - - - - - - -
                //Without model selection
//                Solver sv_part;
//                sv_part.partial_solver(m_new_neigh_p, v_neigh_Vol_p, m_new_neigh_n, v_neigh_Vol_n, sol_coarser.C, sol_coarser.gamma, level,
//                                    v_p_index,v_n_index, uset_SV_index_p, uset_SV_index_n, m_VD_p, m_VD_n,m_VD_both, v_mat_all_predict_validation[iter],
//                                    m_TD, i, v_mat_all_predict_TD[iter]);

                //with model selection
                ModelSelection ms_partition;
                ms_partition.uniform_design_index_base_separate_validation(m_new_neigh_p, v_neigh_Vol_p, m_new_neigh_n, v_neigh_Vol_n,
                                true, sol_coarser.C, sol_coarser.gamma, level, v_p_index, v_n_index, uset_SV_index_p, uset_SV_index_n,
                                m_VD_p, m_VD_n, m_VD_both, v_mat_all_predict_validation[iter], m_TD, i, v_mat_all_predict_TD[iter]);



            }

            paramsInst->update_levels_models_info(level, v_groups.size());        // @072617
            t_all_parts_training.stop_timer("[RF][main] training for all partitions");

            /// - - - - - - - calculate the quality of the models on Validation Data (boosting, majority voting,...) - - - - - - -
            MatAssemblyBegin(v_mat_all_predict_validation[iter], MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(v_mat_all_predict_validation[iter], MAT_FINAL_ASSEMBLY);


            /// - - - - - - - report the final evaluation on Test Data (boosting, majority voting,...) - - - - - - -
            // I need to skip predicting for the lower levels for preformance // TODO, #Performance
            MatAssemblyBegin(v_mat_all_predict_TD[iter], MAT_FINAL_ASSEMBLY);
            MatAssemblyEnd(v_mat_all_predict_TD[iter], MAT_FINAL_ASSEMBLY);

        }// end of       for(int iter=0; iter < 2; iter++){  in line 81



        summary curr_level_validation_summary;
        pt.calc_performance_measure(m_VD_both, v_mat_avg_centers, v_mat_all_predict_validation,curr_level_validation_summary);

        // - - - - - - calculate and report the performance quality of all the trained model on the test data at the current level - - - - -
        pt.calc_performance_measure(m_TD, v_mat_avg_centers, v_mat_all_predict_TD,summary_TD);

        // - - - - - - Add validation information for this level to the vector of whole results for all levels on validation data - - - - -
        ref_results current_level_refinement_results;
        current_level_refinement_results.validation_data_summary = curr_level_validation_summary;
        current_level_refinement_results.test_data_summary = summary_TD;
        current_level_refinement_results.level = level;
        v_ref_results.push_back(current_level_refinement_results);

        MatDestroy(&m_TD);
        MatDestroy(&m_VD_both);
        for(int iter=0; iter < num_iter_refinement; iter++){
            MatDestroy(&v_mat_all_predict_TD[iter]);
        }

        paramsInst->print_summary(curr_level_validation_summary, "[RF][main] VD",level,-2);
        paramsInst->print_summary(summary_TD, "[RF][main] TD",level,-2);

        // - - - - - - - - - - prepare the solution - - - - - - - - - -  #7
        if(level > 1 ){
            std::cout << "[RF][main] prepareing the solution except for the finest level" << std::endl;
//            std::cout << "[RF][main] before prepare solution minority" << std::endl;
            for(auto it = uset_SV_index_p.begin(); it!=uset_SV_index_p.end(); it++){
                    sol_refine.p_index.push_back(*it);
                }
//            std::cout << "[RF][main] before prepare solution majority" << std::endl;
            for(auto it = uset_SV_index_n.begin(); it!=uset_SV_index_n.end(); it++){
                    sol_refine.n_index.push_back(*it);
                }
            std::cout << "[RF][main] after prepare solution majority" << std::endl;
            sol_refine.C = sol_coarser.C;
            sol_refine.gamma = sol_coarser.gamma;
        #if dbl_RF_main_with_partition >=3
            std::cout << "[RF][main] nSV+:"<< uset_SV_index_p.size() << " nSV-:"<< uset_SV_index_n.size() << std::endl;
        #endif


        }// end of "if(level > 1 )" for preparing the solution for levels except the finest

        MatDestroy(&m_neigh_WA_p);
        MatDestroy(&m_neigh_WA_n);
        VecRestoreSubVector(v_vol_p, IS_neigh_p, &v_neigh_Vol_p);
        VecRestoreSubVector(v_vol_n, IS_neigh_n, &v_neigh_Vol_n);
        VecDestroy(&v_vol_p);   // I shouldn't destroy them as long as I haven't restore the subvector
        VecDestroy(&v_vol_n);
    #if dbl_RF_main_with_partition >= 3
        std::cout << "\n[MS][PS] ------------ end of partitioning at level " << level << " ------------\n" << std::endl;
    #endif
    }
    else
    {
        /***********************************************************************************************/
        /*                                     No Partitioning                                         */
        /***********************************************************************************************/
#if dbl_RF_main_no_partition >=1
        printf("\n[RF][main] * * * * No Partitioning, level:%d * * * * \n",level);
#endif

        if(paramsInst->get_ms_status() &&
            (num_neigh_row_p_ + num_neigh_row_n_) < paramsInst->get_ms_limit()    ){
            // ------- call Model Selection (SVM) -------
            ModelSelection ms_refine;
            ms_refine.uniform_design_separate_validation(m_new_neigh_p, v_vol_p, m_new_neigh_n, v_vol_n, true,
                                                         sol_coarser.C, sol_coarser.gamma, m_VD_p, m_VD_n, level, sol_refine, v_ref_results);
#if dbl_RF_main_no_partition >=1
            std::cout << "[RF]{no partitioning} ms_active uniform design is finished!\n";
#endif

        }else{  // No model selection either because it is disabled or the threshold is reached
            Solver sv_refine;
            svm_model * trained_model;
            trained_model = sv_refine.train_model(m_new_neigh_p, v_vol_p, m_new_neigh_n, v_vol_n, 1, sol_coarser.C,sol_coarser.gamma) ;
            sv_refine.evaluate_testdata(level,summary_TD);

            std::cout << "[RF]{no partitioning}{no model selection} the validation should be add, and the result should be saved!, EXIT!\n";
            std::cout << "[RF]{no partitioning}{no model selection} Either increase the model selection threshold or reduce the start of partitioning to fix this.\n";
            exit(1); // #todolist
//            v_perf_per_level_refinement.push_back(summary_TD);           //collect the final best model at each level
            sv_refine.prepare_solution_single_model(trained_model,num_neigh_row_p_,sol_refine);
            sv_refine.free_solver("[RF]");
            // models are exported in the solver class if it is needed
        }

    #if dbl_RF_main >=5
        if(level > 1 ){         //skip this info for the finest level (since in the MS, they are not provided)
            printf("[RF][main] sol_refine.C:%g, sol_refine.gamma:%g\n",sol_refine.C,sol_refine.gamma);
            printf("[RF][main] sol_refine.p size:%lu, sol_refine.n size:%lu\n",sol_refine.p_index.size() ,sol_refine.n_index.size());
        }
    #endif
        MatDestroy(&m_WA_p);
        MatDestroy(&m_WA_n);
        VecDestroy(&v_vol_p);
        VecDestroy(&v_vol_n);
    }
    // free ISs for both situation with or without partitioning
    ISDestroy(&IS_neigh_p);
    ISDestroy(&IS_neigh_n);
    return sol_refine;
}



/*
 * @input:
 *      cc_name: class name used for logging information
 * @output:
 *      m_neighbors is the matrix of neighbor points
 *      neigh_id is IS type which contain the indices for neighbor points
 */

//void Refinement::find_SV_neighbors(Mat& m_data, Mat& m_P, std::vector<int>& seeds_ind,
//                                      Mat& m_SV, Mat& m_neighbors, std::string cc_name,
//                                      IS& IS_neigh_id){
void Refinement::find_SV_neighbors(Mat& m_data, Mat& m_P, std::vector<int>& seeds_ind,
                                      Mat& m_WA, Mat& m_neighbors, std::string cc_name,
                                      IS& IS_neigh_id){

    // create the index set to get the sub matrix in the end
    PetscInt        * ind_;         //arrays of Int that contains the row indices
    PetscInt        num_row_fine_points=0;
    unsigned int num_seeds;
    PetscInt ncols=0;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;


    MatGetSize(m_data,&num_row_fine_points,NULL);
#if dbl_RF_FSN >=5
    std::cout << "[RF][FSN]{" << cc_name << "} m_data num row as num_row_fine_points:"<< num_row_fine_points <<std::endl;
#endif
    /// - - - - - - - - - Create P Transpose matrix - - - - - - - -
    // P' : find fine points in rows instead of columns due to performance issues with Aij matrix
    num_seeds = (int) seeds_ind.size();
#if dbl_RF_FSN >=3
    std::cout  << "[RF][FSN]{" << cc_name << "} initialize num_seeds:" << num_seeds << "\n";
#endif
    PetscMalloc1(num_row_fine_points,&ind_);
    Mat m_Pt_;
    MatTranspose(m_P,MAT_INITIAL_MATRIX,&m_Pt_);
    MatDestroy(&m_P);                           //to make sure I don't use it by mistake

    PetscInt num_row_m_Pt_, num_col_m_Pt_;
    MatGetSize(m_Pt_,&num_row_m_Pt_,&num_col_m_Pt_);
#if dbl_RF_FSN >=5
    std::cout << "[RF][FSN]{" << cc_name << "} P transpose dim ["<< num_row_m_Pt_ <<","<< num_col_m_Pt_ << "]" <<std::endl;
    std::cout << "[RF][FSN]{" << cc_name << "} m_data num rows:"<< num_row_fine_points << std::endl;
    #if dbl_RF_FSN >=7            //should be above 7
        std::cout << "[RF][FSN]{" << cc_name << "} list of all SVs are:\n";
//        if(cc_name == "Majority"){
            std::cout << "[RF][FSN]{" << cc_name << "} [HINT]for no fake point, they should start from zero, not couple hundreds:\n";
//        }
        // num_seeds comes from number of SV from the solution from model selection
        for(unsigned int i=0; i < num_seeds ; i++){
            printf("%d, ",seeds_ind[i]);
        }
        printf("\n");
    #endif
#endif

    // a temporary vector for parts in a selected aggregate
    //maximum number of points(columns) in each row of P'
    std::vector<selected_agg > v_agg_;
    v_agg_.reserve(num_row_fine_points);

    /// - - - - - reserve as the number of rows in finer data set (for each class) - - - - -
    std::vector<int> v_fine_neigh_id(num_row_fine_points);

    /// - - - - - - - - - Select fine points - - - - - - - -
    // Loop over indices of SV's in coarser level in P' matrix (Oct 2, #bug, fixed)
    for(unsigned int i=0; i < num_seeds ; i++){
        MatGetRow(m_Pt_,seeds_ind[i],&ncols, &cols, &vals);

#if dbl_RF_FSN >=1
        if(ncols == 0){
            std::cout  << "[RF][FSN]{" << cc_name << "} empty row in P' at row i:"<< i
                       << " seeds_ind[i]:" << seeds_ind[i] << " ncols:" << ncols << std::endl;
            exit(1);
        }
        #if dbl_RF_FSN >=3
            std::cout  << "[RF][FSN]{" << cc_name << "} MatGetRow of P' matrix in loop seeds_ind[i]:"
                           << seeds_ind[i] << " ncols:" << ncols << std::endl;
        #endif
#endif
        // - - - - if there is only one node in this aggregate, select it - - - -
        if(ncols == 1){
            v_fine_neigh_id[cols[0]] = 1;
        }
        else {                  // multiple nodes participate this aggregate
            for(int j=0; j < ncols ; j++){  // for each row
                // - - - - - create a vector of pairs - - - - -
                // (fine index, participation in aggregate)
                v_agg_.push_back( selected_agg(cols[j], vals[j]) );
            }

            // - - - sort the vector of multiple participants in this aggregate - - -
            std::sort(v_agg_.begin(), v_agg_.end(), std::greater<selected_agg>());

#if dbl_RF_FSN >=7
    printf("==== [MR][inside selecting agg]{after sort each row of P'} i:%d ====\n",i);
    for (auto it = v_agg_.begin(); it != v_agg_.end(); it++){
        printf("index:%d, value:%g\n", it->index, it-> value);
    }
//index is the column number and important part
//value is only used to find the important indices (selected indices)
#endif

            // - - - select fraction of participants - - -
            float add_frac_ = ceil(paramsInst->get_rf_add_fraction() * ncols); // read add_fraction from parameters
            for (auto it = v_agg_.begin(); it != v_agg_.begin() + add_frac_ ; it++){
                v_fine_neigh_id[it->index] =1 ;
            }

            v_agg_.clear();
        } // end of else for multiple participants in this aggregate

        MatRestoreRow(m_Pt_,seeds_ind[i],&ncols, &cols, &vals);
    }
    MatDestroy(&m_Pt_);
#if dbl_RF_FSN >=9
    std::cout<<"[RF][find_SV_neighbors] num_seeds:"<<num_seeds<<std::endl;
#endif
    /// - - - - - - - - - Add distant points - - - - - - - - -

    // - - - - - calc average edge weight - - - - -
    Vec     v_sum_edge_weight;
    VecCreateSeq(PETSC_COMM_SELF,num_row_fine_points,&v_sum_edge_weight);
    MatGetRowSum(m_WA,v_sum_edge_weight);
    PetscScalar         * a_sum_edge_weight;
    PetscMalloc1(num_row_fine_points, &a_sum_edge_weight);
    VecGetArray(v_sum_edge_weight, &a_sum_edge_weight);

    double rf_add_dist_threshold = 0;                           // should be a parameter
    for(unsigned int i=0; i < v_fine_neigh_id.size(); i++){
        if(v_fine_neigh_id[i] == 1 ){        // if this row is selected
            MatGetRow(m_WA,i, &ncols, &cols, &vals);            // Get the selected row
            double avg_edge_weight = a_sum_edge_weight[i] / ncols;
            for(int j=0; j<ncols; j++){
                /// - - - - If I want to add another parameter, I can use the vals[j] to select ratio of the closest points - - - - - //TODO
                if(  (vals[j] >= rf_add_dist_threshold * avg_edge_weight) && (v_fine_neigh_id[cols[j]]!=1)  ){
                        v_fine_neigh_id[cols[j]]=2;
                }
            }
            MatRestoreRow(m_WA,i, &ncols, &cols, &vals);
        }
    }

    /// - - - - - - - - - Get the output submatrix - - - - - - - - -
    // Add points which are selected to the array for the output submatrix
    int cnt_agg_part_distant_neighbor = 0;
    int cnt_total=0;
    for(unsigned int i=0; i < v_fine_neigh_id.size(); i++){
        if(paramsInst->get_rf_add_distant_point_status()){
            if(v_fine_neigh_id[i] > 0 ){        // it participate directly(1) or it is a distant neighbor(2)
                ind_[cnt_total] = i;
                cnt_total++;
                if(v_fine_neigh_id[i] == 2 ){
                    cnt_agg_part_distant_neighbor++;
                }
            }
        }else{                                  // distant neighbors are ignored
            if(v_fine_neigh_id[i] == 1 ){        // if this row is selected
                ind_[cnt_total] = i;
                cnt_total++;
            }
        }
    }   // the ind_ is sorted as it fills in sorted order (i is sorted in the above loop)


    // Using WA matrix, find the neighbors of points which are participated in SV's aggregate
#if dbl_RF_FSN >=1      // this should be 1
    if(paramsInst->get_rf_add_distant_point_status()){
        std::cout  << "[RF][FSN]{" << cc_name << "} num of points participated in SV aggregates are: "<< cnt_total - cnt_agg_part_distant_neighbor  << std::endl;
        std::cout  << "[RF][FSN]{" << cc_name << "} num of distant 1 neighbor of above points are::"<< cnt_agg_part_distant_neighbor << std::endl;
    }else{
        std::cout  << "[RF][FSN]{" << cc_name << "} num of points participated in SV aggregates are: "<< cnt_total <<
                      " and distant neighbors are ignored!" << std::endl;
    }

#endif




    //create the IS (Index Set)
    ISCreateGeneral(PETSC_COMM_SELF,cnt_total,ind_,PETSC_COPY_VALUES,&IS_neigh_id);
    PetscFree(ind_);      //free the indices as I have created the IS

#if dbl_RF_FSN >=7          //default is 7
    printf("[MR] IS is created \n");               //$$debug
    ISView(IS_neigh_id,PETSC_VIEWER_STDOUT_WORLD);
//        MatGetSize(m_data,&num_row_fine_points,NULL);
//        printf("[MR] m_data num rows: %d\n",num_row_fine_points);
#endif


    MatGetSubMatrix(m_data,IS_neigh_id, NULL,MAT_INITIAL_MATRIX,&m_neighbors);


#if dbl_RF_FSN >=3
    PetscInt m_neighbors_num_row =0, m_neighbors_num_col;
    MatGetSize(m_neighbors ,&m_neighbors_num_row,&m_neighbors_num_col);

    std::cout  << "[RF][FSN]{" << cc_name
                  << "} new sub matrix dimension #row:" << m_neighbors_num_row
                  << ",#col:" <<m_neighbors_num_col << std::endl;
#endif
#if dbl_RF_FSN >=7      //default is 7
    std::cout  << "[RF][FSN]{" << cc_name << "} m_neighbors matrix:\n";                       //$$debug
    MatView(m_neighbors,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif
//    ISDestroy(&IS_neigh_id);  // Don't destroy it. It is required later in the partitioning
}






void Refinement::process_coarsest_level(Mat& m_data_p, Vec& v_vol_p, Mat& m_data_n, Vec& v_vol_n, Mat& m_VD_p, Mat& m_VD_n, int level,
                                        solution& sol_coarsest, std::vector<ref_results>& v_ref_results){
    PetscInt check_num_row_VD;
    MatGetSize(m_VD_p, &check_num_row_VD, NULL );
    if(!check_num_row_VD){
        std::cout << "[RF][PCL] empty validation data for minority class, Exit!" << std::endl;
        exit(1);
    }

    MatGetSize(m_VD_n, &check_num_row_VD, NULL );
    if(!check_num_row_VD){
        std::cout << "[RF][PCL] empty validation data for majority class, Exit!" << std::endl;
        exit(1);
    }

    // - - - - for long runs - - - -
    bool l_inh_param=false;
    double local_param_c=1;
    double local_param_gamma=1;
    if(paramsInst->get_best_params_status()){     // use the best parameters from past trainings
        l_inh_param = true;
        local_param_c = paramsInst->get_best_C();
        local_param_gamma = paramsInst->get_best_gamma();
    }
                                            // - - - - - load the validation data - - - - -
    if(paramsInst->get_ms_status()){      // - - - - model selection - - - -

//        std::cout << "\n\n refinement ----  Iteration "<<
//                     paramsInst->get_main_current_kf_id() <<
//                     ", level: " << paramsInst->get_main_current_level_id() << std::endl;

        // call model selection method
        ModelSelection ms_coarsest;
        ms_coarsest.uniform_design_separate_validation(m_data_p, v_vol_p, m_data_n, v_vol_n, l_inh_param, local_param_c, local_param_gamma,
                                                       m_VD_p, m_VD_n, level, sol_coarsest,v_ref_results);
//        std::cout << "[RF][PCL] nSV+:" << sol_coarsest.p_index.size() << std::endl;     //$$debug

    }else{                                          // - - - - No model selection (call solver directly) - - - -

        Solver sv_coarsest;
        struct svm_model *coarsest_model;
        coarsest_model = sv_coarsest.train_model(m_data_p, v_vol_p, m_data_n, v_vol_n, l_inh_param, local_param_c, local_param_gamma);

        PetscInt num_row_p;
        MatGetSize(m_data_p, &num_row_p, NULL);
        prepare_single_solution(&coarsest_model, num_row_p, sol_coarsest);
        std::cout << "[RF][PCL] the process_coarsest_level without model selection is incomplete, EXIT!" << std::endl;
        exit(1);
        sv_coarsest.free_solver("[RF][PCL]");        // Notice, the sv_coarsest is only availabel in this scope, and not accessible outside the else clause
    }
}


void Refinement::prepare_single_solution(svm_model **svm_trained_model, int num_row_p, solution& result_solution){
    PetscInt i;
    result_solution.C = (*svm_trained_model)->param.C;
    result_solution.gamma = (*svm_trained_model)->param.gamma;
    std::cout << "[RF][PSS] params are set C:"<< result_solution.C << std::endl ;

    result_solution.p_index.reserve((*svm_trained_model)->nSV[0]);   //reserve the space for positive class
    for (i=0; i < (*svm_trained_model)->nSV[0];i++){
        // -1 because sv_indice start from 1, while petsc row start from 0
        result_solution.p_index.push_back((*svm_trained_model)->sv_indices[i] - 1);
    }
    std::cout << "[RF][PSS] P class is prepared\n";

    result_solution.n_index.reserve((*svm_trained_model)->nSV[1]);   //reserve the space for negative class
    // start from 0 to #SV in majority
    // add the index in the model for it after subtract from number of minority in training data
    for (i=0; i < (*svm_trained_model)->nSV[1];i++){
        // -1 the same as pos class, p_num_row because they are after each other
        result_solution.n_index.push_back((*svm_trained_model)->sv_indices[(*svm_trained_model)->nSV[0] + i] - 1 - num_row_p);
    }
    std::cout << "[RF][PSS] N class is prepared\n";
}


struct BetterGmean
{
    bool operator () (const ref_results& a, const ref_results& b) const
    {
        return (a.validation_data_summary.perf.at(Gmean) > b.validation_data_summary.perf.at(Gmean));         //a has larger gmean than b
    }
};


void Refinement::add_best_model(std::vector<ref_results>& v_ref_results) const{
    // - - - - - find the best model - - - - -
#if dbl_RF_ABM >=5
    printf("\n[RF][SBM] final model at each level of refinement (before sort)\n");
    for(auto it=v_ref_results.begin() ; it != v_ref_results.end(); ++it){
        paramsInst->print_summary(it->validation_data_summary, "[RF][SBM] (A v-cycle) VD", it-> level);
    }
    for(auto it=v_ref_results.begin() ; it != v_ref_results.end(); ++it){
        paramsInst->print_summary(it->test_data_summary, "[RF][SBM] (A v-cycle) TD", it-> level);
    }
#endif

    std::sort(v_ref_results.begin(),v_ref_results.end(),BetterGmean());   // select the model with best G-mean or any other preferred performance measure

#if dbl_RF_ABM >=3
    printf("\n[RF][SBM] final model at each level of refinement (after sort)\n");
    for(auto it=v_ref_results.begin() ; it != v_ref_results.end(); ++it){
        paramsInst->print_summary(it->validation_data_summary, "[RF][SBM] (A v-cycle) VD", it-> level);
    }
    for(auto it=v_ref_results.begin() ; it != v_ref_results.end(); ++it){
        paramsInst->print_summary(it->test_data_summary, "[RF][SBM] (A v-cycle) TD", it-> level);
    }
#endif
    // - - - - - add the best model to final results of experiment for this v-cycle - - - - -
    paramsInst->add_final_summary(v_ref_results[0].test_data_summary, v_ref_results[0].level);

}
