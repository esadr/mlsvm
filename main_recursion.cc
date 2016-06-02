#include "main_recursion.h"
#include <vector>
#include <algorithm>
#include <cmath>        // round (ceil)
#include "config_logs.h"
#include "config_params.h"
#include "etimer.h"
#include <metis.h>
#include "common_funcs.h"       //only for debugging
#include "chrono"
#include "thread"
#include "loader.h"

struct selected_agg
{
    int index;
    double value;

    selected_agg(int col_index, double fraction_val) : index(col_index), value(fraction_val) {}

    bool operator > (const selected_agg sel_agg) const {
        return (value > sel_agg.value);
    }
};


solution MainRecursion::main(Mat& p_data, Mat& m_P_p_f, Mat& p_WA, Vec& p_vol,
                             Mat& n_data, Mat& m_P_n_f, Mat& n_WA, Vec& n_vol, int level){
    level++;
    int c_limit = Config_params::getInstance()->get_coarse_threshold();
    Mat p_data_c,p_WA_c, n_data_c, n_WA_c,m_P_p,m_P_n;  //data_c is aggregated data (coarse data point)
    Vec p_vol_c, n_vol_c;
    PetscInt p_num_row, n_num_row;
    std::vector<NodeId> v_p_seeds_indices, v_n_seeds_indices;

    MatGetSize(p_data, &p_num_row,0);    //m returns the number of rows globally
    MatGetSize(n_data, &n_num_row,0);    //m returns the number of rows globally

    /// ----------------- Both classes are small enough, Now call the ModelSelection (Recursive Condition) -----------------
    if ((p_num_row <  c_limit) && (n_num_row <  c_limit))  {          //both classes are small enough (coarsest level)
        printf("\t[MR]======== End of Coarsening ========= level:%d\n",level);
        MatGetSize(p_data,&p_num_row,0);    //m returns the number of rows globally
        MatGetSize(n_data,&n_num_row,0);    //m returns the number of rows globally
        printf("[MR][main] num points P:%d, N:%d\n",p_num_row, n_num_row);      //$$debug
        printf("[MR][main] start to solve SVM for level:%d\n",level);      //$$debug

        solution sol_coarsest;
        ModelSelection ms_coarsest;

        if(Config_params::getInstance()->get_best_params_status()){
            sol_coarsest =  ms_coarsest.UD(p_data, p_vol, n_data, n_vol, 1, Config_params::getInstance()->get_best_C() ,
                                  Config_params::getInstance()->get_best_gamma(),level);
        }else{
            printf("[MR][main] coarsest level is calling MS.UD at level:%d\n",level);      //$$debug
            sol_coarsest =  ms_coarsest.UD(p_data, p_vol, n_data, n_vol, 0,0,0,level);
        }
        // free resources
        ms_coarsest.free_model_selection("[MR][Main] in the coarsest level");
        MatDestroy(&p_data);
        MatDestroy(&n_data);
        MatDestroy(&p_WA);
        MatDestroy(&n_WA);

//        MatDestroy(&p_data_c);
//        MatDestroy(&n_data_c);
//        MatDestroy(&m_P_p);
//        MatDestroy(&m_P_n);

        return sol_coarsest;                                // return the coarsest solution

    }else{      ///-------------- Coarsening -------------------
        ETimer t_coarse;
        if(level > Config_params::getInstance()->get_cp_max_coarse_level())     {
            printf("!!! the coarsening is not converged after %d levels, skip this run !!!\n", level);
            solution empty_solution;
            empty_solution.C = -1;
            return empty_solution;
        }
        double filter_threshold = Config_params::getInstance()->get_cs_weak_edges_ft();
        Coarsening p_coarser("Minority") ;
        Coarsening n_coarser("Majority");
        printf("\t[MR]============ Coarsening level:%d ============\n",level);
        if(p_num_row > c_limit){
            printf("++++ Positive class ++++\n");
            m_P_p = p_coarser.calc_P(p_WA, p_vol, v_p_seeds_indices); //m_P_p measn P matrix for positive label (minority class)
            t_coarse.stop_timer("[MR][Main]{1} from start of both class calc_p minority, level:",std::to_string(level));

//            CommonFuncs cf;
//            cf.exp_matrix(p_WA,"./data","WA_matrix_min_"+Config_params::getInstance()->get_ds_name(),"MR");
//            exit(1);

            p_data_c = p_coarser.calc_aggregate_data(m_P_p, p_data,p_vol, v_p_seeds_indices);
            t_coarse.stop_timer("[MR][Main]{2} from start of both class calc Agg Data minority, level:",std::to_string(level));

            p_WA_c = p_coarser.calc_WA_c(m_P_p, p_WA);
            t_coarse.stop_timer("[MR][Main]{2} from start of both class calc Agg Data minority, level:",std::to_string(level));

            p_coarser.filter_weak_edges(p_WA_c, filter_threshold);

            p_vol_c = p_coarser.calc_coarse_volumes(m_P_p, p_vol);
            t_coarse.stop_timer("[MR][Main]{3} from start of both class calc Agg Vol minority, level:",std::to_string(level));
        }else{
            if(level == 1){ // if the minority class don't need any coarsening
                std::cout << "the minority class is too small, reduce the coarse threshold in parameters, Exit!" << std::endl;
                // Create a fake P matrix
                exit(1);
            }
            //don't coarse
            std::cout << "[MR][Main] The minority class is not coarsen anymore" << std::endl;
            // duplicate the P matrix of finer level since I don't need to calculate it
            MatDuplicate(m_P_p_f, MAT_COPY_VALUES, &m_P_p);
            m_P_p = p_coarser.calc_P(p_WA, p_vol, v_p_seeds_indices); //m_P_p measn P matrix for positive label (minority class)

            MatDuplicate(p_WA, MAT_COPY_VALUES, &p_WA_c);
            MatDuplicate(p_data, MAT_COPY_VALUES, &p_data_c);
            VecDuplicate(p_vol,&p_vol_c);
            VecCopy(p_vol,p_vol_c);
        }

        printf("-------- Negative class --------\n");
        m_P_n = n_coarser.calc_P(n_WA, n_vol, v_n_seeds_indices); // same for majority class
        t_coarse.stop_timer("[MR][Main]{4} from start of both class calc_p majority",std::to_string(level));

        n_data_c = n_coarser.calc_aggregate_data(m_P_n, n_data, n_vol, v_n_seeds_indices);
        t_coarse.stop_timer("[MR][Main]{5} from start of both class calc Agg Data majority",std::to_string(level));

        n_WA_c = n_coarser.calc_WA_c(m_P_n, n_WA);
        t_coarse.stop_timer("[MR][Main]{6} from start of both class calc Agg Vol majority",std::to_string(level));

        n_coarser.filter_weak_edges(n_WA_c, filter_threshold);

        n_vol_c = n_coarser.calc_coarse_volumes(m_P_n, n_vol);
        t_coarse.stop_timer("[MR][Main]{7} from start of both class calc Agg Vol majority",std::to_string(level));



#if debug_export == 1
    if (Config_params::getInstance() -> get_debug_exp_MR() >= 1){
        //export Minority(p) matrices
        PetscViewer exp_viewer_p_aggdt;
        char tmp_p_f_name[300];
        sprintf(tmp_p_f_name, "./debug/exp_MR_agg_data_P_l_%d",level);
        const char* exp_p_name_aggdt = tmp_p_f_name;
        PetscViewerBinaryOpen(PETSC_COMM_WORLD,exp_p_name_aggdt,FILE_MODE_WRITE,&exp_viewer_p_aggdt);
        MatView(p_data_c,exp_viewer_p_aggdt);
        PetscViewerDestroy(&exp_viewer_p_aggdt);

        //export Majority(n) matrices
        PetscViewer     viewer_n_aggdt;
        char tmp_n_f_name[300];
        sprintf(tmp_n_f_name, "./debug/exp_MR_agg_data_N_l_%d",level);
        const char* exp_n_name_aggdt = tmp_n_f_name;
        PetscViewerBinaryOpen(PETSC_COMM_WORLD,exp_n_name_aggdt,FILE_MODE_WRITE,&viewer_n_aggdt);
        MatView(n_data_c,viewer_n_aggdt);
        PetscViewerDestroy(&viewer_n_aggdt);
    }
#endif
        t_coarse.stop_timer("coarsening at level",std::to_string(level));


        solution sol_coarser ;
        sol_coarser = main(p_data_c, m_P_p, p_WA_c, p_vol_c, n_data_c, m_P_n, n_WA_c, n_vol_c,level); // recursive call
//        std::cout << "\n\n [MR][Main] after returning from model selection\n\n";
        if(sol_coarser.C == -1 ){       // the coarsening didn't converge, this is going back
            //free the memory
            MatDestroy(&p_data);
            MatDestroy(&n_data);
            MatDestroy(&p_data_c);
            MatDestroy(&n_data_c);
            MatDestroy(&p_WA_c);
            MatDestroy(&n_WA_c);
            VecDestroy(&p_vol);
            VecDestroy(&n_vol);

            // add summary to final reports
            solution empty_sol_refine;
            empty_sol_refine.C = -1;
            return empty_sol_refine;
        }

        ///------------------------- Refinement ----------------------------
        ETimer t_refine;
        printf("[info]coarse solution from level:%d \n",level+1); //because it comes from coarser level
        printf("\n================= Refinement level:%d  =================\n",level);

        solution sol_refine;
        sol_refine = refinement(p_data,m_P_p,p_vol, p_WA, n_data,m_P_n,n_vol, n_WA, sol_coarser,level);

        MatDestroy(&p_data);
        MatDestroy(&n_data);
        MatDestroy(&p_data_c);
        MatDestroy(&n_data_c);
        MatDestroy(&p_WA_c);
        MatDestroy(&n_WA_c);
        VecDestroy(&p_vol);
        VecDestroy(&n_vol);

        t_refine.stop_timer("[MR] Refinement at level:",std::to_string(level));
        return sol_refine;
    }
}


solution MainRecursion::refinement(Mat& m_data_p, Mat& m_P_p, Vec& v_vol_p, Mat&m_WA_p,
                                   Mat& m_data_n, Mat& m_P_n, Vec& v_vol_n, Mat&m_WA_n,
                                   solution& sol_coarser,int level ){
#if dbl_MR_refinement >=5
    PetscInt num_row_p_data =0, num_row_n_data =0, num_row_p_SV =0, num_row_n_SV =0;
    MatGetSize(m_data_p,&num_row_p_data,NULL);
    MatGetSize(m_data_n,&num_row_n_data,NULL);
    printf("[MR][Ref]{beginnig} num fine data p:%d, n:%d\n",num_row_p_data,num_row_n_data);
    MatGetSize(sol_coarser.m_p_SV,&num_row_p_SV,NULL);
    MatGetSize(sol_coarser.m_n_SV,&num_row_n_SV,NULL);
    PetscPrintf(PETSC_COMM_WORLD,"[MR][Ref]{beginnig} num SV data p:%d, n:%d\n",num_row_p_SV,num_row_n_SV);

    PetscInt num_row_p_P =0, num_row_n_P =0, num_col_p_P =0, num_col_n_P =0;
    MatGetSize(m_P_p,&num_row_p_P, &num_col_p_P);
    MatGetSize(m_P_n,&num_row_n_P, &num_col_n_P);
    PetscPrintf(PETSC_COMM_WORLD,"[MR][Ref]{beginnig} Minority class P dim [%d,%d]\n",num_row_p_P,num_col_p_P);
    PetscPrintf(PETSC_COMM_WORLD,"[MR][Ref]{beginnig} Majority class P dim [%d,%d]\n",num_row_n_P,num_col_n_P);

    PetscPrintf(PETSC_COMM_WORLD,"[MR][Ref] sol_coarser.p_index.size():%lu, sol_coarser.n_index.size():%lu\n",
                                sol_coarser.p_index.size(),sol_coarser.n_index.size());
    PetscPrintf(PETSC_COMM_WORLD,"[MR][Ref] call find_SV_neighbors for minority (P) at level:%d\n",level);
#endif

    Mat m_new_neigh_p, m_new_neigh_n;
    IS IS_neigh_p, IS_neigh_n;
    /// - - - - - - - get new points for finer level - - - - - - -
    find_SV_neighbors(m_data_p,m_P_p,sol_coarser.p_index, m_WA_p, m_new_neigh_p,"Minority",IS_neigh_p);
    find_SV_neighbors(m_data_n,m_P_n,sol_coarser.n_index, m_WA_n, m_new_neigh_n,"Majority",IS_neigh_n);

    // - - - - get the size of neighbors - - - -
    PetscInt num_neigh_row_p_ =0, num_neigh_row_n_ =0;
    MatGetSize(m_new_neigh_p,&num_neigh_row_p_,NULL);
    MatGetSize(m_new_neigh_n,&num_neigh_row_n_,NULL);
    PetscPrintf(PETSC_COMM_WORLD,"[MR][Ref] num new neighbor p:%d, n:%d\n",num_neigh_row_p_,num_neigh_row_n_);
    // check for errors
    if(num_neigh_row_p_ == 0 || num_neigh_row_n_ == 0){
        PetscPrintf(PETSC_COMM_WORLD,"\n[Error]:[MR][refinement] Empty matrices for new neighbors,\n Exit!\n");
        exit(1);
    }

    solution sol_refine;

    /// - - - - check to start partitioning - - - -
    PetscInt num_WA_p_ =0, num_WA_n_ =0;
    MatGetSize(m_WA_p,&num_WA_p_,NULL);
    MatGetSize(m_WA_n,&num_WA_n_,NULL);

    if( (num_neigh_row_p_ + num_neigh_row_n_)  > Config_params::getInstance()->get_rf_start_partitioning() ){
    /// - - - - - - Partition the data for each class - - - - - - -
        // check if I pass the WA, vol for the finer level
        // I only need to send the neighbor points not all of them:
        // First, get_parts(m_WA_p,v_vol_p, p_parts);
    //I have a problem to summarize all the solutions from multiple ms.UD() to 1 solution

        /*  - The information for partitioning is related to points in the finer level(current)
         *      which are going to pass to the SVM to creat a model
         *  - The WA, Vol contains all the points which are not needed.
         *  - Only the information for the points which are neighbor to SV of coarser level are required
         *  m_neigh_WA: only specific rows and columns
         *  v_nei_vol: volumes for the points in the m_neigh_WA
         */

        Mat m_neigh_WA_p, m_neigh_WA_n;
        Vec v_neigh_Vol_p, v_neigh_Vol_n;

        printf("[MR][Ref] Get the neighbors WA, Vol at level:%d\n",level);
        MatGetSubMatrix(m_WA_p, IS_neigh_p, IS_neigh_p,MAT_INITIAL_MATRIX,&m_neigh_WA_p);
        MatGetSubMatrix(m_WA_n, IS_neigh_n, IS_neigh_n,MAT_INITIAL_MATRIX,&m_neigh_WA_n);
        VecGetSubVector(v_vol_p, IS_neigh_p, &v_neigh_Vol_p);
        VecGetSubVector(v_vol_n, IS_neigh_n, &v_neigh_Vol_n);
        MatDestroy(&m_WA_p);
        MatDestroy(&m_WA_n);

#if dbl_MR_refinement >=5   //5
    PetscInt num_row_neigh_WA_p, num_row_neigh_WA_n, num_col_neigh_WA_p, num_col_neigh_WA_n;
    MatGetSize(m_neigh_WA_p,&num_row_neigh_WA_p, &num_col_neigh_WA_p);
    MatGetSize(m_neigh_WA_n,&num_row_neigh_WA_n, &num_col_neigh_WA_n);
    PetscPrintf(PETSC_COMM_WORLD,"[MR][Ref]{beginnig} Minority class neigh_WA dim is:[%dx%d]\n",num_row_neigh_WA_p,num_col_neigh_WA_p);
    PetscPrintf(PETSC_COMM_WORLD,"[MR][Ref]{beginnig} Majority class neigh_WA dim is:[%dx%d]\n",num_row_neigh_WA_n,num_col_neigh_WA_n);
#endif
        /// - - - - - find partitions - - - -

        int partition_max_size = Config_params::getInstance()->get_rf_partition_max_size();

        PetscInt num_vertex_p, num_vertex_n;
        MatGetSize(m_neigh_WA_p, &num_vertex_p,NULL);
        MatGetSize(m_neigh_WA_n, &num_vertex_n,NULL);
        int num_partitions_p , num_partitions_n;
        num_partitions_p = ceil( (float)num_vertex_p / (float)partition_max_size ); // ceil is used to make sure we get enough number of partitions
        num_partitions_n = ceil( (float)num_vertex_n / (float)partition_max_size ); // ceil is used to make sure we get enough number of partitions

        Mat m_parts_p, m_centers_p;
        if(num_partitions_p < 2){             // if the number of minority points are not large enough, manually set nparts to 2
            // call single partition
            single_part_matrix(m_new_neigh_p, m_parts_p);
            calc_single_center(m_new_neigh_p, v_neigh_Vol_p, m_centers_p);
        }else{
            std::vector<std::vector<PetscInt>> vv_parts_p(num_partitions_p);    
            get_parts(m_neigh_WA_p,v_neigh_Vol_p, vv_parts_p, m_parts_p);   // get partitions (Metis)
            calc_center(m_new_neigh_p, v_neigh_Vol_p, vv_parts_p, m_parts_p, m_centers_p);
        }
#if dbl_MR_ref_partition >=3
        printf("\n[MR][Ref] centers calculations for minority class is finished\n");
#endif
        Mat m_parts_n, m_centers_n;
        if(num_partitions_n < 2){             // if the number of minority points are not large enough, manually set nparts to 2
            // call single partition
            single_part_matrix(m_new_neigh_n, m_parts_n);
            calc_single_center(m_new_neigh_n, v_neigh_Vol_n, m_centers_n);
        }else{
            std::vector<std::vector<PetscInt>> vv_parts_n(num_partitions_n);
            get_parts(m_neigh_WA_n,v_neigh_Vol_n, vv_parts_n, m_parts_n);
            calc_center(m_new_neigh_n, v_neigh_Vol_n, vv_parts_n, m_parts_n, m_centers_n);
        }
#if dbl_MR_ref_partition >=3
        printf("\n[MR][Ref] centers calculations for majority class is finished\n");
#endif
        
        // - - - - Selection of partitions  - - - -
        // Multiple things to remember
        // at the end, I need to test the test data against multiple hyperplane (Not high pariority)
        // this part are going to change a lot I guess, because there are multiple research questions
        // make it more parametric to have more flexibility

        /// - - - - - calc distances - - - - -
        PetscInt            num_part_p, num_part_n, num_features;
        MatGetSize(m_centers_p, &num_part_p, &num_features);
        MatGetSize(m_centers_n, &num_part_n, NULL);

//        std::vector<std::vector<std::pair<int, double>>> dist_p(num_part_p);
        Mat m_dist;
        calc_distances(num_part_p, num_part_n, m_centers_p, m_centers_n, m_dist);
#if dbl_MR_ref_partition >=3
        printf("[MR][Ref] Distance calculations from minority class to majority class is finished, and the reverse is not needed\n");
#endif
        /// - - - - - for all groups, select set of parts as a group - - - - -
        std::vector<std::vector<PetscInt> > vv_groups;
        find_groups(num_part_p, num_part_n, m_dist, vv_groups );
#if dbl_MR_ref_partition >=3
        printf("[MR][Ref] find groups is finished\n");
#endif

        // I need to skip predicting for the lower levels for preformance // TODO, #Performance
        Mat m_TD ;
        Loader test_loader;
        m_TD = test_loader.load_norm_data_sep(this->test_dataset_f_name);
        PetscInt num_row_TD;
        MatGetSize(m_TD, &num_row_TD, NULL);

        std::unordered_set<PetscInt> uset_SV_index_p;
        std::unordered_set<PetscInt> uset_SV_index_n;
        uset_SV_index_p.reserve(2*num_neigh_row_p_);
        uset_SV_index_p.reserve(2*num_neigh_row_n_);
        Mat m_all_predict;
        MatCreateSeqDense(PETSC_COMM_SELF, num_part_p+num_part_n, num_row_TD,NULL, &m_all_predict);

        bool responce = false;
        int cnt_models = 0;
        ETimer t_all_parts_training;
        for(int i = 0; i < num_part_p+num_part_n ; i++ ){
            std::vector<PetscInt> v_p_index;
            std::vector<PetscInt> v_n_index;
            responce = create_group_index(i, num_part_p, num_part_n, vv_groups, m_parts_p, m_parts_n, v_p_index, v_n_index);
#if dbl_MR_ref_partition >= 3
            std::cout << "[MR][Ref] create_group_index respose for target "<< i << " is: "<< responce << std::endl;
#endif
            if(responce){
                // - - - - - - Train & predict Model - - - - - - -
                ModelSelection ms_partition;
                ms_partition.partial_solver(m_new_neigh_p, v_vol_p, m_new_neigh_n, v_vol_n, sol_coarser.C, sol_coarser.gamma, level,
                                  v_p_index,v_n_index, uset_SV_index_p, uset_SV_index_n);
                // I need to skip predicting for the lower levels for preformance // TODO, #Performance
                ms_partition.predict_label(m_TD, i, m_all_predict);     //predict labels using the model in MS instance

                ms_partition.free_model_selection("ms_partition in [MR][Ref]");
                cnt_models++;
            }
        }
        std::cout << "[MR][Ref] total number of partition groups are: " << cnt_models << std::endl;
        t_all_parts_training.stop_timer("[MR][Ref] training for all partitions");
#if dbl_MR_ref_partition >= 3
        std::cout << "\n[MR][Ref] after all "<< cnt_models << " models are trained \n" << std::endl;
#endif


        /// - - - - - - - report the final evaluation (boosting, majority voting,...) - - - - - - -
        // I need to skip predicting for the lower levels for preformance // TODO, #Performance
        MatAssemblyBegin(m_all_predict, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(m_all_predict, MAT_FINAL_ASSEMBLY);

        std::map<measures,double> final_results;

        final_results = calc_performance_measure(m_TD, m_all_predict);
        MatDestroy(&m_TD);
        MatDestroy(&m_all_predict);

#if dbl_MR_ref_partition >=3
            std::cout << "[MR][Ref] performance measures are calculated successfully!\n" << std::endl;
#endif

        if(level == 1){
            std::cout << "[MR][Ref] recording the final summary " << std::endl;
            iter_summary final_summary;
            final_summary.C = sol_coarser.C;
            final_summary.gamma = sol_coarser.gamma;
            final_summary.result[Acc] = final_results[Acc];
            final_summary.result[Sens] = final_results[Sens];
            final_summary.result[Spec] = final_results[Spec];
            final_summary.result[Gmean] = final_results[Gmean];
            Config_params::getInstance()->add_final_summary(final_summary);
            printf("[MR][Ref] TD, L:%d, Acc:%.2f, Sens:%.2f, Spec:%.2f, Gmean:%.2f, C:%.2f, G:%.4f\n",
                            level, final_results[Acc], final_results[Sens], final_results[Spec], final_results[Gmean],
                            sol_coarser.C, sol_coarser.gamma );     //$$debug
        }else{
#if rpt_TD_only_l1 == 0
            printf("[MR][Ref] TD, L:%d, Acc:%.2f, Sens:%.2f, Spec:%.2f, Gmean:%.2f, C:%.2f, G:%.4f\n",
                        level, final_results[Acc], final_results[Sens], final_results[Spec], final_results[Gmean],
                        sol_coarser.C, sol_coarser.gamma );     //$$debug
#endif
        }


        if(level > 1 ){
            std::cout << "[MR][Ref] prepareing the solution except for the finest level" << std::endl;
        /// - - - - - - - prepare the solution - - - - - - -
            std::cout << "[MR][Ref] before prepare solution minority" << std::endl;
            for(auto it = uset_SV_index_p.begin(); it!=uset_SV_index_p.end(); it++){
                    sol_refine.p_index.push_back(*it);
                }
            std::cout << "[MR][Ref] before prepare solution majority" << std::endl;
            for(auto it = uset_SV_index_n.begin(); it!=uset_SV_index_n.end(); it++){
                    sol_refine.n_index.push_back(*it);
                }
            std::cout << "[MR][Ref] after prepare solution majority" << std::endl;
            sol_refine.C = sol_coarser.C;
            sol_refine.gamma = sol_coarser.gamma;
#if dbl_MR_ref_partition >=3
            std::cout << "[MR][Ref] nSV+:"<< uset_SV_index_p.size() << " nSV-:"<< uset_SV_index_n.size() << std::endl;
#endif
        }

#if dbl_MR_ref_partition >= 3
        std::cout << "\n[MS][PS] ------------ end of partitioning at level" << level << " ------------\n" << std::endl;
#endif
    }else{
        /// - - - - - - No Partitioning - - - - - - -
#if dbl_MR_refinement >=1
        printf("[MR][Ref] Call model selection (No Partitioning) at level:%d\n",level);
        printf("[MR][Ref] SVM parameters are: C=%g, gamma=%g\n",sol_coarser.C,sol_coarser.gamma);     //$$debug
#endif
        // ------- call Model Selection (SVM) -------
        ModelSelection ms_refine;
//        sol_refine= ms_refine.UD(m_new_neigh_p, v_vol_p, m_new_neigh_n, v_vol_n,
//                                   1,sol_coarser.C,sol_coarser.gamma,level,
//                                   1, sol_coarser.p_index.size(), sol_coarser.n_index.size()) ;
        sol_refine= ms_refine.UD(m_new_neigh_p, v_vol_p, m_new_neigh_n, v_vol_n, 1, sol_coarser.C,sol_coarser.gamma,level) ;
        ms_refine.free_model_selection("ms_refine in [MR][Ref]");
#if dbl_MR_refinement >=5
    if(level > 1 ){         //skip this info for the finest level (since in the MS, they are not provided)
//        printf("\n\n\n----[MR][Ref] return solution Problem??? ----\n");

        printf("\n----[MR][Ref] return solution ----\n");
        printf("[MR][Ref] sol_refine.C:%g, sol_refine.gamma:%g\n",sol_refine.C,sol_refine.gamma);
        printf("[MR][Ref] sol_refine.p size:%lu, sol_refine.n size:%lu\n",sol_refine.p_index.size() ,sol_refine.n_index.size());

        PetscInt num_row_p_sol =0, num_row_n_sol =0, num_col_p_sol =0, num_col_n_sol =0;
        MatGetSize(sol_refine.m_p_SV, &num_row_p_sol, &num_col_p_sol);
        MatGetSize(sol_refine.m_n_SV, &num_row_n_sol, &num_col_n_sol);
        printf("[MR][Ref] sol_refine minority SV dim [%d,%d]\n",num_row_p_sol,num_col_p_sol);
        printf("[MR][Ref] sol_refine majority SV dim [%d,%d]\n",num_row_n_sol,num_col_n_sol);
    }
#endif

    }
    // free ISs for both situation with or without partitioning
    ISDestroy(&IS_neigh_p);
    ISDestroy(&IS_neigh_n);
    return sol_refine;
}


std::map<measures,double> MainRecursion::calc_performance_measure(Mat& m_TD, Mat& m_predicted_labels){
    // read the target labels from m_TD to a vector(test data matrix has the labels in the first column)
    /// - - - - - - - - get test data labels (target labels) - - - - - - - -
    PetscInt i=0, ncols=0,num_points=0,num_col=0;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;
    MatGetSize(m_TD,&num_points,&num_col);   //get the number of data points
    std::vector<double> v_target_labels(num_points);

    for (i=0; i< num_points;i++){
        MatGetRow(m_TD,i,&ncols, &cols, &vals);
        v_target_labels[i] = vals[0];             //read the label in the first column(0)
        MatRestoreRow(m_TD,i,&ncols, &cols, &vals);
    }

    /// - - - - - - - - calc predicted labels - - - - - - - -
    // it would be different techniques to calculate the final predicted labels
    std::vector<double> v_final_predicted_label(num_points);  // same for all methods
    // 1- majority voting
    calc_majority_voting(m_predicted_labels, num_points, v_final_predicted_label);

    /// - - - - - - - - compare the target labels and predicted - - - - - - - -
    int correct = 0;
    double tp =0, tn =0, fp =0, fn=0;

    for(int i=0; i < num_points; i++){
#if dbl_MR_calc_performance >= 5
        std::cout << "[MR][CPM] labels: target: "<< v_target_labels[i] << " predicted: " << v_final_predicted_label[i] <<std::endl;
#endif
        if(v_target_labels[i] == 1){                            // ++++ positive class ++++
            if (v_final_predicted_label[i] > 1)
                tp++;                   // true positive
            else
                fn++;
        }else{                                                  // ---- predict negative ----
            if (v_final_predicted_label[i] < 0)
                tn++;                   // true negative
            else
                fp++;                   // false positive
        }
    }
    /// - - - - - - - - calc different performance measures (Acc, Gmean,...)  - - - - - - - -
    correct = tp+tn;                //sum both True
    std::map<measures,double> results_;
    results_[Sens] = tp / (tp+fn) ;
    results_[Spec] = tn / (tn+fp) ;
    results_[Gmean] = sqrt(results_[Sens] * results_[Spec]);
    results_[Acc] = (double)correct / (num_points) ;
    return results_;
}

void MainRecursion::calc_majority_voting(Mat& m_predicted_labels, int num_TD_points, std::vector<double>& v_final_predicted_label ){
    Mat m_pl_t;
    MatTranspose(m_predicted_labels,MAT_INITIAL_MATRIX,&m_pl_t);
    Vec v_sum_labels;
    VecCreateSeq(PETSC_COMM_SELF, num_TD_points, &v_sum_labels);
    MatGetRowSum(m_pl_t, v_sum_labels);    //calc sum of each row of predict_label transpose
    MatDestroy(&m_pl_t);

    PetscScalar     *arr_predicted_labels;
    PetscMalloc1(num_TD_points, &arr_predicted_labels);
    VecGetArray(v_sum_labels, &arr_predicted_labels);
    for(int i=0; i <num_TD_points; i++){
        v_final_predicted_label[i] = arr_predicted_labels[i];
    }
    VecRestoreArray(v_sum_labels, &arr_predicted_labels);
    PetscFree(arr_predicted_labels);
}


/*
 * Notice the second part for N class is a little tricky
 * @input:
 *          target_group range is {0 to (num_part_p + num_part_n)
 *          v_group for n class works with target_group,
 *              but m_parts_n works with (target_group - num_part_p )
 *          parts matrices, groups
 * @output:
 *          two vector for this target group        (I skip calculating the IS directly, since I need the vector for finding the real index of SV_indices)
 *          bool value which represent if this is a valid group
 */
bool MainRecursion::create_group_index(int target_group, int num_part_p, int num_part_n, std::vector<std::vector<PetscInt> >& vv_groups,
                                       Mat& m_parts_p, Mat& m_parts_n, std::vector<PetscInt>& v_p_index,
                                       std::vector<PetscInt>& v_n_index){
    if(vv_groups[target_group].size() == 0) // prevent to recreate duplicate groups (in find_groups function they're inside vector is empty)
        return false;
    else{
        PetscInt j, ncols;
        const PetscInt    *cols;
        const PetscScalar *vals;
        // - - - - - - create vector of close parts if target belong to class P - - - - - -
        if(target_group < num_part_p){      //target is p class
            // add itself to p index
            MatGetRow(m_parts_p,target_group,&ncols, &cols, &vals);
            v_p_index.reserve(ncols);       //it might get bigger but I try to use limited knowledge here
            for(j=0; j< ncols; j++ ){
                v_p_index.push_back(cols[j]);
            }
            MatRestoreRow(m_parts_p,target_group,&ncols, &cols, &vals);

            // add close opposite class (N) parts
            for(unsigned int l=0; l < vv_groups[target_group].size(); l++){
                MatGetRow(m_parts_n, vv_groups[target_group][l],&ncols, &cols, &vals);
                for(j=0; j< ncols; j++ ){
                    v_n_index.push_back(cols[j]);
                }
                MatRestoreRow(m_parts_n, vv_groups[target_group][l],&ncols, &cols, &vals);
            }
        }
        // - - - - - - create vector of close parts if target belong to class N - - - - - -
        else{                              //target is n class
            // add itself to n index
            MatGetRow(m_parts_n,target_group - num_part_p ,&ncols, &cols, &vals);
            v_n_index.reserve(ncols);       //it might get bigger but I try to use limited knowledge here
            for(j=0; j< ncols; j++ ){
                v_n_index.push_back(cols[j]);
            }
            MatRestoreRow(m_parts_n,target_group - num_part_p ,&ncols, &cols, &vals);
            // add close opposite class (P) parts
            for(unsigned int l=0; l < vv_groups[target_group].size(); l++){
                MatGetRow(m_parts_p, vv_groups[target_group][l],&ncols, &cols, &vals);
                for(j=0; j< ncols; j++ ){
                    v_p_index.push_back(cols[j]);
                }
                MatRestoreRow(m_parts_p, vv_groups[target_group][l],&ncols, &cols, &vals);
            }
        }
    #if dbl_MR_CGI >= 5      // default 5
        // - - - - - - Print the final vectors - - - - - -
        std::cout<< "target is: "<< target_group <<std::endl;
        std::cout<< "v_p_index size: "<< v_p_index.size() <<std::endl;
        for(auto it = v_p_index.begin(); it != v_p_index.end(); it++ ){
            std::cout<< *it << ",";
        }
        std::cout<< std::endl;
        std::cout<< "v_n_index size: "<< v_n_index.size() <<std::endl;
        for(auto it = v_n_index.begin(); it != v_n_index.end(); it++ ){
            std::cout<< *it << ",";
        }
        std::cout<< std::endl;
    #endif
        return true;
    }
}

/*
 * @input:
 *      number of partitions(partitions) in each class, number of features in data
 *      distances matrix
 * @output:
 *      vector of groups (id represent the self partition, and inside vector represent the opposite partitions from other class)
 */
void MainRecursion::find_groups(int num_part_p, int num_part_n, Mat& m_dist, std::vector<std::vector<PetscInt> >& vv_groups){

    PetscInt i,j, ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    // - - - - - - create vector of close parts for class P - - - - - -
    for(i=0; i< num_part_p; i++ ){
        MatGetRow(m_dist,i,&ncols, &cols, &vals);    //P row
        std::vector<PetscInt> v_current_group;                //temporary vector
        std::vector<std::pair<int, double>> v_opposite_class;
        // - - - - create vector for opposite parts - - - - -
        for(j=0; j< ncols; j++){
            if(cols[j] != i){                //ignore the loop to itself
                if(cols[j] > num_part_p){
                    v_opposite_class.push_back( std::pair<int, double>(cols[j] - num_part_p, vals[j]) );
                }
            }
        }
        // - - - - sort both maps - - - - -
        sort(v_opposite_class.begin(), v_opposite_class.end(), [](std::pair<int, double> x, std::pair<int, double> y) {return x.second < y.second;});
        // - - - - create groups based on parameters - - - - -
        for(int k = 0; k < Config_params::getInstance()->get_rf_num_opposite_partitions(); k++ ){
            v_current_group.push_back(v_opposite_class[k].first);
        }
        vv_groups.push_back(v_current_group);
        MatRestoreRow(m_dist,i,&ncols, &cols, &vals);
    }
    // - - - - - - create vector of close parts for class N - - - - - -
    for(i=num_part_p; i< num_part_n + num_part_p; i++ ){
        MatGetRow(m_dist,i,&ncols, &cols, &vals);    //N row
        std::vector<PetscInt> v_current_group;
        std::vector<std::pair<int, double>> v_opposite_class;
        // - - - - create vector for opposite parts - - - - -
        for(j=0; j< ncols; j++){
            if(cols[j] != i){                //ignore the loop to itself
                if(cols[j] < num_part_p){
                    v_opposite_class.push_back( std::pair<int, double>(cols[j], vals[j]) );   //left bottom part of m_dist matrix(N->P)
                }
            }
        }
        // - - - - sort both maps - - - - -
        sort(v_opposite_class.begin(), v_opposite_class.end(), [](std::pair<int, double> x, std::pair<int, double> y) {return x.second < y.second;});
        // - - - - create groups based on parameters - - - - -
        for(int k = 0; k < Config_params::getInstance()->get_rf_num_opposite_partitions(); k++ ){
            // Prevent adding duplicate groups             // Note: this works only with 1 opposite part due to zero
            if(  (i - num_part_p) != vv_groups[v_opposite_class[k].first][0] ){
                v_current_group.push_back(v_opposite_class[k].first);
            }
        }
        vv_groups.push_back(v_current_group);
        MatRestoreRow(m_dist,i,&ncols, &cols, &vals);
    }


#if dbl_MR_find_groups >= 5     // default 5
    for(int l=0; l < vv_groups.size(); l++){
        if(l<num_part_p){
            std::cout<< "P vv_groups "<< l <<" opposite:";
            for(int m=0; m< vv_groups[l].size(); m++){
                std::cout<< vv_groups[l][m] <<",";
            }
            std::cout <<std::endl;
        }else{
            std::cout<< "N vv_groups "<< l - num_part_p <<" opposite:";
            for(int m=0; m< vv_groups[l].size(); m++){
                std::cout<< vv_groups[l][m] <<",";
            }
            std::cout <<std::endl;
        }
    }
#endif
}


/*
 * Calculates the Euclidean distance between all minority class(P) and majority class(N) and themselves
 * @input: centers for both classes
 * @output: m_dist is a square matrix which first #p rows and #p columns and then #n rows and #n columns
 *       - P - - - - N -
 *     P| P        P->N |
 *     N| N-> P     N   |
 *       - - - - - - - -
 */
void MainRecursion::calc_distances(int num_part_p, int num_part_n, Mat& m_centers_p,Mat& m_centers_n, Mat& m_dist){
    MatCreateSeqDense(PETSC_COMM_SELF,num_part_p+num_part_n,num_part_p+num_part_n,NULL, &m_dist);
    PetscInt            i, j, ncols_p, ncols_n;
    const PetscInt      *cols_p, *cols_n;
    const PetscScalar   *vals_p, *vals_n;
    CommonFuncs cf;
#if dbl_MR_calc_distances >=7    //default is 7
    std::cout  << "[MR][calc_distances] m_centers_p matrix:\n";                       //$$debug
    MatView(m_centers_p,PETSC_VIEWER_STDOUT_WORLD);
    std::cout  << "[MR][calc_distances] m_centers_n matrix:\n";                       //$$debug
    MatView(m_centers_n,PETSC_VIEWER_STDOUT_WORLD);
#endif
    // - - - - - - calculate the distance between different classes (P->N, N->P)
    for(i=0; i< num_part_p; i++ ){
        MatGetRow(m_centers_p,i,&ncols_p, &cols_p, &vals_p);    //P row
        for(j=0; j< num_part_n; j++){
            MatGetRow(m_centers_n,j,&ncols_n, &cols_n, &vals_n);    //N row

            MatSetValue(m_dist,i, num_part_p + j, cf.calc_euclidean_dist(ncols_p,ncols_n,cols_p,cols_n,vals_p,vals_n), INSERT_VALUES);
            MatRestoreRow(m_centers_n,i,&ncols_n, &cols_n, &vals_n);
        }
        MatRestoreRow(m_centers_p,i,&ncols_p, &cols_p, &vals_p);
    }
    // - - - - - - calculate the distance between class (P)
    for(i=0; i< num_part_p; i++ ){
        MatGetRow(m_centers_p,i,&ncols_p, &cols_p, &vals_p);    //P row
        for(j=i+1; j< num_part_p; j++){
            MatGetRow(m_centers_p,j,&ncols_n, &cols_n, &vals_n);    //P row
//            std::cout  << "[MR][calc_distances] P only i:" << i << " j:" << j << "\n";
            MatSetValue(m_dist,i, j, cf.calc_euclidean_dist(ncols_p,ncols_n,cols_p,cols_n,vals_p,vals_n), INSERT_VALUES);
            MatRestoreRow(m_centers_p,i,&ncols_n, &cols_n, &vals_n);
        }
        MatRestoreRow(m_centers_p,i,&ncols_p, &cols_p, &vals_p);
    }
    // - - - - - - calculate the distance between class (N)
    for(i=0; i< num_part_n; i++ ){
        MatGetRow(m_centers_n,i,&ncols_p, &cols_p, &vals_p);    //N row
        for(j=i+1; j< num_part_n; j++){
            MatGetRow(m_centers_n,j,&ncols_n, &cols_n, &vals_n);    //N row
//            std::cout  << "[MR][calc_distances] N only i:" << num_part_p + i << " j:" << num_part_p + j << "\n";
            MatSetValue(m_dist,num_part_p + i,num_part_p+ j, cf.calc_euclidean_dist(ncols_p,ncols_n,cols_p,cols_n,vals_p,vals_n), INSERT_VALUES);
            MatRestoreRow(m_centers_n,i,&ncols_n, &cols_n, &vals_n);
        }
        MatRestoreRow(m_centers_n,i,&ncols_p, &cols_p, &vals_p);
    }
    MatAssemblyBegin(m_dist, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_dist, MAT_FINAL_ASSEMBLY);
    Mat m_dist_t;
    MatTranspose(m_dist,MAT_INITIAL_MATRIX,&m_dist_t);
    MatAYPX(m_dist,1,m_dist_t,DIFFERENT_NONZERO_PATTERN);
#if dbl_MR_calc_distances >=7    //default is 7
    std::cout  << "[MR][calc_distances] m_dist matrix:\n";                       //$$debug
    MatView(m_dist,PETSC_VIEWER_STDOUT_WORLD);
//    cf.exp_matrix(m_centers_p,"./debug","m_centers_p_may17.dat","[MR][CD]");
//    cf.exp_matrix(m_centers_n,"./debug","m_centers_n_may17.dat","[MR][CD]");
//    cf.exp_matrix(m_dist,"./debug","m_dist_may17.dat","[MR][CD]");
#endif
}




/*
 *  - - - - - - calculate the centroid of each partition - - - - - -
 * @method: for now I calculate the centroid based on Euclidean distance
 * @input: neigh_Data, neigh_vol, and the partitions
 * @output: matrix with parts number of rows and for columns the same number of features as WA
 */
void MainRecursion::calc_center(Mat& m_neigh_Data, Vec& v_vol, std::vector<std::vector<PetscInt>>& vv_parts, Mat& m_parts, Mat& m_centers){
    //---- Normalize the Volumes #1 -----   @@
    Vec v_vol_normal;
    Coarsening co;
    co.normalize_vector(v_vol,v_vol_normal);

//    //---- get sum of normalized volumes #2 -----
//    PetscScalar sum_normalized_vol;
//    VecSum(v_vol_normal, &sum_normalized_vol);

    //---- create diagonal matrix from normalized volume #3 -----
    PetscInt num_point, num_features;
    MatGetSize(m_neigh_Data, &num_point, &num_features);
    Mat m_VnormDiag;
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_point,num_point,1,PETSC_NULL, &m_VnormDiag);
    MatDiagonalSet(m_VnormDiag,v_vol_normal,INSERT_VALUES);


    //---- Data_Vol = VD * Data #4 -----
    Mat m_Data_Vol;
    MatMatMult(m_VnormDiag, m_neigh_Data, MAT_INITIAL_MATRIX,PETSC_DEFAULT,&m_Data_Vol);
    MatDestroy(&m_VnormDiag);

    //---- Sigma Columns = Parts * Data_Vol #5 -----
    Mat m_Sum_Col_Part;     // (i,j) some of all values in column j for partition i
    MatMatMult(m_parts, m_Data_Vol, MAT_INITIAL_MATRIX,PETSC_DEFAULT,&m_Sum_Col_Part);
    MatDestroy(&m_Data_Vol);
#if dbl_MR_calc_center >=7    //default is 7
    std::cout  << "[MR][calc_center] m_Sum_Col_Part matrix:\n";                       //$$debug
    MatView(m_Sum_Col_Part,PETSC_VIEWER_STDOUT_WORLD);
#endif
//    CommonFuncs cf;
//    cf.exp_matrix(m_Sum_Col_Part,"./debug","m_Sum_Col_Part.dat");


    //---- calculate the sum of normalized volume for  each partition #6 -----
    PetscInt num_parts, i;
    unsigned int j;
    PetscScalar sum_curr =0 ;
    MatGetSize(m_parts, &num_parts,NULL);
//    printf("[MR][calc_center] num_parts:%d\n",num_parts);
    std::vector<PetscScalar> v_sum_norm_vol(num_parts);

    PetscScalar * arr_norm_vol;
    PetscMalloc1(num_point, &arr_norm_vol);
    VecGetArray(v_vol_normal,&arr_norm_vol);

    for(i=0; i <num_parts; i++){
        sum_curr = 0;                       //reset it for current partition
        for(j=0; j < vv_parts[i].size(); j++){
            sum_curr += arr_norm_vol [vv_parts[i][j] ];     // vv_parts[i][j] is the index of j-th point in partition i, and we get the volume from array of v_vol_norm
        }
        v_sum_norm_vol[i] = sum_curr;       //save the sum in the end
//        printf("[MR][calc_center] v_sum_norm_vol[%d] is:%g \n",i, v_sum_norm_vol[i]);
    }
    VecRestoreArray(v_vol_normal,&arr_norm_vol);
    PetscFree(arr_norm_vol);
    VecDestroy(&v_vol_normal);      // free vol_normal but don't touch v_vol (important)
    //---- divide each row to Sum_normalized_vol related to that row (partition) #7 -----
    // get sum of each column, divide it to sum of normalized volumes
    PetscInt            ncols;
    const PetscInt      *cols;
    const PetscScalar   *vals;
    //prepare the output
    MatCreateSeqAIJ(PETSC_COMM_SELF, num_parts, num_features , num_features,PETSC_NULL, &m_centers);


    for(i=0; i < num_parts; i++) {
        MatGetRow(m_Sum_Col_Part,i,&ncols,&cols,&vals);
        for(j=0; j < ncols; j++){
            // set the ( vals[j] /v_sum_norm_vol[i] ) into the center matrix
            if(v_sum_norm_vol[i] != 0){ //skip the partitions with zero volume (empty partitions)
                MatSetValue(m_centers, i, cols[j], vals[j] /v_sum_norm_vol[i] ,INSERT_VALUES);
            }
        }
        MatRestoreRow(m_Sum_Col_Part,i,&ncols,&cols,&vals);
    }
    MatDestroy(&m_Sum_Col_Part);
    MatAssemblyBegin(m_centers,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_centers,MAT_FINAL_ASSEMBLY);
#if dbl_MR_calc_center >=7    //default is 7
    std::cout  << "[MR][calc_center] m_centers matrix:\n";                       //$$debug
    MatView(m_centers,PETSC_VIEWER_STDOUT_WORLD);
#endif
}



/*
 * collect the numbers from input data (not all of them, just neibours of SV's)
 * initialize the parameters for meits
 * run metis
 * make a vector of vector (num_partition * size_each_part)
 * Filter empty partitions
 * output:
 *          vv_parts
 *          m_parts: row i represent part i and the nnz in row i are the points which are belong to part i
 */
void MainRecursion::get_parts(Mat& m_WA, Vec& v_vol, std::vector<std::vector<PetscInt>>& vv_parts, Mat& m_parts){
    // - - - - - - define metis parameters - - - - - -
    PetscInt num_vertex, num_edge;
    MatGetSize(m_WA, &num_vertex,NULL);
    MatInfo info;
    int num_partitions = vv_parts.size();

    /// * * * For parallelization consider other options instead of MAT_GLOBAL_MAX * * *
    /// Also notice this is only user for preallocation, and I calculate the number of edges
    /// manualy later
    MatGetInfo(m_WA, MAT_GLOBAL_MAX, &info);
    num_edge = info.nz_used;

    idx_t nvtxs= num_vertex, ncon=1;
    idx_t *xadj=NULL, *vsize=NULL, *adjncy=NULL;
    idx_t *vwgt=NULL, *adjwgt=NULL;
    idx_t nparts = num_partitions;

#if dbl_MR_get_parts >= 1
    std::cout << "[MR][get_parts] number of parts are: " << nparts << std::endl;
#endif
    real_t *tpwgts = NULL, *ubvec= NULL;
    idx_t *options=NULL;
//    idx_t options[METIS_NOPTIONS];
//    METIS_SetDefaultOptions(options);
//    options[METIS_OPTION_DBGLVL] = 1;

    idx_t objval;
    idx_t part[num_vertex];         //this is a vector of part as a result of metis

    xadj = new idx_t[num_vertex + 1 ];      // n + 1 page 23 manual (n is number of vertices)
    adjncy = new idx_t[2 * num_edge];       // 2m page 23 manual (m is number of edges)
    adjwgt= new idx_t[2 * num_edge];        // 2m page 24 manual (m is number of edges)
    vwgt= new idx_t[num_vertex];            // n page 24 manual (n is number of vertices)

#if dbl_MR_get_parts >=5    //default is 5
    // Number of edges are less than below because I remove the loop to itself, but the below value is useful for malloc
    PetscPrintf(PETSC_COMM_WORLD,"[MR][get_parts] num_vertex:%d, nnz in WA num_edge:%d\n",num_vertex, num_edge);
#endif
#if dbl_MR_get_parts >=7    //default is 7
    std::cout  << "[MR][get_parts] m_neigh_WA matrix:\n";                       //$$debug
    MatView(m_WA,PETSC_VIEWER_STDOUT_WORLD);
#endif

    // - - - - - - Set parameters - - - - - -
    PetscInt            i, j, ncols;
    PetscInt            curr_adj_ind_ =0;       // keep track of edge indices in adjncy array
    const PetscInt      *cols;
    const PetscScalar   *vals;

    PetscScalar         *arr_Vol_;
    PetscMalloc1(num_vertex, &arr_Vol_);
    VecGetArray(v_vol,&arr_Vol_);

//    bool weight_inveresed = Config_params::getInstance()->get_inverse_weight() ;
    bool weight_inveresed = 1;
    for(i=0; i< num_vertex; i++){
        MatGetRow(m_WA,i,&ncols,&cols,&vals);
        xadj[i] = curr_adj_ind_;       // starts with zero
        vwgt[i] = arr_Vol_[i];
#if dbl_MR_get_parts >=9    //default is 9
        PetscPrintf(PETSC_COMM_WORLD,"xadj[%d]=%d,vwgt[i]=%d",i,xadj[i],vwgt[i]);     //$$debug
#endif
        for(j=0; j < ncols ; j++){
            if(i != cols[j] ){   // skip the loop to iteself (which cause metis to crash)
                adjncy[curr_adj_ind_] = cols[j];
                if (weight_inveresed ){
                    adjwgt[curr_adj_ind_] = round ( (1.0 / (vals[j] + 0.0001)) * 10000 ) ;
                }else{
                    adjwgt[curr_adj_ind_] = vals[j];
                }
#if dbl_MR_get_parts >=9    //default is 9
    //notice that the adjwgt is integer and George told me it is the only way with metis
//    PetscPrintf(PETSC_COMM_WORLD,"{adjncy[%d]=%d, adjwgt[%d]=%d}, ",curr_adj_ind_,cols[j], curr_adj_ind_,adjwgt[curr_adj_ind_] );
    PetscPrintf(PETSC_COMM_WORLD,"curr_adj_ind_=%d, vals[j]=%g, adjwgt[curr_adj_ind_]=%d\n",curr_adj_ind_,vals[j], adjwgt[curr_adj_ind_] );
#endif
                curr_adj_ind_++;
            }
        }
        MatRestoreRow(m_WA,i,&ncols,&cols,&vals);
    }
    xadj[num_vertex]=curr_adj_ind_;       // edge critic (last value)
#if dbl_MR_get_parts >=9    //default is 9
    PetscPrintf(PETSC_COMM_WORLD,"\nlast xadj[%d]=%d\n",num_vertex,curr_adj_ind_); //$$debug    // the last one shows the end of array and it is N+1
#endif

    VecRestoreArray(v_vol,&arr_Vol_);
    PetscFree(arr_Vol_);

    PetscPrintf(PETSC_COMM_WORLD,"[MR][get_parts] before running Metis\n");       //$$debug
    // - - - - - - Run Metis - - - - - -
    int result;
    try{
//        result = METIS_PartGraphRecursive( &nvtxs, &ncon, xadj, adjncy, vwgt,
//                                       vsize, adjwgt, &nparts, tpwgts,
//                                       ubvec, options, &objval, part);        // This one is not working, for the second class

        result = METIS_PartGraphKway( &nvtxs, &ncon, xadj, adjncy, vwgt,
                                       vsize, adjwgt, &nparts, tpwgts,
                                       ubvec, options, &objval, part);
    }
    catch(...){
        PetscPrintf(PETSC_COMM_WORLD,"[MR][get_parts] Metis faced a problem \n");
    }

#if dbl_MR_get_parts >= 3    //default is 3
    std::cout << "[MR][get_parts] Metis result: " << result << std::endl;
    #if dbl_MR_get_parts >= 7    //default is 7
    PetscPrintf(PETSC_COMM_WORLD,"\n[MR][get_parts] [dev-test] [part index, selected partition]\n");
    for(i=0; i<num_vertex ;i++){
        PetscPrintf(PETSC_COMM_WORLD,"[%d:%d], ",i, part[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD,"\n\n");
    #endif
#endif


    // - - - - These check expected conditions - - - -
    if(curr_adj_ind_ > num_edge) {
        PetscPrintf(PETSC_COMM_WORLD,"[MR][get_parts] curr_adj_ind_:%d is larger than number of non zeros in meigh_WA:%d which is wrong\n",curr_adj_ind_, num_edge);
        exit(1);
    }
    if(result != 1){    // 1 means ok (I email for confirmation)
        PetscPrintf(PETSC_COMM_WORLD,"[MR][get_parts] Metis failed, return value is:%d!\n",result);
        exit(1);
    }

    //prepare the output
    MatCreateSeqAIJ(PETSC_COMM_SELF, num_partitions, num_vertex , num_vertex,PETSC_NULL, &m_parts);
    // part[i] value determine the partition, and i is the index of the point in data matrix
    for(i=0; i<num_vertex ;i++){
        vv_parts[part[i]].push_back(i);
        MatSetValue(m_parts, part[i], i, 1,INSERT_VALUES);
    }
    MatAssemblyBegin(m_parts,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_parts,MAT_FINAL_ASSEMBLY);

    // - - - - Filter empty partitions- - - -
    for(i=0; i< num_partitions; i++){
        if(vv_parts[i].size() == 0){
            printf("[MR][get_parts] part %d is empty\n",i);
            exit(1);
        }
    }

#if dbl_MR_get_parts >=7    //default is 7
    PetscPrintf(PETSC_COMM_WORLD,"[MR][get_parts] m_parts matrix:\n");                       //$$debug
    MatView(m_parts,PETSC_VIEWER_STDOUT_WORLD);

    PetscPrintf(PETSC_COMM_WORLD,"[MR][get_parts] vv_parts are:\n");
    int curr_idx_size;
    for(i=0; i < num_partitions; i++){
        curr_idx_size = vv_parts[i].size();
        PetscPrintf(PETSC_COMM_WORLD,"[MR][get_parts] Result Parts of group:%d, size:%d\n",i, curr_idx_size) ;
        for(int j = 0; j < curr_idx_size ; j++){
            PetscPrintf(PETSC_COMM_WORLD,"%d,",vv_parts[i][j]) ;
        }
        PetscPrintf(PETSC_COMM_WORLD,"\n");
    }
#endif
    // free metis parameters
    delete xadj;
    delete adjncy;
    delete adjwgt;
    delete vwgt;

}



/*
 * output:
 *          m_parst: a matrix with 1 row with number of points as features, all of them are 1
 */
void MainRecursion::single_part_matrix(Mat& m_neigh_Data, Mat& m_parts){
    PetscInt num_row;
    MatGetSize(m_neigh_Data, &num_row, NULL);
    MatCreateSeqAIJ(PETSC_COMM_SELF, 1, num_row , num_row,PETSC_NULL, &m_parts);
    for(int i=0; i < num_row; i++){
        MatSetValue(m_parts, 0, i, 1, INSERT_VALUES);
    }
    MatAssemblyBegin(m_parts,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_parts,MAT_FINAL_ASSEMBLY);
#if dbl_MR_single_part_matrix >=7    //default is 7
    PetscPrintf(PETSC_COMM_WORLD,"[MR][SPM] m_parts matrix:\n");                       //$$debug
    MatView(m_parts,PETSC_VIEWER_STDOUT_WORLD);
#endif
    //@@
}


/*
 * output:
 *          m_centers: calculated 1 center with respect to volumes of points
 */
void MainRecursion::calc_single_center(Mat& m_neigh_Data, Vec& v_vol, Mat& m_center){
    //---- Normalize the Volumes #1 -----
    Vec v_vol_normal;
    Coarsening co;
    co.normalize_vector(v_vol,v_vol_normal);

    //---- get sum of normalized volumes #2 -----
    PetscScalar sum_normalized_vol;
    VecSum(v_vol_normal, &sum_normalized_vol);

    //---- create diagonal matrix from normalized volume #3 -----
    PetscInt num_point, num_features;
    MatGetSize(m_neigh_Data, &num_point, &num_features);
    Mat m_VnormDiag;
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_point,num_point,1,PETSC_NULL, &m_VnormDiag);
    MatDiagonalSet(m_VnormDiag,v_vol_normal,INSERT_VALUES);


    //---- Data_Vol = VD * Data #4 -----
    Mat m_Data_Vol;
    MatMatMult(m_VnormDiag, m_neigh_Data, MAT_INITIAL_MATRIX,PETSC_DEFAULT,&m_Data_Vol);
    MatDestroy(&m_VnormDiag);

    //---- Transpose Data_Vol #5 -----
    // I continue to comment on columns which is the real logic, but in code I use row because of CSR matrix performance
    Mat m_Data_Vol_trans;
    MatTranspose(m_Data_Vol,MAT_INITIAL_MATRIX,&m_Data_Vol_trans);
    MatDestroy(&m_Data_Vol);

    //---- Calc Sum of each column of Data_Vol (each row of Data_vol_trans) #6 -----
    Vec v_sum_DV_col;
    VecCreateSeq(PETSC_COMM_SELF,num_features,&v_sum_DV_col);
    MatGetRowSum(m_Data_Vol_trans, v_sum_DV_col);
    MatDestroy(&m_Data_Vol_trans);

    //---- Divide sum of each column of Data_Vol to sum_normalized_vol #6 -----
    PetscScalar * arr_sum_DV_col;
    PetscMalloc1(num_features, &arr_sum_DV_col);
    VecGetArray(v_sum_DV_col, &arr_sum_DV_col);
    MatCreateSeqAIJ(PETSC_COMM_SELF, 1, num_features , num_features,PETSC_NULL, &m_center);
//    std::cout << "[MR][CSC] sum_normalized_vol: "<< sum_normalized_vol << std::endl;
    for(int i=0; i<num_features; i++){
//        std::cout << "[MR][CSC] arr_sum_DV_col[i]: " << arr_sum_DV_col[i] << ", calc: " <<
//                     arr_sum_DV_col[i]/sum_normalized_vol << " at i: " << i << std::endl;
        MatSetValue(m_center, 0, i, arr_sum_DV_col[i]/sum_normalized_vol, INSERT_VALUES );
    }
    VecRestoreArray(v_sum_DV_col, &arr_sum_DV_col);
    PetscFree(arr_sum_DV_col);
    MatAssemblyBegin(m_center, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_center, MAT_FINAL_ASSEMBLY);
#if dbl_MR_CSC >=7
    printf("[MR][CSC] Center matrix:\n");                                     //$$debug
    MatView(m_center,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif
}



/*
 * @input:
 *      cc_name: class name used for logging information
 * @output:
 *      m_neighbors is the matrix of neighbor points
 *      neigh_id is IS type which contain the indices for neighbor points
 */

//void MainRecursion::find_SV_neighbors(Mat& m_data, Mat& m_P, std::vector<int>& seeds_ind,
//                                      Mat& m_SV, Mat& m_neighbors, std::string cc_name,
//                                      IS& IS_neigh_id){
void MainRecursion::find_SV_neighbors(Mat& m_data, Mat& m_P, std::vector<int>& seeds_ind,
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
#if dbl_MR_fSN >=5
    std::cout << "[MR][fSN]{" << cc_name << "} m_data num row as num_row_fine_points:"<< num_row_fine_points <<std::endl;
#endif
    /// - - - - - - - - - Create P Transpose matrix - - - - - - - -
    // P' : find fine points in rows instead of columns due to performance issues with Aij matrix
    num_seeds = (int) seeds_ind.size();
    std::cout  << "[MR][fSN]{" << cc_name << "} initialize num_seeds:" << num_seeds << "\n";
    PetscMalloc1(num_row_fine_points,&ind_);
    Mat m_Pt_;
    MatTranspose(m_P,MAT_INITIAL_MATRIX,&m_Pt_);
    MatDestroy(&m_P);                           //to make sure I don't use it by mistake

    PetscInt num_row_m_Pt_, num_col_m_Pt_;
    MatGetSize(m_Pt_,&num_row_m_Pt_,&num_col_m_Pt_);
#if dbl_MR_fSN >=5
    std::cout << "[MR][fSN]{" << cc_name << "} P transpose dim ["<< num_row_m_Pt_ <<","<< num_col_m_Pt_ << "]" <<std::endl;
    std::cout << "[MR][fSN]{" << cc_name << "} m_data num rows:"<< num_row_fine_points << std::endl;
    #if dbl_MR_fSN >=7            //should be above 7
        std::cout << "[MR][fSN]{" << cc_name << "} list of all SVs are:\n";
        if(cc_name == "Majority"){
            std::cout << "[MR][fSN]{" << cc_name << "} [HINT]for no fake point, they should start from zero, not couple hundreds:\n";
        }
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

#if dbl_MR_fSN >=1
        if(ncols == 0){
            std::cout  << "[MR][fSN]{" << cc_name << "} empty row in P' at row i:"<< i
                       << " seeds_ind[i]:" << seeds_ind[i] << " ncols:" << ncols << std::endl;
            exit(1);
        }
        #if dbl_MR_fSN >=3
            std::cout  << "[MR][fSN]{" << cc_name << "} MatGetRow of P' matrix in loop seeds_ind[i]:"
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

#if dbl_MR_fSN >=7
    printf("==== [MR][inside selecting agg]{after sort each row of P'} i:%d ====\n",i);
    for (auto it = v_agg_.begin(); it != v_agg_.end(); it++){
        printf("index:%d, value:%g\n", it->index, it-> value);
    }
//index is the column number and important part
//value is only used to find the important indices (selected indices)
#endif

            // - - - select fraction of participants - - -
            float add_frac_ = ceil(Config_params::getInstance()->get_rf_add_fraction() * ncols); // read add_fraction from parameters
            for (auto it = v_agg_.begin(); it != v_agg_.begin() + add_frac_ ; it++){
                v_fine_neigh_id[it->index] =1 ;
            }

            v_agg_.clear();
        } // end of else for multiple participants in this aggregate

        MatRestoreRow(m_Pt_,seeds_ind[i],&ncols, &cols, &vals);
    }
    MatDestroy(&m_Pt_);
#if dbl_MR_fSN >=9
    std::cout<<"[MR][find_SV_neighbors] num_seeds:"<<num_seeds<<std::endl;
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
        if(Config_params::getInstance()->get_rf_add_distant_point_status()){
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
#if dbl_MR_fSN >=1      // this should be 3

    if(Config_params::getInstance()->get_rf_add_distant_point_status()){
        std::cout  << "[MR][FSN]{" << cc_name << "} num of points participated in SV aggregates are: "<< cnt_total - cnt_agg_part_distant_neighbor  << std::endl;
        std::cout  << "[MR][FSN]{" << cc_name << "} num of distant 1 neighbor of above points are::"<< cnt_agg_part_distant_neighbor << std::endl;
    }else{
        std::cout  << "[MR][FSN]{" << cc_name << "} num of points participated in SV aggregates are: "<< cnt_total <<
                      " and distant neighbors are ignored!" << std::endl;
    }

#endif




    //create the IS (Index Set)
    ISCreateGeneral(PETSC_COMM_SELF,cnt_total,ind_,PETSC_COPY_VALUES,&IS_neigh_id);
    PetscFree(ind_);      //free the indices as I have created the IS

#if dbl_MR_fSN >=7          //default is 7
    printf("[MR] IS is created \n");               //$$debug
    ISView(IS_neigh_id,PETSC_VIEWER_STDOUT_WORLD);
//        MatGetSize(m_data,&num_row_fine_points,NULL);
//        printf("[MR] m_data num rows: %d\n",num_row_fine_points);
#endif


    MatGetSubMatrix(m_data,IS_neigh_id, NULL,MAT_INITIAL_MATRIX,&m_neighbors);


#if dbl_MR_fSN >=3
    PetscInt m_neighbors_num_row =0, m_neighbors_num_col;
    MatGetSize(m_neighbors ,&m_neighbors_num_row,&m_neighbors_num_col);

    std::cout  << "[MR][fSN]{" << cc_name
                  << "} new sub matrix dimension #row:" << m_neighbors_num_row
                  << ",#col:" <<m_neighbors_num_col << std::endl;
#endif
#if dbl_MR_fSN >=7      //default is 7
    std::cout  << "[MR][fSN]{" << cc_name << "} m_neighbors matrix:\n";                       //$$debug
    MatView(m_neighbors,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif
//    ISDestroy(&IS_neigh_id);  // Don't destroy it. It is required later in the partitioning
}

