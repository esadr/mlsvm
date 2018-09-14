#include "main_recursion.h"
#include <vector>
#include "config_logs.h"
#include "config_params.h"
#include "etimer.h"

#include <thread>
#include <cassert>

#define paramsInst Config_params::getInstance()
using std::cout; using std::endl;

solution MainRecursion::main(Mat& p_data, Mat& m_P_p_f, Mat& p_WA, Vec& p_vol,
                             Mat& n_data, Mat& m_P_n_f, Mat& n_WA, Vec& n_vol,
                             Mat& m_VD_p, Mat& m_VD_n, int level,
                             std::vector<ref_results>& v_ref_results){
    std::string funcIdx = "[MR][main] ";
    PetscInt check_num_row_VD;
    MatGetSize(m_VD_p, &check_num_row_VD, NULL );
    assert (check_num_row_VD && "empty validation data for minority class");

    MatGetSize(m_VD_n, &check_num_row_VD, NULL );
    assert (check_num_row_VD && "empty validation data for majority class");
    level++;
    paramsInst->set_main_current_level_id(level);
    int c_limit = paramsInst->get_coarse_threshold();
    //data_c is aggregated data (coarse data point)
    Mat p_data_c,p_WA_c, n_data_c, n_WA_c,m_P_p,m_P_n;
    Vec p_vol_c, n_vol_c;
    PetscInt p_num_row, n_num_row;
    std::vector<NodeId> v_p_seeds_indices, v_n_seeds_indices;

    MatGetSize(p_data, &p_num_row,0);    //m returns the number of rows globally
    MatGetSize(n_data, &n_num_row,0);    //m returns the number of rows globally

    /// ----------------- Both classes are small enough, -----------------
    ///            Now call the ModelSelection (Recursive Condition)
    //both classes are small enough (coarsest level)
    if ((p_num_row <  c_limit) && (n_num_row <  c_limit))  {
        printf("\n\\   /\\   /\\   / End of Coarsening \\   /\\   /\\   /\n");
        printf(  " \\_/  \\_/  \\_/       level:%d       \\_/  \\_/  \\_/ \n",
                 level);
        cout << funcIdx + "num points P:" << p_num_row
                << ", N:" << n_num_row << endl;

        paramsInst->set_timer_end_coarsening();
        // Set the number of models at each level to zero as default
        // The current level is used to set the vector size and initialize it
        paramsInst->set_levels_models_info();

        solution sol_coarsest;
        Refinement rf;
#if dbl_MR_main >= 3
        cout << funcIdx + "DEBUG level id before calling"
            << "rf.process_coarsest_level:"
            << std::to_string(paramsInst->get_main_current_level_id()) << endl;
#endif
        rf.process_coarsest_level(p_data, p_vol, n_data, n_vol, m_VD_p, m_VD_n, level ,sol_coarsest, v_ref_results);

#if dbl_MR_main >= 3
        cout << funcIdx + "DEBUG level id after calling rf.process_coarsest_level:" +
                     std::to_string(paramsInst->get_main_current_level_id()) << endl;
#endif
        // free resources
        MatDestroy(&p_data);
        MatDestroy(&n_data);
        MatDestroy(&p_WA);
        MatDestroy(&n_WA);

#if dbl_MR_main >= 3
        cout << funcIdx + "{coarsest} before returning the sol_coarsest C:"<< sol_coarsest.C <<
        ", nSV+:"<< sol_coarsest.p_index.size() << ", nSV-:"<< sol_coarsest.n_index.size() << endl;
#endif
        return sol_coarsest;                                // return the coarsest solution

    }else{      ///-------------- Coarsening -------------------
        ETimer t_coarse;
        if(level > paramsInst->get_cs_max_coarse_level())     {
            cout<< "!!! the coarsening is not converged after "
                   << level << " levels, skip this run !!!" << endl;
            solution empty_solution;
            empty_solution.C = -1;
            return empty_solution;
        }
        double filter_threshold = paramsInst->get_cs_weak_edges_ft();
        Coarsening p_coarser("Minority") ;
        Coarsening n_coarser("Majority");
        cs_info ref_info_p, ref_info_n;
#if dbl_MR_main >= 1
        printf("\n\\  /\\  /\\  /       Coarsening    \\  /\\  /\\  /\n");
        printf(  " \\/  \\/  \\/         level:%d       \\/  \\/  \\/ \n"
                 , level);
#endif
        if(p_num_row >= c_limit){
#if dbl_MR_main >= 1
            cout << funcIdx + "+ + + + + + + + Positive class + + + + + + + + "
                 << endl;
#endif
            m_P_p = p_coarser.calc_P(p_WA, p_vol, v_p_seeds_indices, ref_info_p); //m_P_p measn P matrix for positive label (minority class)
//            t_coarse.stop_timer(funcIdx + "{1} from start of both class calc_p minority, level:",std::to_string(level));

            p_data_c = p_coarser.calc_aggregate_data(m_P_p, p_data,p_vol, v_p_seeds_indices);
//            t_coarse.stop_timer(funcIdx + "{2} from start of both class calc Agg Data minority, level:",std::to_string(level));

            p_WA_c = p_coarser.calc_WA_c(m_P_p, p_WA);
//            t_coarse.stop_timer(funcIdx + "{2} from start of both class calc Agg Data minority, level:",std::to_string(level));

            p_coarser.calc_stat_nnz(p_WA,0, level, "minority");
//            p_coarser.calc_stat_nnz(p_WA_c);

            p_coarser.calc_real_weight(p_WA_c, p_data_c);       //recalculate the weights in adjacency matrix from the data
            p_coarser.filter_weak_edges(p_WA_c, filter_threshold, level);


            p_vol_c = p_coarser.calc_coarse_volumes(m_P_p, p_vol);
//            t_coarse.stop_timer(funcIdx + "{3} from start of both class calc Agg Vol minority, level:",std::to_string(level));
        }else{
            if(level == 1){ // if the minority class don't need any coarsening
                cout << "\n\nNo coarsening for minority class since its size is less than the threshold!" << endl;
                m_P_p = p_coarser.calc_P_without_shrinking(p_WA, p_vol, v_p_seeds_indices, ref_info_p);
            }else{
                // - - - - don't coarse the minority class anymore - - - -
                cout << funcIdx + "The minority class is not coarsen anymore" << endl;
                // duplicate the P matrix of finer level since I don't need to calculate it

                MatDuplicate(m_P_p_f, MAT_COPY_VALUES, &m_P_p);
                m_P_p = p_coarser.calc_P(p_WA, p_vol, v_p_seeds_indices, ref_info_p); //m_P_p measn P matrix for positive label (minority class)
            }

            MatDuplicate(p_WA, MAT_COPY_VALUES, &p_WA_c);
            MatDuplicate(p_data, MAT_COPY_VALUES, &p_data_c);
            VecDuplicate(p_vol,&p_vol_c);
            VecCopy(p_vol,p_vol_c);
        }
#if dbl_MR_main >= 1
        printf("\n[MR][main] - - - - - - - - Negative class - - - - - - - -\n");
#endif
        m_P_n = n_coarser.calc_P(n_WA, n_vol, v_n_seeds_indices, ref_info_n); // same for majority class
//        t_coarse.stop_timer(funcIdx + "{4} from start of both class calc_p majority",std::to_string(level));

        n_data_c = n_coarser.calc_aggregate_data(m_P_n, n_data, n_vol, v_n_seeds_indices);
//        t_coarse.stop_timer(funcIdx + "{5} from start of both class calc Agg Data majority",std::to_string(level));

        n_WA_c = n_coarser.calc_WA_c(m_P_n, n_WA);
//        t_coarse.stop_timer(funcIdx + "{6} from start of both class calc Agg Vol majority",std::to_string(level));

        p_coarser.calc_stat_nnz(n_WA,0, level, "majority");
//        p_coarser.calc_stat_nnz(n_WA_c);

        n_coarser.calc_real_weight(n_WA_c, n_data_c);       //recalculate the weights in adjacency matrix from the data
        n_coarser.filter_weak_edges(n_WA_c, filter_threshold,level);


        n_vol_c = n_coarser.calc_coarse_volumes(m_P_n, n_vol);
//        t_coarse.stop_timer(funcIdx + "{7} from start of both class calc Agg Vol majority",std::to_string(level));

        t_coarse.stop_timer("[MR] total coarsening for both class at level",std::to_string(level));


        solution sol_coarser ;
        sol_coarser = main(p_data_c, m_P_p, p_WA_c, p_vol_c, n_data_c, m_P_n, n_WA_c, n_vol_c, m_VD_p, m_VD_n, level, v_ref_results); // recursive call
        if(sol_coarser.C == -1 ){       // the coarsening didn't converge after maximum number of levels, so we skip this v-cycle completely
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
        //because it comes from coarser level
//        printf(funcIdx + "coarse solution from level:%d \n",level+1);
        printf("\n /\\  /\\  /\\        Refinement     /\\  /\\  /\\ \n");
        printf(  "/  \\/  \\/  \\        level:%d      /  \\/  \\/  \\ \n"
                 , level);

#if dbl_RF_INFO >= 1
        printf("                    Minority                        Majority              ");
        printf("\n         #points:%d, #edges:%d "
               , ref_info_p.num_point, ref_info_p.num_edge);
        printf("\t  #points:%d, #edges:%d"
               , ref_info_n.num_point, ref_info_n.num_edge);
        printf("\n=====================================================\n");
        printf(funcIdx + "coarse solution C:%g, G:%g, nSV+:%lu \n\n",
               sol_coarser.C, sol_coarser.gamma, sol_coarser.p_index.size());
#endif
        solution sol_refine;
        Refinement rf;

        sol_refine = rf.main(p_data, m_P_p, p_vol, p_WA
                             , n_data, m_P_n, n_vol, n_WA
                             , m_VD_p, m_VD_n
                             , sol_coarser, level
                             , v_ref_results);

        MatDestroy(&p_data); 
        MatDestroy(&n_data);
        MatDestroy(&p_data_c);
        MatDestroy(&n_data_c);
        MatDestroy(&p_WA_c);
        MatDestroy(&n_WA_c);
        VecDestroy(&p_vol);
        VecDestroy(&n_vol);

        t_refine.stop_timer(funcIdx +"Refinement at level:",
                            std::to_string(level));
        return sol_refine;
    }
}
