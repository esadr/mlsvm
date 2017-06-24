#include "personalized.h"
#include "etimer.h"
#include "solver.h"
#include <algorithm>
#include "common_funcs.h"


void personalized::uniform_design_index_base(int id, Mat& m_data, std::vector<int>& v_lbl, bool inh_params, double last_c,
                        double last_gamma, const PetscScalar * arr_train_index, PetscInt num_nnz, std::vector<int>& v_predict_lbl){
    ETimer t_sv_ps;

//    double train_fraction = 1 - (1 / Config_params::getInstance()->get_main_num_kf_iter());
    double train_fraction = 0.8;

    PetscInt iter_train_end = ceil(num_nnz * train_fraction);    //number of poitns in training data

    unsigned int num_iter_st1 = Config_params::getInstance()->get_ms_first_stage();
    unsigned int num_iter_st2 = Config_params::getInstance()->get_ms_second_stage();
    std::vector<Solver> v_solver;
    v_solver.reserve(num_iter_st1 + num_iter_st2);
    unsigned int solver_id=0;

    int stage = 1;
    std::vector<ud_point> ud_params_st_1;
    summary current_summary;
    std::vector<summary> v_summary;
    ud_params_st_1 = ud_param_generator(1, inh_params, last_c, last_gamma);
    // - - - - 1st stage - - - -
    for(unsigned int i =0; i < num_iter_st1;i++){
        Solver sv;
        sv.PD_train_model_index_base(m_data,v_lbl,arr_train_index, iter_train_end,true, ud_params_st_1[i].C, ud_params_st_1[i].G);
        v_solver.push_back(sv);
        sv.PD_test_predict_index_base(m_data, v_lbl, arr_train_index, iter_train_end, num_nnz, current_summary,solver_id);
        v_summary.push_back(current_summary);
        ++solver_id;
    }
    int best_1st_stage = select_best_model(v_summary,1,1);

    // - - - - 2nd stage - - - -
    stage = 2 ;
    std::vector<ud_point> ud_params_st_2;
    ud_params_st_2 = ud_param_generator(2,true, ud_params_st_1[best_1st_stage].C , ud_params_st_1[best_1st_stage].G);

    for(unsigned int i =0; i < num_iter_st2; i++){
        //skip the center of second stage(duplicate)
        if(ud_params_st_2[i].C == ud_params_st_1[best_1st_stage].C && ud_params_st_2[i].G == ud_params_st_1[best_1st_stage].G)
            continue;

        Solver sv;
        sv.PD_train_model_index_base(m_data, v_lbl, arr_train_index, iter_train_end,true, ud_params_st_1[i].C, ud_params_st_1[i].G);
        v_solver.push_back(sv);
        sv.PD_test_predict_index_base(m_data, v_lbl, arr_train_index, iter_train_end, num_nnz, current_summary,solver_id);
        v_summary.push_back(current_summary);
        ++solver_id;
    }

    int best_of_all =  select_best_model(v_summary,1,2);
//    t_sv_ps.stop_timer("[PD][UDIB] model training");

    // - - - - - - - - prepare the solution for refinement - - - - - - - - -
    Solver best_sv = v_solver[best_of_all];
    svm_parameter svm_params = best_sv.get_model_params();
//    std::cout << "[PD][UDIB] id:" << id << " C:" << svm_params.C << " G:" << svm_params.gamma << std::endl;

    Solver sv ;
    sv.PD_train_model_index_base(m_data, v_lbl, arr_train_index, num_nnz,true, svm_params.C, svm_params.gamma); //train using whole data
    v_predict_lbl[id] = sv.PD_predict_a_label(m_data, id);
//    svm_model * check_model = sv.get_model();
//    svm_parameter check_svm_params = sv.get_model_params();
//    svm_save_model("./debug/dap_d.model", sv.get_model());

    sv.free_solver("[PD][UIDB]");
//    v_predict_lbl[id] = best_sv.PD_predict_a_label(m_data, id);
    std::cout << "[PD][UDIB] id:" << id << " target:" << v_lbl[id] << " predict:" << v_predict_lbl[id] << std::endl;
    for(auto it=v_solver.begin(); it!= v_solver.end(); ++it){
        it->free_solver("[PD][UDIB] ");   //free all solvers
    }
}









void personalized::Single_training_index_base(int id, Mat& m_data, std::vector<int>& v_lbl, bool inh_params, double last_c,
                        double last_gamma, const PetscScalar * arr_train_index, PetscInt num_nnz, std::vector<int>& v_predict_lbl){

    Solver sv ;
    double paramC = Config_params::getInstance()->get_svm_C();
    double paramG = Config_params::getInstance()->get_svm_gamma();
    std::cout << "[PD][STIB] id:" << id << " C:" << paramC << " G:" << paramG << std::endl;

//    //just to make sure the train for both class is ok
//    Mat m_p_data, m_n_data;
//    MatCreateSeqAIJ(PETSC_COMM_SELF,num_nnz,40,40, PETSC_NULL, &m_p_data);
//    MatCreateSeqAIJ(PETSC_COMM_SELF,num_nnz,40,40, PETSC_NULL, &m_n_data);
//    get_separate_matrices(m_data, v_lbl, arr_train_index,num_nnz,m_p_data, m_n_data);

//    CommonFuncs cf;         /// - - - - export information to debug the problem - - - -
//    std::string str_output_p_data_fname = "m_p_data_STIB.dat";
//    std::string str_output_n_data_fname = "m_n_data_STIB.dat";
//    cf.exp_matrix(m_p_data, "./data",str_output_p_data_fname ,"[PD]");
//    cf.exp_matrix(m_n_data, "./data",str_output_n_data_fname ,"[PD]");


    sv.PD_train_model_index_base(m_data, v_lbl, arr_train_index, num_nnz,true, paramC, paramG); //train using whole data
//    sv.stand_alone_train(m_p_data, m_n_data, "./debug/single_train_IB.model");
    v_predict_lbl[id] = sv.PD_predict_a_label(m_data, id);
    sv.free_solver("[PD][STDB]");
    std::cout << "[PD][STIB] id:" << id << " target:" << v_lbl[id] << " predict:" << v_predict_lbl[id] << std::endl;
}


void personalized::evaluate_results(const std::vector<int>& v_target_lbl, const std::vector<int>& v_predict_lbl, summary& result_summary ){
    double tp =0, tn =0, fp =0, fn=0;

    for(int i=0; i < v_target_lbl.size() ; ++i){
        if(v_target_lbl[i] == 1){   //positive class
            if (v_predict_lbl[i] == 1)     //correct
                tp++;
            else                                //predict negative
                fn++;                   //false
        }
        else{                    //negative class
            if (v_predict_lbl[i] == -1)    //correct
                tn++;
            else                                //predict positive
                fp++;                   //false
        }
    }

    result_summary.perf[Sens] = tp / (tp+fn) ;
    result_summary.perf[Spec] = tn / (tn+fp) ;
    result_summary.perf[Gmean] = sqrt(result_summary.perf[Sens] * result_summary.perf[Spec]);
    result_summary.perf[Acc] = (double)(tp+tn) /  (double)(v_target_lbl.size()) ;

    if(tp+fp == 0)              //prevent nan case
        result_summary.perf[PPV] = 0;
    else
        result_summary.perf[PPV] = tp/ (tp+fp);

    if(tn+fn == 0)              //prevent nan case
        result_summary.perf[NPV] = 0;
    else
        result_summary.perf[NPV] = tn/ (tn+fn);

    result_summary.perf[F1] = 2*tp / (2*tp+fp+fn);

    result_summary.perf[TP] = tp;
    result_summary.perf[FP] = fp;
    result_summary.perf[TN] = tn;
    result_summary.perf[FN] = fn;
    result_summary.C = 0;
    result_summary.gamma = 0;
    result_summary.num_SV_p = 0;
    result_summary.num_SV_n = 0;
    result_summary.iter = -1;

//#if dbl_SV_test_predict >= 1    // 1 default
    Config_params::getInstance()->print_summary(result_summary,"[PD][evaluate_results]");
//#endif
}



void personalized::calc_statistics(int id, Mat& m_stats, std::vector<int>& v_target_lbl,std::vector<int>& v_predict_lbl,
                                   const PetscScalar * arr_train_index, PetscInt num_nnz){
    PetscInt i, ncols, min=0, maj=0;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;

    for(i=0; i< num_nnz; ++i){
//        std::cout << "calc_statatstics inside loop i:" << i << " val" << v_target_lbl[arr_train_index[i]] << std::endl;
        if(v_target_lbl[arr_train_index[i]] == 1)
            min++;
        else
            maj++;
    }

    MatSetValue(m_stats, id, 0, min, INSERT_VALUES);
    MatSetValue(m_stats, id, 1, maj, INSERT_VALUES);
    MatSetValue(m_stats, id, 2, v_target_lbl[id], INSERT_VALUES);
    MatSetValue(m_stats, id, 3, v_predict_lbl[id], INSERT_VALUES);
//    std::cout << "calc_statatstics " << id << " " << min << " " << maj << " " << v_target_lbl[id] << " " << v_predict_lbl[id] << std::endl;
}




/*
 * Remove diagonal from the NN matrix which separate the test data (a test point at each row) from the training data
 */
void personalized::filter_test_point(Mat& m_NN, Mat& m_filtered_NN){

    PetscInt i, ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscInt num_col, num_row;

    MatGetSize(m_NN, &num_row, &num_col);   //set the number of columns
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row ,num_col,num_col,PETSC_NULL, &m_filtered_NN); //+1 is for label


    for(i =0; i < num_row ; i++){
        MatGetRow(m_NN,i,&ncols,&cols,&vals);
        for(int j=0; j < ncols ; j++){
            if(vals[j] != i)
                MatSetValue(m_filtered_NN, i, cols[j], vals[j],INSERT_VALUES);
//            else
//                std::cout << "problem in row: " << i << " ,j:"<<j << " ,cols[j] " << cols[j] << std::endl;
        }
        MatRestoreRow(m_NN,i,&ncols,&cols,&vals);
    }
    MatAssemblyBegin(m_filtered_NN, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_filtered_NN, MAT_FINAL_ASSEMBLY);
    MatDestroy(&m_NN);
}



void personalized::multi_level_solver(int id, Mat& m_data, Mat& m_NN_filtered, std::vector<int>& v_lbl,
                                      const PetscScalar * arr_train_index, PetscInt num_nnz, std::vector<int>& v_predict_lbl){
    PetscInt            num_row_, num_col_;
    MatGetSize(m_data, &num_row_, &num_col_);

    Mat m_p_data, m_n_data;
//    MatCreateSeqAIJ(PETSC_COMM_SELF,num_nnz,num_col_,num_col_, PETSC_NULL, &m_p_data);
//    MatCreateSeqAIJ(PETSC_COMM_SELF,num_nnz,num_col_,num_col_, PETSC_NULL, &m_n_data);

    Solver sv_separate_data;
    sv_separate_data.get_separate_matrices(m_data,v_lbl,arr_train_index,num_nnz,m_p_data, m_n_data );
    exit(1);
}
