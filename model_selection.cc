#include "model_selection.h"
#include <algorithm>    // std::random_shuffle

#include "config_logs.h"
#include "loader.h"
#include <cmath>


struct svm_node *x;


void ModelSelection::cross_fold_data(const Mat data_p, const Mat data_n,
                                        Mat& train_p_, Mat& train_n_, Mat& test_total_){
//devide data to train and test randomely
    Mat             test_p_, test_n_;
    IS              isrow_train_p_, isrow_train_n_;
    IS              isrow_test_p_, isrow_test_n_;
    PetscInt        * ind_train_p_, * ind_train_n_;                 //arrays of Int that contains the row indices (Training)
    PetscInt        * ind_test_p_, * ind_test_n_;                 //arrays of Int that contains the row indices (Test)
    PetscInt        size_total_p_=0, size_total_n_=0;
    PetscInt        num_col_p_=0, num_col_n_=0, max_num_col=0;
    PetscInt        size_train_p_=0, size_train_n_=0;
    PetscInt        size_test_p_=0, size_test_n_=0;       //test = total - train

    MatGetSize(data_p,&size_total_p_,&num_col_p_);                  //get the size of data in positive (minority) class
    MatGetSize(data_n,&size_total_n_,&num_col_n_);

    // training part is rest of validation
    double train_part_percentage =  1 - Config_params::getInstance()->get_ms_validation_part();
    size_train_p_ = floor(size_total_p_ * train_part_percentage) ;
    size_train_n_ = floor(size_total_n_ * train_part_percentage) ;

    size_test_p_ = size_total_p_ - size_train_p_;
    size_test_n_ = size_total_n_ - size_train_n_;
#if dbl_MS_cross_fold >= 1
    printf("[MS][cross_fold]  tr_p :%d, tr_n :%d, number of columns:%d\n",size_train_p_,size_train_n_,num_col_p_);
#endif

    PetscMalloc1(size_train_p_, &ind_train_p_);
    PetscMalloc1(size_test_p_, &ind_test_p_);

    PetscMalloc1(size_train_n_, &ind_train_n_);
    PetscMalloc1(size_test_n_, &ind_test_n_);
// Random generator without duplicates
                                    //---- Positive class training ----
    std::vector<int> vec_p_indices;
    vec_p_indices.reserve(size_total_p_);
    for (int i_p=0; i_p<size_total_p_; ++i_p){       //create a vector of all possible nodes
        vec_p_indices.push_back(i_p);
    }

    srand(std::stoll(Config_params::getInstance()->get_cpp_srand_seed()));
    std::random_shuffle ( vec_p_indices.begin(), vec_p_indices.end() ); //shuffle all nodes

    for (int i =0;i != size_train_p_ ; i++){        // only pick the train size number of them
        ind_train_p_[i] = vec_p_indices[i];
    }

    // ind_train_p should sort
    std::sort(ind_train_p_,ind_train_p_ + size_train_p_);   //this is critical for MatGetSubMatrix method

                                    ///---- Positive class test ----
    for (int i =0;i != size_test_p_; i++){        // only pick the rest of nodes (after training to the end)
        ind_test_p_[i] = vec_p_indices[i+size_train_p_];
    }

    // ind_test_p should sort
    std::sort(ind_test_p_,ind_test_p_ + size_test_p_);       // this is critical for MatGetSubMatrix method

                                    //---- Negative class  training ----
    std::vector<int> vec_n_indices;
    vec_n_indices.reserve(size_total_n_);
    for (int i_n=0; i_n<size_total_n_; ++i_n)
        vec_n_indices.push_back(i_n);
    srand(std::stoll(Config_params::getInstance()->get_cpp_srand_seed()));
    std::random_shuffle ( vec_n_indices.begin(), vec_n_indices.end() );


    for (int i =0;i != size_train_n_ ; i++){
        ind_train_n_[i] = vec_n_indices[i];
    }

    // ind_train_n should sort
    std::sort(ind_train_n_,ind_train_n_ + size_train_n_);

                                    ///---- Negative class test ----
    for (int i =0;i != size_test_n_; i++){        // only pick the rest of nodes (after training to the end)
    ind_test_n_[i] = vec_n_indices[i+size_train_n_];
    }

    // ind_test_n should sort
    std::sort(ind_test_n_,ind_test_n_ + size_test_n_);       // this is critical for MatGetSubMatrix method

    ISCreateGeneral(PETSC_COMM_SELF,size_train_p_,ind_train_p_,PETSC_COPY_VALUES,&isrow_train_p_);
    ISCreateGeneral(PETSC_COMM_SELF,size_test_p_,ind_test_p_,PETSC_COPY_VALUES,&isrow_test_p_);

    ISCreateGeneral(PETSC_COMM_SELF,size_train_n_,ind_train_n_,PETSC_COPY_VALUES,&isrow_train_n_);
    ISCreateGeneral(PETSC_COMM_SELF,size_test_n_,ind_test_n_,PETSC_COPY_VALUES,&isrow_test_n_);
//    ISView(isrow_train_p_,PETSC_VIEWER_STDOUT_SELF);

    PetscFree(ind_train_p_);
    PetscFree(ind_test_p_);

    PetscFree(ind_train_n_);
    PetscFree(ind_test_n_);

    MatGetSubMatrix(data_p,isrow_train_p_, NULL,MAT_INITIAL_MATRIX,&train_p_);
    MatGetSubMatrix(data_p,isrow_test_p_, NULL,MAT_INITIAL_MATRIX,&test_p_);

    MatGetSubMatrix(data_n,isrow_train_n_, NULL,MAT_INITIAL_MATRIX,&train_n_);
    MatGetSubMatrix(data_n,isrow_test_n_, NULL,MAT_INITIAL_MATRIX,&test_n_);

    ISDestroy(&isrow_train_p_);
    ISDestroy(&isrow_test_p_);

    ISDestroy(&isrow_train_n_);
    ISDestroy(&isrow_test_n_);

    if(num_col_p_ >= num_col_n_)
        max_num_col = num_col_p_;
    else
        max_num_col = num_col_n_;
#if dbl_MS_cross_fold >= 1
    printf("[MS][cross_fold] max number of columns :%d\n",max_num_col);
#endif

    MatGetSize(train_p_,&size_total_p_,NULL);                  //get the size of data in positive (minority) class
    MatGetSize(test_p_,&size_test_p_,NULL);                  //get the size of data in positive (minority) class

    MatGetSize(train_n_,&size_total_n_,NULL);
    MatGetSize(test_n_,&size_test_n_,NULL);
#if dbl_MS_cross_fold >= 1
    printf("[MS][cross_fold]train_p size :%d\n",size_train_p_);                                               //$$debug
    printf("[MS][cross_fold]test_p size :%d\n",size_test_p_);                                               //$$debug

    printf("[MS][cross_fold]train_n size :%d\n",size_train_n_);                                               //$$debug
    printf("[MS][cross_fold]test_n size :%d\n",size_test_n_);                                               //$$debug
#endif
#if dbl_MS_cross_fold >= 7
    printf("[MS][cross_fold]train_p Matrix:\n");                                               //$$debug
    MatView(train_p_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

    printf("[MS][cross_fold]train_n Matrix:\n");                                               //$$debug
    MatView(train_n_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif

    /// Mix both test matrices to one matrix
    combine_test_data(test_total_,test_p_,size_test_p_,test_n_,size_test_n_,max_num_col);


}
//=================== combine_test_data ==============================
// Get 2 matrices and make a new one while destroy both initial matrices
void ModelSelection::combine_test_data(Mat& test_total, Mat& dt_test_p,
                                       PetscInt size_p, Mat& dt_test_n, PetscInt size_n, PetscInt max_num_col_){
    PetscInt num_row = size_p + size_n;
    PetscInt i, ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
//    PetscInt total_curr_row = 0;
//    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row ,max_num_col_ + 1 ,pre_init_loader_matrix,PETSC_NULL, &test_total); //+1 is for label

    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row ,max_num_col_ + 1 ,(max_num_col_ + 1 ),PETSC_NULL, &test_total); //+1 is for label
    for(i =0; i < size_p ; i++){
        MatSetValue(test_total, i, 0, +1,INSERT_VALUES);        //Insert positive label
        MatGetRow(dt_test_p,i,&ncols,&cols,&vals);
        for(int j=0; j < ncols ; j++){
            MatSetValue(test_total,i,cols[j]+1, vals[j],INSERT_VALUES) ;    //+1 shifts the columns 1 to the right
        }
        MatRestoreRow(dt_test_p,i,&ncols,&cols,&vals);
    }

    for(i =0; i < size_n ; i++){
        MatSetValue(test_total, i + size_p, 0, -1, INSERT_VALUES);        //Insert negative label
        MatGetRow(dt_test_n,i,&ncols,&cols,&vals);
        for(int j=0; j < ncols ; j++){
            MatSetValue(test_total, i + size_p, cols[j]+1, vals[j],INSERT_VALUES) ;    //+1 shifts the columns 1 to the right
        }
        MatRestoreRow(dt_test_n,i,&ncols,&cols,&vals);
    }
    MatAssemblyBegin(test_total, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(test_total, MAT_FINAL_ASSEMBLY);
    MatDestroy(&dt_test_p);                             //release the separated class of test data
    MatDestroy(&dt_test_n);

#if dbl_MS_combine_test_data > 7
    PetscViewer     viewer_testdata;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Coarsest_test_data.dat",FILE_MODE_WRITE,&viewer_testdata);
    MatView(test_total,viewer_testdata);
    PetscViewerDestroy(&viewer_testdata);

    printf("[combine_test_data]total test Matrix:\n");                                               //$$debug
    MatView(test_total ,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif

}


void ModelSelection::partial_solver(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n, double last_c, double last_gamma,
                        int level, std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
                        std::unordered_set<PetscInt>& uset_SV_index_p, std::unordered_set<PetscInt>& uset_SV_index_n){
    ETimer t_ms_ps;
    // - - - - - - set parameters - - - - - -
    read_parameters();
    param.C = last_c;
    param.gamma = last_gamma;

    const char *error_msg;                                  //check parameters
    error_msg = svm_check_parameter(&prob,&param);
    if(error_msg) {
        printf("[MS][PS]  ERROR: %s\n",error_msg);
        fprintf(stderr,"ERROR: %s\n",error_msg);
        exit(1);
    }
    // - - - - - - read the training part - - - - - -
    read_problem_index_base(p_data,n_data,v_p_index, v_n_index);
//    std::cout<< "[MS][PS] read_problem_index_base finished successfully! " <<std::endl;

    PetscInt p_num_row_ = v_p_index.size();
    PetscInt n_num_row_ = v_n_index.size();
    // - - - - - - - set weights only for Weighted SVM - - - - - - - -
    if(Config_params::getInstance()->get_ms_svm_id()==2){
        alloc_memory_for_weights(param, 0);     // 0 means don't free old memory and it makes sense because this is the first time

        if(Config_params::getInstance()->get_rf_weight_vol()){      //based on volume
            //calc sum of volumes       //TODO
            std::cout<< "[MS][PS]  WSVM weight based on volume has not implemented yet! Exit!!!" <<std::endl; exit(1);
        }else{                                                      //based on number of points
            set_weights_num_points(param, p_num_row_, n_num_row_);
        }
    }else{
        param.weight = NULL;
        param.weight_label=NULL;
        param.nr_weight=0;
    }

    model = svm_train(&prob,&param);
//    t_ms_ps.stop_timer("[MS][PS] model training");

/// - - - - - - - - prepare the solution for refinement - - - - - - - - -

    if(level > 1 ){     // at the finest level, we need to save the model (SV, C, gamma) for unseen points
//        ETimer t_solution;
        // ----- create the index of SVs in data points for each class seperately ----
        PetscInt i;
        for (i=0; i < model->nSV[0];i++){
            // -1 because sv_indice start from 1, while petsc row start from 0
            // the original index in data matrix is saved in v_p_index
            uset_SV_index_p.insert(v_p_index[model->sv_indices[i] - 1]);
        }

        // add the index in the model for it after subtract from number of minority in training data

        for (int i=0; i < model->nSV[1];i++){
            // -1 the same as pos class, p_num_row because they are after each other
            // the original index in data matrix is saved in v_n_index
            uset_SV_index_n.insert(v_n_index[ model->sv_indices[model->nSV[0] + i] - 1 - p_num_row_]);
        }

#if dbl_MS_PS >= 3
        printf("[MS][PS] partial solver has nSV+:%d, nSV-:%d\n", model->nSV[0], model->nSV[1]);
#endif
//        t_solution.stop_timer("[MS][PS] prepare the partial solution (except finest level) at level:",std::to_string(level));
    }else{
        if(Config_params::getInstance()->get_ms_save_final_model()){
            printf("[MS][PS] at the finest level the partial models needs to be saved!!! Exit \n");
            exit(1);
            //            svm_save_model("./final_model.svm",model);
        }
        //TODO: multiple hyperplain
    }

// - - - - - - - - Report partial evaluation results - - - - - - - - -
//    #if rpt_TD_only_l1 == 1
//        return;     // in case to report only at level 1, ignore the rest of the code
//    #endif
//    printf("---------------- partial evaluation of the test data -------------------- level:%d\n",level);
//    std::map<measures,double> partial_results;
//    partial_results = test_predict(m_testdata);
//    printf("[MS][PS] Partial result for TestData L:%d Acc:%g,\tGmean:%g,\tnSV+:%d,\tnSV-:%d,\tC:%g,\tG:%g\n",
//                                    level, partial_results[Acc],partial_results[Gmean],
//                                    model->nSV[0],model->nSV[1],model->param.C, model->param.gamma);     //$$debug
}


void ModelSelection::set_range(){
    if(Config_params::getInstance()->get_ms_svm_id()==1){                   //normal SVM
        range_c.min = -6.64385619 ;
        range_c.max =  6.64385619 ;
    }else{                                                      //Weighted SVM (ms_svm_id == 2)
        range_c.min = -6.64385619 ;
        range_c.max =  10.5507468 ;
    }
    range_gamma.min = -7.6439 ;
    range_gamma.max =  1.5850 ;
}

void ModelSelection::set_center(bool inherit_last_params, double last_c, double last_gamma){
    if(inherit_last_params != 1){    //if last C, Gamma is not passed as parameters
        point_center.x = (range_c.min + range_c.max) /2;
        point_center.y = (range_gamma.min + range_gamma.max) /2;
        point_center.z = 0.25;
        printf("[MS][set_center] last parameters are not inherited, Center Point C:%g, gamma:%g\n",
                                                        point_center.x,point_center.y);         //$$debug
    }
    else{                           // inherit last C, Gamma
        point_center.x = log2(last_c);
        point_center.y = log2(last_gamma);
        point_center.z = 0.25;
        printf("[MS][set_center] inherited parametes, Center Point C:%g, gamma:%g\n",
                                                        point_center.x,point_center.y); //(log2 calculated)
    }
}

//============================= UD ===================================
//inherit_last_params: 0 means not inherit
//                     1 means inherit the last C, gamma
solution ModelSelection::UD(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n,
                            int inherit_last_params, double last_c, double last_gamma, int level){

    ETimer t_ms_ud;
    PetscInt p_num_row_, n_num_row_, num_col_;  //num_col_ is the number of features that is exactly the same for both class
    int total_num_point_ = 0;
    MatGetSize(p_data, &p_num_row_, &num_col_);
    MatGetSize(n_data, &n_num_row_, NULL);
    total_num_point_ = p_num_row_ + n_num_row_;
#if dbl_MS_UD >= 7
    printf("[MS][UD]p_data Matrix in top of model selection:\n");                   //$$debug //print Minority(p) matrix
    MatView(p_data,PETSC_VIEWER_STDOUT_WORLD);
    printf("[MS][UD]n_data Matrix in top of model selection:\n");                   //$$debug    //print Majority(n) matrix
    MatView(n_data,PETSC_VIEWER_STDOUT_WORLD);
#endif
    bool ms_is_active = 1;                  //Control Flags
    bool ms_limit_not_reached = 1;

    if(!Config_params::getInstance()->get_ms_status()){   //model selection is deactive     //TODO
        printf("Model selection is disabled\n");
        ms_is_active = 0;
    }
    else{                       // model selection is active
        if(total_num_point_ > Config_params::getInstance()->get_ms_limit() ){
            ms_limit_not_reached = 0;
        }
        else{                                                                   /// search for new C, Gamma
            set_range();
            set_center(inherit_last_params,last_c,last_gamma);
        }                                                                       ///end of search for new C, Gamma
    }

    if(ms_is_active){
        if(ms_limit_not_reached){
            ETimer t_1st_stage_total;
            printf(" ------------- 1st Stage at level:%d---------------\n", level);
            std::vector<ud_point> ud_params_st_1;
            std::map<measures,double> predict_result_1;     //map that contains all the measures
            std::vector<summary> measures_st_1;
            measures_st_1.reserve(Config_params::getInstance()->get_ms_first_stage());
            ud_params_st_1 = ud_param_generator(range_c,range_gamma,1, Config_params::getInstance()->get_ms_first_stage(), point_center);
        // Cross fold data (the input data is the remaininag data after I exclude the test data before the v-cycle)
        // Here a new test data is created that is random at each stage and it is used to improve the model
        // It is not the same as untouched test data.

            Mat train_p_1_, train_n_1_, test_total_1_;
            cross_fold_data(p_data, n_data, train_p_1_, train_n_1_, test_total_1_);

            PetscInt num_row_p1=0, num_row_n1=0,num_row_t1=0;
            MatGetSize(train_p_1_,&num_row_p1,NULL);                   //num_row returns the number of rows globally
            MatGetSize(train_n_1_,&num_row_n1,NULL);
            MatGetSize(test_total_1_,&num_row_t1,NULL);
#if dbl_MS_UD >= 5
    printf("[MS][UD] Inside model selection p:%d n:%d t:%d\n",num_row_p1,num_row_n1,num_row_t1);
#endif
            read_parameters();
            read_problem(train_p_1_, train_n_1_);                   // the crossed fold data is used for this

            const char *error_msg;                                  //check parameters
            error_msg = svm_check_parameter(&prob,&param);
            if(error_msg) {
                printf("[MS][UD] ERROR: %s\n",error_msg);
                fprintf(stderr,"ERROR: %s\n",error_msg);
                exit(1);
            }                                                       // now prob and param are loaded and checked

            if(Config_params::getInstance()->get_ms_svm_id()==2){                   //Weighted SVM
                alloc_memory_for_weights(param, 0);     // 0 means don't free old memory and it makes sense because this is the first time
            }else{
                param.weight = NULL;
                param.weight_label=NULL;
                param.nr_weight=0;
            }


            for(unsigned int i =0; i < ud_params_st_1.size();i++){
                //  set the parameters
                param.gamma = pow(2,ud_params_st_1[i].G);
                if(Config_params::getInstance()->get_ms_svm_id()==2){                   //Weighted SVM
                    /// The weight are set once for both stage and the test data after it
                    if(Config_params::getInstance()->get_rf_weight_vol()){      //based on volume
                        //calc sum of volumes       //TODO
                        std::cout<< "[MS][UD] WSVM weight based on volume has not implemented yet! Exit!!!" <<std::endl; exit(1);
                    }else{                                                      //based on number of points
                        set_weights_num_points(param, num_row_p1, num_row_n1);
                    }
                }
                param.C = pow(2,ud_params_st_1[i].C) ; // this is needed for both SVM, WSVM

                model = svm_train(&prob,&param);
                predict_result_1 = test_predict(test_total_1_);
#if dbl_MS_UD >= 1
                printf("[MS][UD] level:%d, stage:1, i:%d, Acc:%g, Gmean:%g, c:%g, gamma:%g)\n",
                                level,i, predict_result_1[Acc],predict_result_1[Gmean], param.C, param.gamma);
#endif
                measures_st_1.push_back(make_summary(i,predict_result_1[Gmean], model->nSV[0],
                                model->nSV[1], model->param.C, model->param.gamma));     // save the Gmean for stage 1 results
            }
            int best_1st_stage = select_best_1st(measures_st_1,level);
            printf("[MS][UD] best selected C:%g, gamma:%g at 1st stage\n", measures_st_1[best_1st_stage].C,
                                                     measures_st_1[best_1st_stage].gamma);
            t_1st_stage_total.stop_timer("Whole 1st stage at level:", std::to_string(level));
            printf("------------- 2nd Stage at level:%d---------------\n",level);
            ETimer t_2nd_stage_total;
            // call UDsample to find the best params        //Set best of first stage to the center point of 2nd stage
            point_center.x = measures_st_1[best_1st_stage].C;
            point_center.y = measures_st_1[best_1st_stage].gamma;
#if dbl_MS_UD >= 1
            printf("[MS][UD] center for second stage-> C:%g, gamma:%g\n",pow(2,point_center.x), pow(2,point_center.y));          //$$debug
//            printf("[MS][UD] center for second stage-> C:%g, gamma:%g\n",point_center.x, point_center.y);          //$$debug
#endif
            std::vector<ud_point> ud_params_st_2;
            std::map<measures,double> predict_result_2;
            std::vector<summary> measures_st_2;
            //reserve the space to work with indices instead of push_back
            measures_st_2.reserve(Config_params::getInstance()->get_ms_second_stage());
            ud_params_st_2 = ud_param_generator(range_c,range_gamma,2,
                                                    Config_params::getInstance()->get_ms_second_stage(), point_center);

            // - - - - - Find the center of 1st stage - - - - -
            int center_index =0;
            for (unsigned int i=0; i != ud_params_st_2.size() ; i++){
                if(  (point_center.x == ud_params_st_2[i].C) && (point_center.y == ud_params_st_2[i].G)   ){
                    center_index = i;
                    printf("[MS][UD] new center for second stage was in index : %d \n", center_index);
                    break;
                }
            }
            // - - - - -  Erase the center from second stage - - - - -
#if dbl_MS_UD >= 9
    for(unsigned int i =0; i < ud_params_st_2.size();i++){
        printf("[MS][UD] before remove center i:%d, C:%g, gamma:%g\n", i,ud_params_st_2[i].C, ud_params_st_2[i].G);
    }
#endif
            //it needs a pointer to exact location
            ud_params_st_2.erase(ud_params_st_2.begin()+ center_index);
#if dbl_MS_UD >= 9
    for(unsigned int i =0; i < ud_params_st_2.size();i++){
        printf("[MS][UD] after remove center i:%d, C:%g, gamma:%g\n", i,ud_params_st_2[i].C, ud_params_st_2[i].G);
    }
#endif
            for(unsigned int i =0; i < ud_params_st_2.size();i++){
                // - - - - - - set the parameters - - - - - - -
                //  the weights are not going to change since the data is the same(no need to set weights)
                param.gamma = pow(2,ud_params_st_2[i].G);
                param.C = pow(2,ud_params_st_2[i].C);

                model = svm_train(&prob,&param);
                predict_result_2 = test_predict(test_total_1_); //  test the data and keep the results
#if dbl_MS_UD >= 1
                printf("[MS][UD] level:%d, stage 2, i:%d, Acc:%g, Gmean:%g, c:%g, gamma:%g)\n",
                                level, i, predict_result_2[Acc],predict_result_2[Gmean], param.C, param.gamma);
#endif
                measures_st_2.push_back(make_summary(i,predict_result_2[Gmean], model->nSV[0],
                                model->nSV[1], model->param.C, model->param.gamma));     // save the Gmean for stage 1 results
            }
            // - - - - - - Add summary of the center to measures_st_2 - - - - - -
            // As I select the best of 1st stage in the center of second, I just add results to 2nd stage
            measures_st_2.push_back(make_summary(ud_params_st_2.size(),measures_st_1[best_1st_stage].gmean,
                            measures_st_1[best_1st_stage].num_SV_p, measures_st_1[best_1st_stage].num_SV_n,
                            measures_st_1[best_1st_stage].C, measures_st_1[best_1st_stage].gamma));

            int best_summary_both_stage = select_best_both_stage(measures_st_2,level);
            // - - - - - - - Set final parameters - - - - - - -
            param.C = measures_st_2[best_summary_both_stage].C;
            param.gamma = measures_st_2[best_summary_both_stage].gamma;
#if dbl_MS_UD >= 3
            printf("[MS][UD] best params are set to train the model for last time\n");   //$$debug
            printf("[MS][UD] C:%g, gamma:%g\n",param.C, param.gamma);
#endif
            MatDestroy(&train_p_1_);                                //free resources
            MatDestroy(&train_n_1_);
            MatDestroy(&test_total_1_);
#if dbl_MS_UD >= 3
            printf("Train the model for last time with best C, gamma and full datasets\n");
#endif
            t_2nd_stage_total.stop_timer("Whole 2nd stage at level:", std::to_string(level));

        }else{                           // else for {"if(ms_limit_not_reached)"}
            /// - - - model selection is disabeld due to the size and parameters are inherit from coarser level - - -
            read_parameters();
            if(last_c == 0){
                param.C = Config_params::getInstance()->get_svm_C();
                param.gamma = Config_params::getInstance()->get_svm_gamma() ;
                printf("[MS][UD] No recent parameter for SVM and ms_limit is reached C:%g, gamma:%g\n",param.C, param.gamma);
                // @05202016-1206
                if(Config_params::getInstance()->get_ms_svm_id()==2){                   //Weighted SVM (reserve memory)
                    alloc_memory_for_weights(param, 0);     // 1 means free old memory, here we pass the 0, because it is the first time for this model
                    set_weights_num_points(param, p_num_row_, n_num_row_);              //Weighted SVM (set weights)
                }
            }else{
                printf("[MS][UD] ms limit is reached, so params are inherited\n");
                param.C = last_c;
                param.gamma = last_gamma;
                printf("[MS][UD] Inherited params are C:%g, gamma:%g\n",param.C, param.gamma);
                if(Config_params::getInstance()->get_ms_svm_id()==2){                   //Weighted SVM (reserve memory)
                    alloc_memory_for_weights(param, 0);     // 1 means free old memory, here we pass the 0, because it is the first time for this model
                    set_weights_num_points(param, p_num_row_, n_num_row_);              //Weighted SVM (set weights)
                }
            }
        }


    }else{              //else for if(ms_is_active)  <==  ms_status = 0
        read_parameters();
        param.C = Config_params::getInstance()->get_svm_C();
        param.gamma = Config_params::getInstance()->get_svm_gamma() ;

        printf("[MS][UD] param C:%g, gamma:%g\n",param.C, param.gamma);       //$$debug
        printf("[MS][UD] No model selection\n");
        // @05202016-1206
        if(Config_params::getInstance()->get_ms_svm_id()==2){                   //Weighted SVM (reserve memory)
            alloc_memory_for_weights(param, 0);     // 1 means free old memory, here we pass the 0, because it is the first time for this model
            set_weights_num_points(param, p_num_row_, n_num_row_);              //Weighted SVM (set weights)
        }
    }



    ETimer t_svm_train;
    MatGetSize(p_data, &p_num_row_, NULL);
    MatGetSize(n_data, &n_num_row_, NULL);
    printf("[MS][UD] number of points in minority: %d, majority:%d \n", p_num_row_, n_num_row_);       //$$debug

    read_problem(p_data, n_data);
    printf("[MS][UD] read_problem_finished!\n");       //$$debug

    model = svm_train(&prob,&param);
    t_svm_train.stop_timer("[MS][UD] MS Validation (only svm train)");
#if report_MS_untouched_ds == 1
    #if rpt_TD_only_l1 == 1         // Technical Debt
        if(level == 1){
    #endif

        printf("---------------- test the untouched test data --------------------\n");

        std::map<measures,double> final_results;
        Mat untouched_test_data ;
        Loader test_loader;
        untouched_test_data = test_loader.load_norm_data_sep(this->test_dataset_f_name);

        final_results = test_predict(untouched_test_data);

        printf("[MS][UD] TD , L:%d, Acc:%.2f, Sens:%.2f, Spec:%.2f, Gmean:%.2f, nSV+:%d, nSV-:%d, C:%.2f, G:%.4f\n",
                                        level, final_results[Acc],final_results[Sens], final_results[Spec], final_results[Gmean],
                                        model->nSV[0],model->nSV[1],model->param.C, model->param.gamma);     //$$debug
        if(level == 1){
            iter_summary curr_summary;
            curr_summary.C = model->param.C;
            curr_summary.gamma = model->param.gamma;
            curr_summary.result[Acc] = final_results[Acc];
            curr_summary.result[Sens] = final_results[Sens];
            curr_summary.result[Spec] = final_results[Spec];
            curr_summary.result[Gmean] = final_results[Gmean];
            Config_params::getInstance()->add_final_summary(curr_summary);
        }
        MatDestroy(&untouched_test_data);
    #if rpt_TD_only_l1 == 1
    }
    #endif

#endif
#if dbl_MS_UD >= 1
    printf("[MS][UD] level:%d {Final} nSV_c0:%d, nSV_c1:%d, l:%d, c:%g, gamma:%g\n",
                    level, model->nSV[0],model->nSV[1], model->l,
                    model->param.C, model->param.gamma);  //$$debug
#endif

    ETimer t_solution;
    /// prepare the solution for refinement
    solution sol_result;
    if(level > 1 ){     // at the finest level, we need to save the model (SV, C, gamma) for unseen points
        // ----- create the index of SVs in data points for each class seperately ----
        MatCreateSeqAIJ(PETSC_COMM_SELF,model->nSV[0], num_col_, PETSC_DECIDE,PETSC_NULL, &sol_result.m_p_SV);    // performance problem
        sol_result.p_index.reserve(model->nSV[0]);   //reserve the space for positive class

        PetscInt i;
        for (i=0; i < model->nSV[0];i++){
            // -1 because sv_indice start from 1, while petsc row start from 0
            sol_result.p_index.push_back(model->sv_indices[i] - 1);
        }
        t_solution.stop_timer("MS prepare the solution: (from start to end of minority class) ");
    #if dbl_MS_UD >=5
        printf("[MS][UD] number of row for minority(P) data (loop counter for majority points): %d\n",p_num_row_);
    #endif
        MatCreateSeqAIJ(PETSC_COMM_SELF,model->nSV[1], num_col_, PETSC_DECIDE,PETSC_NULL, &sol_result.m_n_SV);    // performance problem
        sol_result.n_index.reserve(model->nSV[1]);   //reserve the space for negative class


        // start from 0 to #SV in majority
        // add the index in the model for it after subtract from number of minority in training data
        // # of minority points -> p_num_row_ which is calculated in the beginning
        for (int i=0; i < model->nSV[1];i++){
            // -1 the same as pos class, p_num_row because they are after each other
            sol_result.n_index.push_back(model->sv_indices[model->nSV[0] + i] - 1 - p_num_row_);
#if dbl_MS_UD >=9
            printf("[MS][UD]majority(N) i:%d [model->sv_indices[model->nSV[0]] -1 +i - p_num_row_]:%d\n",
                                            i, model->sv_indices[model->nSV[0]] -1 +i - p_num_row_);       //@
#endif
        }
        t_solution.stop_timer("[TODO] MS prepare the solution: (from start to end of both classes) ");

#if dbl_MS_UD >=7       // 7 default
        if(Config_params::getInstance()->get_rf_fake_points()){
            printf("[MS][UD] sol_result.m_n_SV:\n");                                               //$$debug
            MatView(sol_result.m_n_SV,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
        }
#endif
        sol_result.C = model->param.C;
        sol_result.gamma = model->param.gamma;

#if dbl_MS_UD >= 9
        printf("[MS][UD] p_num_row:%d, model->nSV[0]:%d\n",p_num_row_, model->nSV[0]);
        for (int i=0; i < model->nSV[0];i++){
            printf("[MS][UD] sol positive index[%d]:%d\n", i, sol_result.p_index[i]);
        }

        for (int i=0; i < model->nSV[1];i++){
            printf("[MS][UD] sol negative index[%d]:%d\n", i, sol_result.n_index[i]);
        }
#endif
    }else{
        if(Config_params::getInstance()->get_ms_save_final_model()){
            printf("[MS][UD] at the finest level the model is save to ./final_model.svm \n");
            svm_save_model("./final_model.svm",model);
        }
        //TODO: multiple hyperplain
    }

    t_solution.stop_timer("[MS][UD] prepare the solution(save the indices and SVs for refinement) at level:",std::to_string(level));
    t_ms_ud.stop_timer("[MS][UD] All Model Selection(1st,2ns,validation,solution are included) at level:",std::to_string(level));

    return sol_result;
}

void ModelSelection::set_weights_num_points(svm_parameter& param_, PetscInt num_p_point, PetscInt num_n_point){
    param_.weight[0]= 1.0 / num_p_point;
    param_.weight[1]= 1.0 / num_n_point;
#if dbl_MS_set_weights >= 1
        printf("[MS][set_weights_num_points] Min class P(label :%d,\t # row:%d,\t weight:%g)\n",param_.weight_label[0], num_p_point, param_.weight[0]);
        printf("[MS][set_weights_num_points] Maj class N(label :%d,\t # row:%d,\t weight:%g)\n",param_.weight_label[1], num_n_point, param_.weight[1]);
#endif
}


void ModelSelection::alloc_memory_for_weights(svm_parameter& in_param, bool free_first){
    if(free_first){
        free(in_param.weight_label);
        free(in_param.weight);
    }
    in_param.nr_weight = 2;
    in_param.weight_label = (int *) malloc(2 * sizeof(int));
    in_param.weight = (double *) malloc(2 * sizeof(double));
    in_param.weight_label[0] = 1;                      // Minority class (P)
    in_param.weight_label[1] = -1;                     // Maj            (N)
#if dbl_MS_malloc_weights >= 1
        printf("[MS][alloc_memory_for_weights] nr_weight:%d,weight_label[0]:%d, weight_label[1]:%d\n",
               in_param.nr_weight, in_param.weight_label[0], in_param.weight_label[1]);
#endif
}


/* use UD table to determine the parameter values for model selection
 * @Input params:
 * svm_id :         type of SVM (svm_id=1: WSVM and SVM,     svm_id=2: WRSVM)
 * range_c :        range of C (minimum and maximum) (this are not the real values for C, these are log(c) )
 * range gamma :    range of Gamma OR range of epsilon if svm_id==1
 *                      In Regression, the searching range of episilon is equal to [0 , 0.5]
 * stage :          the stage of nested UDs (1 or 2)
 * pattern :        the UD pattern, 5-5runs,9-9runs,and 13-13runs
 * point center :   the UD center for current stage  (default= center of the searching box)
 * @Outputs :
 * Vector of ud_point  : the UD sampling points for current stage
 */
//std::vector<ud_point> ModelSelection::ud_param_generator(int svm_id, ms_range range_c,
//                                                         ms_range range_g, int stage, int pattern, ms_point p_center){

std::vector<ud_point> ModelSelection::ud_param_generator(ms_range range_c, ms_range range_g,
                                                         int stage, int pattern, ms_point p_center){

    double c_len = (range_c.max - range_c.min) ;                // length for C range
    double g_len = (range_g.max - range_g.min) ;                // length for gamma range
//    std::cout << "clen " << c_len << ",g_len " << g_len  <<"\n";
    std::vector<ud_point> params(pattern);
    double cen_c = p_center.x - (c_len /pow(2,stage));
    double cen_g = p_center.y - (g_len /pow(2,stage));
    double pow_2_stage_minus_one = pow(2,stage -1);
    double pattern_minus_one = (pattern -1);
    for(int i=0; i < pattern;i++){        //loop over all rows of the UD Table
        params[i].C = (  (UDTable[pattern][i][0] - 1) / pattern_minus_one / pow_2_stage_minus_one   * c_len ) + cen_c;
        params[i].G = (  (UDTable[pattern][i][1] - 1) / pattern_minus_one / pow_2_stage_minus_one   * g_len ) + cen_g;

        if (params[i].C > range_c.max ){           //keep it inside the scope (set the outside value to max on that side)
            printf("[MS][ud_param_generator] stage: %d, params[i].C: %g\n", stage, params[i].C);
                params[i].C = range_c.max;
        }
        if(params[i].C < range_c.min ){
                params[i].C = range_c.min;
        }

        if (params[i].G > range_g.max ){
                params[i].G = range_g.max;
        }
        if(params[i].G < range_g.min ){
                params[i].G = range_g.min;
        }
#if dbl_MS_ud_param_generator >= 3
//        printf("[MS][ud_param_generator] stage: %d, log(C): %g, log(gamma): %g\n",stage, params[i].C,params[i].G); //$$debug
        printf("[MS][ud_param_generator] stage: %d, C: %g, gamma: %g\n",stage, pow(2,params[i].C), pow(2,params[i].G)); //$$debug
#endif
    }
    return params;
}


void ModelSelection::predict_label(Mat& test_data, int target_row, Mat& m_predicted_label){
#if dbl_MS_predict_label >= 7
    printf("[MS][predict_label] test_predict_data Matrix:\n");                                       //$$debug
    MatView(test_data,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif
//    PetscErrorCode ierr;
//    int correct = 0;
//    int total = 0;
//    double error = 0;
//    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
//    double tp =0, tn =0, fp =0, fn=0;
    int svm_type=svm_get_svm_type(model);
//    int nr_class=svm_get_nr_class(model);
    double *prob_estimates=NULL;
//    std::map<measures,double> results_;
// - - - - - - - - - start of reading the test points - - - - - - - - -
    PetscInt i=0, k=0, ncols=0,num_points=0,num_col=0;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;

    MatGetSize(test_data,&num_points,&num_col);   //get the number of data points
#if dbl_MS_predict_label >= 3
    printf("[MS][predict_label] test data points rows:%d cols:%d \n", num_points,num_col);
#endif
// - - - - - - - - - read test points from matrix to array - - - - - - - - -
    for (i=0; i< num_points;i++){
//        double target_label, predict_label;//target is the one that is in the vector
        double predict_label;
        MatGetRow(test_data,i,&ncols, &cols, &vals);

//        target_label = vals[0];             //read the label in the first column(0)
        //allocate required memory
        x = (struct svm_node *) realloc(x, (ncols + 1 ) * sizeof(struct svm_node));

        for (k=1; k< ncols; k++) {  //zero is the label
            x[k-1].index = cols[k];   //since I add the label in the first column +1 is not required anymore
            x[k-1].value = vals[k];
#if dbl_MS_predict_label >= 5
            printf("[MS][predict_label] i: %d, k-1: %d, x[k-1].index: %d, x[k-1].value: %g\n",i, k-1,x[k-1].index, x[k-1].value);    //$$debug
#endif
        }
        //create the end element of each node (-1,0)
        x[k].index = -1;
        x[k].value = 0;
#if dbl_MS_predict_label >= 5
        printf("k:%d, x[k].index:%d, x[k].value:%g\n", k,x[k].index, x[k].value);    //$$debug
#endif
        MatRestoreRow(test_data,i,&ncols, &cols, &vals);

        if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))  {    // Not used
            predict_label = svm_predict_probability(model,x,prob_estimates);
        }
        else {
            predict_label = svm_predict(model,x);
        }
        // target row indicate the row index and i indicate the column index as  each row in the matrix belongs to a specific model
        MatSetValue(m_predicted_label,target_row,i,predict_label,INSERT_VALUES);
    }
}

std::map<measures,double> ModelSelection::test_predict(Mat& test_data){
#if dbl_MS_test_predict >= 7
    printf("[MS][test_predict] test_predict_data Matrix:\n");                                       //$$debug
    MatView(test_data,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif
    PetscErrorCode ierr;
    int correct = 0;
    int total = 0;
    double error = 0;
    double sump = 0, sumt = 0, sumpp = 0, sumtt = 0, sumpt = 0;
    double tp =0, tn =0, fp =0, fn=0;
    int svm_type=svm_get_svm_type(model);
//    int nr_class=svm_get_nr_class(model);
    double *prob_estimates=NULL;
    std::map<measures,double> results_;
//start of reading the test points
    PetscInt i=0, k=0, ncols=0,num_points=0,num_col=0;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;

    MatGetSize(test_data,&num_points,&num_col);   //get the number of data points
#if dbl_MS_test_predict >= 3
    printf("[MS][test_predict] test data points rows:%d cols:%d \n", num_points,num_col);
#endif
//read test points from matrix to array
    for (i=0; i< num_points;i++){
        double target_label, predict_label;//target is the one that is in the vector
        ierr =  MatGetRow(test_data,i,&ncols, &cols, &vals);
//            printf("[MS][test_predict] ierr:%d, i:%d\n",ierr,i);
        if(ierr != 0){
            printf("[MS][test_predict] cannot access row %d from test data\n",i);
            printf("[MS][test_predict] ***** Exit *****\n");
            exit(1);
        }

        target_label = vals[0];             //read the label in the first column(0)
//            printf("target label is %g\n",target_label);        //$$debug
        //allocate required memory
//        x = (struct svm_node *) realloc(x, (ncols+1) * sizeof(struct svm_node));    //+1 is for the last index (-1)
//            printf("ncols:%d\n",ncols);                         //$$debug
        x = (struct svm_node *) realloc(x, (ncols + 1 ) * sizeof(struct svm_node));    //No need for +1 as I add the label in index 0 that add one non-zero value to each row
#if dbl_MS_test_predict >= 3
//                printf("================ [MS][test_predict] i:%d=============\n",i);      //$$debug
        printf("i:%d, ",i);
#endif
        for (k=1; k< ncols; k++) {  //zero is the label
            x[k-1].index = cols[k];   //since I add the label in the first column +1 is not required anymore
            x[k-1].value = vals[k];
#if dbl_MS_test_predict >= 5
            printf("i:%d, k-1:%d, x[k-1].index:%d, x[k-1].value:%g\n",i, k-1,x[k-1].index, x[k-1].value);    //$$debug
#endif
        }
        //create the end element of each node (-1,0)
        x[k].index = -1;
        x[k].value = 0;
#if dbl_MS_test_predict >= 5
        printf("k:%d, x[k].index:%d, x[k].value:%g\n", k,x[k].index, x[k].value);    //$$debug
#endif
        MatRestoreRow(test_data,i,&ncols, &cols, &vals);

        if (predict_probability && (svm_type==C_SVC || svm_type==NU_SVC))  {    // Not used
            predict_label = svm_predict_probability(model,x,prob_estimates);
        }
        else {
            predict_label = svm_predict(model,x);
        }

        if(target_label == 1)   //positive class
        {
            if (predict_label == 1)     //correct
                tp++;
            else                                //predict negative
                fn++;                   //false
        }
        else                    //negative class
        {
            if (predict_label == -1)    //correct
                tn++;
            else                                //predict positive
                fp++;                   //false
        }

        error += (predict_label-target_label)*(predict_label-target_label);
        sump += predict_label;
        sumt += target_label;
        sumpp += predict_label * predict_label;
        sumtt += target_label  * target_label;
        sumpt += predict_label * target_label;
        ++total;
    }
    correct = tp+tn;                //sum both True

    results_[Sens] = tp / (tp+fn) ;
    results_[Spec] = tn / (tn+fp) ;
    results_[Gmean] = sqrt(results_[Sens] * results_[Spec]);
    results_[Acc] = (double)correct / (total) ;
#if dbl_MS_test_predict >= 1
    printf("tp:%g,tn:%g,fp:%g,fn:%g,Acc:%g, Sens:%g, Spec:%g, Gmean:%g\n",tp,tn,fp,fn,results_[Acc], results_[Sens],results_[Spec],results_[Gmean]);
#endif
//end of loop for each test point
    if (svm_type==NU_SVR || svm_type==EPSILON_SVR)       {
        printf("Mean squared error = %g (regression)\n",error/total);
        printf("Squared correlation coefficient = %g (regression)\n",
            ((total*sumpt-sump*sumt)*(total*sumpt-sump*sumt))/
            ((total*sumpp-sump*sump)*(total*sumtt-sumt*sumt))             );
    }
    else{
#if dbl_MS_test_predict >= 1
        printf("Accuracy = %g%% (%d/%d) (classification)\n",                  (double)correct/total*100,correct,total);
#endif
    }
    if(predict_probability)
        free(prob_estimates);
    return results_;
}


//=========== read the training data using the vector of indices ============
void ModelSelection::read_problem_index_base(Mat& p_train_data, Mat& n_train_data,
                                             std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index){
    PetscInt i=0, j=0, l=0, k=0, ncols;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;
    PetscInt num_col=0, num_total_nodes=0;  //num_row=0,
    PetscInt p_num_node_=0, n_num_node_=0, num_elements_=0;
#if dbl_MS_read_problem >= 7
    printf("[MS][RPIB] p_train_data matrix:\n");                   //$$debug
    MatView(p_train_data, PETSC_VIEWER_STDOUT_WORLD);
    printf("[MS][RPIB] n_train_data matrix:\n");                   //$$debug
    MatView(n_train_data, PETSC_VIEWER_STDOUT_WORLD);
#endif


// - - - - - - find number of nodes and elements - - - - - - -
    MatGetSize(p_train_data,NULL,&num_col);    // we need the number of columns
    p_num_node_ = v_p_index.size();
    n_num_node_ = v_n_index.size();
    num_total_nodes = p_num_node_ + n_num_node_;
#if dbl_MS_read_problem >= 1
    printf("[MS][RPIB] number of P_data: %d, N_data: %d, total_nodes :%d \n",
                               p_num_node_,n_num_node_,num_total_nodes);     //$$debug
#endif
//---- Count number of non zero elements -----
    // for positive class
    for (i=0; i< p_num_node_;i++){
//        MatGetRow(p_train_data,i,&ncols, &cols, &vals);
        MatGetRow(p_train_data, v_p_index[i],&ncols, &cols, &vals);
        num_elements_ += ncols + 1;     // +1 : for the end of line index
        MatRestoreRow(p_train_data, v_p_index[i],&ncols, &cols, &vals);
    }
#if dbl_MS_read_problem >= 3
    printf("[MS][RPIB] number of p elems:%d\n",num_elements_);
#endif
    // for negative class
    for (i=0; i< n_num_node_;i++){
        MatGetRow(n_train_data, v_n_index[i],&ncols, &cols, &vals);
        num_elements_ += ncols + 1;     // +1 : for the end of line index
        MatRestoreRow(n_train_data, v_n_index[i],&ncols, &cols, &vals);
    }
#if dbl_MS_read_problem >= 5
    printf("[MS][RPIB] total number of elems for both classes (one column as terminator [-1,0]):%d\n",num_elements_);
#endif

//---- read the data to prob for libsvm -----
    prob.y = Malloc(double, num_total_nodes );
    prob.x = Malloc(struct svm_node *, num_total_nodes );
    x_space = Malloc(struct svm_node, num_elements_ );

#if dbl_MS_read_problem >= 3
    printf("[MS][RPIB] After Malloc svm objects\n");
#endif

    prob.l = num_total_nodes;
    //read positive data
    j=0;                //set the j as an index to go through the x_space
    for (i=0; i< p_num_node_;i++){
        prob.y[i] = 1;
        prob.x[i] = &x_space[j];
        MatGetRow(p_train_data, v_p_index[i],&ncols, &cols, &vals);
        l=0;
#if dbl_MS_read_problem >= 3
        if(ncols == 0){
            printf("[MS][RPIB]  *** Error *** Empty row at %d row in p_train_data! Exit\n",i);
            exit(1);
        }
#endif
        for (k=0; k< num_col; k++) {    //note this is num_col instead of ncols because of j increament
            if(k == cols[l])
            {
                x_space[j].index = k+1;   //the libsvm use 1 index instead of zero
                x_space[j].value = vals[l];
                l++;
            }
#if dbl_MS_read_problem >= 7            //default is 7
            printf("[MS][RPIB] {P} i:%d j:%d prob.y[i]:%g x_space[j].index:%d x_space[j].value:%g\n",
                                        i, j, prob.y[i], x_space[j].index, x_space[j].value);     //$$debug
            printf("k:%d ,l:%d, cols[l]:%d, vals[l]:%g\n",
                   k, l, cols[l], vals[l]);
#endif
            ++j;
        }
        //create the end element of each node (-1,0)
        x_space[j].index = -1;
        x_space[j].value = 0;
        ++j;

        MatRestoreRow(p_train_data, v_p_index[i],&ncols, &cols, &vals);
    }
#if dbl_MS_read_problem >= 3
    printf("[MS][RPIB] end of positive class, i is :%d,num of elements(j) is :%d\n",i,j);
#endif
    //read negative data
    for (i=0; i< n_num_node_;i++){
        prob.y[i+p_num_node_] = -1;
        prob.x[i+p_num_node_] = &x_space[j];
        MatGetRow(n_train_data, v_n_index[i],&ncols, &cols, &vals);

        l=0;
//        for (k=0; k< ncols; k++) {
        for (k=0; k< num_col; k++) {
            if(k == cols[l])
            {
                x_space[j].index = k+1;   //the libsvm use 1 index instead of zero
                x_space[j].value = vals[l];
                l++;
            }
#if dbl_MS_read_problem >= 7
            printf("[MS][RPIB] {N} i:%d j:%d prob.y[i+p_num_node_]:%g x_space[j].index:%d x_space[j].value:%g\n",
                                        i, j, prob.y[i+p_num_node_], x_space[j].index, x_space[j].value);     //$$debug
//            printf("k:%d ,l:%d, cols[l]:%d, vals[l]:%g\n",
//                   k, l, cols[l], vals[l]);
#endif
            ++j;
        }
        //create the end element of each node (-1,0)
        x_space[j].index = -1;
        x_space[j].value = 0;
        ++j;

        MatRestoreRow(n_train_data, v_n_index[i],&ncols, &cols, &vals);
    }
#if dbl_MS_read_problem >= 5
    printf("[MS][RPIB] end of negative class, i is :%d,num of elements(j) is :%d\n",i,j);
#endif
//    Dont Destroy input matrices at all    ( They are deleted after 2nd stage of model selection )
}

//=========== read the training data (both of the matricex are training data not include test data)============
//void ModelSelection::read_problem(Mat& p_train_data, Mat& n_train_data, Mat& p_SVs, Mat& n_SVs){
void ModelSelection::read_problem(Mat& p_train_data, Mat& n_train_data){
    PetscInt i=0, j=0, l=0, k=0, ncols;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;
    PetscInt num_col=0, num_total_nodes=0;  //num_row=0,
    PetscInt p_num_node_=0, n_num_node_=0, num_elements_=0;
#if dbl_MS_read_problem >= 7
    printf("[MS][RP] p_train_data matrix:\n");                   //$$debug
    MatView(p_train_data, PETSC_VIEWER_STDOUT_WORLD);
    printf("[MS][RP] n_train_data matrix:\n");                   //$$debug
    MatView(n_train_data, PETSC_VIEWER_STDOUT_WORLD);
#endif


/// find number of nodes and elements
    MatGetSize(p_train_data,&p_num_node_,&num_col);    //m returns the number of rows globally
//    p_num_node_ = num_row;
    MatGetSize(n_train_data,&n_num_node_,NULL);    //m returns the number of rows globally
//    n_num_node_ = num_row;
    num_total_nodes = p_num_node_ + n_num_node_;
#if dbl_MS_read_problem >= 1
    printf("[MS][read_problem] number of P_data: %d, N_data: %d, total_nodes :%d \n",
                               p_num_node_,n_num_node_,num_total_nodes);     //$$debug
#endif
//---- Count number of non zero elements -----
    // for positive class
    for (i=0; i< p_num_node_;i++){
        MatGetRow(p_train_data,i,&ncols, &cols, &vals);
//        printf("i:%d\n",i);                                               //$$should clear
//        for (j=0; j< ncols; j++) {
//            num_elements_++;
//        }
//        num_elements_++;    //for the end of line index
        num_elements_ += ncols + 1;
        MatRestoreRow(p_train_data,i,&ncols, &cols, &vals);
    }
#if dbl_MS_read_problem >= 3
    printf("[read_problem]number of p elems:%d\n",num_elements_);
#endif
    // for negative class
    for (i=0; i< n_num_node_;i++){
        MatGetRow(n_train_data,i,&ncols, &cols, &vals);
//        printf("i:%d\n",i);                                               //$$should clear
//        for (j=0; j< ncols; j++) {
//            num_elements_++;
//        }
//        num_elements_++;    //for the end of line index
        num_elements_ += ncols + 1;
        MatRestoreRow(n_train_data,i,&ncols, &cols, &vals);
    }
#if dbl_MS_read_problem >= 5
    printf("[MS][RP]{correct _ NNZ} number of total elems(one column as terminator [-1,0]):%d\n",num_elements_);
#endif

//---- read the data to prob for libsvm -----
    prob.y = Malloc(double, num_total_nodes );
    prob.x = Malloc(struct svm_node *, num_total_nodes );
    x_space = Malloc(struct svm_node, num_elements_ );

#if dbl_MS_read_problem >= 3
    printf("[MS][RP]After Malloc\n");
#endif

    prob.l = num_total_nodes;
    //read positive data
    j=0;                //set the j as an index to go through the x_space
    for (i=0; i< p_num_node_;i++){
        prob.y[i] = 1;
        prob.x[i] = &x_space[j];
        MatGetRow(p_train_data,i,&ncols, &cols, &vals);
        l=0;
#if dbl_MS_read_problem >= 3
        if(ncols == 0){
            printf("[MS][RP] *** Error *** Empty row at %d row in p_train_data\n",i);
            exit(1);
        }
#endif
        for (k=0; k< num_col; k++) {    //note this is num_col instead of ncols because of j increament
            if(k == cols[l])
            {
                x_space[j].index = k+1;   //the libsvm use 1 index instead of zero
                x_space[j].value = vals[l];
                l++;
            }
#if dbl_MS_read_problem >= 7            //default is 7
            printf("[MS][RP]{P} i:%d j:%d prob.y[i]:%g x_space[j].index:%d x_space[j].value:%g\n",
                                        i, j, prob.y[i], x_space[j].index, x_space[j].value);     //$$debug
//            printf("k:%d ,l:%d, cols[l]:%d, vals[l]:%g\n",
//                   k, l, cols[l], vals[l]);
#endif
            ++j;
        }
        //create the end element of each node (-1,0)
        x_space[j].index = -1;
        x_space[j].value = 0;
        ++j;

        MatRestoreRow(p_train_data,i,&ncols, &cols, &vals);
    }
#if dbl_MS_read_problem >= 3
    printf("end of positive class, i is :%d,num of elements(j) is :%d\n",i,j);
#endif
    //read negative data
    for (i=0; i< n_num_node_;i++){
        prob.y[i+p_num_node_] = -1;
        prob.x[i+p_num_node_] = &x_space[j];
        MatGetRow(n_train_data,i,&ncols, &cols, &vals);

        l=0;
//        for (k=0; k< ncols; k++) {
        for (k=0; k< num_col; k++) {
            if(k == cols[l])
            {
                x_space[j].index = k+1;   //the libsvm use 1 index instead of zero
                x_space[j].value = vals[l];
                l++;
            }
#if dbl_MS_read_problem >= 7
            printf("[MS][RP]{N} i:%d j:%d prob.y[i+p_num_node_]:%g x_space[j].index:%d x_space[j].value:%g\n",
                                        i, j, prob.y[i+p_num_node_], x_space[j].index, x_space[j].value);     //$$debug
//            printf("k:%d ,l:%d, cols[l]:%d, vals[l]:%g\n",
//                   k, l, cols[l], vals[l]);
#endif
            ++j;
        }
        //create the end element of each node (-1,0)
        x_space[j].index = -1;
        x_space[j].value = 0;
        ++j;

        MatRestoreRow(n_train_data,i,&ncols, &cols, &vals);
    }
#if dbl_MS_read_problem >= 5
    printf("end of negative class, i is :%d,num of elements(j) is :%d\n",i,j);
#endif
//    Destroy p_data, n_data    ( They are deleted after 2nd stage of model selection )

}


//======================================================================
void ModelSelection::read_parameters(){
    param.svm_type = Config_params::getInstance()->get_svm_svm_type();
    param.kernel_type = Config_params::getInstance()->get_svm_kernel_type();
    param.degree = Config_params::getInstance()->get_svm_degree();
    param.gamma = Config_params::getInstance()->get_svm_gamma();
//    param.coef0 = Config_params::getInstance()->get_svm_coef0();
//    param.nu = Config_params::getInstance()->get_svm_nu();
    param.cache_size = Config_params::getInstance()->get_svm_cache_size();
    param.C = Config_params::getInstance()->get_svm_C();
    param.eps = Config_params::getInstance()->get_svm_eps();
//    param.p = Config_params::getInstance()->get_svm_p();
    param.shrinking = Config_params::getInstance()->get_svm_shrinking();
    param.probability = Config_params::getInstance()->get_svm_probability();
    param.nr_weight = Config_params::getInstance()->get_svm_nr_weight();
    param.weight_label = NULL;
    param.weight = NULL;
}

//======================================================================
void ModelSelection::free_model_selection(std::string caller_name){
#if dbl_MS_free_model_selection >= 1
    std::cout <<"free model selection is called by: "<< caller_name << " (free method is in [MS])" << std::endl;
#endif
    svm_free_and_destroy_model(&model);
    svm_destroy_param(&param);
    free(prob.y);
    free(prob.x);
    free(x_space);
}


//======================================================================
summary ModelSelection::make_summary(int iter,double Gmean, int num_SV_p, int num_SV_n, double C, double gamma){
    summary s = {iter, Gmean, num_SV_p, num_SV_n, num_SV_p + num_SV_n , C, gamma};
    return s;
}

//======================================================================
//int ModelSelection::select_best_1st(std::map<int, summary>& map_summary, int level){
int ModelSelection::select_best_1st(std::vector<summary>& map_summary, int level){

#if dbl_MS_SB1 >=5
    printf("\nbefore gmean sort level:%d\n",level);
    for(unsigned int i=0; i< map_summary.size();i++){
        printf("[MS][BS1] iter:%d, gmean:%g, nSV_min:%d, nSV_maj:%d, nSV_t:%d C:%g, gamma:%g loop i:%d\n",
                                        map_summary[i].iter, map_summary[i].gmean, map_summary[i].num_SV_p,
                                        map_summary[i].num_SV_n,map_summary[i].num_SV_t,map_summary[i].C,map_summary[i].gamma,i);
    }
#endif

    std::sort(map_summary.begin(),map_summary.end(),std::greater<summary>());

#if dbl_MS_SB1 >=5
    printf("\nafter gmean sort level:%d\n",level);
    for(unsigned int i=0; i< map_summary.size();i++){
        printf("[MS][BS1] iter:%d, gmean:%g, nSV_min:%d, nSV_maj:%d, nSV_t:%d C:%g, gamma:%g loop i:%d\n",
                                        map_summary[i].iter, map_summary[i].gmean, map_summary[i].num_SV_p,
                                        map_summary[i].num_SV_n, map_summary[i].num_SV_t, map_summary[i].C, map_summary[i].gamma,i);
    }
#endif

    //keep the best Gmean, remove the rest
    double best_gmean = map_summary[0].gmean;
    int n_SV_t = map_summary[0].num_SV_t;   //this is not the best
    int best_iter = map_summary[0].iter;
    int response = 0;
    /// - - - - - - more freedome to select a slightly slower gmean while the nSV is smaller - - - - - - -
//    if(Config_params::getInstance()->get_ms_gmean_sel_rel_diff() == 0){
    if(Config_params::getInstance()->get_ms_best_selection() == 0){
        for(unsigned i=1; i< map_summary.size();i++){    // start from 1 as I keep the n_SV_t for first one
            if (  ((best_gmean - map_summary[i].gmean)/best_gmean ) <   Config_params::getInstance()->get_ms_bs_gm_threshold() ){
                printf("[MS][BS1] inside loop select lower SV,i:%d, iter:%d, gmean diff:%g\n",
                                            i,map_summary[i].iter, (best_gmean - map_summary[i].gmean));
                if (n_SV_t > map_summary[i].num_SV_t){
#if dbl_MS_SB1 >=9
                    printf("[MS][BS1] inside loop n_SV_t:%d, new_n_SV_t:%d\n",n_SV_t,map_summary[i].num_SV_t);
#endif
                    n_SV_t = map_summary[i].num_SV_t;   // keep the smaller SV
                    best_iter = map_summary[i].iter;
                    response = i;
                }
            }
        }
    }

    /// - - - - - - Consider the best gmean and less number of support vectors(nSV) - - - - - - -
//    if(Config_params::getInstance()->get_ms_gmean_sel_rel_diff() == 1){
    if(Config_params::getInstance()->get_ms_best_selection() == 1){
        for(unsigned i=1; i< map_summary.size();i++){    // start from 1 as I keep the n_SV_t for first one
            if (  (best_gmean - map_summary[i].gmean) <
                                                    Config_params::getInstance()->get_ms_bs_gm_threshold() ){
#if dbl_MS_SB1 >=9
                printf("[MS][BS1] inside loop select lower SV,i:%d, iter:%d, gmean diff:%g\n",
                                            i,map_summary[i].iter,  (best_gmean - map_summary[i].gmean));
#endif
                if (n_SV_t > map_summary[i].num_SV_t){
#if dbl_MS_SB1 >=9
                    printf("[MS][BS1] inside loop n_SV_t:%d, new_n_SV_t:%d\n",
                                            n_SV_t,map_summary[i].num_SV_t);
#endif
                    n_SV_t = map_summary[i].num_SV_t;   // keep the smaller SV
                    best_iter = map_summary[i].iter;
                    response = i;
                }
            }
        }
    }//end of if for get_ms_gmean_sel_rel_diff selection

    /// - - - - - - Consider the ONLY best gmean (nSV is not important) - - - - - - -
//    if(Config_params::getInstance()->get_ms_gmean_sel_rel_diff() == 2){
    if(Config_params::getInstance()->get_ms_best_selection() == 2){
        for(unsigned i=1; i< map_summary.size();i++){    // start from 1 as I keep the n_SV_t for first one
            if (  best_gmean < map_summary[i].gmean ){
#if dbl_MS_SB1 >=9
                    printf("[MS][BS1] inside loop earlier best Gmean:%g is switched with %g\n",
                                            best_gmean,map_summary[i].gmean);
#endif
                best_iter = map_summary[i].iter;
                response = i;
            }
        }
    }

#if dbl_MS_SB1 >=5
    printf("[MS][BS1] Best selected parameter(response:%d iter:%d level:%d)\n", response, best_iter, level);
#endif
    return response;
}



//======================================================================
//int ModelSelection::select_best_both_stage(std::map<int, summary>& map_summary, int level){
int ModelSelection::select_best_both_stage(std::vector<summary>& map_summary, int level){
    return select_best_1st(map_summary,level);
}

//int ModelSelection::calc_sum_vol(Vec& v_vol, std::vector<int>& v_index, int limit ){
//    int sum = 0;
//    for (int i =0;i != limit ; i++){
//        sum += v_index[i];
//    }
//    return sum;
//}

//int ModelSelection::calc_sum_vol(Vec& v_vol, std::vector<int>& v_index ){
//    int sum = 0;
//    for (unsigned int i =0;i != v_index.size() ; i++){
//        sum += v_index[i];
//    }
//    return sum;
//}
