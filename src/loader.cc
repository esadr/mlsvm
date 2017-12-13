#include "loader.h"
#include "common_funcs.h"
//#include "//    ETimer.h"

//Loader::Loader(const char * f_indices_file_name, const char * f_dists_file_name){
Loader::Loader(const std::string &f_indices_file_name, const std::string &f_dists_file_name){
    f_indices_name_ = f_indices_file_name.c_str();
    f_dists_name_   = f_dists_file_name.c_str();
    std::cout << "[LD][Constructor] indice file: "<< f_indices_file_name <<std::endl;
}

/**
 * @param : WA
 *
 */
Loader::Loader(const char * f_indices_name, const char * ds_indices_name,
               const char * f_dists_name, const char * ds_dists_name){
//Loader::Loader(char * f_indices_name){
    f_indices_name_ = f_indices_name;
    ds_indices_name_ = ds_indices_name;
    f_dists_name_ = f_dists_name;
    ds_dists_name_ = ds_dists_name;
}



Mat Loader::load_WA_binary(const char * f_name){    //Import WA directly
    Mat             WA_;
    PetscViewer     viewer_WA_;
    PetscInt num_row=0;
    
    ETimer t_read_matrices;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_name,FILE_MODE_READ,&viewer_WA_);
    MatCreate(PETSC_COMM_WORLD,&WA_);
    MatLoad(WA_,viewer_WA_);
    PetscViewerDestroy(&viewer_WA_);        //destroy the viewer

    t_read_matrices.stop_timer("[Loader::load_WA_binary] reading WA matrix");

    MatGetSize(WA_,&num_row,0);    //m returns the number of rows globally
//    size_ = num_row;

#if dbl_LD_LWAB >= 7
//    printf("WA Matrix:\n");                                               //$$debug
//    MatView(WA_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif

    ETimer t_WA_complete;
    Mat WA_t;
    MatTranspose(WA_,MAT_INITIAL_MATRIX,&WA_t);
    MatAYPX(WA_,1,WA_t,DIFFERENT_NONZERO_PATTERN);
    t_WA_complete.stop_timer("t_WA_complete");

#if dbl_LD_LWAB >= 7
//    PetscViewer     viewer_WA;
//    PetscViewerBinaryOpen(PETSC_COMuM_WORLD,"WA_export.dat",FILE_MODE_WRITE,&viewer_WA);
//    MatView(WA,viewer_WA);

   printf("WA Matrix After add to it's transpose:\n");                   //$$debug
   MatView(WA_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
   PetscViewerDestroy(&viewer_WA_);
#endif
    return WA_;
}




Mat Loader::load_flann_binary(){
    Mat             m_ind_, m_dis_, WA;
    PetscViewer     viewer_ind_, viewer_dis_;

    ETimer t_read_matrices;
//load indices vector
    std::cout <<  "[LD][LFB] f_indices_name_ :"<< f_indices_name_ << std::endl;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_indices_name_,FILE_MODE_READ,&viewer_ind_);
    MatCreate(PETSC_COMM_WORLD,&m_ind_);
    MatLoad(m_ind_,viewer_ind_);
    PetscViewerDestroy(&viewer_ind_);        //destroy the viewer

//load dists vector
    std::cout <<  "[LD][LFB] f_dists_name_ :"<< f_dists_name_ << std::endl;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_dists_name_,FILE_MODE_READ,&viewer_dis_);
    MatCreate(PETSC_COMM_WORLD,&m_dis_);
    MatLoad(m_dis_,viewer_dis_);
    PetscViewerDestroy(&viewer_dis_);        //destroy the viewer
    t_read_matrices.stop_timer("[Loader::load_flann_binary] reading both matrices");


#if dbl_LD_LFB >= 7
    printf("m_ind Matrix:\n");                                               //$$debug
    MatView(m_ind_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

    printf("m_dis Matrix:\n");                                               //$$debug
    MatView(m_dis_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debu
#endif

    PetscInt num_row=0,i,j, ncols_ind,ncols_dis;
    const PetscInt    *cols_ind, *cols_dis;                        //if not NULL, the column numbers
    const PetscScalar *vals_ind, *vals_dis;
    PetscScalar weight_ = 0;    //for differnet distance
    MatGetSize(m_ind_,&num_row,0);    //m returns the number of rows globally = number of nodes
//    size_ = num_row;
    printf("number of rows(nodes) in matrix: %d \n",num_row);  //$$debug

    /// -------- Calc upper bound of number of non-zeros in each row ---------------
    //approximate method, since there might be only i->j or both (i,j) and (j,i)
    //for simplicity, we don't detect the both condition as one which might increase the number as large as twice
    // while it is not the exact number, it is larger number (not more than twice as large) which lead the number of non-zeros
    ETimer t_calc_nnz;
    std::vector<PetscInt> node_stat_approximate(num_row,0);       //make a vector of zero's for all data points

    for(i =0; i <num_row; i++){                         //i would be the row number
        MatGetRow(m_ind_,i,&ncols_ind,&cols_ind,&vals_ind);             //ncols : number if non-zeros in the row
        MatGetRow(m_dis_,i,&ncols_dis,&cols_dis,&vals_dis);

        for (j=0; j<ncols_ind; j++) {//Notice: as I use indices for j, for dists I should reduce it by one
            //if it's not a loop to itself      &&   the distance is not exact zero between i,j
            if(i != (vals_ind[j] - 1) && (vals_dis[j]) ){
                node_stat_approximate[i]+=1;    //we increase both i and j
                node_stat_approximate[j]+=1;
            }
        }
        MatRestoreRow(m_ind_,i,&ncols_ind,&cols_ind,&vals_ind);
        MatRestoreRow(m_dis_,i,&ncols_dis,&cols_dis,&vals_dis);
    }
    t_calc_nnz.stop_timer("[LD][LFB] calc upper bound of number of non-zeros in each row for WA ");

    /// ------------- Create the WA matrix ----------------------
    //Create Matrix WA : Weighted Adjancency
//    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row, Config_params::getInstance()->get_pre_init_loader_matrix(),PETSC_NULL, &WA); //depreacted v0.0.45 121117_1200
    PetscInt* nnz = &node_stat_approximate[0];  //https://stackoverflow.com/a/2923290/2674061
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row, PETSC_NULL,nnz, &WA);


    ETimer t_init_WA;
    CommonFuncs cf;
    cf.set_weight_type(Config_params::getInstance()->get_ld_weight_type(), Config_params::getInstance()->get_ld_weight_param());
    for(i =0; i <num_row; i++){                         //i would be the row number
        MatGetRow(m_ind_,i,&ncols_ind,&cols_ind,&vals_ind);             //ncols : number if non-zeros in the row
        MatGetRow(m_dis_,i,&ncols_dis,&cols_dis,&vals_dis);

        for (j=0; j<ncols_ind; j++) {//Notice: as I use indices for j, for dists I should reduce it by one
            weight_ = cf.convert_distance_to_weight(vals_dis[j-1]);
            //if it's not a loop to itself    && the distance is not exact zero between i,j
            if(i != (vals_ind[j] - 1)  && vals_dis[j]){
                if (vals_ind[j] - 1 > i ){      //if it's in upper triangular
                    MatSetValue(WA,i,vals_ind[j] - 1 ,weight_,INSERT_VALUES);

                }else{
                    MatSetValue(WA,vals_ind[j] - 1, i ,weight_,INSERT_VALUES);  // switched item (needed when I fill only a triangular)
                }
            }
        }
        MatRestoreRow(m_ind_,i,&ncols_ind,&cols_ind,&vals_ind);
        MatRestoreRow(m_dis_,i,&ncols_dis,&cols_dis,&vals_dis);
    }

    MatAssemblyBegin(WA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(WA,MAT_FINAL_ASSEMBLY);
#if dbl_LD_LFB >=3
    printf("[LD][LFB] insert values to WA is finished\n");      //$$debug
    #if dbl_LD_LFB >= 9
        printf("[LD][LFB] WA Matrix (only triangular part):\n");                                               //$$debug
        MatView(WA,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
#endif
    t_init_WA.stop_timer("[LD][LFB] insert triangular part of WA");

    ETimer t_WA_complete;
    Mat WA_t;
    MatTranspose(WA,MAT_INITIAL_MATRIX,&WA_t);
    MatAYPX(WA,1,WA_t,DIFFERENT_NONZERO_PATTERN);
    t_WA_complete.stop_timer("[LD][LFB] t_WA_complete");
    MatDestroy(&WA_t);
#if dbl_LD_LFB >=3
    #if dbl_LD_LFB >= 7
        printf("[LD][LFB] WA Matrix After add to it's transpose:\n");                   //$$debug
        MatView(WA,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
    printf("[LD][LFB] WA is created completely \n");
#endif
    return WA;
}







void Loader::create_WA_matrix(Mat& m_NN_idx,Mat& m_NN_dis,Mat& m_WA,const std::string& info,bool debug_status){
    ETimer t_all;

    PetscInt num_row=0,i,j, ncols_ind,ncols_dis;
    const PetscInt    *cols_ind, *cols_dis;                        //if not NULL, the column numbers
    const PetscScalar *vals_ind, *vals_dis;
    PetscScalar weight_ = 0;    //for differnet distance
    MatGetSize(m_NN_idx,&num_row,0);    //m returns the number of rows globally = number of nodes
#if dbl_LD_CWAM >=1
    printf("[LD][CWAM] number of rows(nodes) in NN matrix: %d \n",num_row);  //$$debug
#endif
    if(debug_status){
        printf("[LD][CWAM] m_NN_idx Matrix :\n");                   //$$debug
        MatView(m_NN_idx,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

        printf("[LD][CWAM] m_NN_dis Matrix :\n");                   //$$debug
        MatView(m_NN_dis,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    }

    /// -------- Calc upper bound of number of non-zeros in each row ---------------
    //exact method using vector of set
    ETimer t_calc_nnz;
    std::vector<std::set<int>> node_stat_approximate(num_row,std::set<int>());       //make a vector of zero's for all data points
//    std::cout << "[LD][CWAM] number of columns: " << num_row << std::endl;

    for(i =0; i <num_row; i++){                         //i would be the row number
        MatGetRow(m_NN_idx,i,&ncols_ind,&cols_ind,&vals_ind);             //ncols : number if non-zeros in the row
        MatGetRow(m_NN_dis,i,&ncols_dis,&cols_dis,&vals_dis);
//        std::cout << "row " << i << std::endl;
        for (j=0; j<ncols_ind; j++) {//Notice: as I use indices for j, for dists I should reduce it by one
//            std::cout << "col " << j << std::endl;

            //it's not a loop to itself    (the [KF][filterNN] filters the loops) and distance zero
//            if (vals_ind[j] > i ){      //if it's in upper triangular
//                std::cout << "if " << i << ","<< vals_ind[j] << std::endl;
                node_stat_approximate[i].insert(vals_ind[j]);
//            }
//            else{
//                std::cout << "else " << vals_ind[j] <<","<< i << std::endl;
                node_stat_approximate[vals_ind[j]].insert(i);
//            }
        }
        MatRestoreRow(m_NN_idx,i,&ncols_ind,&cols_ind,&vals_ind);
        MatRestoreRow(m_NN_dis,i,&ncols_dis,&cols_dis,&vals_dis);
    }
    t_calc_nnz.stop_timer("[LD][CWAM] calc number of non-zeros in each row for WA ");

//    std::cout << "[LD][CWAM] number of columns: " << num_row <<", non_zero's array:" << std::endl;
    std::vector<PetscInt> v_nnz(num_row,0);
    int cnt=0;
    for(std::set<int> s:node_stat_approximate){
//        std::cout << cnt <<","<< s.size() << std::endl;
        v_nnz[cnt] = s.size() ;
        cnt+=1;
    }

    PetscInt* nnz = &v_nnz[0];
#if dbl_LD_CWAM >=7
    for(int i =0; i < v_nnz.size(); i++){
        std::cout <<"malloc nnz " << i <<","<< nnz[i] << std::endl;
    }
#endif

    /// ------------- Create the WA matrix ----------------------
    //Create Matrix WA : Weighted Adjancency
//    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row, Config_params::getInstance()->get_pre_init_loader_matrix(),PETSC_NULL, &WA); //depreacted v0.0.45 121117_1200
      //https://stackoverflow.com/a/2923290/2674061
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row, PETSC_NULL,nnz, &m_WA);

    ETimer t_init_WA;
    CommonFuncs cf;
    cf.set_weight_type(Config_params::getInstance()->get_ld_weight_type(), Config_params::getInstance()->get_ld_weight_param());
    for(i =0; i <num_row; i++){                         //i would be the row number
        MatGetRow(m_NN_idx,i,&ncols_ind,&cols_ind,&vals_ind);             //ncols : number if non-zeros in the row
        MatGetRow(m_NN_dis,i,&ncols_dis,&cols_dis,&vals_dis);

//        std::cout << "\n[LD][CWAM] i:"<< i << std::endl;
        for (j=0; j<ncols_ind; j++) {//Notice: as I use indices for j, for dists I should reduce it by one
#if dbl_LD_CWAM >=7
            std::cout << "\nj:"<< j <<", cols_ind[j]:" << cols_ind[j] << ", cols_dis[j]:" << cols_dis[j]
                      << ", other node:"<< vals_ind[j] << ", distance:"<< vals_dis[j];
#endif
            weight_ = cf.convert_distance_to_weight(vals_dis[j]);
            //it's not a loop to itself    (the [KF][filterNN] filters the loops) and distance zero
//            if (i < vals_ind[j]){      //if it's in upper triangular
                MatSetValue(m_WA,i, vals_ind[j],weight_,INSERT_VALUES);
//                printf("if setvalue: (%d,%d)=%g\n",i, vals_ind[j], vals_dis[j]);
//            }else{
                MatSetValue(m_WA, vals_ind[j], i ,weight_,INSERT_VALUES);  // switched item (needed when I fill only a triangular)
//                printf("else setvalue: (%d,%d)=%g\n", vals_ind[j], i, vals_dis[j]);
//            }
        }

        MatRestoreRow(m_NN_idx,i,&ncols_ind,&cols_ind,&vals_ind);
        MatRestoreRow(m_NN_dis,i,&ncols_dis,&cols_dis,&vals_dis);
    }
//    exit(1);
//if(debug_status)
//    exit(1);

    MatAssemblyBegin(m_WA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_WA,MAT_FINAL_ASSEMBLY);
//#if dbl_LD_CWAM >=3
//    printf("[LD][CWAM] insert values to m_WA is finished\n");      //$$debug
//    #if dbl_LD_CWAM >= 7
//        printf("[LD][CWAM] WA Matrix (only triangular part):\n");                                               //$$debug
//        MatView(m_WA,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
//    #endif
//#endif
    t_init_WA.stop_timer("[LD][CWAM] insert triangular part of WA");

//    exit(1);

//    ETimer t_WA_complete;
//    Mat m_WA_t;
//    MatTranspose(m_WA,MAT_INITIAL_MATRIX,&m_WA_t);
//    MatAYPX(m_WA,1,m_WA_t,DIFFERENT_NONZERO_PATTERN);
//    t_WA_complete.stop_timer("[LD][CWAM] t_m_WA_complete");
//    MatDestroy(&m_WA_t);
#if dbl_LD_CWAM >=3
    #if dbl_LD_CWAM >= 7
        printf("[LD][CWAM] WA Matrix After add to it's transpose:\n");                   //$$debug
        MatView(m_WA,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
    printf("[LD][CWAM] WA is created completely \n");
#endif
    t_all.stop_timer("[LD][CWAM] whole WA matrix for class ", info);
}



















Mat Loader::load_norm_data_sep(const std::string f_name){    //load normalized data for each class seperately
    Mat             m_data_;
    PetscViewer     viewer_data_;


    ETimer t_read_matrix;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_name.c_str(),FILE_MODE_READ,&viewer_data_);
    MatCreate(PETSC_COMM_WORLD,&m_data_);
    MatLoad(m_data_,viewer_data_);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer

    t_read_matrix.stop_timer("[LD][LNDS] reading data matrix");

#if dbl_LD_LNDS >= 5
    PetscInt num_row=0;
    MatGetSize(m_data_,&num_row,0);    //m returns the number of rows globally
    printf("[LD][LNDS] number of rows in data: %d \n", num_row);
#endif
#if dbl_LD_LNDS >= 7
    printf("[LD][LNDS] data Matrix:\n");                                               //$$debug
    MatView(m_data_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif

    return m_data_;
}




/*
 * This will be used for the first level of coarsening to initialize the volume (usually with 1)
 * @param val
 * PetscScalar type that use to initialize all components of the vector
 * @return
 * vector with size that come from load_flann and the value of all components are val
 */
Vec Loader::init_volume(PetscScalar val, PetscInt num_elements){
    Vec vol_;
    VecCreateSeq(PETSC_COMM_SELF, num_elements,&vol_);
    VecSet(vol_,val);           //Sets all components of a vector to a single scalar value.
    VecAssemblyBegin(vol_);
    VecAssemblyEnd(vol_);
    return vol_;
}

//Vec Loader::init_volume(PetscScalar val, int pref_size){
//    this->size_ = pref_size;
//    return init_volume(val);
//}


Mat Loader::load_test_data(const char * f_name, int& num_node, int& num_elem){    //load data
    Mat             t_data_;
    PetscViewer     viewer_data_;
    PetscInt num_row=0;
    PetscInt i, j, ncols;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, f_name,FILE_MODE_READ,&viewer_data_);
    MatCreate(PETSC_COMM_WORLD,&t_data_);
    MatLoad(t_data_,viewer_data_);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer


    MatGetSize(t_data_,&num_row,0);    //m returns the number of rows globally
    printf("number of rows in test data: %d \n", num_row);
    num_node = num_row;
    num_elem =0 ;
    for (i=0; i< num_row;i++){
        MatGetRow(t_data_,i,&ncols, &cols, &vals);
        for (j=0; j< ncols; j++) {
            num_elem++;
        }
        num_elem++;    //for the end of line index
        MatRestoreRow(t_data_,i,&ncols, &cols, &vals);
    }
    printf("number of test nodes:%d, elems:%d\n",num_node,num_elem);
//    printf("test Matrix:\n");                                               //$$debug
//    MatView(t_data_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug


    return t_data_;
}



Mat Loader::load_train_data(const char * f_name, int& num_node, int& num_elem){    //load data
    Mat             t_data_;
    PetscViewer     viewer_data_;
    PetscInt num_row=0;
    PetscInt i, j, ncols;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, f_name,FILE_MODE_READ,&viewer_data_);
    MatCreate(PETSC_COMM_WORLD,&t_data_);
    MatLoad(t_data_,viewer_data_);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer


    MatGetSize(t_data_,&num_row,0);    //m returns the number of rows globally
    printf("number of rows in data: %d \n", num_row);
    num_node = num_row;
    num_elem =0 ;
    for (i=0; i< num_row;i++){
        MatGetRow(t_data_,i,&ncols, &cols, &vals);
        for (j=0; j< ncols; j++) {
            num_elem++;
        }
        num_elem++;    //for the end of line index
        MatRestoreRow(t_data_,i,&ncols, &cols, &vals);
    }
    printf("nodes:%d, elems:%d\n",num_node,num_elem);
//    printf("t_data Matrix:\n");                                               //$$debug
//    MatView(t_data_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

    return t_data_;
}

Mat Loader::read_input_matrix(const std::string f_name){
    Mat             t_data_;
    PetscViewer     viewer_data_;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, f_name.c_str(),FILE_MODE_READ,&viewer_data_);
    MatCreate(PETSC_COMM_WORLD,&t_data_);
    MatLoad(t_data_,viewer_data_);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer


//    printf("Sample Matrix:\n");                                               //$$debug
//    MatView(t_data_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    return t_data_;
}

Vec Loader::read_input_vector(const std::string f_name){
    Vec             t_data_;
    PetscViewer     viewer_data_;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, f_name.c_str(),FILE_MODE_READ,&viewer_data_);
    VecCreate(PETSC_COMM_WORLD,&t_data_);
    VecLoad(t_data_,viewer_data_);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer


//    printf("Sample Vector:\n");                                               //$$debug
//    VecView(t_data_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    return t_data_;
}


//Vec Loader::convert_matrix_to_vector(Mat& m_input){
//    PetscInt num_row, num_col;
//    MatGetSize(m_input,&num_row,&num_col);
//    if (num_col != 1){
//        printf("You should use this to convert a matrix with only 1 column to vector\n[Exit]");
//        exit(1);
//    }

//    PetscInt i, ncols;
//    const PetscInt    *cols;                        //if not NULL, the column numbers
//    const PetscScalar *vals;
//    PetscScalar arr_vals[num_row];
//    Vec v_res;
//    VecCreateSeq(PETSC_COMM_SELF,num_row,&v_res);
//    for(i=0; i < num_row; i++){
//        MatGetRow(m_input,i,&ncols, &cols, &vals);
//        arr_vals[i] = vals[0];
//        MatRestoreRow(m_input,i,&ncols, &cols, &vals);
//    }
//    VecPlaceArray(v_res, arr_vals);
//    printf("[LD] convert matrix to vector:\n");                                            //$$debug
//    VecView(v_res,PETSC_VIEWER_STDOUT_WORLD);                             //$$debug
//    return v_res;
//}




































