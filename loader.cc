#include "loader.h"
//#include "//    ETimer.h"

//Loader::Loader(const char * f_indices_file_name, const char * f_dists_file_name){
Loader::Loader(const std::string f_indices_file_name, const std::string f_dists_file_name){
    f_indices_name_ = f_indices_file_name.c_str();
    f_dists_name_   = f_dists_file_name.c_str();

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
    size_ = num_row;

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



Mat Loader::load_flann_binary(){                        //the current method July 2015
    Mat             m_ind_, m_dis_,m_ind_t_, m_dis_t_, WA;
    PetscViewer     viewer_ind_, viewer_dis_;
//    PetscBool      flg;
//    PetscErrorCode ierr;

    ETimer t_read_matrices;
//load indices vector
//    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_indices_name_,FILE_MODE_READ,&viewer_ind_);
    std::cout <<  "[LD][LFB] f_indices_name_ :"<< f_indices_name_ << std::endl;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_indices_name_,FILE_MODE_READ,&viewer_ind_);
//    CHKERRQ(ierr);
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

    ETimer t_transpose_matrices;
/* the flann result for each node is saved as a column format,
 * and I use CSR(row) so, I should use their transpose */
    MatTranspose(m_ind_,MAT_INITIAL_MATRIX,&m_ind_t_);
    MatTranspose(m_dis_,MAT_INITIAL_MATRIX,&m_dis_t_);
    MatDestroy(&m_ind_);
    MatDestroy(&m_dis_);

#if dbl_LD_LFB >= 7
    printf("m_ind_t Matrix:\n");                                               //$$debug
    MatView(m_ind_t_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

    printf("m_dis_t Matrix:\n");                                               //$$debug
    MatView(m_dis_t_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debu
#endif
    t_transpose_matrices.stop_timer("transposed");

    PetscInt num_row=0,i,j, ncols_ind,ncols_dis;
    const PetscInt    *cols_ind, *cols_dis;                        //if not NULL, the column numbers
    const PetscScalar *vals_ind, *vals_dis;
    PetscScalar weight_ = 0;    //for differnet distance
    MatGetSize(m_ind_t_,&num_row,0);    //m returns the number of rows globally = number of nodes
    size_ = num_row;
    printf("number of rows(nodes) in matrix: %d \n",num_row);  //$$debug

/// ------------- Create the WA matrix ----------------------

    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row,
                    Config_params::getInstance()->get_pre_init_loader_matrix(),
                    PETSC_NULL, &WA);  //Create Matrix WA : Weighted Adjancency
//    pre_init_loader_matrix is defined on the config_params
    ETimer t_init_WA;
    for(i =0; i <num_row; i++){                         //i would be the row number
        MatGetRow(m_ind_t_,i,&ncols_ind,&cols_ind,&vals_ind);             //ncols : number if non-zeros in the row
        MatGetRow(m_dis_t_,i,&ncols_dis,&cols_dis,&vals_dis);

        for (j=0; j<ncols_ind; j++) {//Notice: as I use indices for j, for dists I should reduce it by one

            weight_ = calc_distance(vals_dis[j-1]);

            if(i != (vals_ind[j] - 1)  ){   //if it's not a loop to itself
                if (vals_ind[j] - 1 > i ){      //if it's in upper triangular
                    MatSetValue(WA,i,vals_ind[j] - 1 ,weight_,INSERT_VALUES);
                }else{
                    MatSetValue(WA,vals_ind[j] - 1, i ,weight_,INSERT_VALUES);  // switched item (needed when I fill only a triangular)
                }
            }
        }
        MatRestoreRow(m_ind_t_,i,&ncols_ind,&cols_ind,&vals_ind);
        MatRestoreRow(m_dis_t_,i,&ncols_dis,&cols_dis,&vals_dis);
    }

    MatAssemblyBegin(WA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(WA,MAT_FINAL_ASSEMBLY);
#if dbl_LD_LFB >=3
    printf("[LD][LFB] insert values to WA is finished\n");      //$$debug
    #if dbl_LD_LFB >= 9
        printf("WA Matrix (only triangular part):\n");                                               //$$debug
        MatView(WA,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
#endif
    t_init_WA.stop_timer("insert triangular part of WA");

    ETimer t_WA_complete;
    Mat WA_t;
    MatTranspose(WA,MAT_INITIAL_MATRIX,&WA_t);
    MatAYPX(WA,1,WA_t,DIFFERENT_NONZERO_PATTERN);
    t_WA_complete.stop_timer("t_WA_complete");
    MatDestroy(&WA_t);
#if dbl_LD_LFB >=3
    #if dbl_LD_LFB >= 7
        printf("WA Matrix After add to it's transpose:\n");                   //$$debug
        MatView(WA,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
    printf("[loader] WA is created completely \n");
#endif
    return WA;
}


PetscScalar Loader::calc_distance(PetscScalar raw_weight){  // for now it suppose the input is:  (Euclidean Dist)^2
    if(this->weight_type == 1 ){ //Flann (square Euclidean distance)
        return ( 1 / (sqrt(raw_weight) + 0.00001 ) );    //
    }

    if(this->weight_type == 2 ){ // Gaussian distance)
//        std::cout << "[LD][calc_distance] exit" << std::endl;
//        exit(1);
        return ( exp((-1) * raw_weight * this->weight_gamma)  );    //
    }
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





Vec Loader::init_volume(PetscScalar val){
    /*
     * This will be used for the first level of coarsening to initialize the volume (usually with 1)
     * @param val
     * PetscScalar type that use to initialize all components of the vector
     * @return
     * vector with size that come from load_flann and the value of all components are val
     */
    Vec vol_;
    VecCreateSeq(PETSC_COMM_SELF,size_,&vol_);
    VecSet(vol_,val);           //Sets all components of a vector to a single scalar value.
    VecAssemblyBegin(vol_);
    VecAssemblyEnd(vol_);

    return vol_;
}

Vec Loader::init_volume(PetscScalar val, int pref_size){
    this->size_ = pref_size;
    return init_volume(val);
}


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




































