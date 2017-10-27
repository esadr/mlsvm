#include "loader.h"
#include "common_funcs.h"

#include <fstream>      //read and write from/to files
#include <sstream>      //read and write from/to files

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



Mat Loader::load_flann_binary_old(){                        //deprecated Jan 20, 2017
    Mat             m_ind_, m_dis_,m_ind_t_, m_dis_t_, WA;
    PetscViewer     viewer_ind_, viewer_dis_;
//    PetscBool      flg;
//    PetscErrorCode ierr;

    ETimer t_read_matrices;
//load indices vector
//    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_indices_name_,FILE_MODE_READ,&viewer_ind_);
    std::cout <<  "[LD][LFB] f_indices_name_ :"<< f_indices_name_ << std::endl;
    
//    std::string test = Config_params::getInstance()->get_p_indices_f_name();
//    f_indices_name_ = test.data();
//    std::cout <<  "[LD][LFB] f_indices_name_ :"<< f_indices_name_ << std::endl;
//    std::cout <<  "[LD][LFB] f_indices_name_  in string format:"<< Config_params::getInstance()->get_p_indices_f_name() << std::endl;
//    exit(1);
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
//    size_ = num_row;
    printf("number of rows(nodes) in matrix: %d \n",num_row);  //$$debug

/// ------------- Create the WA matrix ----------------------
    //Create Matrix WA : Weighted Adjancency
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row, Config_params::getInstance()->get_pre_init_loader_matrix(),PETSC_NULL, &WA);

    ETimer t_init_WA;
    CommonFuncs cf;
    cf.set_weight_type(Config_params::getInstance()->get_ld_weight_type(), Config_params::getInstance()->get_ld_weight_param());
    for(i =0; i <num_row; i++){                         //i would be the row number
        MatGetRow(m_ind_t_,i,&ncols_ind,&cols_ind,&vals_ind);             //ncols : number if non-zeros in the row
        MatGetRow(m_dis_t_,i,&ncols_dis,&cols_dis,&vals_dis);

        for (j=0; j<ncols_ind; j++) {//Notice: as I use indices for j, for dists I should reduce it by one
            weight_ = cf.convert_distance_to_weight(vals_dis[j-1]);

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

/// ------------- Create the WA matrix ----------------------
    //Create Matrix WA : Weighted Adjancency
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row, Config_params::getInstance()->get_pre_init_loader_matrix(),PETSC_NULL, &WA);

    ETimer t_init_WA;
    CommonFuncs cf;
    cf.set_weight_type(Config_params::getInstance()->get_ld_weight_type(), Config_params::getInstance()->get_ld_weight_param());
    for(i =0; i <num_row; i++){                         //i would be the row number
        MatGetRow(m_ind_,i,&ncols_ind,&cols_ind,&vals_ind);             //ncols : number if non-zeros in the row
        MatGetRow(m_dis_,i,&ncols_dis,&cols_dis,&vals_dis);

        for (j=0; j<ncols_ind; j++) {//Notice: as I use indices for j, for dists I should reduce it by one
            weight_ = cf.convert_distance_to_weight(vals_dis[j-1]);

            if(i != (vals_ind[j] - 1)  ){   //if it's not a loop to itself
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




/*
 * input: Metis v4 file format weighted, the distance are probably distance?
 *
 * Note1: the values as the weight is the strenght which is calculated as the inverse of the distance
 * Note2: the indices in the input starts at 1 instead of 0, so for counting we can ignore it
 *          but for loading the values we need to reduce them by 1
 * output: weighted Adjacency matrix
 */

// - - - - read the Metis file - - - -
// Number of rows and columns is the first value in the first line of the input file
// The nnz for each row is required which is not known in advance
// Solution: 2 passes through the file,
//          first: for counting number of edges for each node,
//          second: initialize the values in the WA matrix

Mat Loader::load_metis(){
    printf("[LD] load metis (not tested) Exit!!!\n");
    exit(1);
    std::string in_file_name = Config_params::getInstance()->get_ds_name(); //@@
    /// -------------- 1st pass (over the input file) --------------
    ETimer t_readfile;
    // - - - - read the Metis file - - - -
    std::ifstream file(in_file_name);
    std::string str, item;

    //read the first line to get number of rows
    std::getline(file, str);
    std::stringstream ss(str);
    std::vector<std::string> graph_info;
    while (getline(ss, item, ' ')) {    //split by space, tokens[0] is label, and the rest of them are index:value
        graph_info.push_back(item);
    }
    PetscInt num_vertices = stoi(graph_info[0]);
    //PetscInt num_edges = stoi(graph_info[1]);

    /* determine number of nonzeros per row in the new matrix */
    PetscInt *cnt, curr_vertex=0;
    PetscMalloc1(num_vertices, &cnt);

    while (std::getline(file, str))// starts from the 2nd row in the file which is counted in index 0 of cnt
    {
        std::stringstream ss(str);
        item = "";
        std::vector<std::string> edge_info;
        while (getline(ss, item, ' ')) {    //split by space, tokens[0] is label, and the rest of them are index:value
            edge_info.push_back(item);
        }
        cnt[curr_vertex] = edge_info.size() / 2;
        curr_vertex++;
    }
    file.close();

    //Create Matrix WA : Weighted Adjancency
    Mat WA;
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_vertices, num_vertices, 0, cnt, &WA);

    /// -------------- 2nd pass (over the input file) --------------
    CommonFuncs cf;
    cf.set_weight_type(Config_params::getInstance()->get_ld_weight_type(), Config_params::getInstance()->get_ld_weight_param());
    // - - - - read the Metis file - - - -
    file.open(in_file_name);

    //ignore the first line
    std::getline(file, str);
    curr_vertex = 0;
    while (std::getline(file, str))// starts from the 2nd row in the file which is counted in index 0 of cnt
    {
        std::stringstream ss(str);
        item = "";
        std::vector<std::string> edge_info;
        while (getline(ss, item, ' ')) {    //split by space, tokens[0] is label, and the rest of them are index:value
            edge_info.push_back(item);
        }

        for(unsigned int i=0; i+1 < edge_info.size(); i+=2){ //check i+1 to make sure inside the loop there is no out of bound index
            //neighbour vertex: edge_info[i] - 1
            //edge distance: edge_info[i+1]
            int neighbour_vertex = stoi(edge_info[i]) - 1;
            double edge_weight = cf.convert_distance_to_weight(stod(edge_info[i+1]));

            // the self loop and lower triangular elements are ignored
            // the lower triangular will be build from the uppper triangular later
            if (curr_vertex < neighbour_vertex )      //only upper triangular
                MatSetValue(WA, curr_vertex, neighbour_vertex, edge_weight, INSERT_VALUES);
        }
        curr_vertex++;
    }
    file.close();
    t_readfile.stop_timer("[LD][LM] reading input file twice");

    ETimer t_WA_assembly;
    MatAssemblyBegin(WA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(WA,MAT_FINAL_ASSEMBLY);
#if dbl_LD_LM >=3
    printf("[LD][LM] insert values to WA is finished\n");      //$$debug
    #if dbl_LD_LM >= 9
        printf("WA Matrix (only triangular part):\n");                                               //$$debug
        MatView(WA,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
#endif
    t_WA_assembly.stop_timer("[LD][LM] WA assembly");

    ETimer t_WA_complete;
    Mat WA_t;
    MatTranspose(WA,MAT_INITIAL_MATRIX,&WA_t);
    MatAYPX(WA,1,WA_t,DIFFERENT_NONZERO_PATTERN);
    t_WA_complete.stop_timer("[LD][LM] WA add to its transpose");
    MatDestroy(&WA_t);
#if dbl_LD_LM >=3
    #if dbl_LD_LM >= 7
        printf("[LD][LM] WA Matrix After add to it's transpose:\n");                   //$$debug
        MatView(WA,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
    printf("[LD][LM] WA is created completely \n");
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
        printf("m_NN_idx Matrix :\n");                   //$$debug
        MatView(m_NN_idx,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

        printf("m_NN_dis Matrix :\n");                   //$$debug
        MatView(m_NN_dis,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    }


// ------------- Create the WA matrix ----------------------
    //Create Matrix WA : Weighted Adjancency
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row, Config_params::getInstance()->get_pre_init_loader_matrix(),PETSC_NULL, &m_WA);

    ETimer t_init_WA;
    CommonFuncs cf;
    cf.set_weight_type(Config_params::getInstance()->get_ld_weight_type(), Config_params::getInstance()->get_ld_weight_param());
    for(i =0; i <num_row; i++){                         //i would be the row number
        MatGetRow(m_NN_idx,i,&ncols_ind,&cols_ind,&vals_ind);             //ncols : number if non-zeros in the row
        MatGetRow(m_NN_dis,i,&ncols_dis,&cols_dis,&vals_dis);

        for (j=0; j<ncols_ind; j++) {//Notice: as I use indices for j, for dists I should reduce it by one
            weight_ = cf.convert_distance_to_weight(vals_dis[j-1]);
//            std::cout << "[LD][CWAM] i:"<< i <<", j"<< j << ",(vals_ind[j] - 1):" << (vals_ind[j] - 1) << std::endl;
            if(i != (vals_ind[j] - 1)  ){   //if it's not a loop to itself
                if (vals_ind[j] - 1 > i ){      //if it's in upper triangular
                    MatSetValue(m_WA,i,vals_ind[j] - 1 ,weight_,INSERT_VALUES);
                }else{
                    MatSetValue(m_WA,vals_ind[j] - 1, i ,weight_,INSERT_VALUES);  // switched item (needed when I fill only a triangular)
                }
            }
        }

        MatRestoreRow(m_NN_idx,i,&ncols_ind,&cols_ind,&vals_ind);
        MatRestoreRow(m_NN_dis,i,&ncols_dis,&cols_dis,&vals_dis);
    }
if(debug_status) exit(1);
    MatAssemblyBegin(m_WA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_WA,MAT_FINAL_ASSEMBLY);
#if dbl_LD_CWAM >=3
    printf("[LD][CWAM] insert values to m_WA is finished\n");      //$$debug
    #if dbl_LD_CWAM >= 7
        printf("[LD][CWAM] WA Matrix (only triangular part):\n");                                               //$$debug
        MatView(m_WA,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
#endif
    t_init_WA.stop_timer("insert triangular part of WA");

    ETimer t_WA_complete;
    Mat m_WA_t;
    MatTranspose(m_WA,MAT_INITIAL_MATRIX,&m_WA_t);
    MatAYPX(m_WA,1,m_WA_t,DIFFERENT_NONZERO_PATTERN);
    t_WA_complete.stop_timer("t_m_WA_complete");
    MatDestroy(&m_WA_t);
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




































