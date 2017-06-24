#include "partitioning.h"
#include "loader.h"
#include "common_funcs.h"       //only for debugging
#include "coarsening.h"
#include "string"
#include <limits>           //std::numeric_limits<double>::max()
//#include "chrono"
//#include "thread"





//void Partitioning::main(Mat& m_data_p, Vec& v_vol_p, IS& is_p, Mat& m_data_n, Vec& v_vol_n, IS& is_n, solution& refined_solution){

//}


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

int Partitioning::get_parts(Mat& m_WA, Vec& v_vol, int num_req_parts, int level, std::vector<std::vector<PetscInt>>& vv_parts, Mat& m_parts, unsigned int metis_seed){
    // - - - - - - define metis parameters - - - - - -
    PetscInt num_vertex, num_edge=0, num_col;
    MatGetSize(m_WA, &num_vertex, &num_col);

    //metis API needs to count both (i,j) and (j,i) and later it double it again which is 4 times unique edges in my case
    PetscInt            i,j, ncols;
    const PetscInt      *cols;
    const PetscScalar   *vals;
    for(i=0; i < num_vertex; i++){
        MatGetRow(m_WA,i,&ncols,&cols,&vals);
        for(j=0; j <ncols; j++){
            if(vals[j] != 0)
                num_edge++;
        }
        MatRestoreRow(m_WA,i,&ncols,&cols,&vals);
    }

    idx_t nvtxs= num_vertex, ncon=1;
    idx_t *xadj=NULL, *vsize=NULL, *adjncy=NULL;
    idx_t *vwgt=NULL, *adjwgt=NULL;
    idx_t nparts = num_req_parts;

#if dbl_PR_get_parts >= 1
    std::cout << "[PR][get_parts] number of parts are: " << nparts << std::endl;
#endif
    real_t *tpwgts = NULL, *ubvec= NULL;
//    idx_t *options=NULL;
    // - - - - set the options - - - -
    idx_t options[METIS_NOPTIONS];
    METIS_SetDefaultOptions(options);
//    options[METIS_OPTION_DBGLVL] = 1+2;   //set the level of debug information
    options[METIS_OPTION_OBJTYPE] = METIS_OBJTYPE_CUT;
    options[METIS_OPTION_NUMBERING] = 0;
    options[METIS_OPTION_UFACTOR] = 30;
    options[METIS_OPTION_SEED] = metis_seed;

    idx_t objval;
    idx_t part[num_vertex];         //this is a vector of part as a result of metis

    xadj = new idx_t[num_vertex + 1 ];      // n + 1 page 23 manual (n is number of vertices)
//    int size_adj = 2 * num_edge * 1.1;
    adjncy = new idx_t[2 * num_edge ];       // 2m page 23 manual (m is number of edges)
    adjwgt= new idx_t[2 * num_edge];        // 2m page 24 manual (m is number of edges)
//    adjncy = new idx_t[size_adj];       // 2m page 23 manual (m is number of edges)
//    adjwgt= new idx_t[size_adj];        // 2m page 24 manual (m is number of edges)

//    adjwgt = NULL;
    vwgt= new idx_t[num_vertex];            // n page 24 manual (n is number of vertices)
//    vwgt= NULL;

#if dbl_PR_get_parts >=5    //default is 5
    // Number of edges are less than below because I remove the loop to itself, but the below value is useful for malloc
    PetscPrintf(PETSC_COMM_WORLD,"[PR][get_parts] num_vertex:%d, (nnz in WA which is double of unique num edges) num_edge:%d\n",num_vertex, num_edge);
    #if dbl_PR_get_parts >=7    //default is 7
        std::cout  << "[PR][get_parts] m_neigh_WA matrix:\n";                       //$$debug
        MatView(m_WA,PETSC_VIEWER_STDOUT_WORLD);
    #endif
#endif

    // - - - - - - Set parameters - - - - - -
    PetscInt            curr_adj_ind_ =0;       // keep track of edge indices in adjncy array

    PetscScalar         *arr_adjwgt_;       //use to pass the edge weights to scale_edge_weight()
    PetscMalloc1(num_edge, &arr_adjwgt_);

    PetscScalar         *arr_Vol_;
    PetscMalloc1(num_vertex, &arr_Vol_);
    VecGetArray(v_vol,&arr_Vol_);

    for(i=0; i< num_vertex; i++){
        MatGetRow(m_WA,i,&ncols,&cols,&vals);
        xadj[i] = curr_adj_ind_;       // starts with zero
        vwgt[i] = arr_Vol_[i];
#if dbl_PR_get_parts >=9    //default is 9
        PetscPrintf(PETSC_COMM_WORLD,"xadj[%d]=%d,vwgt[i]=%d",i,xadj[i],vwgt[i]);     //$$debug
#endif
        for(j=0; j < ncols ; j++){
            if(i != cols[j] ){   // skip the loop to iteself (which cause metis to crash)
                adjncy[curr_adj_ind_] = cols[j];
                arr_adjwgt_[curr_adj_ind_] = vals[j] ;  // scale_edge_weight will fix the values
                curr_adj_ind_++;
            }
        }
        MatRestoreRow(m_WA,i,&ncols,&cols,&vals);
    }
    xadj[num_vertex]=curr_adj_ind_;       // edge critic (last value)
#if dbl_PR_get_parts >=9    //default is 9
    PetscPrintf(PETSC_COMM_WORLD,"\nlast xadj[%d]=%d\n",num_vertex,curr_adj_ind_); //$$debug    // the last one shows the end of array and it is N+1
#endif

    VecRestoreArray(v_vol,&arr_Vol_);
    PetscFree(arr_Vol_);


//    scale_edge_weights(adjwgt, num_edge, arr_adjwgt_);
    scale_edge_weights(adjwgt, curr_adj_ind_, arr_adjwgt_);     //since I don't add loops to itself, there should be fewer edge weights in the array
    PetscFree(arr_adjwgt_);

//    for(int k=0; k < num_edge ; k++){
//        PetscPrintf(PETSC_COMM_WORLD,"[PR][get_part]debug adjwgt[%d]=%d\n",k,adjwgt[k]); //$$debug
//    }

    PetscPrintf(PETSC_COMM_WORLD,"[PR][get_parts] before running Metis\n");       //$$debug
//    if(Config_params::getInstance()->get_ds_name() == "higgs"){
//    if(level == 1 ){
//        CommonFuncs cf;
//        const std::string s_level = std::to_string(level) + ".dat";
//        const std::string m_fname = "m_neigh_WA_" + s_level;
//        const std::string v_fname = "v_neigh_vol_" + s_level;
//        cf.exp_matrix(m_WA, "./data", m_fname, "get_part");
//        cf.exp_vector(v_vol, "./data", v_fname, "get_part");
//        std::cout << "number of partitions was " << num_req_parts << ", m_WA, v_vol are exported!" << std::endl;

//        for(i=0; i< curr_adj_ind_; i++){
//            PetscPrintf(PETSC_COMM_WORLD,"[PR][get_parts] i:%d, adjwgt:%d\n",i, adjwgt[i]);       //$$debug
//        }

//    }
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
        PetscPrintf(PETSC_COMM_WORLD,"[PR][get_parts] Metis faced a problem \n");
    }

#if dbl_PR_get_parts >= 3    //default is 3
    std::cout << "[PR][get_parts] Metis result: " << result << std::endl;
    #if dbl_PR_get_parts >= 7    //default is 7
    PetscPrintf(PETSC_COMM_WORLD,"\n[PR][get_parts] [dev-test] [part index, selected partition]\n");
    for(i=0; i<num_vertex ;i++){
        PetscPrintf(PETSC_COMM_WORLD,"[%d:%d], ",i, part[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD,"\n\n");
    #endif
#endif


    // - - - - These check expected conditions - - - -
    if(curr_adj_ind_ > num_edge) {
        PetscPrintf(PETSC_COMM_WORLD,"[PR][get_parts] curr_adj_ind_:%d is larger than number of non zeros in meigh_WA:%d which is wrong\n",curr_adj_ind_, num_edge);
        exit(1);
    }
    if(result != 1){    // 1 means ok (I email for confirmation)
        PetscPrintf(PETSC_COMM_WORLD,"[PR][get_parts] Metis failed, return value is:%d!\n",result);
        exit(1);
    }

    //prepare the output
    MatCreateSeqAIJ(PETSC_COMM_SELF, num_req_parts, num_vertex , num_vertex,PETSC_NULL, &m_parts);
    // part[i] value determine the partition, and i is the index of the point in data matrix
    for(i=0; i<num_vertex ;i++){
        vv_parts[part[i]].push_back(i);
        MatSetValue(m_parts, part[i], i, 1,INSERT_VALUES);
    }
    MatAssemblyBegin(m_parts,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_parts,MAT_FINAL_ASSEMBLY);

    // - - - - Filter empty partitions- - - -
    int num_non_empty_parts = 0;
    for(i=0; i< num_req_parts; i++){
        if(vv_parts[i].size() != 0){
            num_non_empty_parts++;
        }else{
            std::cout << "[PR][get_part] empty partition at index:" << i << ", Exit!" << std::endl;
            CommonFuncs cf;
            cf.exp_matrix(m_WA, "./data", "m_neigh_WA.dat","get_part");
            cf.exp_vector(v_vol, "./data", "v_neigh_vol.dat","get_part");
            std::cout << "number of partitions was " << num_req_parts << ", m_WA, v_vol are exported!" << std::endl;
            exit(1);
        }
    }

#if dbl_PR_get_parts >=7    //default is 7
    PetscPrintf(PETSC_COMM_WORLD,"[PR][get_parts] m_parts matrix:\n");                       //$$debug
    MatView(m_parts,PETSC_VIEWER_STDOUT_WORLD);

    PetscPrintf(PETSC_COMM_WORLD,"[PR][get_parts] vv_parts are:\n");
    int curr_idx_size;
    for(i=0; i < num_req_parts; i++){
        curr_idx_size = vv_parts[i].size();
        PetscPrintf(PETSC_COMM_WORLD,"[PR][get_parts] Result Parts of group:%d, size:%d\n",i, curr_idx_size) ;
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
    return num_non_empty_parts ;
}



//If weights are inversed in the beginning, the higher values (inversed weights) means stronger relation
//and are closer to each other
//The metis is going to minimize the edge cut, so it keeps the heavy edge weights inside the partitions
//and there is no need to inverse the weights again, but if the values were distances, we need to inverse them
void Partitioning::scale_edge_weights(idx_t *adjwgt, PetscInt num_edge, PetscScalar *arr_adjwgt){
    bool weight_inversed = Config_params::getInstance()->get_inverse_weight() ;
    PetscInt i;
    PetscScalar tmp_sum=0, scale_rate=0;
    if (weight_inversed ){
        CommonFuncs cf;
        cf.zscore_array(arr_adjwgt,num_edge);
        for(i=0; i < num_edge; i++){
            tmp_sum += arr_adjwgt[i];
        }
        tmp_sum += num_edge * 4;
        //max number for 64 bit is:  9,223,372,036,854,775,807, but I used the max of 32 bits
        scale_rate = (4294967295 / tmp_sum) * 0.2;
#if dbl_PR_SEW >=3    //default is 3
        PetscPrintf(PETSC_COMM_SELF,"[PR][SEW] total sum before scaling:%g, scale_rate:%g \n", tmp_sum, scale_rate);                       //$$debug
#endif
        for(i=0; i < num_edge; i++){
            if(arr_adjwgt[i] > -4){
                adjwgt[i] = round( (arr_adjwgt[i] + 4) * scale_rate);   // range is larger than 1 and less than 800
            }else{                          //handle large negative values (outliers in negative side)
                adjwgt[i] = 0;
            }
#if dbl_PR_SEW >=9    //default is 9
            PetscPrintf(PETSC_COMM_SELF,"[PR][SEW] not inversing i:%d, adjwgt[i]:%d \n", i, adjwgt[i]);                       //$$debug
#endif
        }
    }else{
        for(i=0; i < num_edge; i++){
//            adjwgt[i] = round ( (1.0 / (arr_adjwgt[i] + 0.0001)) * 10000 )  ;
            adjwgt[i] = 1;
#if dbl_PR_SEW >=9    //default is 9
            PetscPrintf(PETSC_COMM_SELF,"[PR][SEW] inversing i:%d, adjwgt[i]:%d \n", i, adjwgt[i]);                       //$$debug
#endif

        }
    }
}

/*
 * output:
 *          m_parst: a matrix with 1 row with number of points as features, all of them are 1
 */
void Partitioning::single_part_matrix(Mat& m_neigh_Data, Mat& m_parts){
    PetscInt num_row;
    MatGetSize(m_neigh_Data, &num_row, NULL);
    MatCreateSeqAIJ(PETSC_COMM_SELF, 1, num_row , num_row,PETSC_NULL, &m_parts);
    for(int i=0; i < num_row; i++){
        MatSetValue(m_parts, 0, i, 1, INSERT_VALUES);
    }
    MatAssemblyBegin(m_parts,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_parts,MAT_FINAL_ASSEMBLY);
#if dbl_PR_single_part_matrix >=7    //default is 7
    PetscPrintf(PETSC_COMM_WORLD,"[PR][SPM] m_parts matrix:\n");                       //$$debug
    MatView(m_parts,PETSC_VIEWER_STDOUT_WORLD);
#endif
    //@@
}


/*
 * output:
 *          m_centers: calculated 1 center with respect to volumes of points
 *          sum_vol: sum of the volumes in the partition
 */
void Partitioning::calc_single_center(Mat& m_neigh_Data, Vec& v_vol, Mat& m_center, std::vector<PetscScalar>& v_sum_vol_parts){
    // calculate the sum of volumes for this partition (without normalization)
    PetscScalar sum_vol;
    VecSum(v_vol, &sum_vol);
//    std::cout << "[PR][CSC] sum vol:" << sum_vol << std::endl;
    v_sum_vol_parts[0] = sum_vol;

    //---- Normalize the Volumes #1 -----
    Vec v_vol_normal;
    Coarsening co;
    co.normalize_vector(v_vol,v_vol_normal);

    //---- get sum of normalized volumes #2 -----
    PetscScalar sum_normalized_vol;
    VecSum(v_vol_normal, &sum_normalized_vol);
    if(sum_normalized_vol == 0) {
        std::cout << "[PR][CSC] sum normalized volume is zero which cause division by zero!, exit " << std::endl;
        exit(1);
    }
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
    for(int i=0; i<num_features; i++){
//        std::cout << "[PR][CSC] arr_sum_DV_col[i]: " << arr_sum_DV_col[i] << ", calc: " <<
//                     arr_sum_DV_col[i]/sum_normalized_vol << " at i: " << i << std::endl;
        MatSetValue(m_center, 0, i, arr_sum_DV_col[i]/sum_normalized_vol, INSERT_VALUES );
    }
    VecRestoreArray(v_sum_DV_col, &arr_sum_DV_col);
    PetscFree(arr_sum_DV_col);
    MatAssemblyBegin(m_center, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_center, MAT_FINAL_ASSEMBLY);
#if dbl_PR_CSC >=7
    printf("[PR][CSC] Center matrix:\n");                                     //$$debug
    MatView(m_center,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif
}






/*
 *  - - - - - - calculate the centroid of each partition - - - - - -
 * @method: for now I calculate the centroid based on Euclidean distance
 * I consider the volumes in the calculating the centers and I normalize the center using sum of all volumes in this part
 * @input: neigh_Data, neigh_vol, and the partitions
 * @output: matrix with parts number of rows and for columns the same number of features as WA
 */
void Partitioning::calc_center(Mat& m_neigh_Data, Vec& v_neigh_vol, std::vector<std::vector<PetscInt>>& vv_parts, Mat& m_parts,
                                                                Mat& m_centers, std::vector<PetscScalar>& v_sum_vol_parts){
    //---- create diagonal matrix from volumes #1 -----
    PetscInt num_point, num_features;
    MatGetSize(m_neigh_Data, &num_point, &num_features);
    Mat m_VnormDiag;
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_point,num_point,1,PETSC_NULL, &m_VnormDiag);
    MatDiagonalSet(m_VnormDiag,v_neigh_vol,INSERT_VALUES);

    //---- Data_Vol = VD * Data #2 -----
    Mat m_Data_Vol;
    MatMatMult(m_VnormDiag, m_neigh_Data, MAT_INITIAL_MATRIX,PETSC_DEFAULT,&m_Data_Vol);
    MatDestroy(&m_VnormDiag);

    //---- Sigma Columns = Parts * Data_Vol #3 -----
    Mat m_Sum_Col_Part;     // (i,j) some of all values in column j for partition i
    MatMatMult(m_parts, m_Data_Vol, MAT_INITIAL_MATRIX,PETSC_DEFAULT,&m_Sum_Col_Part);
    MatDestroy(&m_Data_Vol);
#if dbl_PR_calc_center >=7    //default is 7
    std::cout  << "[PR][calc_center] m_Sum_Col_Part matrix:\n";                       //$$debug
    MatView(m_Sum_Col_Part,PETSC_VIEWER_STDOUT_WORLD);
#endif

    //---- calculate the sum of normalized volume for  each partition #4 -----
    PetscInt num_parts, i;
    unsigned int j;
    PetscScalar sum_curr =0 ;
    MatGetSize(m_parts, &num_parts,NULL);
//    printf("[PR][calc_center] num_parts:%d\n",num_parts);
//    std::vector<PetscScalar> v_sum_norm_vol(num_parts);
    PetscScalar * arr_norm_vol;
    PetscMalloc1(num_point, &arr_norm_vol);
    VecGetArray(v_neigh_vol,&arr_norm_vol);

    for(i=0; i <num_parts; i++){
        sum_curr = 0;                       //reset it for current partition
        for(j=0; j < vv_parts[i].size(); j++){
            // vv_parts[i][j] is the index of j-th point in partition i, and we get the volume from array of v_neigh_vol_norm
            sum_curr += arr_norm_vol [vv_parts[i][j] ];
        }
        v_sum_vol_parts[i] = sum_curr;       //save the sum in the end
        if(sum_curr == 0){
            printf("[PR][calc_center] sum_curr is:%g \n",sum_curr);
            exit(1);
        }
#if dbl_PR_calc_center >=1    //default is 7
        printf("[PR][calc_center] v_sum_vol_parts[%d] is:%g \n",i, v_sum_vol_parts[i]);
#endif
    }
    VecRestoreArray(v_neigh_vol,&arr_norm_vol);
    PetscFree(arr_norm_vol);

    //---- divide each row to Sum_normalized_vol related to that row (partition) #5 -----
    // get sum of each column, divide it to sum of normalized volumes
    PetscInt            ncols;
    const PetscInt      *cols;
    const PetscScalar   *vals;
    //prepare the output
    MatCreateSeqAIJ(PETSC_COMM_SELF, num_parts, num_features , num_features,PETSC_NULL, &m_centers);


    for(i=0; i < num_parts; i++) {
        MatGetRow(m_Sum_Col_Part,i,&ncols,&cols,&vals);
        for(j=0; j < ncols; j++){
            // set the ( vals[j] /v_sum_vol_parts[i] ) into the center matrix
            if(v_sum_vol_parts[i] != 0){ //skip the partitions with zero volume (empty partitions)
                MatSetValue(m_centers, i, cols[j], vals[j] /v_sum_vol_parts[i] ,INSERT_VALUES);
            }
        }
        MatRestoreRow(m_Sum_Col_Part,i,&ncols,&cols,&vals);
    }
    MatDestroy(&m_Sum_Col_Part);
    MatAssemblyBegin(m_centers,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_centers,MAT_FINAL_ASSEMBLY);
#if dbl_PR_calc_center >=7    //default is 7
    std::cout  << "[PR][calc_center] m_centers matrix:\n";                       //$$debug
    MatView(m_centers,PETSC_VIEWER_STDOUT_WORLD);
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
void Partitioning::calc_distances(int num_part_p, int num_part_n, Mat& m_centers_p,Mat& m_centers_n, Mat& m_dist){
    MatCreateSeqDense(PETSC_COMM_SELF,num_part_p+num_part_n,num_part_p+num_part_n,NULL, &m_dist);
    PetscInt            i, j, ncols_p, ncols_n;
    const PetscInt      *cols_p, *cols_n;
    const PetscScalar   *vals_p, *vals_n;
    CommonFuncs cf;
#if dbl_PR_calc_distances >=7    //default is 7
    std::cout  << "[PR][calc_distances] m_centers_p matrix:\n";                       //$$debug
    MatView(m_centers_p,PETSC_VIEWER_STDOUT_WORLD);
    std::cout  << "[PR][calc_distances] m_centers_n matrix:\n";                       //$$debug
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
//            std::cout  << "[PR][calc_distances] P only i:" << i << " j:" << j << "\n";
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
//            std::cout  << "[PR][calc_distances] N only i:" << num_part_p + i << " j:" << num_part_p + j << "\n";
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
#if dbl_PR_calc_distances >=7    //default is 7
    std::cout  << "[PR][calc_distances] m_dist matrix:\n";                       //$$debug
    MatView(m_dist,PETSC_VIEWER_STDOUT_WORLD);
//    cf.exp_matrix(m_centers_p,"./debug","m_centers_p_may17.dat","[PR][CD]");
//    cf.exp_matrix(m_centers_n,"./debug","m_centers_n_may17.dat","[PR][CD]");
//    cf.exp_matrix(m_dist,"./debug","m_dist_may17.dat","[PR][CD]");
#endif
}





/*
 * @input:
 *      number of partitions(partitions) in each class, number of features in data
 *      distances matrix
 * @output:
 *      vector index represents nothing
 *      the first represent the index of P class partition and the second represent the index of N class partition
 */
void Partitioning::find_groups(int num_part_p, int num_part_n, Mat& m_dist, std::vector<std::pair<int, int> >& v_groups){

    // edge case: number of partition for any class is 1
    if(num_part_p == 1){
        if(num_part_n == 1){
            v_groups.push_back(std::pair<int, double>(0,0));
//            std::cout<< "[PR][FG] v_groups size:" << v_groups.size() << std::endl;
        }else{      //multiple N class partitions
            for(int i=0; i < num_part_n; i++){      //add all the N class partitions with the only one P class partition
                v_groups.push_back(std::pair<int, double>(0,i));
            }
            std::cout<< "[PR][FG] v_groups size:" << v_groups.size() << std::endl;
        }
#if dbl_PR_find_groups >= 9     // default 9
        for(auto it=v_groups.begin(); it != v_groups.end(); it++){
            std::cout<< "[PR][FG] group P part index: "<< it->first <<", N part index: " << it->second << std::endl;
        }
#endif
        std::cout<< "[PR][FG] end of only 1 minotiry part!\n";
        return; //stop the rest
    }
    if(num_part_n == 1){
        if(num_part_p == 1){
            v_groups.push_back(std::pair<int, double>(0,0));
        }else{      //multiple P class partitions
            for(int i=0; i < num_part_p; i++){      //add all the P class partitions with the only one N class partition(N goes in second)
                v_groups.push_back(std::pair<int, double>(i,0));
            }
        }
#if dbl_PR_find_groups >= 9     // default 9
        for(auto it=v_groups.begin(); it != v_groups.end(); it++){
            std::cout<< "[PR][FG] group P part index: "<< it->first <<", N part index: " << it->second << std::endl;
        }
#endif
        std::cout<< "[PR][FG] end of only 1 majotiry part!\n";
        return; //stop the rest
    }

    // - - - - - - - normal scenario with multiple partition in both class - - - - - - - @@060616-1817
    PetscInt i,j, ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    // - - - - - - create vector of opposite closest part for class P - - - - - -
    for(i=0; i< num_part_p; i++ ){
        MatGetRow(m_dist,i,&ncols, &cols, &vals);    //P row
        std::vector<std::pair<int, double>> v_opposite_class;
        // - - - - create vector for opposite parts - - - - -
        for(j=0; j< ncols; j++){
            if( (cols[j] != i) && (cols[j] >= num_part_p)  ){                //ignore the loop to itself, and distance to P class
                v_opposite_class.push_back( std::pair<int, double>(cols[j] - num_part_p, vals[j]) );
//                std::cout<< "[PR][FG] inside P class part, N index:" << (cols[j] - num_part_p) <<
//                            ", dist:"<< vals[j] << ", cols[j]:" << cols[j] << ", num_part_p:" << num_part_p << std::endl;
            }
        }
        // - - - - sort partitions on distance - - - - -
        sort(v_opposite_class.begin(), v_opposite_class.end(), [](std::pair<int, double> x, std::pair<int, double> y) {return x.second < y.second;});
        v_groups.push_back(std::pair<int,int>(i,v_opposite_class[0].first) );
#if dbl_PR_find_groups >= 9     // default 9
        std::cout<< "[PR][FG] P class (normal scenario) (P:"<< i <<", N:"<<  v_opposite_class[0].first <<")!\n";
#endif
        MatRestoreRow(m_dist,i,&ncols, &cols, &vals);
    }
    //debug only
//    for(int i = 0; i < v_groups.size(); i++){
//        std::cout << v_groups[i].first << ", " << v_groups[i].second << std::endl;
//    }

    // - - - - - - create vector of close parts for class N - - - - - -
    for(i=num_part_p; i< num_part_n + num_part_p ; i++ ){
//        std::cout<< "[PR][FG]  i:"<< i << std::endl;
        MatGetRow(m_dist,i,&ncols, &cols, &vals);    //N row
        std::vector<std::pair<int, double>> v_opposite_class;
        // - - - - create vector for opposite parts - - - - -
        for(j=0; j< ncols; j++){
            if( (cols[j] != i) && (cols[j] < num_part_p)   ){                //ignore the loop to itself
                    v_opposite_class.push_back( std::pair<int, double>(cols[j], vals[j]) );   //left bottom part of m_dist matrix(N->P)
//                    std::cout<< "[PR][FG] inside N class part, P index:" << cols[j] << ", dist:"<< vals[j] << std::endl;
                }
        }
        // - - - - sort partitions on distance - - - - -
        sort(v_opposite_class.begin(), v_opposite_class.end(), [](std::pair<int, double> x, std::pair<int, double> y) {return x.second < y.second;});

        //in the pairs it is assumed that the first is the P_index and second is N_index for partitions
        std::pair<int, int> check_duplicate = std::pair<int, int> ( v_opposite_class[0].first, (i-num_part_p) ) ;
//        std::cout<< "[PR][FG] N class (normal scenario) check duplicate (P:"<< check_duplicate.first <<", N:"<< check_duplicate.second    <<")!\n";
        if(std::find(v_groups.begin(), v_groups.end(), check_duplicate ) == v_groups.end() ) { // ignore duplicate groups
            v_groups.push_back(std::pair<int,int>(v_opposite_class[0].first, (i-num_part_p) ) );
        }
        MatRestoreRow(m_dist,i,&ncols, &cols, &vals);
    }


#if dbl_PR_find_groups >= 3     // default 3
    #if dbl_PR_find_groups >= 5     // default 5
    std::cout<< "[PR][FG] final group v_groups.size():"<< v_groups.size() << std::endl;
    for(auto it=v_groups.begin(); it != v_groups.end(); it++){
        std::cout<< "[PR][FG] final group P part index: "<< it->first <<", N part index: " << it->second << std::endl;
    }
    #endif
    std::cout<< "[PR][FG] End of final group"<< std::endl;
#endif

}


/*
 * @output:
 *      m_avg_centers : matrix of avg centers for each group inside the v_groups vector
 */

void Partitioning::calc_avg_center(Mat& m_centers_p, Mat& m_centers_n, std::vector<std::pair<int, int> >& v_groups
                       , std::vector<PetscScalar>& v_sum_vol_parts_p, std::vector<PetscScalar>& v_sum_vol_parts_n, Mat& m_avg_centers){
    // - - - - - calculate the centers with respect to sum of volumes in partitions - - - - -
    Mat m_center_vol_p, m_center_vol_n;

    calc_center_volume(m_centers_p, v_sum_vol_parts_p, m_center_vol_p);
    calc_center_volume(m_centers_n, v_sum_vol_parts_n, m_center_vol_n);

    // - - - - - calculate the avg centers with respect to normalizing back using sum of volumes in partitions - - - - -
    PetscInt it_grp, it_p_id, it_n_id, ncols_p, ncols_n, num_features;
    const PetscInt    *cols_p, *cols_n;
    const PetscScalar *vals_p, *vals_n;
    PetscScalar sum_group_volumes;
    MatGetSize(m_centers_p, NULL, &num_features);
    MatCreateSeqAIJ(PETSC_COMM_SELF,v_groups.size(), num_features, num_features,PETSC_NULL, &m_avg_centers);

    for(it_grp=0; it_grp < v_groups.size(); it_grp++) {
        MatGetRow(m_center_vol_p, v_groups[it_grp].first, &ncols_p, &cols_p, &vals_p);
        MatGetRow(m_center_vol_n, v_groups[it_grp].second, &ncols_n, &cols_n, &vals_n);
        it_p_id = 0;
        it_n_id = 0;
        sum_group_volumes = v_sum_vol_parts_p[v_groups[it_grp].first] + v_sum_vol_parts_n[v_groups[it_grp].second];
#if dbl_PR_CAC >=9
        printf("[PR][CAC]{debug} sum_group_volumes for group id %d is:%g\n", it_grp, sum_group_volumes);
#endif
        while( it_p_id < ncols_p && it_n_id < ncols_n){
            if(cols_p[it_p_id] == cols_n[it_n_id] ){
                MatSetValue(m_avg_centers, it_grp, cols_p[it_p_id], (vals_p[it_p_id] + vals_n[it_n_id]) / sum_group_volumes, INSERT_VALUES);
                it_p_id++;
                it_n_id++;
            }else{
                if(cols_p[it_p_id] < cols_n[it_n_id] ){     // P has a nnz and N doesn't have it
                    MatSetValue(m_avg_centers, it_grp, cols_p[it_p_id], (vals_p[it_p_id]) / sum_group_volumes ,INSERT_VALUES);
                    it_p_id++;
                }else{                                      // N has a nnz and P doesn't have it
                    MatSetValue(m_avg_centers, it_grp, cols_n[it_n_id], (vals_n[it_n_id]) / sum_group_volumes ,INSERT_VALUES);
                    it_n_id++;
                }
            }
        }
        MatRestoreRow(m_center_vol_p, v_groups[it_grp].first, &ncols_p, &cols_p, &vals_p);
        MatRestoreRow(m_center_vol_n, v_groups[it_grp].second, &ncols_n, &cols_n, &vals_n);

    }
    MatAssemblyBegin(m_avg_centers, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_avg_centers, MAT_FINAL_ASSEMBLY);
#if dbl_PR_CAC >=7
    printf("[PR][CAC] m_avg_centers matrix:\n");                                     //$$debug
    MatView(m_avg_centers,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif
}



void Partitioning::calc_center_volume(Mat& m_centers, std::vector<PetscScalar>& v_sum_vol_parts, Mat& m_center_vol){

    PetscInt num_point;
    MatGetSize(m_centers, &num_point, NULL);
    if( num_point != v_sum_vol_parts.size())    {
        std::cout << "[PR][CCV] mismatch size m_centers:"<< num_point <<", v_vol_parts.size():"<< v_sum_vol_parts.size() << std::endl;
//        exit(1);
    }

    Vec v_sum_vol_;
    VecCreateSeq(PETSC_COMM_SELF,num_point,&v_sum_vol_);
    for(int i=0; i < num_point; i++){
//        std::cout << "\n\n [DEBUG] v_sum_vol_parts_p[i]:" << v_sum_vol_parts[i] << std::endl;
        VecSet(v_sum_vol_, v_sum_vol_parts[i]);
    }
    //---- create diagonal matrix from sum of volumes # -----
    Mat m_V_Diag;
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_point, num_point, 1,PETSC_NULL, &m_V_Diag);
    MatDiagonalSet(m_V_Diag,v_sum_vol_,INSERT_VALUES);


    //---- centers_vol = V_Diag * centers #4 -----
    MatMatMult(m_V_Diag, m_centers, MAT_INITIAL_MATRIX,PETSC_DEFAULT,&m_center_vol);
    MatDestroy(&m_V_Diag);

    #if dbl_PR_CCV >=7
//    printf("[PR][CCV] m_centers matrix:\n");                                     //$$debug
//    MatView(m_centers,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
//    printf("[PR][CCV] v_sum_vol_ vector:\n");                                     //$$debug
//    VecView(v_sum_vol_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    printf("[PR][CCV] m_center_vol matrix:\n");                                     //$$debug
    MatView(m_center_vol,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif

    VecDestroy(&v_sum_vol_);
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
 *
 */
void Partitioning::create_group_index(int target_group, std::vector<std::pair<int, int> > v_groups,
                                       Mat& m_parts_p, Mat& m_parts_n, std::vector<PetscInt>& v_p_index,
                                       std::vector<PetscInt>& v_n_index){
    PetscInt j, ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;


    // fill the P index

    MatGetRow(m_parts_p, v_groups[target_group].first, &ncols, &cols, &vals);
    v_p_index.reserve(ncols);       //it might get bigger but I try to use limited knowledge here
    for(j=0; j< ncols; j++ ){
        v_p_index.push_back(cols[j]);
    }
    MatRestoreRow(m_parts_p, v_groups[target_group].first, &ncols, &cols, &vals);

    // fill the N index
    MatGetRow(m_parts_n, v_groups[target_group].second, &ncols, &cols, &vals);
    for(j=0; j< ncols; j++ ){
        v_n_index.push_back(cols[j]);
    }
    MatRestoreRow(m_parts_n, v_groups[target_group].second, &ncols, &cols, &vals);


#if dbl_PR_CGI >= 5      // default 5
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
}




void Partitioning::calc_performance_measure(Mat& m_TD, std::vector<Mat>& v_mat_avg_center,
                                            std::vector<Mat>& v_mat_predicted_labels, summary& result_summary){
    // read the target labels from m_TD to a vector(test data matrix has the labels in the first column)
    /// - - - - - - - - get test data labels (target labels) - - - - - - - -
    PetscInt i=0, ncols=0, num_points=0,num_col=0;
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
    std::vector<double> v_final_predicted_label(num_points,0);  // same for all methods


    switch(Config_params::getInstance()->get_pr_maj_voting_id()){
        case 1:             // 1- majority voting all the same
            calc_majority_voting(v_mat_predicted_labels, num_points, v_final_predicted_label);
            break;
        case 2:             // 2- majority voting with inverse Euclidean distance weight (to avg_centers)
            calc_majority_voting_distance_weight(v_mat_predicted_labels, num_points, m_TD, v_mat_avg_center, 2, v_final_predicted_label);
            break;
        case 3:             // 3- majority voting with inverse Euclidean distance weight to the 2nd power (to avg_centers)
            calc_majority_voting_distance_weight(v_mat_predicted_labels, num_points, m_TD, v_mat_avg_center, 3, v_final_predicted_label);
            break;
        case 4:             // 4- majority voting with inverse Manhatan distance weight (to avg_centers)
            calc_majority_voting_distance_weight(v_mat_predicted_labels, num_points, m_TD, v_mat_avg_center, 4, v_final_predicted_label);
            break;
        case 5:             // not implemented yet
            calc_majority_voting_nearest(v_mat_predicted_labels, num_points, m_TD, v_mat_avg_center, 5, v_final_predicted_label);
            break;

    }
    std::cout << "[PR][CPM] v_final_predicted_label.size() "<< v_final_predicted_label.size() <<
                 " v_final_predicted_label[0]: " << v_final_predicted_label[0] <<std::endl;
    exit(1);
    /// - - - - - - - - compare the target labels and predicted - - - - - - - -
    int correct = 0;
    double tp =0, tn =0, fp =0, fn=0;

    for(int i=0; i < num_points; i++){
#if dbl_PR_CPM >= 5
        std::cout << "[PR][CPM] labels =>  target: "<< v_target_labels[i] << " predicted: " << v_final_predicted_label[i] <<std::endl;
#endif
        if(v_target_labels[i] == 1){                            // ++++ positive class ++++
            if (v_final_predicted_label[i] >= 0)
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
//    std::map<measures,double> results_;
    result_summary.perf[Sens] = tp / (tp+fn) ;
    result_summary.perf[Spec] = tn / (tn+fp) ;
    result_summary.perf[Gmean] = sqrt(result_summary.perf[Sens] * result_summary.perf[Spec]);
    result_summary.perf[Acc] = (double)correct / (num_points) ;

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

    result_summary.num_SV_p = 0;    //to ignore them
    result_summary.num_SV_n = 0;
    result_summary.C = 0;
    result_summary.gamma = 0;
}








// I decided to not use the below method, which needs to overload other functions I called

///*
// * this overload is create at 021517-1300 specifically for [RF] with partitioning
// * Evaluate the performance quality over the validation data from multiple models which is recoreded inside the v_mat_predicted_labels
// * The validation data is not required since the size of them are passed and the prediction are exist in v_mat_predicted_labels as mentioned above
// */
//void Partitioning::calc_performance_measure(PetscInt num_VD_p, PetscInt num_VD_n, std::vector<Mat>& v_mat_avg_center,
//                                            std::vector<Mat>& v_mat_predicted_labels, summary& result_summary){
//    // read the target labels only needs the size of validation data for each class
//    /// - - - - - - - - get validation data labels  - - - - - - - -
//    PetscInt i=0, ncols=0, num_points = num_VD_p+num_VD_n;
//    std::vector<double> v_target_labels(num_points);

//    for (i=0; i< num_VD_p;i++){
//        v_target_labels[i] = 1;             //label of minority class is +1
//    }

//    for (i=0; i< num_VD_n;i++){
//        v_target_labels[num_VD_p + i] = 1;             //label of minority class is -1
//    }

//    /// - - - - - - - - calc predicted labels - - - - - - - -
//    // it would be different techniques to calculate the final predicted labels
//    std::vector<double> v_final_predicted_label(num_points);  // same for all methods

//    switch(Config_params::getInstance()->get_pr_maj_voting_id()){
//        case 1:             // 1- majority voting all the same
//            calc_majority_voting(v_mat_predicted_labels, num_points, v_final_predicted_label);
//            break;
//        case 2:             // 2- majority voting with inverse Euclidean distance weight (to avg_centers)
//            calc_majority_voting_distance_weight(v_mat_predicted_labels, num_points, m_TD, v_mat_avg_center, 2, v_final_predicted_label);
//            break;
//        case 3:             // 3- majority voting with inverse Euclidean distance weight to the 2nd power (to avg_centers)
//            calc_majority_voting_distance_weight(v_mat_predicted_labels, num_points, m_TD, v_mat_avg_center, 3, v_final_predicted_label);
//            break;
//        case 4:             // 4- majority voting with inverse Manhatan distance weight (to avg_centers)
//            calc_majority_voting_distance_weight(v_mat_predicted_labels, num_points, m_TD, v_mat_avg_center, 4, v_final_predicted_label);
//            break;

//    }

//    /// - - - - - - - - compare the target labels and predicted - - - - - - - -
//    int correct = 0;
//    double tp =0, tn =0, fp =0, fn=0;

//    for(int i=0; i < num_points; i++){
//#if dbl_PR_CPM >= 5
//        std::cout << "[PR][CPM] labels =>  target: "<< v_target_labels[i] << " predicted: " << v_final_predicted_label[i] <<std::endl;
//#endif
//        if(v_target_labels[i] == 1){                            // ++++ positive class ++++
//            if (v_final_predicted_label[i] >= 0)
//                tp++;                   // true positive
//            else
//                fn++;
//        }else{                                                  // ---- predict negative ----
//            if (v_final_predicted_label[i] < 0)
//                tn++;                   // true negative
//            else
//                fp++;                   // false positive
//        }
//    }
//    /// - - - - - - - - calc different performance measures (Acc, Gmean,...)  - - - - - - - -
//    correct = tp+tn;                //sum both True
////    std::map<measures,double> results_;
//    result_summary.perf[Sens] = tp / (tp+fn) ;
//    result_summary.perf[Spec] = tn / (tn+fp) ;
//    result_summary.perf[Gmean] = sqrt(result_summary.perf[Sens] * result_summary.perf[Spec]);
//    result_summary.perf[Acc] = (double)correct / (num_points) ;

//    if(tp+fp == 0)              //prevent nan case
//        result_summary.perf[PPV] = 0;
//    else
//        result_summary.perf[PPV] = tp/ (tp+fp);

//    if(tn+fn == 0)              //prevent nan case
//        result_summary.perf[NPV] = 0;
//    else
//        result_summary.perf[NPV] = tn/ (tn+fn);

//    result_summary.perf[F1] = 2*tp / (2*tp+fp+fn);
//    result_summary.perf[TP] = tp;
//    result_summary.perf[FP] = fp;
//    result_summary.perf[TN] = tn;
//    result_summary.perf[FN] = fn;

//    result_summary.num_SV_p = 0;    //to ignore them
//    result_summary.num_SV_n = 0;
//    result_summary.C = 0;
//    result_summary.gamma = 0;
//}











//void Partitioning::calc_majority_voting(Mat& m_predicted_labels, int num_TD_points, std::vector<double>& v_final_predicted_label ){
void Partitioning::calc_majority_voting(std::vector<Mat>& v_mat_predicted_labels, int num_TD_points, std::vector<double>& v_final_predicted_label ){
    for(unsigned int i =0; i < v_mat_predicted_labels.size(); i++){
        Mat m_pl_t;
        MatTranspose(v_mat_predicted_labels[i],MAT_INITIAL_MATRIX,&m_pl_t);
        Vec v_sum_labels;
        VecCreateSeq(PETSC_COMM_SELF, num_TD_points, &v_sum_labels);
        MatGetRowSum(m_pl_t, v_sum_labels);    //calc sum of each row of predict_label transpose
        MatDestroy(&m_pl_t);


        PetscScalar     *arr_predicted_labels;
        PetscMalloc1(num_TD_points, &arr_predicted_labels);
        VecGetArray(v_sum_labels, &arr_predicted_labels);   //the vector od sum_labels is calculated separately for each iteration
        for(int it_TD=0; it_TD <num_TD_points; it_TD++){
            //the v_final_predicted_label sums the summation for all iterations
            v_final_predicted_label[it_TD] += arr_predicted_labels[it_TD];
        }
        VecRestoreArray(v_sum_labels, &arr_predicted_labels);
        PetscFree(arr_predicted_labels);
    }
}




//void Partitioning::calc_majority_voting_distance_weight(Mat& m_predicted_labels, int num_TD, Mat& m_TD, Mat& m_AC
//                                                        , int distance_id, std::vector<double>& v_final_predicted_label ){
void Partitioning::calc_majority_voting_distance_weight(std::vector<Mat>& v_mat_predicted_labels, int num_TD, Mat& m_TD,
                                        std::vector<Mat>& v_mat_AC, int distance_id, std::vector<double>& v_final_predicted_label ){

    ETimer t_cmvdw;
    PetscInt            i,j, ncols_TD, ncols_AC, ncols_PLt, num_AC;
    const PetscInt      *cols_TD, *cols_AC, *cols_PLt;
    const PetscScalar   *vals_TD, *vals_AC, *vals_PLt;
    unsigned int num_iter = v_mat_predicted_labels.size();
    CommonFuncs cf;
    std::vector<double> v_sum_norm_prediction(num_TD,0.0);// the default is also initialize it with zeros
    // loop over multiple prediction matrix and avg_center
    for(unsigned int iter =0; iter < num_iter; iter++){
        MatGetSize(v_mat_AC[iter], &num_AC, NULL);
//        std::cout << "[PR][CMVDW] [DEBUG] m_AC_num_row:" << num_AC << " num_TD_row:" << num_TD  <<
//                     " v_mat_AC.size():" << v_mat_AC.size() << std::endl;        //$$debug
        Mat m_PLt;
        //Transpose the PL, to get predicted labels for TD points as row
        MatTranspose(v_mat_predicted_labels[iter],MAT_INITIAL_MATRIX,&m_PLt);

        double tmp_dist;        //keep the temporary distance value
        double tmp_final_label;
        std::vector<double> v_tmp_inv_dist(num_AC);     //vector of temporary inverse distances from a test point to all avg_centers

        //Calculate the inverse of the distance between the test point and all the avg_centers into v_tmp_inv_dist
        for(i=0; i< num_TD; i++){
            MatGetRow(m_TD,i,&ncols_TD, &cols_TD, &vals_TD);        //Test Data points
            MatGetRow(m_PLt,i,&ncols_PLt, &cols_PLt, &vals_PLt);    //Predicted Labels
            for(j=0; j< num_AC; j++){
                MatGetRow(v_mat_AC[iter],j,&ncols_AC, &cols_AC, &vals_AC);    //Average Centers
                switch(distance_id){    //case 1 is handeled in calc_majority_voting function
                    case 2:       //Euclidean distance
                        tmp_dist = cf.calc_euclidean_dist(ncols_TD, ncols_AC, cols_TD, cols_AC, vals_TD, vals_AC); //get distance
                        v_tmp_inv_dist[j]= 1.0 / tmp_dist;                //save inversed distance
                        break;
                    case 3:       //(Euclidean distance)^2
                        tmp_dist = cf.calc_euclidean_dist(ncols_TD, ncols_AC, cols_TD, cols_AC, vals_TD, vals_AC); //get distance
                        v_tmp_inv_dist[j]= 1.0 / pow(tmp_dist, 2);                //save inversed distance
                        break;
                    case 4:       // Manhatan distance (Norm 1)
                        tmp_dist = cf.calc_manhatan_dist(ncols_TD, ncols_AC, cols_TD, cols_AC, vals_TD, vals_AC); //get distance
                        v_tmp_inv_dist[j]= 1.0 / tmp_dist;                //save inversed distance
                        break;
                }
                MatRestoreRow(v_mat_AC[iter],j,&ncols_AC, &cols_AC, &vals_AC);
            }
            // calc dot product of v_tmp_inv_dist and vector of predicted labels from PLt matrix
            tmp_final_label = cf.vec_vec_dot_product(ncols_PLt, cols_PLt, vals_PLt, v_tmp_inv_dist);

            //normalize the weight vector(v_tmp_inv_dist)
            double sum_v_weights=0;
            std::for_each(v_tmp_inv_dist.begin(), v_tmp_inv_dist.end(), [&] (double n) {
                sum_v_weights += n;
            });
//            tmp_final_label = tmp_final_label / sum_v_weights;
            // v_sum_norm_prediction is initialized with zero in the beginning and sum up all the normalized predictions
            v_sum_norm_prediction[i] += tmp_final_label / sum_v_weights;

//            std::cout << "[PR][CMVDW] i:" << i  << " before restore row"<< std::endl;

            MatRestoreRow(m_PLt,i,&ncols_PLt, &cols_PLt, &vals_PLt);
            MatRestoreRow(m_TD,i,&ncols_TD, &cols_TD, &vals_TD);
//            if(i == 2)
//                exit(1);

        } // end of prediction for all the test data points
        MatDestroy(&m_PLt);

        std::cout << "[PR][CMVDW] iter:" << iter << std::endl;      //$$debug
    }// end of iterations

    // - - - - - - average the predictions over multiple iterations and filter them - - - -
    for(i=0; i< num_TD; i++){
        //check with different threshold (0, 0.5)
        if( (v_sum_norm_prediction[i] / (double)num_iter) > 0)       //average the predictions
            v_final_predicted_label[i]= 1;
        else
            v_final_predicted_label[i]= -1;
    }

//        v_final_predicted_label[i]= tmp_final_label;
//        std::cout << "[PR][CMVDW] final_predicted_label:" << v_final_predicted_label[i] << std::endl;

    t_cmvdw.stop_timer("[PR] calc_majority_voting_distance_weight");
}



/* 051917-2350
 * @input:
 * so far the distance_id is not used (it could be used in future to try different distances rather than Euclidean distance)
 * std::vector<Mat>& v_mat_predicted_labels this should be modify to 1 matrix rather than vector of matrices, but since
 *      the refinement has not modified yet, I keep it like this for now
 * the predicted labels have been calculated by all the models and now we just need to select/filter/aggregate them to one label
 * @output: vector of labels is the output (v_final_predicted_label)
 */
void Partitioning::calc_majority_voting_nearest(std::vector<Mat>& v_mat_predicted_labels, int num_TD, Mat& m_TD,
                                        std::vector<Mat>& v_mat_AC, int distance_id, std::vector<double>& v_final_predicted_label ){

    ETimer t_cmvdw;
    PetscInt            i,j, ncols_TD, ncols_AC, ncols_PLt, num_AC;
    const PetscInt      *cols_TD, *cols_AC, *cols_PLt;
    const PetscScalar   *vals_TD, *vals_AC, *vals_PLt;
    unsigned int num_iter = v_mat_predicted_labels.size();
    CommonFuncs cf;


    std::vector<double> v_sum_norm_prediction(num_TD,0.0);// the default is also initialize it with zeros
    // loop over multiple prediction matrix and avg_center
    for(unsigned int iter =0; iter < num_iter; iter++){
        MatGetSize(v_mat_AC[iter], &num_AC, NULL);
        std::cout << "[PR][CMVN] [DEBUG] m_AC_num_row:" << num_AC << " num_TD_row:" << num_TD  <<
                     " v_mat_AC.size():" << v_mat_AC.size() << std::endl;        //$$debug
        Mat m_PLt;
        //Transpose the PL, to get predicted labels for TD points as row
        MatTranspose(v_mat_predicted_labels[iter],MAT_INITIAL_MATRIX,&m_PLt);

        //debug
        CommonFuncs cf;
        const std::string m_fname = "m_PLt.dat";
        cf.exp_matrix(m_PLt, "./data/", m_fname, "calc_majority_voting_nearest");

        double tmp_dist;        //keep the temporary distance value
        double min_distance;

        //Find the nearest avg_center to the test data point
        for(i=0; i< num_TD; i++){
            MatGetRow(m_TD,i,&ncols_TD, &cols_TD, &vals_TD);        //Test Data points
            MatGetRow(m_PLt,i,&ncols_PLt, &cols_PLt, &vals_PLt);    //Predicted Labels
            min_distance= std::numeric_limits<double>::max();       //set the min_distance to +infinity for each data point
            for(j=0; j< num_AC; j++){           //number of models
                MatGetRow(v_mat_AC[iter],j,&ncols_AC, &cols_AC, &vals_AC);    //Average Centers
                //calculate the distance of current test point to avg_center of current level
                tmp_dist = cf.calc_euclidean_dist(ncols_TD, ncols_AC, cols_TD, cols_AC, vals_TD, vals_AC); //get distance
                std::cout << "[PR][CMVN] [DEBUG] i:"<<i<<", j:"<<j<<", tmp_dist:" << tmp_dist << ", min_dist:" << min_distance;        //$$debug
                std::cout << ", v_final_predicted_label[i]:" << v_final_predicted_label[i] << std::endl;        //$$debug
                if(tmp_dist <= min_distance){
                    min_distance = tmp_dist;                        // update the minimum
                    //make sure the i is the test point id which is predicted current model (we are in the right row for sure)
                    v_final_predicted_label[i] = vals_PLt[i];      // update the final label
                    std::cout << "[PR][CMVN] [DEBUG] update the label with closer model, new lable" << v_final_predicted_label[i] <<
                                 ", vals_PLt[i]:" << vals_PLt[i] << ",ncols_PLt:" << ncols_PLt <<
                                 ", cols_PLt[i]:" << cols_PLt[i] << std::endl;        //$$debug
                }
                MatRestoreRow(v_mat_AC[iter],j,&ncols_AC, &cols_AC, &vals_AC);
            }
            std::cout << "[PR][CMVN] [DEBUG] final prediction -> v_final_predicted_label[i]:" << v_final_predicted_label[i] << std::endl;        //$$debug
            MatRestoreRow(m_PLt,i,&ncols_PLt, &cols_PLt, &vals_PLt);
            MatRestoreRow(m_TD,i,&ncols_TD, &cols_TD, &vals_TD);
        } // end of prediction for all the test data points
        MatDestroy(&m_PLt);
        exit(1);
        std::cout << "[PR][CMVN] iter:" << iter << std::endl;      //$$debug
    }// end of iterations
    t_cmvdw.stop_timer("[PR] calc_majority_nearest");
}

