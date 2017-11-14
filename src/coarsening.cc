#include "coarsening.h"
#include "etimer.h"
#include "config_params.h"
#include "config_logs.h"
#include <set>
#include "common_funcs.h"
#include <unordered_map>
#include <iomanip>  //for setprecision


Mat Coarsening::calc_P(Mat& WA, Vec& vol,std::vector<NodeId>& v_seeds_indices, cs_info& ref_info, bool debug) {

    PetscInt num_row;
    MatGetSize(WA,&num_row,0);    //num_row returns the number of rows globally
//    printf("number of nodes : %d\n",num_row );        //$$debug
    ETimer t_WD;
    Vec     D_vec;                  //sum of neighbors' weight for each node in the WA matrix
    PetscInt vol_rstart, vol_rend, i,j,D_vec_rstart, D_vec_rend,ncols;
    PetscScalar * vol_array, * D_vec_array;

    VecCreateSeq(PETSC_COMM_SELF,num_row,&D_vec);
//    MatGetDiagonal(WA,D_vec);
    MatGetRowSum(WA,D_vec);                                 //Calculate sum of each row of W matrix
#if dbl_CO_calcP >= 7
    printf("[Coarsening] D_vec Vector:\n");
    VecView(D_vec,PETSC_VIEWER_STDOUT_WORLD);
#endif

    VecGetOwnershipRange(vol,&vol_rstart,&vol_rend);        //read the volume from vol vector
    VecGetArray(vol,&vol_array);

    VecGetOwnershipRange(D_vec,&D_vec_rstart,&D_vec_rend);
    VecGetArray(D_vec,&D_vec_array);
//====================  update nodes information =========================
    Graph vertexs;
    //Set sum_neighbors_weights using D_vec
    for (i=0; i< num_row; ++i){
        Node new_node(i,vol_array[i]);
        new_node.setSumNeighborsWeight(D_vec_array[i]);
        vertexs.addNode(new_node);
    }
    VecRestoreArray(vol,&vol_array);        //Free the vol_array
    VecRestoreArray(D_vec,&D_vec_array);        //Free the D_vec_array
    VecDestroy(&D_vec);                         //Free the D_vector
//==================== Calculate the future volume =======================
    Volume      tmp_fut_vol =0;                     //temporary variable for future volume of each node
    Volume      sum_future_volume = 0;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;

#if dbl_CO_calcP >= 1         //calculate the stat for number of edges
    int sum_nnz =0;
    std::vector<tmp_degree> stat_degree_;
    stat_degree_.reserve(num_row);
#endif
    // For each row
    for(i =0; i <num_row; i++){
        tmp_fut_vol = vertexs.getNode(i).getVolume();       // V_i
        MatGetRow(WA,i,&ncols,&cols,&vals); //ncols : number if non-zeros in the row

#if dbl_CO_calcP >= 1         //calculate the stat for number of edges
        sum_nnz += ncols;                       //$$debug
        stat_degree_.push_back(tmp_degree(i,(int)ncols));
#endif
        // For each non-zero item in row i
        for (j=0; j<ncols; j++) {
            if(vertexs.getNode(cols[j]).getSumNeighborsWeight() != 0){
                tmp_fut_vol += vertexs.getNode(cols[j]).getVolume() * ( vals[j] / vertexs.getNode(cols[j]).getSumNeighborsWeight() );
            }
            else {
                tmp_fut_vol += vertexs.getNode(cols[j]).getVolume();
            }
        }
        vertexs.getNode(i).setFutureVolume(tmp_fut_vol);
        MatRestoreRow(WA,i,&ncols,&cols,&vals);       //Frees any temporary space allocated by MatGetRow()
        sum_future_volume += tmp_fut_vol;
#if dbl_CO_calcP >= 9
        printf("[CO][calc_p]row: %d Fv: %.4f \n",i,vertexs.getNode(i).getFutureVolume());
#endif
    }//end of for each row (num_row)


#if dbl_CO_calcP >= 1         //calculate the stat for number of edges
    ref_info.num_point = num_row;
    ref_info.num_edge = sum_nnz / 2;
    std::cout <<"[CO][calc_p]{" << this->cc_name <<"} number of rows:"<< num_row <<
                "\t\tedges:"<< ref_info.num_edge << std::endl;
    std::sort(stat_degree_.begin(), stat_degree_.end(), std::greater<tmp_degree>());     //Sort all edges in descending order

    ref_info.min_num_edge = stat_degree_[num_row-1].degree_;
    ref_info.avg_num_edge = (PetscScalar)sum_nnz / (PetscScalar)num_row;    // this is average degree of nodes, which I should not consider unique edges
    ref_info.max_num_edge = stat_degree_[0].degree_;
    if(num_row%2)   //odd number of rows
        ref_info.median_num_edge =  stat_degree_[num_row/2].degree_;
    else            //even number of rows
        ref_info.median_num_edge =  (stat_degree_[num_row/2 - 1].degree_ + stat_degree_[num_row/2 ].degree_) / 2;
    std::cout <<"[CO][calc_p]{" << this->cc_name <<"} Degrees\t Max:" << ref_info.max_num_edge <<
                "\t\tMin:" << ref_info.min_num_edge << "\t\tAvg:"<< ref_info.avg_num_edge  <<
                "\t\tMedian:"<< ref_info.median_num_edge  << std::endl;
#endif

    vertexs.setAvgFutureVolume( sum_future_volume / vertexs.getSize());

#if dbl_CO_calcP >= 3
    std::cout <<"[CO][calc_p]{" << this->cc_name <<"} Average Future Volume:"<< sum_future_volume / vertexs.getSize() << std::endl;
    t_WD.stop_timer("[CO][calc_p]Calc future volume");
#endif

//==================== Select strong Seeds ============================
//        ETimer t_sseed;

#if dbl_CO_calcP >=5
    int num_strong_seeds=0;
    num_strong_seeds = vertexs.selectSeeds();      //NOTE: Always notice that set avg-future-volume in advance
    printf("number of strong seeds: %d\n",num_strong_seeds);    //$$debug
#endif
//        t_sseed.stop_timer("select strong seeds:");
#if dbl_CO_calcP >=7
    std::cout << "list of seeds that has larger future volume than average \n";                 //$$debug
    vertexs.printSeeds();                                                                       //$$debug
#endif

//================ Recalculate the future volume =======================
// (based on 2-sum formula) for non seed nodes (checked 07162015-1227)
    ETimer t_recalc_fv;
    std::vector<tmp_future_volume> F_nodes_;                        // a vector consist of index and future volume of each node belongs to F
    tmp_fut_vol =0;                                                 // temporary variable for future volume of each node
    for(i =0; i <num_row; i++){
        if (!vertexs.getNode(i).getIsSeed()){                   // only recalcualte the Non seed nodes (notice the ! sign)
            tmp_fut_vol = vertexs.getNode(i).getVolume();       // V_i
//            printf("recalc fv at i:%d Vi:%g\n",i,vertexs.getNode(i).getVolume());     //$$debug
            MatGetRow(WA,i,&ncols,&cols,&vals);                                         //ncols : number if non-zeros in the row
            for (j=0; j<ncols; j++) {                           // SIGMA (j belong to F)
                if(!vertexs.getNode(cols[j]).getIsSeed()){      // check that it is not a seed (check: j belong to F)
                    if(vertexs.getNode(cols[j]).getSumNeighborsWeight() != 0){  // SIGMA W_jk (prevent Division by zero)
                        tmp_fut_vol += vertexs.getNode(cols[j]).getVolume() *
                                ( vals[j] / vertexs.getNode(cols[j]).getSumNeighborsWeight() );
#if dbl_CO_calcP >=9
//                        printf("recalc fv at row:%d col:%d V_j(vol):%g Wji:%g Sigma Wjk:%g\n",i,j,
//                                vertexs.getNode(cols[j]).getVolume(), vals[j], vertexs.getNode(cols[j]).getSumNeighborsWeight());
#endif
                    }
                    else {
                        tmp_fut_vol += vertexs.getNode(cols[j]).getVolume();
                    }
                }
            }
            MatRestoreRow(WA,i,&ncols,&cols,&vals);
            vertexs.getNode(i).setFutureVolume(tmp_fut_vol);
            F_nodes_.push_back(tmp_future_volume(i,tmp_fut_vol));
        }
    }
#if dbl_CO_calcP >=3
    t_recalc_fv.stop_timer("[CO][calc_p]Recalc FV");
#endif

    std::sort(F_nodes_.begin(), F_nodes_.end(), std::greater<tmp_future_volume>());     //Sort all F nodes in descending order

#if dbl_CO_calcP >=7
        std::cout<< "\nPrint temporary vector of nodes After sort :\n" ;        //$$debug
        for (auto it = F_nodes_.begin(); it != F_nodes_.end(); ++it) {
            std::cout << "Index : "<<it->node_index << " new FV :"<< it->future_volume << "\n";
        }
#endif
#if dbl_CO_calcP >=3
    printf("[CO][calc_p]after sort FV\n");
#endif
//================ Add points from F to C =======================
    ETimer t_update_C;
    Volume sigma_C_, sigma_V_;
    for (auto it = F_nodes_.begin(); it != F_nodes_.end(); ++it) {                      //go through all F_nodes
        sigma_C_ = 0 ;
        sigma_V_ = 0;

        MatGetRow(WA, it->node_index ,&ncols,&cols,&vals);        //ncols : number if non-zeros in the row
#if dbl_CO_calcP >=9
        printf("check to see if it is eligible to add from F to C node_index:%lu ncols:%d\n", it->node_index, ncols );
#endif
        for (j=0; j<ncols; j++) {
            if(it->node_index != (unsigned) cols[j]){                // to ignore the diagonal
                if(vertexs.getNode(cols[j]).getIsSeed()){   //Only for nodes belongs to C (this is updated nodes with recent changes)
                    sigma_C_ += vals[j];
                }
                sigma_V_ += vals[j];                        //for all nodes in V (including the C)
            }
        }
        if ( (sigma_C_/sigma_V_) <= Config_params::getInstance()->get_coarse_q() ){          //condition for moving a node from F to C
            vertexs.getNode(it->node_index).setSeed(1);  //add the node to C
#if dbl_CO_calcP >=9
            printf("#new seed:%lu\n",vertexs.getNode(it->node_index).getIndex());
#endif
        }
#if dbl_CO_calcP >=9
        printf("N: %lu FV: %g calc (SigmaC/SigmaV): %g\n",it->node_index,it->future_volume ,(sigma_C_/sigma_V_));
        printf("Sigma_C: %g Sigma_V: %g\n",sigma_C_,sigma_V_);
#endif
    }
#if dbl_CO_calcP >= 3
    t_update_C.stop_timer("[CO][calc_p]Add points from F to C");
#endif

    if (num_row < Config_params::getInstance()->get_coarse_threshold() ){       //when the data doesn't need coarsening anymore, move all nodes to C
        printf("[CO][calc_p]\t\t *** Notice *** \nnumber of seeds are equal to number of fine points due to small size of this class!!!\n");
        for (auto it = F_nodes_.begin(); it != F_nodes_.end(); ++it) {                      //go through all F_nodes
            vertexs.getNode(it->node_index).setSeed(1);         //set all nodes to seed
        }
    }
//========================= List Seeds indices ===========================
    std::map<int,int> seeds_indices;
    int num_seeds = 0;
    vertexs.getSeedsIndices(seeds_indices,num_seeds);
    this->num_coarse_points = (PetscInt) num_seeds;;        //used also in other methods

    v_seeds_indices.reserve(num_seeds);
    vertexs.find_seed_indices(v_seeds_indices);

#if dbl_CO_calcP >= 3
    printf("[CO][calc_p]after find seeds indices\n");
#endif

//========================= Create the P matrix ===========================
// rows : fine points        => row_dimension : num_rows
// columns : coarse points   => col_dimension : number of the coarse points => num_col
    Mat         P;
    std::vector<tmp_filter_p> filter_nodes_p;

    double sigma_w_ik=0 , sigma_filter_p=0;
    int max_fraction=0;          // find the maximum number of fraction for each node in P table (Bug #1)
//    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,this->num_coarse_points, Config_params::getInstance()->get_coarse_r() ,PETSC_NULL, &P);// try to reserve space for only number of final non zero entries for each fine node (e.g. 4)
    float threshold=Config_params::getInstance()->get_cs_boundary_points_threshold();    //Experiment boundary points (Jan 9, 2017);
    size_t ultimate_estimated_fractions=Config_params::getInstance()->get_cs_boundary_points_max_num();    //Experiment boundary points (Jan 9, 2017)
//    std::cout << "[CO][calc_p] DEBUG boundary points t:" << threshold << ", max fractions:" << ultimate_estimated_fractions << std::endl;
    
//#if boundary_points == 0    //Experiment boundary points (Jan 9, 2017)
    bool is_boundary_points_active = Config_params::getInstance()->get_cs_boundary_points_status();
//    std::cout << "[CO][calc_p] DEBUG boundary points status:" << is_boundary_points_active << std::endl;
    if(! is_boundary_points_active){
        MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,this->num_coarse_points, Config_params::getInstance()->get_coarse_r() ,PETSC_NULL, &P);// try to reserve space for only number of final non zero entries for each fine node (e.g. 4)
//        std::cout << "[CO][calc_p] DEBUG P matrix is created with fixed number of fractions (no boundary) nnz:" << Config_params::getInstance()->get_coarse_r() << std::endl;
//#else       //boundary_points == 1 , 20 is constant as the maximum number of estimated fractions for a fine point
    }else{
        MatCreateSeqAIJ(PETSC_COMM_SELF,num_row, this->num_coarse_points, ultimate_estimated_fractions, PETSC_NULL, &P);
        std::cout << "[CO][calc_p] DEBUG P matrix is created for boundary points with nnz:" << ultimate_estimated_fractions << std::endl;
    }
//#endif

#if dbl_CO_calcP >= 3
    printf("[CO][calc_p]after MatCreate for P matrix\n");
#endif
#if dbl_CO_calcP >= 3
    printf("[CO][calc_p]{Create the P matrix} MatCreateSeqAIJ num_row:%d num_coarse_points:%d nnz(coarse_r):%d\n",
           num_row,this->num_coarse_points,Config_params::getInstance()->get_coarse_r());
#endif
    for(i =0; i <num_row; ++i){                             //All nodes in V (i == node id )
        if(debug) std::cout << i << ", ";
        if(vertexs.getNode(i).getIsSeed()){                 //if the node is seed ==> value == 1
            MatSetValue(P,i,seeds_indices[i],1,INSERT_VALUES);
        }
        else{                                       //nodes belongs to F (not seed)
            MatGetRow(WA,i,&ncols,&cols,&vals);     //ncols : number of non-zeros in the row

            //     $$Alert: This should be less than "r"
            sigma_w_ik = 0;             //reset for each new node
            sigma_filter_p = 0 ;        //clear the filter from last value
            filter_nodes_p.clear();     //clear the vector from last values

            for (j=0; j<ncols; j++) {               //calculate the sigma_W_ik
                if(vertexs.getNode(cols[j]).getIsSeed()){    //if J belongs to N_i
                    sigma_w_ik += vals[j];
                }
            }
            for (j=0; j<ncols; j++) {
                if(vertexs.getNode(cols[j]).getIsSeed()){           //if J belongs to N_i
                    // add the node_index and the value to a vector       // changed at 071616-1710 (this one is based on 2-sum paper)
                    filter_nodes_p.push_back( tmp_filter_p(seeds_indices[cols[j]], vals[j]/sigma_w_ik) ); //equation 2 at page 242 min 2-sum paper Safro
                }
            }

    //Sort all nodes_values in descending order
             std::sort(filter_nodes_p.begin(), filter_nodes_p.end(), std::greater<tmp_filter_p>());

//             #if boundary_points == 0    //Experiment boundary points (Jan 9, 2017)
             if(! is_boundary_points_active){
        // Find the max number of fractions
                if( Config_params::getInstance()->get_coarse_r() <   (filter_nodes_p.end() - filter_nodes_p.begin())    ){
                    max_fraction = Config_params::getInstance()->get_coarse_r();
                }
                else{
                    max_fraction = (filter_nodes_p.end() - filter_nodes_p.begin());
                }

        // Select the "max number" of them
                for (auto it = filter_nodes_p.begin(); it != filter_nodes_p.begin()+max_fraction; it++) {
                    sigma_filter_p += it->p_value;
                }
        // Normalize them
                for (auto it = filter_nodes_p.begin(); it != filter_nodes_p.begin()+max_fraction; it++) {
                    //Insert (( W_ij / sigma_E_ik ) / sigma_filter_p ) to normalize the values that make sum of all of them equal to 1
                    MatSetValue(P,i,it->seed_index,( it->p_value  / sigma_filter_p ) ,INSERT_VALUES);
                }
             }else{
//            #else       //boundary_points == 1
                //check the entropy
                float entropy=0;

                for (auto it = filter_nodes_p.begin(); it != filter_nodes_p.begin()+max_fraction; it++) {
                    entropy -= (it->p_value) * log2(it->p_value);
                }
                if(entropy > threshold){ //select all of the fractions
                    max_fraction = std::min(filter_nodes_p.size() -1 , ultimate_estimated_fractions);  // it should not go beyound ultimate number of fractions which cause memory preallocation error in PETSc matrix
//                    for (auto it = filter_nodes_p.begin(); it != filter_nodes_p.end(); it++) {
//                        MatSetValue(P,i,it->seed_index,( it->p_value ) ,INSERT_VALUES);
//                        if(debug) std::cout << "[CO][calc_p] inside filter nodes, i: " << i <<
//                                               ", seed index:" << it->seed_index  <<
//                                               ", value:" << it->p_value  << std::endl;
//                    }
//                    if(debug) std::cout << "[CO][calc_p] fine row: " << i << " max_fraction:" << max_fraction << std::endl;

                }else{  // Find the max number of fractions
                    if( Config_params::getInstance()->get_coarse_r() <  (filter_nodes_p.end() - filter_nodes_p.begin()) ){
                        max_fraction = Config_params::getInstance()->get_coarse_r();
                    }
                    else{
                        max_fraction = (filter_nodes_p.end() - filter_nodes_p.begin());
                    }
//                    // Select the "max number" of them
//                    for (auto it = filter_nodes_p.begin(); it != filter_nodes_p.begin()+max_fraction; it++) {
//                        sigma_filter_p += it->p_value;
//                    }
//                    // Normalize them
//                    for (auto it = filter_nodes_p.begin(); it != filter_nodes_p.begin()+max_fraction; it++) {
//                        //Insert (( W_ij / sigma_E_ik ) / sigma_filter_p ) to normalize the values that make sum of all of them equal to 1
//                        MatSetValue(P,i,it->seed_index,( it->p_value  / sigma_filter_p ) ,INSERT_VALUES);
//                    }
                }

                // Select the "max number" of them
                for (auto it = filter_nodes_p.begin(); it != filter_nodes_p.begin()+max_fraction; it++) {
                    sigma_filter_p += it->p_value;
                }
                // Normalize them
                for (auto it = filter_nodes_p.begin(); it != filter_nodes_p.begin()+max_fraction; it++) {
                    //Insert (( W_ij / sigma_E_ik ) / sigma_filter_p ) to normalize the values that make sum of all of them equal to 1
                    MatSetValue(P,i,it->seed_index,( it->p_value  / sigma_filter_p ) ,INSERT_VALUES);
                }
             }


//            #endif

            MatRestoreRow(WA,i,&ncols,&cols,&vals);        //Frees any temporary space allocated by MatGetRow()
        }
    }
//    exit(1);
#if dbl_CO_calcP >= 3
    printf("[CO][calc_p] before Assembly for P matrix\n");
#endif

    MatAssemblyBegin(P,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(P,MAT_FINAL_ASSEMBLY);

#if dbl_CO_calcP >=7
    printf("[CO][calc_p] P Matrix:\n");                                               //$$debug
    MatView(P,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif

#if dbl_CO_calcP >=5
    PetscInt m,n;                                   // m is the number of rows (current number of nodes)
    MatGetSize(P,&m,&n);                            // n is the number of columns/seeds (number of rows for next level)
    printf("[CO][calc_p] P dim [%d,%d]\n",m,n);                            //$$debug
#endif
    return P;
}





Mat Coarsening::calc_P_without_shrinking(Mat& WA, Vec& vol,std::vector<NodeId>& v_seeds_indices, cs_info& ref_info) {

    PetscInt num_row,i;
    MatGetSize(WA,&num_row,0);    //num_row returns the number of rows globally
    v_seeds_indices.reserve(num_row);

#if dbl_CO_calcP >= 1         //calculate the stat for number of edges
    int sum_nnz =0;
    std::vector<tmp_degree> stat_degree_;
    stat_degree_.reserve(num_row);
    PetscInt ncols;
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;

    for(i =0; i <num_row; i++){
        MatGetRow(WA,i,&ncols,&cols,&vals); //ncols : number if non-zeros in the row
        sum_nnz += ncols;                       //$$debug
        stat_degree_.push_back(tmp_degree(i,(int)ncols));
        MatRestoreRow(WA,i,&ncols,&cols,&vals);       //Frees any temporary space allocated by MatGetRow()
    }//end of for each row (num_row)

    ref_info.num_point = num_row;
    ref_info.num_edge = sum_nnz / 2;
    std::cout <<"[CO][calc_p]{" << this->cc_name <<"} number of rows:"<< num_row <<
                "\t\tedges:"<< ref_info.num_edge << std::endl;
    std::sort(stat_degree_.begin(), stat_degree_.end(), std::greater<tmp_degree>());     //Sort all edges in descending order

    ref_info.min_num_edge = stat_degree_[num_row-1].degree_;
    ref_info.avg_num_edge = (sum_nnz/2) / num_row;
    ref_info.max_num_edge = stat_degree_[0].degree_;
    std::cout <<"[CO][calc_p]{" << this->cc_name <<"} Degrees\t Max:" << ref_info.max_num_edge <<
                "\t\tMin:" << ref_info.min_num_edge << "\t\tAvg:"<< ref_info.avg_num_edge  <<  std::endl;
#endif



//========================= Create the P matrix ===========================
// rows : fine points        => row_dimension : num_rows
// columns : coarse points   => col_dimension : count the coarse points => num_col
    Mat         m_P;
    // each row has only 1 non zero, which is it self
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row, 1,PETSC_NULL, &m_P);
#if dbl_CO_calcP >= 3
    printf("[CO][calc_p]after MatCreate for P matrix\n");
#endif
    for(i =0; i <num_row; ++i){
        MatSetValue(m_P,i,i,1,INSERT_VALUES);       // create a diagonal matrix
        v_seeds_indices.push_back(i);
    }
    MatAssemblyBegin(m_P,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_P,MAT_FINAL_ASSEMBLY);
#if dbl_CO_calcP >=0
    printf("[CO][calc_p] P Matrix:\n");                                               //$$debug
    MatView(m_P,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif
//    exit(1);
    return m_P;
}





//==================== Calculate Aggregate data matrix====================
Mat Coarsening::calc_aggregate_data(Mat& P, Mat& data, Vec& v_vol, std::vector<NodeId>& v_seed_index) {
    ETimer t_calc_agg_data;
    if(Config_params::getInstance()->get_cs_use_real_points()){
        std::cout << "[CO][CAD] Use Real points" << std::endl;
        Mat m_dt_c;
        IS              is_seed_;
        PetscInt        * ind_seed_;
        PetscMalloc1(v_seed_index.size(), &ind_seed_);

        std::cout << "[CO][CAD] number of seeds:" << v_seed_index.size() << std::endl;
        for (unsigned int i = 0; i != v_seed_index.size(); i++) {
    //            std::cout << i << ":"<< v_seed_index[i]  << std::endl;
            ind_seed_[i]=  v_seed_index[i];
        }

        std::sort(ind_seed_, ind_seed_ + v_seed_index.size());
        ISCreateGeneral(PETSC_COMM_SELF,v_seed_index.size(), ind_seed_,PETSC_COPY_VALUES,&is_seed_);
        PetscFree(ind_seed_);

        MatGetSubMatrix(data, is_seed_, NULL,MAT_INITIAL_MATRIX,&m_dt_c);
        ISDestroy(&is_seed_);
        t_calc_agg_data.stop_timer("[CO][CAD]");
        return m_dt_c;

    }else{      // default method which calculates the fake points for the coarser level
//        std::cout << "[CO][CAD] Calc Fake points" << std::endl;
        Mat m_dt_c;
    #if dbl_CO_CAD >= 7
        printf("[CO][CAD] data matrix:\n");                           //$$debug
        MatView(data,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
        printf("[CO][CAD] P matrix:\n");                              //$$debug
        MatView(P,PETSC_VIEWER_STDOUT_WORLD);                                   //$$debug
    #endif
    #if dbl_CO_CAD >= 3
        PetscInt num_row=0,num_col=0;
        MatGetSize(data,&num_row,&num_col);
        printf("[CO][CAD] INPUT data dimension row:%d, col:%d\n",num_row, num_col);

        MatGetSize(P,&num_row,&num_col);
        printf("[CO][CAD] INPUT P dimension row:%d, col:%d\n",num_row, num_col);


    #endif
    ///---- Normalize the Volumes #1 -----
        Vec v_vol_normal;
        normalize_vector(v_vol,v_vol_normal);

    ///---- create diagonal matrix from normalized volume #2 -----
        PetscInt num_fine;
        VecGetSize(v_vol,&num_fine);
        Mat m_VnormDiag;
        MatCreateSeqAIJ(PETSC_COMM_SELF,num_fine,num_fine,1,PETSC_NULL, &m_VnormDiag);
        MatDiagonalSet(m_VnormDiag,v_vol_normal,INSERT_VALUES);
        VecDestroy(&v_vol_normal);      // free vol_normal but don't touch v_vol (important)
    ///---- PV = VD * P #3 -----
        Mat m_PV;
        MatMatMult(m_VnormDiag,P,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&m_PV);
        MatDestroy(&m_VnormDiag);

    ///---- DataCoarse(dt_c) = PV' * data #4 -----
        MatTransposeMatMult(m_PV,data, MAT_INITIAL_MATRIX,PETSC_DEFAULT,&m_dt_c);

    ///---- create the PV transpose (PV_trans) #5.1 -----
        Mat m_PV_trans, m_PV_trans_sum_diag;
        MatTranspose(m_PV,MAT_INITIAL_MATRIX,&m_PV_trans);
        MatDestroy(&m_PV);
    #if dbl_CO_CAD >=13
        printf("PV transpose:\n");                                        //$$debug
        MatView(m_PV_trans,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
    ///---- create a vector of sum of each row of PV Transpose matrix (column of PV matrix) #5.2 -----
        Vec v_sum_PV_trans_row, v_sum_PV_trans_row_inverse;
        PetscInt num_row_PV_trans=0;
        MatGetSize(m_PV_trans,&num_row_PV_trans,NULL);      //equal to number of coarse points or seeds
        VecCreateSeq(PETSC_COMM_SELF,num_row_PV_trans,&v_sum_PV_trans_row);
        MatGetRowSum(m_PV_trans,v_sum_PV_trans_row);    //calc sum of each row of PV'
        MatDestroy(&m_PV_trans);
    #if dbl_CO_CAD >= 9
        printf("[CO][CAD] v_sum_PV_trans_row Vector:\n");                                            //$$debug
        VecView(v_sum_PV_trans_row,PETSC_VIEWER_STDOUT_WORLD);                             //$$debug
    #endif
    ///---- calculate the inverse of each element in the vector #5.3 -----
        VecCreateSeq(PETSC_COMM_SELF,num_row_PV_trans,&v_sum_PV_trans_row_inverse); // create new vector
        calc_inverse(v_sum_PV_trans_row, v_sum_PV_trans_row_inverse);


    ///---- create diagonal matrix #5.4 -----
        MatCreateSeqAIJ(PETSC_COMM_SELF,num_row_PV_trans,num_row_PV_trans,1,
                                                                PETSC_NULL, &m_PV_trans_sum_diag);
        // initilize the diag with inverse of sum of PV'
        MatDiagonalSet(m_PV_trans_sum_diag,v_sum_PV_trans_row_inverse,INSERT_VALUES);
        VecDestroy(&v_sum_PV_trans_row_inverse);
    #if dbl_CO_CAD >=7
        printf("[CO][CAD] m_PV_trans_sum_diag matrix:\n");                                        //$$debug
        MatView(m_PV_trans_sum_diag,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
    ///---- finalize the aggregate data #6 -----
        MatMatMult(m_PV_trans_sum_diag,m_dt_c,MAT_INITIAL_MATRIX,PETSC_DEFAULT,&m_dt_c);
    #if dbl_CO_CAD >=7
        printf("[CO][CAD] final aggregate data (m_dt_c) matrix (normalized agg data):\n");                                        //$$debug
        MatView(m_dt_c,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
        MatDestroy(&m_PV_trans_sum_diag);
        t_calc_agg_data.stop_timer("[CO][CAD]");
        return m_dt_c;
    }
}




/*
 *  v_raw is the input vector with raw data
 *  v_norm is the output vector which is normilazied by dividing each
 *      element to max value
 */
int Coarsening::normalize_vector(Vec& v_raw, Vec& v_norm ){
    PetscScalar max_val;
    VecMax(v_raw,NULL,&max_val);     //find the max
#if dbl_CO_vNorm >= 3
    printf("[CO][vNorm] VecMax max_val:%g\n",max_val);
#endif

    PetscInt num_row;
    VecGetSize(v_raw,&num_row);
    VecCreateSeq(PETSC_COMM_SELF,num_row,&v_norm);          // create new vector
#if dbl_CO_vNorm >= 3
    printf("[CO][vNorm] v_raw dim:%d\n",num_row);
#endif

    PetscScalar * arr_raw;
    PetscCalloc1(num_row, &arr_raw);

    VecGetArray(v_raw,&arr_raw);

    PetscScalar tmp_val = 0;
    for (int i =0; i < num_row;i++){
        tmp_val = arr_raw[i] / max_val;
        VecSetValues(v_norm, 1, &i, &tmp_val, INSERT_VALUES);
#if dbl_CO_vNorm >= 7
    printf("[CO][vNorm] i:%d, arr_raw[i]:%g, max_val:%g ,tmp_val:%g\n",i,arr_raw[i],max_val,tmp_val);
#endif
    }
    VecRestoreArray(v_raw,&arr_raw);
    VecAssemblyBegin(v_norm);
    VecAssemblyEnd(v_norm);
    PetscFree(arr_raw);
#if dbl_CO_vNorm >= 7
    printf("[CO][vNorm] v_raw Vector:\n");                                            //$$debug
    VecView(v_raw,PETSC_VIEWER_STDOUT_WORLD);                             //$$debug
#endif
#if dbl_CO_vNorm >= 3
    printf("[CO][vNorm] max volume is :%g\n",max_val);
#endif
#if dbl_CO_vNorm >= 7
    printf("[CO][vNorm] v_vol_normal Vector:\n");                                            //$$debug
    VecView(v_norm,PETSC_VIEWER_STDOUT_WORLD);                             //$$debug
#endif

    return 0;   //Everything is OK
}








/*
 *  v_raw is the input vector with raw data
 *  v_inverse is the output vector
 *  model vec/vec/examples/tutorials/ex6.c
 */
int Coarsening::calc_inverse(Vec& v_raw, Vec& v_inv){
    PetscInt num_row =0 ,i =0;
    VecGetSize(v_raw, &num_row);
    PetscScalar * arr_vals;
    PetscCalloc1(num_row, &arr_vals);

    VecGetArray(v_raw,&arr_vals);

    PetscScalar tmp_val = 0;
    for(i = 0; i < num_row; i++){

        tmp_val = 1 / arr_vals[i];
#if dbl_CO_cInv >= 7
        printf("i:%d, tmp_val:%g\n",i,tmp_val);
#endif
        VecSetValues(v_inv, 1, &i, &tmp_val,INSERT_VALUES);
    }
    VecAssemblyBegin(v_inv);
    VecAssemblyEnd(v_inv);
    VecRestoreArray(v_raw,&arr_vals);
    PetscFree(arr_vals);
#if dbl_CO_cInv >= 7
    printf("[CO][calc_inverse] v_inv vector:\n");
    VecView(v_inv,PETSC_VIEWER_STDOUT_WORLD);
#endif
    VecDestroy(&v_raw);
    return 0;   //Everything is OK
}


//==================== Calculate the coarser WA matrix (WA_c)====================
Mat Coarsening::calc_WA_c(Mat& P, Mat& WA) {
//    MatPtAP(Mat A,Mat P,MatReuse scall,PetscReal fill,Mat *C)     //Creates the matrix product C = P^T * A * P
    Mat WA_c;

    MatPtAP(WA,P,MAT_INITIAL_MATRIX,1,&WA_c); ;
//    MatSetOption(WA_c,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);
#if dbl_CO_calc_WA_c >= 9
    printf("WA_c matrix before removing loops:\n");                                               //$$debug
    MatView(WA_c,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif

//==================== Remove loops from WA_c matrix ====================
    ETimer t_remove_loops;
    Vec v_diag_val;
    PetscInt i= 0 ;
    VecCreate(PETSC_COMM_SELF,&v_diag_val);
    VecSetSizes(v_diag_val,PETSC_DECIDE,this->num_coarse_points);        //number of nodes in coarser level
    VecSetFromOptions(v_diag_val);
    MatGetDiagonal(WA_c,v_diag_val);
    Mat m_diag;

    PetscScalar *v_diag_arr;
    VecGetArray(v_diag_val, &v_diag_arr);
    MatCreateSeqAIJ(PETSC_COMM_SELF,this->num_coarse_points,this->num_coarse_points,1,PETSC_NULL, &m_diag);
//    MatSetOption(m_diag,MAT_IGNORE_ZERO_ENTRIES,PETSC_TRUE);
    for(i=0; i< this->num_coarse_points; i++){
        MatSetValue(m_diag,i,i,v_diag_arr[i],INSERT_VALUES);
    }
    VecRestoreArray(v_diag_val,&v_diag_arr);
    VecDestroy(&v_diag_val);
    MatAssemblyBegin(m_diag,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_diag,MAT_FINAL_ASSEMBLY);

    MatAXPY(WA_c,-1,m_diag,DIFFERENT_NONZERO_PATTERN);

    t_remove_loops.stop_timer("remove loops from WA_c");

#if dbl_CO_calc_WA_c >= 9
    printf("WA_c matrix after removing loops:\n");                                               //$$debug
    MatView(WA_c,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif

#if dbl_CO_calc_WA_c >= 9
    /*
    Vec             v_diag;
    PetscScalar     * v_diag_val;
    MatGetDiagonal(WA_c,v_diag);
    VecGetArray(v_diag, &v_diag_val);
    for(i =0; i <num_row; ++i){
        if(v_diag_val[i] != 0){
            printf("row %d has loop with val %g\n",i,v_diag_val[i]);
        }
    }
    VecRestoreArray(v_diag,&v_diag_val);
    VecDestroy(&v_diag);
    */
#endif
    return WA_c;
}


//==================== Calculate the real volume vector ====================
Vec Coarsening::calc_coarse_volumes(Mat & P, Vec & vol){
    //multiply the vector of volumes to P matrix
    Vec coarser_vol;

    VecCreateSeq(PETSC_COMM_SELF,this->num_coarse_points,&coarser_vol);
    //this one is not working unless, I create the coarser_vol vector
    MatMultTranspose(P,vol,coarser_vol);

#if dbl_CO_calc_coarse_vol >=7
    PetscInt v;
    VecGetSize(coarser_vol,&v);                             //$$debug
    printf("Coarse vol size : %d\n",v);                     //$$debug
    printf("coarser_vol Vector:\n");                                            //$$debug
    VecView(coarser_vol,PETSC_VIEWER_STDOUT_WORLD);                             //$$debug
#endif
//    VecDestroy(&vol);                         //Free the vol vector

#if dbl_CO_calc_coarse_vol >=5
    PetscScalar sum;                                                           //$$debug
    VecSum(coarser_vol, &sum);                                                  //$$debug
    printf("Sum of coarser level vector of Volumes:%g \n",sum);                                 //$$debug
#endif
    return coarser_vol;
}


//==================== Filter weak edges from WA matrix ====================
/*
 * Output is filtered matrix which is passed as input A
 */

void Coarsening::filter_weak_edges(Mat &A, double alfa, int level){
    ETimer t_fwe;
    PetscInt num_row, num_col;
    MatGetSize(A,&num_row, &num_col);
#if dbl_CO_FWE >=7
    printf("[CO][FWE] Input matrix:\n");                                     //$$debug
    MatView(A,PETSC_VIEWER_STDOUT_WORLD);
#endif
    // - - - - create sum of all the weights for each row - - - -
    Vec v_sum_row;
    VecCreateSeq(PETSC_COMM_SELF,num_row,&v_sum_row);
    MatGetRowSum(A, v_sum_row);
#if dbl_CO_FWE >=7
    printf("[CO][FWE] v_sum_row vector:\n");                                     //$$debug
    VecView(v_sum_row,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif
    PetscInt            ncols;
    const PetscInt      *cols;                        //if not NULL, the column numbers
    const PetscScalar   *vals;
    double              avg_weight=0;
    PetscScalar         * a_sum_row;
    PetscInt            * a_nnz_per_row;
    // - - - - define vectors - - - -
    std::vector<std::set<PetscInt>> v_weak_edge(num_row);
    PetscMalloc1(num_row, &a_sum_row);
    PetscMalloc1(num_row, &a_nnz_per_row);
    VecGetArray(v_sum_row,&a_sum_row);
    std::unordered_map<std::pair<PetscInt,PetscInt>, bool, pair_hash> umap_marked;
    umap_marked.reserve(num_row*10);      //reserve the space for all rows  // 10 is not optimized
//    t_fwe.stop_timer("[CO][FWE] checkpoint 1: sum of rows are calculated");
//    ETimer t_mark_weak_edges;
//    std::cout << "[CO][FWE] save edges" << std::endl;
    double trs_=0.00001;
    // - - - - 1st round on matrix to find weak edges - - - -
    for(PetscInt i=0; i<num_row; i++){
        MatGetRow(A,i,&ncols,&cols,&vals);
        int nnz_real=0;                                             //not needed if I don't create another matrix in the end
//        for(int k=0; k<ncols; k++){
//            if( fabs(vals[k]) > trs_ ) {   nnz_real++;    }
//        }

//        avg_weight = a_sum_row[i] / nnz_real;  //calc average edge weight
//        std::cout << "[CO][FWE] average edge weight"<<avg_weight << std::endl;
//        a_nnz_per_row[i] = nnz_real;                        // this is not ultimate num_nnz because, the weak edges are going to be removed later
        a_nnz_per_row[i] = ncols;                        // this is not ultimate num_nnz because, the weak edges are going to be removed later
#if dbl_CO_FWE >=9
//        std::cout << " alfa * aveg_weight:"<< alfa * avg_weight <<", nnz_real:"<<nnz_real<< std::endl;
        std::cout << "alfa * sum_weigth:"<< alfa * a_sum_row[i] << std::endl;
#endif
        for(int j=0; j<ncols; j++){
//            std::cout << "i:"<<i<<" vals[j]:"<< vals[j] << std::endl;
//            if( (i != cols[j]) && (vals[j] < (alfa * avg_weight)) ){
            if( (i != cols[j]) && (vals[j] < (alfa * a_sum_row[i])) ){      //since I don't need to divide it to degree of node which is nnz_real
                //add to ds (i, cols[j])
//                v_weak_edge[i].insert(cols[j]);
                std::pair<PetscInt,PetscInt> edge_pair;
                if(i < cols[j])                                   //keep one element for both (i,j) and (j,i)
                    edge_pair = std::make_pair(i,cols[j]);
                else
                    edge_pair = std::make_pair(cols[j],i);

                auto ismarked = umap_marked.find(edge_pair);
                if(ismarked != umap_marked.end())
                    umap_marked[edge_pair] = true;
                else
                    umap_marked[edge_pair] = false;

#if dbl_CO_FWE >=9
                std::cout << "(i:"<< i<<",j:"<< cols[j] <<")\n";
#endif
            }
        }
        MatRestoreRow(A,i,&ncols,&cols,&vals);
    }
//    t_mark_weak_edges.stop_timer("[CO][FWE] only marking weak edges");
//    t_fwe.stop_timer("[CO][FWE] checkpoint 2: all the weak edges are marked, no update on matrix yet");

    VecRestoreArray(v_sum_row,&a_sum_row);
    PetscFree(a_sum_row);

    VecDestroy(&v_sum_row);
    int cnt_filtered=0;
//    ETimer t_zero_weak_edges;
    for(auto it=umap_marked.begin(); it!=umap_marked.end(); ++it){
        if(it->second){
            MatSetValue(A,it->first.first,it->first.second,0,INSERT_VALUES);    //#optimization : this is not optimize
            MatSetValue(A,it->first.second,it->first.first,0,INSERT_VALUES);
            a_nnz_per_row[it->first.first]--;   //as an edge is removed from row i, in case another matrix is created based on this info
            cnt_filtered++;
        #if dbl_CO_FWE >=9
            std::cout << "(i:"<< it->first.first<<",j:"<< it->first.second <<") is removed and vice versa\n";
        #endif

        }
    }
//    t_zero_weak_edges.stop_timer("[CO][FWE] zero weak edges in original adjacency matrix (without Assembly)");
//    t_fwe.stop_timer("[CO][FWE] checkpoint 3: iterate through map to find all weaks(both side weak) and update on matrix, before assembly");

//    ETimer t_zero_weak_edges_assemble;
    MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A,MAT_FINAL_ASSEMBLY);
//    t_zero_weak_edges_assemble.stop_timer("[CO][FWE] zero weak edges in original adjacency (only Assembly part)");
//    t_fwe.stop_timer("[CO][FWE] checkpoint 4: after assembly");
#if dbl_CO_FWE >=3
//    std::cout << "[CO][FWE]{" << this->cc_name <<"} "<< cnt_filtered /2 <<" edges are filtered from WA_c (coarser level)" << std::endl;
    #if dbl_CO_FWE >=7
        printf("[CO][FWE] filtered matrix:\n");                                     //$$debug
        MatView(A,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    #endif
#endif
//    ETimer t_create_new_WA_matrix;
    PetscInt num_real_non_zero=0;
    Mat m_Anz;
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row,num_row, PETSC_NULL, a_nnz_per_row, &m_Anz);
    PetscFree(a_nnz_per_row);
    MatGetSize(m_Anz,&num_row, &num_col);
//    std::cout << "[CO][FWE] final matrix dimension is ("<<num_row <<"x"<<num_col<<")" << std::endl; // it is a symmetric matrix
    t_fwe.stop_timer("[CO][FWE] checkpoint 5: new matrix is created, before loop to copy all elements");
    // - - - - 2nd round : create clean matrix - - - -
    for(int i=0; i<num_row; i++){
        MatGetRow(A,i,&ncols,&cols,&vals);
        for(int j=0; j<ncols; j++){
            if(fabs(vals[j]) > trs_){   //detect real non-zero elements
                MatSetValue(m_Anz,i,cols[j],vals[j],INSERT_VALUES);
                num_real_non_zero++;
            }
        }
        MatRestoreRow(A,i,&ncols,&cols,&vals);
    }
//    t_fwe.stop_timer("[CO][FWE] checkpoint 6: all elements are copied, before assembly");
    MatAssemblyBegin(m_Anz,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_Anz,MAT_FINAL_ASSEMBLY);
//    t_fwe.stop_timer("[CO][FWE] checkpoint 7: after assembly");
    A=m_Anz;
//    t_create_new_WA_matrix.stop_timer("[CO][FWE] create new WA matrix by moving all elements one by one (not optimized)");
#if dbl_CO_FWE >=3
    PetscInt filtered_edges_ = cnt_filtered /2;
    PetscInt total_edges_ = filtered_edges_ + (num_real_non_zero /2);
    float percentage_ = ((double)filtered_edges_ / (double)total_edges_ ) * 100 ;
    std::cout << "[CO][FWE]{" << this->cc_name << "} level:"<< level << ", all edges:"<< total_edges_ <<", filtered:"
              << filtered_edges_ << ", remained:" << num_real_non_zero /2 << ", filtered:" << std::fixed << std::setprecision(2) << percentage_  <<"%\n" ;


//    PetscPrintf(PETSC_COMM_WORLD, (", %2f\%\n", percentage_ );
    #if dbl_CO_FWE >=7
        printf("[CO][FWE] final matrix:\n");                                     //$$debug
        MatView(A,PETSC_VIEWER_STDOUT_WORLD);
    #endif
#endif

    t_fwe.stop_timer("[CO][FWE] filter weak edges takes");
//    t_fwe.stop_timer("[CO][FWE] checkpoint 8: end");
}















Mat Coarsening::calc_real_weight(Mat& m_WA_c, Mat& m_data_c) {
    ETimer t_crw;
    PetscInt num_row,i,k, ncols_WA_c, ncols_dtc_i, ncols_dtc_k;
    MatGetSize(m_WA_c,&num_row,NULL);
    Mat m_tmp_WA;
    MatDuplicate(m_WA_c,MAT_DO_NOT_COPY_VALUES, &m_tmp_WA); //create an empty matrix with same non zero structure

    CommonFuncs cf;

    const PetscInt    *cols_WA_c, *cols_dtc_i, *cols_dtc_k;
    const PetscScalar *vals_WA_c, *vals_dtc_i, *vals_dtc_k;

    for(i =0; i <num_row; i++){
        MatGetRow(m_WA_c,i,&ncols_WA_c,&cols_WA_c,&vals_WA_c);
        MatGetRow(m_data_c,i,&ncols_dtc_i,&cols_dtc_i,&vals_dtc_i); //get data at row i

        for (k=0; k<ncols_WA_c; k++) {
            MatGetRow(m_data_c,k,&ncols_dtc_k,&cols_dtc_k,&vals_dtc_k); //get data at row k
            //calc the distance between i, k
            double distance = cf.calc_euclidean_dist(ncols_dtc_i, ncols_dtc_k, cols_dtc_i, cols_dtc_k, vals_dtc_i, vals_dtc_k);
            MatRestoreRow(m_data_c,k,&ncols_dtc_k,&cols_dtc_k,&vals_dtc_k);
            MatSetValue(m_tmp_WA, i, cols_WA_c[k], cf.convert_distance_to_weight(distance), INSERT_VALUES);
        }
        MatRestoreRow(m_data_c,i,&ncols_dtc_i,&cols_dtc_i,&vals_dtc_i);
        MatRestoreRow(m_WA_c,i,&ncols_WA_c,&cols_WA_c,&vals_WA_c);
    }
    MatAssemblyBegin(m_tmp_WA,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_tmp_WA,MAT_FINAL_ASSEMBLY);
    MatCopy(m_tmp_WA, m_WA_c, SAME_NONZERO_PATTERN);    //copy the temporary WA matrix to m_WA_c
    MatDestroy(&m_tmp_WA);                               //free the temporary matrix
    t_crw.stop_timer("[CO][CRW] calc real weight");
}




