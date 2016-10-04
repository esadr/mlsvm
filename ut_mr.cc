//#include "ut_mr.h"
//#include <iostream>
//#include "main_recursion.h"
//#include "model_selection.h"

////ut_MR::ut_MR()
////{
////}

//void ut_MR::test_calc_center(){
////(Mat& m_neigh_Data, Vec& v_vol, std::vector<std::vector<PetscInt>>& vv_parts, Mat& m_parts, Mat& m_centers){
//    Mat m_dt, m_parts, m_centers ;
//    Vec v_vol;
//    load_matrix("../mlsvm/data/ut/calc_center_data.dat", m_dt,1);
//    load_matrix("../mlsvm/data/ut/calc_center_parts.dat", m_parts,1 );
//    load_vec("../mlsvm/data/ut/calc_center_vol.dat", v_vol,1 );
//    int num_parts = 3;
//    std::vector<std::vector<PetscInt>> vv_parts(num_parts);
//    read_parts(vv_parts, m_parts,1);

////    MainRecursion mr("fake_test_file_name")  ;
//    MainRecursion mr;
//    std::vector<PetscScalar> v_sum_vol_parts(num_parts);
//    mr.calc_center(m_dt,v_vol,vv_parts,m_parts,m_centers, v_sum_vol_parts);

//}

//void ut_MR::test_calc_distances(){
//    Mat m_P, m_N;

//    load_matrix("../mlsvm/data/ut/A.dat", m_P,0);
//    load_matrix("../mlsvm/data/ut/B.dat", m_N,0);
//    PetscInt            num_part_p, num_part_n, num_features;
//    MatGetSize(m_P, &num_part_p, &num_features);
//    MatGetSize(m_N, &num_part_n, NULL);
//    std::cout  << "num_part_p: " << num_part_p <<"\n";
//    std::cout  << "num_part_n: " << num_part_n <<"\n";
//    std::cout  << "num_features: " << num_features <<"\n";
////    std::vector<std::vector<double>> dist_p(num_part_p);
//    std::vector<std::vector<std::pair<int, double>>> dist_p(num_part_p);

////    MainRecursion mr("fake_test_file_name")  ;
//    MainRecursion mr;
////    mr.calc_distances(num_part_p,num_part_n,num_features,m_P,m_N,dist_p);
//    Mat m_dist;
//    mr.calc_distances(num_part_p,num_part_n,m_P,m_N,m_dist);

////    mr.find_opposite_sets(num_part_p,num_part_n,num_features,dist_p);
//}





//void ut_MR::read_parts(std::vector<std::vector<PetscInt>>& vv_parts, Mat& m_parts, bool print){



//    PetscInt            i, j, ncols, num_parts, num_points;
//    const PetscInt      *cols;
//    const PetscScalar   *vals;

//    MatGetSize(m_parts, &num_parts, &num_points);
//    std::cout << "mparts dim "<< num_parts <<"," << num_points <<std::endl ;
//    for(i=0; i < num_parts; i++) {
//        MatGetRow(m_parts,i,&ncols,&cols,&vals);
//        for(j=0; j < ncols; j++){
//            vv_parts[i].push_back(cols[j]);
//        }
//        MatRestoreRow(m_parts,i,&ncols,&cols,&vals);
//    }
//    if(print){
//        std::cout  << "vv_parts is:\n";
//        for(i=0; i<num_parts;i++){
//            for(unsigned int k=0; k < vv_parts[i].size(); k++){
//                std::cout << vv_parts[i][k] <<"," ;
//            }
//            std::cout << std::endl ;
//        }
//    }
//}

//void ut_MR::test_ms_ud(){
//    Mat m_dt_p, m_dt_n;
//    Vec v_vol_p, v_vol_n;
//    load_matrix("../mlsvm/data/ut/check_svm_SV_indices_p.dat", m_dt_p,1);
//    load_matrix("../mlsvm/data/ut/check_svm_SV_indices_n.dat", m_dt_n,1);


//    ModelSelection ms;
//    /* modify:  svm_kernel_type = 0  (for linear kernel)
//     *          ms_status = 0
//     */
//    ms.UD(m_dt_p, v_vol_p, m_dt_n, v_vol_n, 1, 100, 0.7, 2);
//}

//void ut_MR::test_mr_get_part(){
//    Mat m_neigh_WA;
//    Vec v_neigh_vol;
//    load_matrix("./data/m_neigh_WA_n_level3.dat", m_neigh_WA,0);
//    load_vec("./data/v_neigh_Vol_n_level3.dat", v_neigh_vol,0);

//    int num_partitions = 2;
//    Mat m_parts;
//    std::vector<std::vector<PetscInt>> vv_parts(num_partitions);
//    MainRecursion mr;
//    mr.get_parts(m_neigh_WA, v_neigh_vol, vv_parts, m_parts);   // get partitions (Metis)

//}
