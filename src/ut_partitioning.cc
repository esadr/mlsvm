#include "ut_partitioning.h"
#include "vector"
#include "partitioning.h"


void UT_Partitioning::test_find_groups(){
//    int num_part_p, int num_part_n, Mat& m_dist, std::vector<std::pair<int, int> >& v_groups
    int num_part_p = 3;
    int num_part_n = 4;
    std::vector<std::pair<int, int> > v_groups;
    Mat m_dist;
    std::string file_name = "./data/ut/m_dist_7x7.dat";
    UT_Common ut_cm;
    ut_cm.load_matrix(file_name.c_str(), m_dist,1);

    Partitioning pr;
    pr.find_groups(num_part_p,num_part_n, m_dist, v_groups);

}

void UT_Partitioning::test_get_parts(){
    Mat m_WA, m_parts;
    Vec v_vol;
    std::string mat_file_name = "./data/m_neigh_WA_6.dat";
    std::string vec_file_name = "./data/v_neigh_vol_6.dat";
    UT_Common ut_cm;
    ut_cm.load_matrix(mat_file_name.c_str(), m_WA,0);
    ut_cm.load_vec(vec_file_name.c_str(), v_vol,0);
    int num_partitions_p = 13;
    Partitioning pr;
    std::vector<std::vector<PetscInt>> vv_parts_p(num_partitions_p);
    pr.get_parts(m_WA, v_vol, num_partitions_p, 1, vv_parts_p, m_parts );

}
