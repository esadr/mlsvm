#include "ut_cs.h"
#include "ut_common.h"
#include "loader.h"

//UT_CS::UT_CS(){}


void UT_CS::test_filtering_weak_edges(){
    UT_Common utc;
    Mat m_WA;
    utc.load_matrix("./data/WA_for_filtering.dat", m_WA,1);
    filter_weak_edges(m_WA,.5,0);

    printf("[UT_CS][test_filtering_weak_edges] filtered Matrix:\n");                                               //$$debug
    MatView(m_WA,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

}


void UT_CS::test_calc_p(){
    UT_Common utc;
    Mat m_WA, m_P;
    utc.load_matrix("./data/ut/WA_sample_points_10.dat", m_WA,1);
//    utc.load_matrix("./data/ut/WA_sample_points_modified.dat", m_WA,1);
    PetscInt num_nodes;
    MatGetSize(m_WA,&num_nodes,NULL);

    Vec v_vol;
    std::vector<NodeId> v_seeds_indices;
    cs_info ref_info;

    Loader ld;
//    v_vol = ld.init_volume(1,num_nodes);
    v_vol = ld.read_input_vector("./data/ut/sample_points_volume.dat");
    m_P = calc_P(m_WA, v_vol, v_seeds_indices, ref_info);

    printf("[UT_CS][test_calc_p] P Matrix:\n");                                               //$$debug
    MatView(m_P, PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

}

void UT_CS::test_calc_p(Mat& m_WA){
    UT_Common utc;
    Mat m_P;
    PetscInt num_nodes;
    MatGetSize(m_WA,&num_nodes,NULL);

    Vec v_vol;
    std::vector<NodeId> v_seeds_indices;
    cs_info ref_info;

    Loader ld;
//    v_vol = ld.read_input_vector("./data/ut/raw_60_points_5_groups_volume.dat");
    v_vol = ld.read_input_vector("./data/ut/sample_points_volume.dat");
    m_P = calc_P(m_WA, v_vol, v_seeds_indices, ref_info);

    printf("[UT_CS][test_calc_p] P Matrix:\n");                                               //$$debug
    MatView(m_P, PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

}
