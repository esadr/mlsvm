#include "ut_ld.h"
#include <string>
#include "common_funcs.h"

using namespace std;

UT_LD::UT_LD(){}

Mat UT_LD::test_load_flann_binary(){
    Mat m_WA;
//    const string indice_f_name_ {"./data/ut/raw_60_points_5_groups_indices.dat"};
//    const string dist_f_name_ {"./data/ut/raw_60_points_5_groups_dists.dat"};
    const string indice_f_name_ {"./data/ut/sample_points_10_indices.dat"};
    const string dist_f_name_ {"./data/ut/sample_points_10_dists.dat"};
    Loader ld(indice_f_name_,dist_f_name_);
    m_WA = ld.load_flann_binary();

    printf("[UT_LD][test_load_flann_binary] WA Matrix:\n");                                               //$$debug
    MatView(m_WA, PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

    CommonFuncs cf;
    cf.exp_matrix(m_WA,"./data/ut/","WA_sample_points_10.dat","UT_LD::test_load_flann_binary");

    return m_WA;
}
