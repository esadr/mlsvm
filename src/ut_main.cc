#include "ut_common.h"
#include "ut_kf.h"
#include "ut_mr.h"
#include "ut_ms.h"
#include "config_params.h"
#include "ut_partitioning.h"
#include "ut_cf.h"
#include "ut_cs.h"
#include "ut_ld.h"
#include "ut_clustering_rf.h"

Config_params* Config_params::instance = NULL;

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);

//    Config_params::getInstance()->init_to_default();
   Config_params::getInstance()->read_params("./params.xml", argc, argv);
//   Config_params::getInstance()->print_params();

//    ut_MR utmr;
//    utmr.test_calc_distances();
//    utmr.test_calc_center();

//    utmr.test_ms_ud();
//    utmr.test_mr_get_part();
  

    /* 061016-1711 test kfold */
//    UT_KF utkf;
//    utkf.cross_validation_simple();

    /* 061216-1900 test kfold */
//    Config_params::getInstance()->debug_only_set_p_norm_data_path_file_name("./datasets/ut_kfold_20_zsc_data.dat");
//    Config_params::getInstance()->debug_only_set_n_norm_data_path_file_name("./datasets/ut_kfold_40_zsc_data.dat");
//     UT_KF utkf;
//     utkf.cross_validation();
    
//      UT_MS utms;
//      utms.test_params();
    
//     /* 061516-1734 test [PR][find_group] */
//      UT_Partitioning ut_pr;
// //      ut_pr.test_find_groups();	
//      ut_pr.test_get_parts();

/* 071516-1530 test [CF][zscore] */
//      UT_CF ut;
//      ut_pr.test_find_groups();	
//      ut.t_zscore_array();	
    
//     /* 110316-1104 test [CS][filter_weak_edges] after new changes */    
//    UT_CS utcs;
//	utcs.test_filtering_weak_edges();

//    /* 122116-1744 test [CS][calc_p] test version 0.0.34 */
//    UT_CS utcs;
//    utcs.test_calc_p();

    /* 122216-1244 test [LD][load_flann_binary] test version 0.0.34 */
    /* reused at 010917-1505*/
//    Mat m_WA;
//    UT_LD utld;
//    m_WA = utld.test_load_flann_binary();

//    UT_CS utcs;
//    utcs.test_calc_p();

    ut_Clustering_rf utrf;
    utrf.test_calc_new_center();


    
    PetscFinalize();
    return 0;
}

