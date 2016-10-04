#include "ut_common.h"
#include "ut_kf.h"
#include "ut_mr.h"
#include "ut_ms.h"
#include "config_params.h"
#include "ut_partitioning.h"
#include "ut_cf.h"

Config_params* Config_params::instance = NULL;

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);

   Config_params::getInstance()->init_to_default();
   Config_params::getInstance()->read_params("./params.xml", argc, argv);

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
    
     UT_MS utms;
     utms.test_params();
    
//     /* 061516-1734 test [PR][find_group] */
//      UT_Partitioning ut_pr;
// //      ut_pr.test_find_groups();	
//      ut_pr.test_get_parts();

/* 071516-1530 test [CF][zscore] */
//      UT_CF ut;
//      ut_pr.test_find_groups();	
//      ut.t_zscore_array();	
    
    
    PetscFinalize();
    return 0;
}

