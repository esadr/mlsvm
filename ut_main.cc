#include "ut_mr.h"
#include "config_params.h"

Config_params* Config_params::instance = NULL;

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);


    Config_params::getInstance()->init_to_default();
    Config_params::getInstance()->read_params("./params.xml", argc, argv);



    ut_MR utmr;
//    utmr.test_calc_distances();

    utmr.test_ms_ud();

    PetscFinalize();
    return 0;
}

