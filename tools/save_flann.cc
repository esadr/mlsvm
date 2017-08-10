#include <iostream>
#include "../etimer.h"
#include "../k_fold.h"
#include "../config_params.h"

Config_params* Config_params::instance = NULL;

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    ETimer t_all;
    /* ------------------------- Read Parameters ---------------------------- */
//    Config_params::program_parts prg_name;
//    prg_name = Config_params::program_parts::flann;

    Config_params::getInstance()->read_params("./params.xml", argc, argv, Config_params::flann;); //@ 040317-1842
    /* ------------------------- Prepare FLANN Commands---------------------------- */
    std::string str_NN {std::to_string(Config_params::getInstance()->get_nn_number_of_neighbors())};
    std::string str_nn_distance_type {std::to_string(Config_params::getInstance()->get_nn_distance_type())};
    std::string str_NN_params {" " + str_NN + " "+ str_nn_distance_type} ;

    const char* env_p = std::getenv("PY_PATH");
    if(env_p == NULL ){
        printf("\n[Main] the PY_PATH is not defined. Exit(1)!\n\n");
        printf("Please define the PY_PATH to the python location. For more information, read the README file\n\n");
        exit(1);
    }
    std::string py_path= env_p;

    if(Config_params::getInstance()->get_nn_number_of_classes() == 1){
        std::string sh_command = py_path +"/python ./scripts/flann.py "+ Config_params::getInstance()->get_single_norm_data_f_name() + str_NN_params;
        /* ------------------------- Run FLANN ---------------------------- */
        std::cout << "[Main] sh_command for 1 class:" << sh_command << std::endl;
        system(sh_command.c_str());
    }else{  // two classes
        /* ------------------------- Divide Data ---------------------------- */
        ETimer t_kf;
        k_fold kf;
        kf.read_in_data();
        kf.divide_data(true);   //true: export the divided data into files
        t_kf.stop_timer("[Main] reading and deviding data and writing to files in k-fold class");



        std::string min_command = py_path +"/python ./scripts/flann.py "+ Config_params::getInstance()->get_p_norm_data_f_name()+ str_NN_params;
        std::string maj_command = py_path +"/python ./scripts/flann.py "+ Config_params::getInstance()->get_n_norm_data_f_name()+ str_NN_params;
        /* ------------------------- Run FLANN ---------------------------- */
        ETimer t_flann_min;
        std::cout << "[Main] python min_command:" << min_command << std::endl;
        system(min_command.c_str());
        t_flann_min.stop_timer("[Main] flann for minority class");

        ETimer t_flann_maj;
        std::cout << "[Main] maj_command:" << maj_command << std::endl;
        system(maj_command.c_str());
        t_flann_maj.stop_timer("[Main] flann for majority class");
    }
    t_all.stop_timer("[Main] whole saving flann");
    PetscFinalize();
    return 0;
}
