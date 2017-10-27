#include "config_params.h"
#include "loader.h"

Config_params* Config_params::instance = NULL;


int main(int argc, char **argv){
    PetscInitialize(NULL, NULL, NULL, NULL);
    //read XML parameters
    Config_params::getInstance()->read_params("./params.xml", argc, argv, Config_params::ld_metis);

    Loader ld;
    ld.load_metis();

    std::cout << "Loading the metis PETSc binary format finished successfully!\n";
    PetscFinalize();
    return 0;
}




