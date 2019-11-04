#include "../config_params.h"
#include "../convertor.h"

Config_params* Config_params::instance = NULL;


int main(int argc, char **argv){
    PetscInitialize(&argc, &argv, NULL, NULL);
    //read XML parameters
    Config_params::getInstance()->read_params("./params.xml", argc, argv, Config_params::convert_files);

    Convertor cv;
    cv.CSV_file_to_PETSc_Vector();

    std::cout << "Convert CSV to PETSc Vector format finished successfully!\n";
    PetscFinalize();
    return 0;
}





