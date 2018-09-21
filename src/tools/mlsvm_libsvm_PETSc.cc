#include "../config_params.h"
#include "../convertor.h"

Config_params* Config_params::instance = NULL;

int main(int argc, char **argv){
//    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscInitialize(NULL, NULL, NULL, NULL);
    //read XML parameters
    Config_params::getInstance()->read_params("./params.xml",
                                    argc, argv, Config_params::convert_files);

    Mat A;
    Vec lbl;
    std::string str_in_fname = Config_params::getInstance()->get_ds_path()
                        + Config_params::getInstance()->get_ds_name();
    Convertor cv;
    cv.Libsvm_file_to_PETSc_format( str_in_fname, A,lbl);

    std::cout << "Convert LibSVM to PETSc binary format finished successfully!\n";
    PetscFinalize();
    return 0;
}

