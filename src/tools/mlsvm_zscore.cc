#include "../config_params.h"
#include "../preprocessor.h"
#include "../loader.h"
#include "../common_funcs.h"

Config_params* Config_params::instance = NULL;

int main(int argc, char **argv){
//    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscInitialize(NULL, NULL, NULL, NULL);
    //read XML parameters
    Config_params::getInstance()->read_params("./params.xml", argc, argv, Config_params::zscore );
    const std::string models_path = "./svm_models/";

    Mat m_raw_data, m_raw_data_T, m_normalized_T, m_normalized ;
//    check_input_parameters();
    //read the data as dsname_data.dat
    Loader ld;
    std::string raw_file_path_name = Config_params::getInstance()->get_ds_path() +
                    Config_params::getInstance()->get_ds_name() + "_data.dat";
    m_raw_data = ld.load_norm_data_sep(raw_file_path_name);

    // transpose the raw data matrix
    MatTranspose(m_raw_data,MAT_INITIAL_MATRIX,&m_raw_data_T);
    MatDestroy(&m_raw_data);
    std::cout << "MatTranspose is done\n";

    Preprocessor pr;
    m_normalized_T = pr.normalizeDataZscore_Transposed(m_raw_data_T, 1);
    MatDestroy(&m_raw_data_T);
    std::cout << "Normalization is done\n";

    // transpose the normalized transposed data matrix
    MatTranspose(m_normalized_T,MAT_INITIAL_MATRIX,&m_normalized);
    MatDestroy(&m_normalized_T);
    std::cout << "Transpose back to original form is done\n";


    //export to file
    std::string normalized_file_path_name = Config_params::getInstance()->get_ds_path() +
                    Config_params::getInstance()->get_ds_name() + "_zsc_data.dat";

    CommonFuncs cf;
    cf.exp_matrix(m_normalized, "", normalized_file_path_name , "mlsvm_zscore" );
    MatDestroy(&m_normalized);

    std::cout << "Data is normalized successfully!\n";
    PetscFinalize();
    return 0;
}
