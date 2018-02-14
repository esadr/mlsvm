#include "../config_params.h"
#include "../config_logs.h"
#include "../solver.h"
#include "../OptionParser.h"
#include "../loader.h"

Config_params* Config_params::instance = NULL;

int main(int argc, char **argv){
    PetscInitialize(&argc, &argv, NULL, NULL);
    //read XML parameters
    Config_params::getInstance()->init_to_default();
    Config_params::getInstance()->read_params("./params.xml", argc, argv);

//    std::string in_fname_p = Config_params::getInstance()->get_p_norm_data_f_name();
//    std::string in_fname_n = Config_params::getInstance()->get_n_norm_data_f_name();

    std::string path = "./debug/";
    std::string in_fname_p_data = path + "m_train_data_p.dat";
    std::string in_fname_n_data = path + "m_train_data_n.dat";



    Loader ld;
    Mat m_p_data, m_n_data;
    m_p_data = ld.load_norm_data_sep(in_fname_p_data);
    m_n_data = ld.load_norm_data_sep(in_fname_n_data);
    PetscInt num_p, num_n, num_p_features, num_n_features;
    MatGetSize(m_p_data, &num_p, &num_p_features);
    MatGetSize(m_n_data, &num_n, &num_n_features);
    std::cout << "P size:" << num_p << "x" << num_p_features
              << ",N size:" << num_n << "x" << num_n_features << std::endl;
    Solver sv;

    sv.stand_alone_train_without_instance_weight(m_p_data, m_n_data,"./debug/sat_svm.model");
//    Config_params::getInstance()->print_params();

    std::cout << "Single SVM train finished successfully!\n";
    PetscFinalize();
    return 0;
}





