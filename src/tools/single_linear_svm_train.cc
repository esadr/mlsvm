#include "../config_params.h"
#include "../config_logs.h"
#include "../solver_linear.h"
#include "../OptionParser.h"
#include "../loader.h"

Config_params* Config_params::instance = NULL;

int main(int argc, char **argv){
//    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscInitialize(NULL, NULL, NULL, NULL);
    //read XML parameters
//    Config_params::getInstance()->read_params("./params.xml", argc, argv);
    Config_params::getInstance()->read_params("./params.xml", argc, argv, Config_params::sa_svm_linear);

//    std::string in_fname_p = Config_params::getInstance()->get_p_norm_data_f_name();
//    std::string in_fname_n = Config_params::getInstance()->get_n_norm_data_f_name();
    std::string in_fname_p_data = Config_params::getInstance()->get_ds_path() + "/m_train_data_p.dat";
    std::string in_fname_n_data = Config_params::getInstance()->get_ds_path() + "/m_train_data_n.dat";
    std::string in_fname_p_vol  = Config_params::getInstance()->get_ds_path() + "/v_vol_p.dat";
    std::string in_fname_n_vol  = Config_params::getInstance()->get_ds_path() + "/v_vol_n.dat";

    std::cout << "p data:" << in_fname_p_data  << std::endl;
    std::cout << "n data:" << in_fname_n_data  << std::endl;

    // - - - - - load the input files - - - - -
    Loader ld;
    Mat m_p_data, m_n_data;
    m_p_data = ld.load_norm_data_sep(in_fname_p_data);
    m_n_data = ld.load_norm_data_sep(in_fname_n_data);

    PetscInt num_p, num_n, num_p_features, num_n_features;
    MatGetSize(m_p_data, &num_p, &num_p_features);
    MatGetSize(m_n_data, &num_n, &num_n_features);
    std::cout << "P data size:" << num_p << "x" << num_p_features
              << ",N data size:" << num_n << "x" << num_n_features << std::endl;

    Vec v_p_vol, v_n_vol;
//    v_p_vol = ld.read_input_vector(in_fname_p_vol);
//    v_n_vol = ld.read_input_vector(in_fname_n_vol);
    v_p_vol = ld.init_volume(1,num_p);
    v_n_vol = ld.init_volume(1,num_n);


    VecGetSize(v_p_vol, &num_p);
    VecGetSize(v_n_vol, &num_n);
    std::cout << "P vol size:" << num_p << ",N vol size:" << num_n << std::endl;

    // - - - - - train the model - - - - -
    Solver sv(1); //call stand alone constructor which doesn't need any parameter

    sv.stand_alone_train_linear(m_p_data, v_p_vol, m_n_data,v_n_vol,"./debug/sat_linear_svm.model");
//    Config_params::getInstance()->print_params();

    std::cout << "Single SVM train finished successfully!\n";
    PetscFinalize();
    return 0;
}





