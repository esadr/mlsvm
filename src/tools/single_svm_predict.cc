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

    std::string in_model = "./debug/sat_svm.model";
//    std::string in_model = "./debug//diabet/dap_d.train.model";
//    std::string in_test_data = Config_params::getInstance()->get_test_ds_f_name();
    std::string in_test_data  = "./debug/m_test_data.dat";
//    bool errors = false;
//    if (in_fname_p.empty()){
//        std::cout <<  "Must indicate an train data file name for minority class using --sat_p option" << std::endl;
//        errors = true;
//    }
//    if (in_test_data.empty()){
//        std::cout <<  "Must indicate an train data file name for majority class using --sap_td option" << std::endl;
//        errors = true;
//    }

//    if(errors)
//        exit(1);

//    Config_params::getInstance()->print_params();

//void Solver::stand_alone_train(Mat& m_data_p, Mat& m_data_n, svm_parameter& params, std::string model_fname){
    Loader ld;
    Mat m_test_data;
    m_test_data = ld.load_norm_data_sep(in_test_data);

    svm_model * trained_model = svm_load_model(in_model.c_str());

    summary final_summary;
    Solver sv;
    sv.set_local_model(trained_model);
    sv.test_predict(m_test_data, final_summary );
    Config_params::getInstance()->print_summary(final_summary,"stand alone predict");
    std::cout << "Single SVM predict finished successfully!\n";
    PetscFinalize();
    return 0;
}


