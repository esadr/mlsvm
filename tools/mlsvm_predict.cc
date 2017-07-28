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
    Config_params::getInstance()->read_params_predict("./params.xml", argc, argv);


    // we suppose the name of the test data set is dsname_test_data.dat
    std::string in_test_data  = Config_params::getInstance()->get_ds_name()+ "_test_data.dat";

    Loader ld;
    Mat m_test_data;
    m_test_data = ld.load_norm_data_sep(in_test_data);


    std::string in_model = "./debug/sat_svm.model";
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

