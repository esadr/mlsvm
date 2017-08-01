#include "../config_params.h"
#include "../config_logs.h"
#include "../solver.h"
#include "../OptionParser.h"
#include "../loader.h"
#include "fstream"

Config_params* Config_params::instance = NULL;

int main(int argc, char **argv){
    PetscInitialize(&argc, &argv, NULL, NULL);
    //read XML parameters
//    Config_params::getInstance()->init_to_default();
    Config_params::getInstance()->read_params("./params.xml", argc, argv, Config_params::prediction );


    // we suppose the name of the test data set is dsname_test_data.dat
    std::string in_test_data  = Config_params::getInstance()->get_ds_name()+ "_test_data.dat";

    Loader ld;
    Mat m_test_data;
    m_test_data = ld.load_norm_data_sep(in_test_data);

    /*
     * we need the experiment id and k-fold id to find the model
     * the model file is saved with a name that includes this information
     * Using the models.summary file, we can find the number of models at a specific level
     * which is selected as the best model among all the levels for a specific k-fold of an experiment
     */


    //find the level in the summary
    std::fstream in_file;
    std::string fname = "./svm_models/" + Config_params::getInstance()->get_ds_name() + "_models.summary";
    in_file.open(fname);
    std::vector<std::string> v_lines;
    std::string line;
    while(in_file >> line){
        v_lines.push_back(line);
    }

    in_file.close();

    for(auto it = v_lines.begin(); it != v_lines.end(); it++){
        std::cout << *it << std::endl;
    }
    //make the name for the in_model
    exit(1);


    std::string in_model = "./debug/sat_svm.model";

    // for 1 model




    // for many models


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

