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
    const std::string models_path = "./svm_models/";


    // check the input is passed
    int req_exp = Config_params::getInstance()->get_experiment_id();
    int req_kf = Config_params::getInstance()->get_kfold_id();
    if(req_exp < 0){
        std::cout << "Please specify the experiment id using -x!\n Exit" << std::endl;
        exit(1);
    }
    if(req_kf < 0){
        std::cout << "Please specify the k-fold id using -k!\n Exit" << std::endl;
        exit(1);
    }


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
    std::string fname = models_path + Config_params::getInstance()->get_ds_name() + "_models.summary";
//    std::cout << fname << std::endl;
//exit(1);
    in_file.open(fname);
    std::vector<std::pair<int,int>> v_lines;
    std::string line;

    while(std::getline(in_file, line)){
        std::size_t pos1, pos2, pos3 =0;
        pos1 = line.find(':');
        pos2 = line.find(':',pos1+1);
        pos3 = line.find(',');
//        std::cout << pos1 << " , " << pos3 << " , " << pos2 << std::endl;
//        std::cout << line.substr(pos1+1, pos3-pos1-1) << " -- " << line.substr(pos2+1, line.length()) << std::endl;
        if((pos1 > line.length() || pos2 > line.length() || pos3> line.length() )){
            std::cout << "Wrong line at index "<< v_lines.size() << " in the summary file!" << std::endl;
            exit(1);
        }
        int tmp_level_id = stoi(line.substr(pos1+1, pos3-pos1-1));
        int tmp_num_models = stoi(line.substr(pos2+1, line.length()));
//        std::cout << "l:" << tmp_level_id << ",n:" << tmp_num_models << std::endl;
        v_lines.push_back(std::make_pair(tmp_level_id, tmp_num_models));
    }
    in_file.close();

//    exit(1);

    int total_exp = v_lines[0].first;
    int total_kf = v_lines[0].second;
    std::cout << "Te:" << total_exp<< ",Re:" << req_exp << ",Tk:" << total_kf<< ",Rk:" << req_kf << std::endl;

    //check the input exp_id and kf_id
    if(total_exp < req_exp){
        std::cerr << "experiment id is larger than expected! \n Exit" << std::endl;
        exit(1);
    }
    if(total_kf < req_kf){
        std::cerr << "k-fold id is larger than expected! \n Exit" << std::endl;
        exit(1);
    }

    // all the indices start from 0
    int line_id = req_exp * total_kf + req_kf + 1; //1 adds for the first line of summary
    int level_id = v_lines[line_id].first;
    int num_models = v_lines[line_id].second;
    std::cout << "line_id:"<< line_id << ",l:" << level_id << ",n:" << num_models << std::endl;
    exit(1);


    std::string in_model = "./svm_models/";

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

