#include "../config_params.h"
#include "../config_logs.h"
#include "../solver.h"
#include "../OptionParser.h"
#include "../loader.h"
#include "fstream"

Config_params* Config_params::instance = NULL;


void check_input_parameters(){
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
}


/* ==========================================================================================
 *                                    READ MODEL'S SUMMERY
 * ==========================================================================================*/
void read_model_summary(std::string models_path, std::vector<std::pair<int,int>>& v_lines){
    //find the level in the summary
    std::fstream in_file;
    std::string fname = models_path + Config_params::getInstance()->get_ds_name() + "_models.summary";
    //    std::cout << fname << std::endl;
    in_file.open(fname);
//    std::vector<std::pair<int,int>> v_lines;
    std::string line;

    while(std::getline(in_file, line)){
        std::size_t pos1, pos2, pos3 =0;
        pos1 = line.find(':');
        pos2 = line.find(':',pos1+1);
        pos3 = line.find(',');
    //        std::cout << pos1 << " , " << pos3 << " , " << pos2 << std::endl;
    //        std::cout << line.substr(pos1+1, pos3-pos1-1) << " -- " << line.substr(pos2+1, line.length()) << std::endl;
        if((pos1 > line.length() || pos2 > line.length() || pos3> line.length() )){
            std::cout << "[RMS] Wrong line at index "<< v_lines.size() << " in the summary file!" << std::endl;
            exit(1);
        }
        int tmp_level_id = stoi(line.substr(pos1+1, pos3-pos1-1));
        int tmp_num_models = stoi(line.substr(pos2+1, line.length()));
    //        std::cout << "l:" << tmp_level_id << ",n:" << tmp_num_models << std::endl;
        v_lines.push_back(std::make_pair(tmp_level_id, tmp_num_models));
    }
    in_file.close();
}

/* ==========================================================================================
 *                                CALCULATE NUMBER OF MODELS
 * ==========================================================================================
 * check the parameters with models summary file
 * calculate the level id and number of models at the requested experiment and k-fold
*/
void calc_number_models(std::vector<std::pair<int,int>>& v_lines, int& level_id, int& num_models) {

    int req_exp = Config_params::getInstance()->get_experiment_id();
    int req_kf = Config_params::getInstance()->get_kfold_id();
    int total_exp = v_lines[0].first;
    int total_kf = v_lines[0].second;

    std::cout << "Te:" << total_exp<< ",Re:" << req_exp << ",Tk:" << total_kf<< ",Rk:" << req_kf << std::endl;
    //check the input exp_id and kf_id
    if(total_exp < req_exp){
        std::cerr << "[CNM] experiment id is larger than expected! \n Exit" << std::endl;
        exit(1);
    }
    if(total_kf < req_kf){
        std::cerr << "[CNM] k-fold id is larger than expected! \n Exit" << std::endl;
        exit(1);
    }


    // all the indices start from 0
    int line_id = req_exp * total_kf + req_kf + 1; //1 adds for the first line of summary
    level_id = v_lines[line_id].first;
    num_models = v_lines[line_id].second;
    std::cout << "[CNM] line_id:"<< line_id << ",l:" << level_id << ",n:" << num_models << std::endl;
}





int main(int argc, char **argv){
    PetscInitialize(&argc, &argv, NULL, NULL);
    //read XML parameters
    Config_params::getInstance()->read_params("./params.xml", argc, argv, Config_params::prediction );
    const std::string models_path = "./svm_models/";

    check_input_parameters();

    std::vector<std::pair<int,int>> v_lines;
    read_model_summary(models_path, v_lines);
    int level_id, num_models;
    calc_number_models(v_lines, level_id, num_models);

    /* ==========================================================================================
     *                                      LOAD TEST DATA
     * ==========================================================================================*/
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


//    exit(1);






    /* ==========================================================================================
     *                                          PREDICTION
     * ==========================================================================================*/
    //no model
    if(num_models == 0){
        std::cerr << "Number of models are zero, the summary file is corrupted! \n Exit" << std::endl;
        exit(1);
    }

    // 1 model
    if(num_models == 1){
        std::cout << "Start prediction for a model\n";
        std::string model_name = models_path + Config_params::getInstance()->get_ds_name() +
                "_exp_" + std::to_string(Config_params::getInstance()->get_experiment_id()) +
                "_kf_" + std::to_string(Config_params::getInstance()->get_kfold_id()) +
                "_level_" +  std::to_string(level_id) + ".svmmodel";
        svm_model * trained_model = svm_load_model(model_name.c_str());
        std::cout << "model name:" << model_name << ", nSV:" << *(trained_model->nSV) <<"\n";

//        exit(1);
        summary final_summary;
//exit(1);
        Solver sv;
//exit(1);
        sv.set_local_model(trained_model);
        sv.test_predict(m_test_data, final_summary );
        Config_params::getInstance()->print_summary(final_summary,"stand alone predict");

    }else{      //multiple models
        std::cout << "Start prediction for multiple models\n";
        exit(1);
    }
//exit(1);


    // for many models


//    svm_model * trained_model = svm_load_model(in_model.c_str());

//    summary final_summary;
//    Solver sv;
//    sv.set_local_model(trained_model);
//    sv.test_predict(m_test_data, final_summary );
//    Config_params::getInstance()->print_summary(final_summary,"stand alone predict");
    std::cout << "Single SVM predict finished successfully!\n";
    PetscFinalize();
    return 0;
}

