#include "config_params.h"
#include <iostream>
#include "pugixml.hpp"
#include <time.h>
#include <chrono>


//Initialize all parameters with default values
void Config_params::init_to_default(){
    debug_exp_CS = 0;
    debug_exp_CS_level = 0;
    debug_exp_MS = 0;
    debug_exp_MS_level = 0;
    debug_exp_MR = 0;
    debug_exp_MR_level = 0;

    ds_path = "";
    ds_name = "";
    ds_version = "";
    cpp_srand_seed = "";
    main_num_repeat_exp=0;
    main_num_kf_iter=0;
    pre_init_loader_matrix = 0;
    inverse_weight = 0;
    ld_weight_type = 0;
    ld_weight_param = 0;
    coarse_Eta = 0;
    coarse_threshold = 0;
    coarse_q = 0 ;
    coarse_r = 0;
    cp_max_coarse_level = 0;
    cs_use_real_points = 0;
    cs_weak_edges_ft = 0 ;
    ms_status = false;
    ms_limit = 0;
    ms_svm_id = 0;
    ms_first_stage = 0;
    ms_second_stage = 0;
    ms_print_untouch_reuslts = false;
    ms_bs_gm_threshold = 0;
    ms_best_selection = 0;
    ms_save_final_model = 0;
    svm_type = 0;
    kernel_type = 0;
    degree = 0;
    gamma = 0;
    coef0 = 0;
    nu = 0;
    cache_size = 0;
    C = 0;
    eps = 0;
    p = 0;
    shrinking = 0;
    probability = 0;
    nr_weight = 0;
    rf_add_fraction = 0;
    rf_sel_all_min_limit = 0;
    rf_add_distant_point_status = 0;
    rf_start_partitioning = 0;
    rf_partition_max_size = 0;
    rf_num_opposite_partitions = 0;

    best_C = 0;
    best_gamma = 0;
    best_params_are_set = 0;
}


Config_params* Config_params::getInstance() {
    if ( !instance ) instance = new Config_params;
    return instance;
}

void Config_params::print_params(){
    std::cout << "mlsvm_version:" << mlsvm_version << std::endl;
    std::cout << "============= dataset=============" <<
                 "\nds_path: "              << get_ds_path()          <<
                 "\nds_name: "              << get_ds_name()          <<
                 std::endl;

    std::cout << "--- C++ Paramters ---"    << "\ncpp_srand_seed: " <<
                 get_cpp_srand_seed()             << "\nNotice the seeds for each expriment is different and prints in the beginning of experiment"<<
                 std::endl;

    std::cout << "--- Main file Paramters ---"      <<
                 "\nmain_num_repeat_exp: "          << get_main_num_repeat_exp()   <<
                 "\nmain_num_kf_iter: "             << get_main_num_kf_iter()      <<
                 std::endl;

    std::cout << "--- Loader Paramters ---"     <<
                 "\npre_init_loader_matrix: "   << get_pre_init_loader_matrix()   <<
                 "\ninverse_weight: "           << get_inverse_weight()           <<
                 "\nld_weight_type: "           << get_ld_weight_type()           <<
                 "\nld_weight_param: "           << get_ld_weight_param()         <<
                 std::endl;

    std::cout << "--- Coarsening Paramters ---" <<
                 "\ncoarse_Eta: "           << get_coarse_Eta()           <<
                 "\ncoarse_threshold: "     << get_coarse_threshold()     <<
                 "\ncoarse_q: "             << get_coarse_q()             <<
                 "\ncoarse_r: "             << get_coarse_r()             <<
                 "\ncp_max_coarse_level: "  << get_cp_max_coarse_level()  <<
                 "\ncs_use_real_points: "   << get_cs_use_real_points()   <<
                 "\ncs_weak_edges_ft: "     << get_cs_weak_edges_ft()     <<
                 std::endl;

    std::cout << "--- ModelSelection Paramters ---" <<
                 "\nms_status: "          << get_ms_status()            <<
                 "\nms_limit: "           << get_ms_limit()             <<
                 "\nms_first_stage: "     << get_ms_first_stage()       <<
                 "\nms_second_stage: "    << get_ms_second_stage()      <<
                 "\nms_best_selection: "  << get_ms_best_selection()    <<
                 "\nms_validation_part: " << get_ms_validation_part()   <<
                 "\nms_svm_id: "          << get_ms_svm_id()            <<
                 "\nms_bs_gm_threshold: " << get_ms_bs_gm_threshold()   <<
                 std::endl;

    std::cout << "--- SVM Paramters ---" <<
                 "\nsvm_type: "         << svm_type                 <<
                 "\nkernel_type: "      << get_svm_kernel_type()    <<
                 "\ndegree: "           << degree                   <<
                 "\ngamma: "            << get_svm_gamma()          <<
                 "\ncoef0: "            << coef0                    <<
                 "\nnu: "               << nu                       <<
                 "\ncache_size: "       << cache_size               <<
                 "\nC: "                << get_svm_C()              <<
                 "\neps: "              << get_svm_eps()            <<
                 "\np: "                << p                        <<
                 "\nshrinking: "        << get_svm_shrinking()      <<
                 "\nprobability: "      << get_svm_probability()    <<
                 std::endl;


    std::cout << "--- Refinement Paramters ---" <<
                 "\nadd_fraction: "             << get_rf_add_fraction()          <<
                 "\nrf_sel_all_min_limit: "     << rf_sel_all_min_limit     <<
                 "\nrf_add_distant_point_status: "           << rf_add_distant_point_status           <<
                 "\nrf_weight_vol: "            << get_rf_weight_vol()            <<
                 "\nrf_start_partitioning: "      << get_rf_start_partitioning()      <<
                 "\nrf_partition_max_size: "      << get_rf_partition_max_size()      <<
                 "\nrf_num_opposite_partitions: "    << get_rf_num_opposite_partitions()    <<
                 std::endl;
//    std::cout << "--- Log Paramters ---"                                    <<
//                 "\ntimer_print: "              << timer_print              <<
//                 "\ndebug_level: "              << debug_level              <<
//                 "\ndebug_level_loader: "       << debug_level_loader       <<
//                 "\ndebug_level_coarse: "       << debug_level_coarse       <<
//                 "\ndebug_level_main_rec: "     << debug_level_main_rec     <<
//                 "\ndebug_level_MS: "           << debug_level_MS           <<
//                 "\nreport_MS_untouched_ds: "   << report_MS_untouched_ds   << std::endl;

}



void Config_params::read_params(std::string XML_FILE_PATH,int argc, char * argv[]){
    /// load the file
    std::cout << "[Config_Params][read_params] param.xml path : "<< XML_FILE_PATH << "\n";
    // Create empty XML document within memory
    pugi::xml_document doc;
    // Load XML file into memory
    // Remark: to fully read declaration entries you have to specify
    // "pugi::parse_declaration"
    pugi::xml_parse_result result = doc.load_file(XML_FILE_PATH.c_str(),
        pugi::parse_default|pugi::parse_declaration);
    if (!result)   {
        std::cout << "[Config_Params][read_params] Parse error: " << result.description()
            << ", character pos= " << result.offset << std::endl;
        exit(1);
    }
    // A valid XML document must have a single root node
    pugi::xml_node root = doc.document_element();

    /// Set debug parameters
//    debug_export = root.child("debug_export").attribute("boolVal").as_bool();
    debug_exp_CS       = root.child("debug_exp_CS").attribute("intVal").as_int();
    debug_exp_CS_level = root.child("debug_exp_CS_level").attribute("intVal").as_int();
    debug_exp_MS       = root.child("debug_exp_MS").attribute("intVal").as_int();
    debug_exp_MS_level = root.child("debug_exp_MS_level").attribute("intVal").as_int();
    debug_exp_MR       = root.child("debug_exp_MR").attribute("intVal").as_int();
    debug_exp_MR_level = root.child("debug_exp_MR_level").attribute("intVal").as_int();

//    mlsvm_version = root.child("mlsvm_version").attribute("stringVal").as_string();
    mlsvm_version = root.child("mlsvm_version").attribute("stringVal").value();

    /// Set multilevel parameters
    cpp_srand_seed = root.child("cpp_srand_seed").attribute("stringVal").value();
    if(cpp_srand_seed == ""){                                          //for resimulate the same run later
        cpp_srand_seed =  std::to_string(std::chrono::system_clock::now().time_since_epoch() /std::chrono::milliseconds(1));
//        std::cout << cpp_srand_seed << std::endl;
//        exit(1);
    }

    main_num_repeat_exp = root.child("main_num_repeat_exp").attribute("intVal").as_int();
    main_num_kf_iter = root.child("main_num_kf_iter").attribute("intVal").as_int();

    ds_path = root.child("ds_path").attribute("stringVal").value();
    ds_name = root.child("ds_name").attribute("stringVal").value();
    ds_version = root.child("ds_version").attribute("stringVal").value();
    set_inputs();           // set all the dataset files
    pre_init_loader_matrix = root.child("pre_init_loader_matrix").attribute("intVal").as_int();
    inverse_weight = root.child("inverse_weight").attribute("boolVal").as_bool();
    ld_weight_type = root.child("ld_weight_type").attribute("intVal").as_int();
    ld_weight_param = root.child("ld_weight_param").attribute("doubleVal").as_double();

    coarse_Eta = root.child("coarse_Eta").attribute("doubleVal").as_double();
    coarse_threshold = root.child("coarse_threshold").attribute("intVal").as_int();
    coarse_q = root.child("coarse_q").attribute("doubleVal").as_double();
    coarse_r = root.child("coarse_r").attribute("intVal").as_int();
    cp_max_coarse_level = root.child("cp_max_coarse_level").attribute("intVal").as_int();
    cs_use_real_points = root.child("cs_use_real_points").attribute("intVal").as_int();
    cs_weak_edges_ft = root.child("cs_weak_edges_ft").attribute("doubleVal").as_double();

    ms_status           = root.child("ms_status").attribute("intVal").as_int();
    ms_limit            = root.child("ms_limit").attribute("intVal").as_int();
    ms_svm_id           = root.child("ms_svm_id").attribute("intVal").as_int();
    ms_first_stage      = root.child("ms_first_stage").attribute("intVal").as_int();
    ms_second_stage     = root.child("ms_second_stage").attribute("intVal").as_int();
    ms_validation_part  = root.child("ms_validation_part").attribute("doubleVal").as_double();
    ms_print_untouch_reuslts = root.child("ms_print_untouch_reuslts").attribute("intVal").as_int();
    ms_bs_gm_threshold  = root.child("ms_bs_gm_threshold").attribute("doubleVal").as_double();
    ms_best_selection = root.child("ms_best_selection").attribute("intVal").as_int();
    ms_save_final_model = root.child("ms_save_final_model").attribute("intVal").as_int();

    svm_type    = root.child("svm_svm_type").attribute("intVal").as_int();
    kernel_type = root.child("svm_kernel_type").attribute("intVal").as_int();
    degree      = root.child("svm_degree").attribute("intVal").as_int();
    gamma       = root.child("svm_gamma").attribute("doubleVal").as_double();
    coef0       = root.child("svm_coef0").attribute("doubleVal").as_double();
    nu          = root.child("svm_nu").attribute("doubleVal").as_double();
    cache_size  = root.child("svm_cache_size").attribute("doubleVal").as_double();
    C           = root.child("svm_C").attribute("doubleVal").as_double();
    eps         = root.child("svm_eps").attribute("doubleVal").as_double();
    p           = root.child("svm_p").attribute("doubleVal").as_double();
    shrinking   = root.child("svm_shrinking").attribute("intVal").as_int();
    probability = root.child("svm_probability").attribute("intVal").as_int();
    nr_weight   = root.child("svm_nr_weight").attribute("intVal").as_int();

    rf_add_fraction = root.child("rf_add_fraction").attribute("floatVal").as_float();
    rf_sel_all_min_limit = root.child("rf_sel_all_min_limit").attribute("intVal").as_int();
    rf_add_distant_point_status  = root.child("rf_add_distant_point_status").attribute("boolVal").as_bool();
    rf_weight_vol   = root.child("rf_weight_vol").attribute("intVal").as_int();
    rf_start_partitioning = root.child("rf_start_partitioning").attribute("intVal").as_int();
    rf_partition_max_size = root.child("rf_partition_max_size").attribute("intVal").as_int();
    rf_num_opposite_partitions = root.child("rf_num_opposite_partitions").attribute("intVal").as_int();

    parser_.add_option("-s", "--sr_seed", "--srand_seed")    .dest("cpp_srand_seed")  .set_default(cpp_srand_seed);
    parser_.add_option("-x", "--n_ex", "--num_exp")          .dest("main_num_repeat_exp")  .set_default(main_num_repeat_exp);
    parser_.add_option("-k", "--n_kf", "--num_kfold")        .dest("main_num_kf_iter")  .set_default(main_num_kf_iter);
    parser_.add_option("--ds_p", "--path")                   .dest("ds_path")  .set_default(ds_path);
    parser_.add_option("-f", "--ds_f", "--file")             .dest("ds_name")  .set_default(ds_name);
    parser_.add_option("--dsv", "--version")                 .dest("ds_version")  .set_default(ds_version);
    parser_.add_option("--cspi", "--pre_init_loader_matrix") .dest("pre_init_loader_matrix")  .set_default(pre_init_loader_matrix);
//    parser_.add_option("--iw", "--inverse_weight")           .dest("inverse_weight")  .set_default(inverse_weight);
    parser_.add_option("--cs_eta")                           .dest("coarse_Eta")  .set_default(coarse_Eta);
    parser_.add_option("-t", "--cs_t", "--coarse_threshold") .dest("coarse_threshold")  .set_default(coarse_threshold);
    parser_.add_option("-q", "--cs_q", "--coarse_q")         .dest("coarse_q")  .set_default(coarse_q);
    parser_.add_option("-r", "--cs_r", "--coarse_r")         .dest("coarse_r")  .set_default(coarse_r);
    parser_.add_option("--cs_m", "--cp_max_coarse_level")    .dest("cp_max_coarse_level")  .set_default(cp_max_coarse_level);
    parser_.add_option("--cs_we_ft")                         .dest("cs_weak_edges_ft")  .set_default(cs_weak_edges_ft);
    parser_.add_option("--ms_status")                        .dest("ms_status")     .set_default(ms_status);
    parser_.add_option("-l", "--ms_l")                       .dest("ms_limit")  .set_default(ms_limit);
    parser_.add_option("-i", "--ms_id", "--ms_svm_id")       .dest("ms_svm_id")  .set_default(ms_svm_id);
    parser_.add_option("-a", "--ms_s1", "--ms_first_stage")  .dest("ms_first_stage")  .set_default(ms_first_stage);
    parser_.add_option("-b", "--ms_s2", "--ms_second_stage") .dest("ms_second_stage")  .set_default(ms_second_stage);
    parser_.add_option("--ms_bs")                            .dest("ms_best_selection")  .set_default(ms_best_selection);
    parser_.add_option("-v", "--validation_part")            .dest("ms_validation_part")  .set_default(ms_validation_part);
    parser_.add_option("-p", "--ms_pt", "--ms_prt_testdata") .dest("ms_print_untouch_reuslts")  .set_default(ms_print_untouch_reuslts);
    parser_.add_option("--msk", "--kernel_type")             .dest("kernel_type")  .set_default(kernel_type);
    parser_.add_option("-g", "--ms_g")                       .dest("gamma")  .set_default(gamma);
    parser_.add_option("-c", "--ms_c")                       .dest("C")  .set_default(C);
    parser_.add_option("-e", "--ms_eps")                     .dest("eps")  .set_default(eps);
    parser_.add_option("--ms_shrinking")                     .dest("shrinking")  .set_default(shrinking);
    parser_.add_option("--ms_probability")                   .dest("probability")  .set_default(probability);
    parser_.add_option("-z", "--rff", "--rf_add_fraction")   .dest("rf_add_fraction")  .set_default(rf_add_fraction);
    parser_.add_option("--rf_weight_vol")                    .dest("rf_weight_vol")  .set_default(rf_weight_vol);
    parser_.add_option("--rf_start_partitioning")              .dest("rf_start_partitioning")  .set_default(rf_start_partitioning);
    parser_.add_option("--rf_partition_max_size")              .dest("rf_partition_max_size")  .set_default(rf_partition_max_size);


    this->options_ = parser_.parse_args(argc, argv);
    std::vector<std::string> args = parser_.args();
    std::cout << "[Config_params] input parameters are read" << std::endl;
}


void Config_params::set_ds_path(std::string const new_ds_path){
    this->ds_path = new_ds_path;
}

void Config_params::set_ds_name(std::string const new_ds_name){
    this->ds_name = new_ds_name;
}

//void Config_params::set_inputs(std::string const ds_name, std::string const ds_version){
void Config_params::set_inputs(){
//    p_indices_f_name    = ds_path + ds_name + "/" + ds_version +"/" +"P_indices.dat";
//    p_dist_f_name       = ds_path + ds_name + "/" + ds_version +"/" +"P_dists.dat";
//    n_indices_f_name    = ds_path + ds_name + "/" + ds_version +"/" +"N_indices.dat";
//    n_dist_f_name       = ds_path + ds_name + "/" + ds_version +"/" +"N_dists.dat";
    p_indices_f_name    = "./data/kfold_min_train_indices.dat";
    p_dist_f_name       = "./data/kfold_min_train_dists.dat";
    n_indices_f_name    = "./data/kfold_maj_train_indices.dat";
    n_dist_f_name       = "./data/kfold_maj_train_dists.dat";

//    p_norm_data_f_name  = ds_path + ds_name + "/" + ds_version +"/" +"P_data.dat";
//    n_norm_data_f_name  = ds_path + ds_name + "/" + ds_version +"/" +"N_data.dat";
//    test_ds_f_name      = ds_path + ds_name + "/" + ds_version +"/" +"T_data.dat";
    p_norm_data_f_name  = "./data/kfold_min_train.dat";
    n_norm_data_f_name  = "./data/kfold_maj_train.dat";
    test_ds_f_name      = "./data/kfold_test_data.dat";
}


void Config_params::update_srand_seed(){
//    cpp_srand_seed =  std::to_string(std::chrono::system_clock::now().time_since_epoch() /std::chrono::milliseconds(1));
    cpp_srand_seed = std::to_string(atoll(cpp_srand_seed.c_str()) + 1)  ;
    std::cout << "\n\n * * * New srand seed is: " << cpp_srand_seed <<" * * * \n"<< std::endl;
}

void Config_params::debug_only_set_srand_seed(std::string new_seed){
    cpp_srand_seed =  new_seed;
    std::cout << "\n\n * * * (Only for debug - It shouldn't be used in the real runs) New srand seed is:" << cpp_srand_seed <<" * * * \n"<< std::endl;
}

void Config_params::add_final_summary(iter_summary current_summary){
    this->all_summary.push_back(current_summary);
    std::cout << "[Config_params] summary added to all_summary" << std::endl;
}

void Config_params::set_best_parameters(measures preferred_measure){    //default measure is set to Gmean in the header file
    double max_measure_=0;
    int max_index_ =0;
    for(unsigned int i=0; i < this->all_summary.size(); i++){
        if(this->all_summary[i].result[preferred_measure] > max_measure_){
            std::cout << "[CP][best_parameters] result at "<< i << " is " << this->all_summary[i].result[preferred_measure] << std::endl;
            std::cout << "[CP][best_parameters] C at "<< i << " is " << this->all_summary[i].C << std::endl;
            std::cout << "[CP][best_parameters] gamma at "<< i << " is " << this->all_summary[i].gamma << std::endl;
            max_measure_ = this->all_summary[i].result[preferred_measure];
            max_index_ = i;
        }
    }
    this->best_C = this->all_summary[max_index_].C;
    this->best_gamma = this->all_summary[max_index_].gamma;
    std::cout << "[CP][best_parameters] selected C: "<< this->best_C <<
                 ", gamma: " << this->best_gamma << " and size of summary is :"<<
                 this->all_summary.size() <<std::endl;
    this->best_params_are_set = 1;

}

void Config_params::print_final_results(){
    printf("[CP] final results are: \n");
    double sum_acc=0;
    double sum_gmean=0;
    double sum_sens=0;
    double sum_spec=0;

    for(unsigned int i=0; i< this->all_summary.size(); i++){
        printf("[CP][FP]|it:%d|\tAcc:%g|\tSN:%g|\tSP:%g|\tGmean:%g|\tC:%g|\tGamma:%g\n",i,
               this->all_summary[i].result[Acc], this->all_summary[i].result[Sens],
               this->all_summary[i].result[Spec], this->all_summary[i].result[Gmean],
               this->all_summary[i].C, this->all_summary[i].gamma);
        sum_acc += this->all_summary[i].result[Acc];
        sum_sens += this->all_summary[i].result[Sens];
        sum_spec += this->all_summary[i].result[Spec];
        sum_gmean += this->all_summary[i].result[Gmean];

    }
    double avg_acc = sum_acc / this->all_summary.size();
    double avg_sens = sum_sens / this->all_summary.size();
    double avg_spec = sum_spec / this->all_summary.size();
    double avg_gmean = sum_gmean / this->all_summary.size();
    printf("\n\n[CP][FP]|\tAvg_Acc:%g|\tAvg_SN:%g|\tAvg_SP:%g|\tAvg_Gmean:%g\n\n",avg_acc, avg_sens, avg_spec, avg_gmean);
}
