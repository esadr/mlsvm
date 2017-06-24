#include "config_params.h"
#include <iostream>
#include "pugixml.hpp"
#include <time.h>
#include <chrono>
#include "petscvec.h"

//Initialize all parameters with default values
//void Config_params::init_to_default(){
//    debug_exp_CS = 0;
//    debug_exp_CS_level = 0;
//    debug_exp_MS = 0;
//    debug_exp_MS_level = 0;
//    debug_exp_MR = 0;
//    debug_exp_MR_level = 0;

//    nn_number           =0;
//    nn_distance_type    =0;
//    ds_path             ="";
//    ds_name             ="";
//    exp_info            ="";
//    tmp_path            ="";
//    cpp_srand_seed      ="";
//    main_num_repeat_exp =0;
//    main_num_kf_iter    =0;
//    VD_sample_size_fraction=0;
////    multi_level_status=0;
//    pre_init_loader_matrix =0;
//    inverse_weight      =0;
//    ld_weight_type      =0;
//    ld_weight_param     =0;
//    coarse_Eta          =0;
//    coarse_threshold = 0;
//    coarse_q = 0 ;
//    coarse_r = 0;
//    cs_max_coarse_level = 0;
//    cs_use_real_points = 0;
//    cs_weak_edges_ft = 0 ;
//    ms_status = 0;
//    ms_limit = 0;
//    ms_svm_id = 0;
//    ms_first_stage  = 0;
//    ms_second_stage = 0;
//    ms_print_untouch_reuslts = false;
//    ms_bs_gm_threshold  = 0;
//    ms_best_selection   = 0;
//    ms_save_final_model = 0;
//    svm_type    = 0;
//    kernel_type = 0;
//    degree  = 0;
//    gamma   = 0;
//    coef0   = 0;
//    nu      = 0;
//    cache_size = 0;
//    C       = 0;
//    eps     = 0;
//    p       = 0;
//    shrinking   = 0;
//    probability = 0;
//    nr_weight   = 0;
//    rf_add_fraction = 0;
//    rf_add_distant_point_status = 0;
//    pr_start_partitioning = 0;
//    pr_partition_max_size = 0;

//    pr_maj_voting_id =0;

//    best_C = 0;
//    best_gamma = 0;
//    best_params_are_set = 0;
//}


Config_params* Config_params::getInstance() {
    if ( !instance ) instance = new Config_params;
    return instance;
}

void Config_params::print_classification_params(){
    std::cout << "mlsvm_version:" << mlsvm_version << std::endl;
    std::cout << "============= dataset=============" <<
                 "\nds_path: "              << get_ds_path()          <<
                 "\nds_name: "              << get_ds_name()          <<
                 "\ntmp_path: "             << get_tmp_path()          <<
                 std::endl;

    std::cout << "\ncpp_srand_seed: " <<get_cpp_srand_seed()<< std::endl;

    std::cout << "--- Main file Paramters ---"      <<
                 "\nmain_num_repeat_exp: "          << get_main_num_repeat_exp()        <<
                 "\nmain_num_kf_iter: "             << get_main_num_kf_iter()           <<
                 "\nexp_info: "                     << get_exp_info()                   <<

                 "\nML_status: "                    << get_multi_level_status()         <<
                 std::endl;

    std::cout << "--- NN Paramters ---"     <<
                 "\nnn_number: "            << get_nn_number_of_neighbors()     <<
                 "\nnn_distance_type:"      << get_nn_distance_type()           << std::endl;

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
                 "\ncs_max_coarse_level: "  << get_cs_max_coarse_level()  <<
                 "\ncs_use_real_points: "   << get_cs_use_real_points()   <<
                 "\ncs_weak_edges_ft: "     << get_cs_weak_edges_ft()     <<
                 "\ncs_boundary_points_status: "     << get_cs_boundary_points_status()     <<
                 "\ncs_boundary_points_threshold: "  << get_cs_boundary_points_threshold()  <<
                 "\ncs_boundary_points_max_num: "    << get_cs_boundary_points_max_num()    <<
                 std::endl;

    std::cout << "--- ModelSelection Paramters ---" <<
                 "\nms_status: "                   << get_ms_status()                     <<
                 "\nms_limit: "                    << get_ms_limit()                      <<
                 "\nms_first_stage: "              << get_ms_first_stage()                <<
                 "\nms_second_stage: "             << get_ms_second_stage()               <<
                 "\nms_best_selection: "           << get_ms_best_selection()             <<
                 "\nms_VD_sample_size_fraction: "  << get_ms_VD_sample_size_fraction()    <<
                 "\nms_svm_id: "                   << get_ms_svm_id()                     <<
                 "\nms_bs_gm_threshold: "          << get_ms_bs_gm_threshold()            <<
                 std::endl;
//                 "\nms_validation_part: " << get_ms_validation_part()   <<

    std::cout << "--- SVM Paramters ---" <<
                 "\nsvm_type: "         << get_svm_svm_type()       <<
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
                 "\nadd_fraction: "             << get_rf_add_fraction()                <<
                 "\nrf_add_distant_point_status(2nd): "   << get_rf_add_distant_point_status()      <<
                 "\nrf_weight_vol: "            << get_rf_weight_vol()                  <<
                 "\npr_start_partitioning: "    << get_pr_start_partitioning()          <<
                 std::endl;

    std::cout << "--- Partitioning Paramters ---" <<
                 "\npr_partition_max_size: "      << get_pr_partition_max_size()        <<
                 "\npr_maj_voting_id: "           << get_pr_maj_voting_id()             <<
                 std::endl;
}

void Config_params::print_flann_params(){
    std::cout << "--- NN Paramters ---"         <<
                 "\nnn_number_of_classes: "     << get_nn_number_of_classes()   <<
                 "\nnn_number_of_neighbors: "   << get_nn_number_of_neighbors() <<
                 "\nnn_distance_type: "         << get_nn_distance_type()       << std::endl;

    if(get_nn_number_of_classes() == 2){
        std::cout << "--- Input File names ---"    <<
                 "\nminority data: "                << get_p_norm_data_f_name() <<
                 "\nmajority data: "                << get_n_norm_data_f_name() << std::endl;
    }else{
        std::cout << "--- Input File names ---"    <<
                 "\ndata: "                         << get_single_norm_data_f_name() << std::endl;
    }


}

void Config_params::read_params(std::string XML_FILE_PATH,int argc, char * argv[], program_parts caller_func){

    PetscBool       flg; //@@ 040417-2130
    PetscInt        temp;
//    PetscOptionsGetInt(NULL,NULL,"-help",&temp,&flg);			//newer versions of  PETSc
//    if (!flg){
//        //print help function
//        PetscPrintf(PETSC_COMM_WORLD,"[CP] Help parameters are not ready yet! Please refer to manual! \nExit!\n");
//        exit(1);
//    }

    if(caller_func == program_parts::main){
        PetscOptionsGetInt(NULL,NULL,"-d",&main_function,&flg);			//newer versions of  PETSc
        if (!flg){
//            PetscPrintf(PETSC_COMM_WORLD,"[CP] Must indicate the functionality type you need using -d . \nExit!\n");
//            exit(1);
            PetscPrintf(PETSC_COMM_WORLD,"[WARNING] You can choose the functionality using -d parameter. Multilevel SVM is the default functionality \n");
            main_function=0;
        }
    }

    /// load the file
    std::cout << "[Config_Params][read_params] param.xml path : "<< XML_FILE_PATH << "\n";
    // Create empty XML document within memory
    pugi::xml_document doc;
    // Load XML file into memory
    pugi::xml_parse_result result = doc.load_file(XML_FILE_PATH.c_str(),
        pugi::parse_default|pugi::parse_declaration);
    if (!result)   {
        std::cout << "[Config_Params][read_params] Parse error: " << result.description()
            << ", character pos= " << result.offset << std::endl;
        exit(1);
    }
    // A valid XML document must have a single root node
    pugi::xml_node root = doc.document_element();




    switch(caller_func){
    case main:
    {
        /// - - - - - check the ml_function - - - - -
//        main_function       = root.child("main_function").attribute("intVal").as_int();
//        parser_.add_option("-d") .dest("main_function")  .set_default(main_function);


//        this->options_ = parser_.parse_args(argc, argv);
//        std::vector<std::string> args = parser_.args();

    //    std::cout << "main function:" << get_main_function() << std::endl;

        switch(main_function){
        case 0:
            read_classification_parameters(root, argc, argv);
            set_inputs_file_names();
            print_classification_params();
            break;
        case 1:

            break;
        case 2:
            std::cout << "start reading clustering parameters\n";
            read_clustering_parameters(root, argc, argv);
            break;
        }
    }
        break;
    case flann:
        std::cout << "start reading clustering parameters\n";
        read_flann_parameters(root, argc, argv);
        set_file_names_for_save_flann();
        print_flann_params();
        break;
    }
}



void Config_params::read_classification_parameters(pugi::xml_node& root,int argc, char * argv[]){

    /// Set debug parameters
    debug_exp_CS       = root.child("debug_exp_CS").attribute("intVal").as_int();
    debug_exp_CS_level = root.child("debug_exp_CS_level").attribute("intVal").as_int();
    debug_exp_MS       = root.child("debug_exp_MS").attribute("intVal").as_int();
    debug_exp_MS_level = root.child("debug_exp_MS_level").attribute("intVal").as_int();
    debug_exp_MR       = root.child("debug_exp_MR").attribute("intVal").as_int();
    debug_exp_MR_level = root.child("debug_exp_MR_level").attribute("intVal").as_int();

    mlsvm_version = root.child("mlsvm_version").attribute("stringVal").value();

    /// Set multilevel parameters
    cpp_srand_seed = root.child("cpp_srand_seed").attribute("stringVal").value();
    if(cpp_srand_seed == ""){                                          //for resimulate the same run later
        cpp_srand_seed =  std::to_string(std::chrono::system_clock::now().time_since_epoch() /std::chrono::milliseconds(1));
    }
    /// read the parameters from the XML file (params.xml)
    main_num_repeat_exp = root.child("main_num_repeat_exp").attribute("intVal").as_int();
    main_num_kf_iter    = root.child("main_num_kf_iter").attribute("intVal").as_int();
    multi_level_status  = root.child("multi_level_status").attribute("boolVal").as_bool();
    exp_info            = root.child("exp_info").attribute("stringVal").value();
    exp_info            = root.child("ms_VD_sample_size_fraction").attribute("doubleVal").value();
    nn_number_of_neighbors  = root.child("nn_number_of_neighbors").attribute("intVal").as_int();
    nn_distance_type    = root.child("nn_distance_type").attribute("intVal").as_int();
    ds_path             = root.child("ds_path").attribute("stringVal").value();
    ds_name             = root.child("ds_name").attribute("stringVal").value();
    tmp_path            = root.child("tmp_path").attribute("stringVal").value();
    pre_init_loader_matrix = root.child("pre_init_loader_matrix").attribute("intVal").as_int();
    inverse_weight      = root.child("inverse_weight").attribute("boolVal").as_bool();
    ld_weight_type      = root.child("ld_weight_type").attribute("intVal").as_int();
    ld_weight_param     = root.child("ld_weight_param").attribute("doubleVal").as_double();
    coarse_Eta          = root.child("coarse_Eta").attribute("doubleVal").as_double();
    coarse_threshold    = root.child("coarse_threshold").attribute("intVal").as_int();
    coarse_q            = root.child("coarse_q").attribute("doubleVal").as_double();
    coarse_r            = root.child("coarse_r").attribute("intVal").as_int();
    cs_max_coarse_level = root.child("cs_max_coarse_level").attribute("intVal").as_int();
    cs_use_real_points  = root.child("cs_use_real_points").attribute("intVal").as_int();
    cs_weak_edges_ft    = root.child("cs_weak_edges_ft").attribute("doubleVal").as_double();
    cs_boundary_points_status       = root.child("cs_boundary_points_status").attribute("boolVal").as_bool();
    cs_boundary_points_threshold    = root.child("cs_boundary_points_threshold").attribute("doubleVal").as_double();
    cs_boundary_points_max_num      = root.child("cs_boundary_points_max_num").attribute("intVal").as_int();
    ms_status           = root.child("ms_status").attribute("intVal").as_int();
    ms_limit            = root.child("ms_limit").attribute("intVal").as_int();
    ms_svm_id           = root.child("ms_svm_id").attribute("intVal").as_int();
    ms_first_stage      = root.child("ms_first_stage").attribute("intVal").as_int();
    ms_second_stage     = root.child("ms_second_stage").attribute("intVal").as_int();
    ms_VD_sample_size_fraction  = root.child("ms_VD_sample_size_fraction").attribute("doubleVal").as_double();
    ms_print_untouch_reuslts    = root.child("ms_print_untouch_reuslts").attribute("intVal").as_int();
    ms_bs_gm_threshold  = root.child("ms_bs_gm_threshold").attribute("doubleVal").as_double();
    ms_best_selection   = root.child("ms_best_selection").attribute("intVal").as_int();
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
    rf_add_fraction                 = root.child("rf_add_fraction").attribute("floatVal").as_float();
    rf_add_distant_point_status     = root.child("rf_add_distant_point_status").attribute("boolVal").as_bool();
    rf_weight_vol                   = root.child("rf_weight_vol").attribute("intVal").as_int();
    pr_start_partitioning = root.child("pr_start_partitioning").attribute("intVal").as_int();
    pr_partition_max_size = root.child("pr_partition_max_size").attribute("intVal").as_int();
    pr_maj_voting_id      = root.child("pr_maj_voting_id").attribute("intVal").as_int();

    /// read the parameters from input arguments ()
    /// Notice the values are saved in options_ not the specific parameters in this class
    /// Hence, use the get method to retreive the override values from input arguments
    parser_.add_option("--nn_n")                             .dest("nn_number_of_neighbors")  .set_default(nn_number_of_neighbors) ;
    parser_.add_option("--nn_d")                             .dest("nn_distance_type")  .set_default(nn_distance_type);
    parser_.add_option("-s")                                 .dest("cpp_srand_seed")  .set_default(cpp_srand_seed);
    parser_.add_option("-x")                                 .dest("main_num_repeat_exp")  .set_default(main_num_repeat_exp);
    parser_.add_option("-k")                                 .dest("main_num_kf_iter")  .set_default(main_num_kf_iter);
    parser_.add_option("--ml_s")                             .dest("multi_level_status")  .set_default(multi_level_status);
    parser_.add_option("-u", "--exp_info")                   .dest("exp_info")  .set_default(exp_info);
    parser_.add_option("--ds_p")                             .dest("ds_path")  .set_default(ds_path);
    parser_.add_option("-f", "--ds_f", "--file")             .dest("ds_name")  .set_default(ds_name);
    parser_.add_option("--tmp_p")                            .dest("tmp_path")  .set_default(tmp_path);
    parser_.add_option("--cs_pi")                            .dest("pre_init_loader_matrix")  .set_default(pre_init_loader_matrix);
//    parser_.add_option("--iw", "--inverse_weight")           .dest("inverse_weight")  .set_default(inverse_weight);
    parser_.add_option("--cs_eta")                           .dest("coarse_Eta")  .set_default(coarse_Eta);
    parser_.add_option("-t", "--cs_t" )                      .dest("coarse_threshold")  .set_default(coarse_threshold);
    parser_.add_option("-q", "--cs_q" )                      .dest("coarse_q")  .set_default(coarse_q);
    parser_.add_option("-r", "--cs_r" )                      .dest("coarse_r")  .set_default(coarse_r);
    parser_.add_option("--cs_m")                             .dest("cs_max_coarse_level")  .set_default(cs_max_coarse_level);
    parser_.add_option("--cs_we")                            .dest("cs_weak_edges_ft")  .set_default(cs_weak_edges_ft);
    parser_.add_option("--cs_bp_s")                          .dest("cs_boundary_points_status")  .set_default(cs_boundary_points_status);
    parser_.add_option("--cs_bp_t")                          .dest("cs_boundary_points_threshold")  .set_default(cs_boundary_points_threshold);
    parser_.add_option("--cs_bp_max")                        .dest("cs_boundary_points_max_num")  .set_default(cs_boundary_points_max_num);
    parser_.add_option("--ms_status")                        .dest("ms_status")     .set_default(ms_status);
    parser_.add_option("-l", "--ms_l")                       .dest("ms_limit")  .set_default(ms_limit);
    parser_.add_option("-i", "--ms_id")                      .dest("ms_svm_id")  .set_default(ms_svm_id);
    parser_.add_option("-a", "--ms_s1")                      .dest("ms_first_stage")  .set_default(ms_first_stage);
    parser_.add_option("-b", "--ms_s2")                      .dest("ms_second_stage")  .set_default(ms_second_stage);
    parser_.add_option("--ms_bs")                            .dest("ms_best_selection")  .set_default(ms_best_selection);
    parser_.add_option("-v")                                 .dest("ms_VD_sample_size_fraction")  .set_default(ms_VD_sample_size_fraction);
    parser_.add_option("-p", "--ms_prt")                     .dest("ms_print_untouch_reuslts")  .set_default(ms_print_untouch_reuslts);
    parser_.add_option("--ms_k")                             .dest("kernel_type")  .set_default(kernel_type);
    parser_.add_option("-g", "--ms_g")                       .dest("gamma")  .set_default(gamma);
    parser_.add_option("-c", "--ms_c")                       .dest("C")  .set_default(C);
    parser_.add_option("-e", "--ms_eps")                     .dest("eps")  .set_default(eps);
    parser_.add_option("--ms_shrinking")                     .dest("shrinking")  .set_default(shrinking);
    parser_.add_option("--ms_probability")                   .dest("probability")  .set_default(probability);
    parser_.add_option("-z", "--rf_f")                       .dest("rf_add_fraction")  .set_default(rf_add_fraction);
    parser_.add_option("--rf_2nd")                           .dest("rf_add_distant_point_status")     .set_default(rf_add_distant_point_status);
    parser_.add_option("--rf_weight_vol")                    .dest("rf_weight_vol")  .set_default(rf_weight_vol);
    parser_.add_option("--pr_start")                         .dest("pr_start_partitioning")  .set_default(pr_start_partitioning);
    parser_.add_option("--pr_max")                           .dest("pr_partition_max_size")  .set_default(pr_partition_max_size);
    parser_.add_option("--mv_id")                            .dest("pr_maj_voting_id")     .set_default(pr_maj_voting_id);
    // - - - Tools - - -
    parser_.add_option("--sat_p")                            .dest("p_norm_data_f_name")     .set_default("");
    parser_.add_option("--sat_n")                            .dest("n_norm_data_f_name")     .set_default("");
    parser_.add_option("--sap_td")                           .dest("test_ds_f_name")     .set_default("");
    parser_.add_option("-d")                                 .dest("tmp");

    this->options_ = parser_.parse_args(argc, argv);
    std::vector<std::string> args = parser_.args();
//    set_inputs_file_names();           // set all the dataset files        //because of exp_info, file name's should set after parsing the argv
    check_input_parameters();
    std::cout << "[Config_params] input parameters are read" << std::endl;
}



void Config_params::read_clustering_parameters(pugi::xml_node& root,int argc, char * argv[]){


    mlclustering_version = root.child("mlclustering_version").attribute("stringVal").value();

    /// Set multilevel parameters
    cpp_srand_seed = root.child("cpp_srand_seed").attribute("stringVal").value();
    if(cpp_srand_seed == ""){                                          //for resimulate the same run later
        cpp_srand_seed =  std::to_string(std::chrono::system_clock::now().time_since_epoch() /std::chrono::milliseconds(1));
    }
    /// read the parameters from the XML file (params.xml)
    exp_info            = root.child("exp_info").attribute("stringVal").value();
    nn_number_of_neighbors  = root.child("nn_number_of_neighbors").attribute("intVal").as_int();
    nn_distance_type    = root.child("nn_distance_type").attribute("intVal").as_int();
    ds_path             = root.child("ds_path").attribute("stringVal").value();
    ds_name             = root.child("ds_name").attribute("stringVal").value();
    tmp_path            = root.child("tmp_path").attribute("stringVal").value();
    pre_init_loader_matrix = root.child("pre_init_loader_matrix").attribute("intVal").as_int();
    inverse_weight      = root.child("inverse_weight").attribute("boolVal").as_bool();
    ld_weight_type      = root.child("ld_weight_type").attribute("intVal").as_int();
    ld_weight_param     = root.child("ld_weight_param").attribute("doubleVal").as_double();
    coarse_Eta          = root.child("coarse_Eta").attribute("doubleVal").as_double();
    coarse_threshold    = root.child("coarse_threshold").attribute("intVal").as_int();
    coarse_q            = root.child("coarse_q").attribute("doubleVal").as_double();
    coarse_r            = root.child("coarse_r").attribute("intVal").as_int();
    cs_max_coarse_level = root.child("cs_max_coarse_level").attribute("intVal").as_int();
    cs_use_real_points  = root.child("cs_use_real_points").attribute("intVal").as_int();
    cs_weak_edges_ft    = root.child("cs_weak_edges_ft").attribute("doubleVal").as_double();
    cs_boundary_points_status       = root.child("cs_boundary_points_status").attribute("boolVal").as_bool();
    cs_boundary_points_threshold    = root.child("cs_boundary_points_threshold").attribute("doubleVal").as_double();
    cs_boundary_points_max_num      = root.child("cs_boundary_points_max_num").attribute("intVal").as_int();
    rf_add_fraction                 = root.child("rf_add_fraction").attribute("floatVal").as_float();
    rf_add_distant_point_status     = root.child("rf_add_distant_point_status").attribute("boolVal").as_bool();
    rf_weight_vol                   = root.child("rf_weight_vol").attribute("intVal").as_int();

    /// read the parameters from input arguments ()
    /// Notice the values are saved in options_ not the specific parameters in this class
    /// Hence, use the get method to retreive the override values from input arguments
    parser_.add_option("--nn_n")                             .dest("nn_number_of_neighbors")  .set_default(nn_number_of_neighbors);
    parser_.add_option("--nn_d")                             .dest("nn_distance_type")  .set_default(nn_distance_type);
    parser_.add_option("-s")                                 .dest("cpp_srand_seed")  .set_default(cpp_srand_seed);
    parser_.add_option("-u", "--exp_info")                   .dest("exp_info")  .set_default(exp_info);
    parser_.add_option("--ds_p")                             .dest("ds_path")  .set_default(ds_path);
    parser_.add_option("-f", "--ds_f", "--file")             .dest("ds_name")  .set_default(ds_name);
    parser_.add_option("--tmp_p")                            .dest("tmp_path")  .set_default(tmp_path);
    parser_.add_option("--cs_pi")                            .dest("pre_init_loader_matrix")  .set_default(pre_init_loader_matrix);
    parser_.add_option("--cs_eta")                           .dest("coarse_Eta")  .set_default(coarse_Eta);
    parser_.add_option("-t", "--cs_t" )                      .dest("coarse_threshold")  .set_default(coarse_threshold);
    parser_.add_option("-q", "--cs_q" )                      .dest("coarse_q")  .set_default(coarse_q);
    parser_.add_option("-r", "--cs_r" )                      .dest("coarse_r")  .set_default(coarse_r);
    parser_.add_option("--cs_m")                             .dest("cs_max_coarse_level")  .set_default(cs_max_coarse_level);
    parser_.add_option("--cs_we")                            .dest("cs_weak_edges_ft")  .set_default(cs_weak_edges_ft);
    parser_.add_option("--cs_bp_s")                          .dest("cs_boundary_points_status")  .set_default(cs_boundary_points_status);
    parser_.add_option("--cs_bp_t")                          .dest("cs_boundary_points_threshold")  .set_default(cs_boundary_points_threshold);
    parser_.add_option("--cs_bp_max")                        .dest("cs_boundary_points_max_num")  .set_default(cs_boundary_points_max_num);
    parser_.add_option("-z", "--rf_f")                       .dest("rf_add_fraction")  .set_default(rf_add_fraction);
    parser_.add_option("--rf_2nd")                           .dest("rf_add_distant_point_status")     .set_default(rf_add_distant_point_status);
    parser_.add_option("--rf_weight_vol")                    .dest("rf_weight_vol")  .set_default(rf_weight_vol);
    parser_.add_option("-d")                                 .dest("tmp");

    this->options_ = parser_.parse_args(argc, argv);
    std::vector<std::string> args = parser_.args();
//    set_inputs_file_names();           // set all the dataset files        //because of exp_info, file name's should set after parsing the argv

    check_input_parameters();
    std::cout << "[Config_params] input parameters for clustering are read" << std::endl;

}




void Config_params::read_flann_parameters(pugi::xml_node& root,int argc, char * argv[]){ //@@ 040317-1842
    // read XML values
    nn_number_of_classes    = root.child("nn_number_of_classes").attribute("intVal").as_int();
    nn_number_of_neighbors  = root.child("nn_number_of_neighbors").attribute("intVal").as_int();
    nn_distance_type        = root.child("nn_distance_type").attribute("intVal").as_int();
    ds_path             = root.child("ds_path").attribute("stringVal").value();
    ds_name             = root.child("ds_name").attribute("stringVal").value();
    tmp_path            = root.child("tmp_path").attribute("stringVal").value();
//    nn_path         = root.child("nn_path").attribute("stringVal").value();
//    nn_data_fname1  = root.child("nn_data_fname1").attribute("stringVal").value();
//    nn_data_fname2  = root.child("nn_data_fname2").attribute("stringVal").value();
//    nn_tmp_path     = root.child("nn_tmp_path").attribute("stringVal").value();
    // read input arguments and saved them in options_
    parser_.add_option("--nn_c")                            .dest("nn_number_of_classes")  .set_default(nn_number_of_classes);
    parser_.add_option("--nn_n")                            .dest("nn_number_of_neighbors")  .set_default(nn_number_of_neighbors);
    parser_.add_option("--nn_d")                            .dest("nn_distance_type")  .set_default(nn_distance_type);
    parser_.add_option("--ds_p")                             .dest("ds_path")  .set_default(ds_path);
    parser_.add_option("-f", "--ds_f", "--file")             .dest("ds_name")  .set_default(ds_name);
    parser_.add_option("--tmp_p")                            .dest("tmp_path")  .set_default(tmp_path);
//    parser_.add_option("--nn_p")                            .dest("nn_path")  .set_default(nn_path);
//    parser_.add_option("--nn_f1")                           .dest("nn_data_fname1")  .set_default(nn_data_fname1);
//    parser_.add_option("--nn_f2")                           .dest("nn_data_fname2")  .set_default(nn_data_fname2);
//    parser_.add_option("--tmp_p")                           .dest("nn_tmp_path")  .set_default(nn_tmp_path);

    this->options_ = parser_.parse_args(argc, argv);
    std::vector<std::string> args = parser_.args();
    std::cout << "[Config_params] flann parameters are read" << std::endl;
}











void Config_params::check_input_parameters(){
    if(get_nn_distance_type() > 8 || get_nn_distance_type() <1){
        std::cout << "[Config_params] supported distance types are from 1 to 8!" << std::endl;
        exit(1);
    }
}


void Config_params::set_ds_path(std::string const new_ds_path){
    this->ds_path = new_ds_path;
}

void Config_params::set_ds_name(std::string const new_ds_name){
    this->ds_name = new_ds_name;
}

void Config_params::set_file_names_for_save_flann(){
    if(get_nn_number_of_classes() == 2){
        p_norm_data_f_name  = get_ds_path() + get_ds_name() + "_min_norm_data.dat";
        n_norm_data_f_name  = get_ds_path() + get_ds_name() + "_maj_norm_data.dat";
    }else{
        single_norm_data_f_name  = get_ds_path() + get_ds_name() + "_zsc_data.dat";
    }
}


void Config_params::set_inputs_file_names(){
    p_indices_f_name    = get_tmp_path() +"kfold_min_train_"+get_exp_info()+"_indices.dat";
    p_dist_f_name       = get_tmp_path() +"kfold_min_train_"+get_exp_info()+"_dists.dat";
    n_indices_f_name    = get_tmp_path() +"kfold_maj_train_"+get_exp_info()+"_indices.dat";
    n_dist_f_name       = get_tmp_path() +"kfold_maj_train_"+get_exp_info()+"_dists.dat";

    p_norm_data_f_name  = get_tmp_path() +"kfold_min_train_"+get_exp_info();
    n_norm_data_f_name  = get_tmp_path() +"kfold_maj_train_"+get_exp_info();
    test_ds_f_name      = get_tmp_path() +"kfold_test_data_"+get_exp_info();
}


std::string Config_params::get_tmp_path() const {
    std::string tmp_str (options_["tmp_path"]);   //http://www.cplusplus.com/reference/string/string/rfind/
    std::string key ("/");

    std::size_t found = tmp_str.rfind(key);
    if (found == tmp_str.size() - 1)
        return tmp_str;
    else
        return tmp_str + "/";
}

void Config_params::debug_only_set_p_norm_data_path_file_name(std::string const path_file_name){
    p_norm_data_f_name  = path_file_name;
}
void Config_params::debug_only_set_n_norm_data_path_file_name(std::string const path_file_name){
    n_norm_data_f_name  = path_file_name;
}


void Config_params::update_srand_seed(){
//    cpp_srand_seed =  std::to_string(std::chrono::system_clock::now().time_since_epoch() /std::chrono::milliseconds(1));
    std::string last_srand_seed = get_cpp_srand_seed();
    options_["cpp_srand_seed"] = std::to_string(atoll(last_srand_seed.c_str()) + 1)  ;
    std::cout << "[CP][USS] new srand seed is: " << get_cpp_srand_seed() << std::endl;
}

void Config_params::debug_only_set_srand_seed(std::string new_seed){
    cpp_srand_seed =  new_seed;
    std::cout << "\n\n * * * (Only for debug - It shouldn't be used in the real runs) New srand seed is:" << cpp_srand_seed <<" * * * \n"<< std::endl;
}

void Config_params::add_final_summary(summary current_summary, int selected_level){
    current_summary.selected_level = selected_level;
    this->all_summary.push_back(current_summary);
    std::cout << "[Config_params] summary added to all_summary" << std::endl;
}


void Config_params::print_summary(const summary& summary_in, std::string caller_method, int level, int iter, int stage, int fold) const{
    printf("%s, ",caller_method.c_str());

    if(level != -1)
        printf("l:%d, ",level);

    if(iter >= 0 )
        printf("it passed:%d, ",iter);

    if(summary_in.iter != -1 && iter != -2)
        printf("it set:%d, ",summary_in.iter);

    if(fold != -1)
        printf("fold:%d, ", fold);

    if(stage != -1)
        printf("stage:%d, ",stage);

    printf("AC:%.2f, SN:%.2f, SP:%.2f, PPV:%.2f, NPV:%.2f, F1:%.2f, GM:%.3f, TP:%.0f, TN:%.0f, FP:%.0f, FN:%.0f",
           summary_in.perf.at(Acc), summary_in.perf.at(Sens), summary_in.perf.at(Spec), summary_in.perf.at(PPV),
           summary_in.perf.at(NPV), summary_in.perf.at(F1), summary_in.perf.at(Gmean), summary_in.perf.at(TP),
           summary_in.perf.at(TN), summary_in.perf.at(FP), summary_in.perf.at(FN));

    if(summary_in.C && summary_in.gamma)
        printf(", C:%.2f, Gamma:%.4f", summary_in.C, summary_in.gamma);

    if(summary_in.num_SV_p || summary_in.num_SV_n)
        printf(", nSV+:%d, nSV-:%d", summary_in.num_SV_p, summary_in.num_SV_n);

    printf("\n");

}

void Config_params::set_best_parameters(measures preferred_measure){    //default measure is set to Gmean in the header file
    double max_measure_=0;
    int max_index_ =0;
    for(unsigned int i=0; i < this->all_summary.size(); i++){
        if(this->all_summary[i].perf[preferred_measure] > max_measure_){
            std::cout << "[CP][Set_Best_params] result at "<< i << " is " << this->all_summary[i].perf[preferred_measure] << std::endl;
            std::cout << "[CP][Set_Best_params] C at "<< i << " is " << this->all_summary[i].C << std::endl;
            std::cout << "[CP][Set_Best_params] gamma at "<< i << " is " << this->all_summary[i].gamma << std::endl;
            max_measure_ = this->all_summary[i].perf[preferred_measure];
            max_index_ = i;
        }
    }
    this->best_C = this->all_summary[max_index_].C;
    this->best_gamma = this->all_summary[max_index_].gamma;
    std::cout << "[CP][Set_Best_params] selected C: "<< this->best_C <<
                 ", gamma: " << this->best_gamma << " and size of summary is :"<<
                 this->all_summary.size() <<std::endl;
    this->best_params_are_set = 1;

}

void Config_params::print_final_results() const{
    printf("[CP] final results are: \n");
    double sum_acc=0;
    double sum_gmean=0;
    double sum_sens=0;
    double sum_spec=0;
    double sum_ppv=0;
    double sum_npv=0;
    double sum_f1=0;

    for(unsigned int i=0; i< this->all_summary.size(); i++){
        printf("[CP][PFR],it:%d, BestL:%d, AC:%.2f, SN:%.2f, SP:%.2f, PPV:%.2f, NPV:%.2f, F1:%.2f, GM:%.2f",
               i, this->all_summary[i].selected_level,
               this->all_summary[i].perf.at(Acc), this->all_summary[i].perf.at(Sens),
               this->all_summary[i].perf.at(Spec), this->all_summary[i].perf.at(PPV),
               this->all_summary[i].perf.at(NPV), this->all_summary[i].perf.at(F1),
               this->all_summary[i].perf.at(Gmean));

        if(this->all_summary[i].C && this->all_summary[i].gamma)
            printf(", C:%.2f, G:%.4f",this->all_summary[i].C, this->all_summary[i].gamma);

        printf("\n");

        sum_acc += this->all_summary[i].perf.at(Acc);
        sum_sens += this->all_summary[i].perf.at(Sens);
        sum_spec += this->all_summary[i].perf.at(Spec);
        sum_ppv += this->all_summary[i].perf.at(PPV);
        sum_npv += this->all_summary[i].perf.at(NPV);
        sum_f1 += this->all_summary[i].perf.at(F1);
        sum_gmean += this->all_summary[i].perf.at(Gmean);

    }
    double avg_acc = sum_acc / this->all_summary.size();
    double avg_sens = sum_sens / this->all_summary.size();
    double avg_spec = sum_spec / this->all_summary.size();
    double avg_ppv = sum_ppv / this->all_summary.size();
    double avg_npv = sum_npv / this->all_summary.size();
    double avg_f1 = sum_f1 / this->all_summary.size();
    double avg_gmean = sum_gmean / this->all_summary.size();
    printf("\n\n[CP][PFR], Avg_Acc:%.2f, Avg_SN:%.2f, Avg_SP:%.2f, Avg_PPV:%.2f, Avg_NPV:%.2f, Avg_F1:%.2f, Avg_GM:%.2f\n\n",
           avg_acc, avg_sens, avg_spec, avg_ppv, avg_npv, avg_f1, avg_gmean);
}



void Config_params::print_ref_result(const std::vector<ref_results>& v_ref_results) const{
    for(auto it= v_ref_results.begin(); it!=v_ref_results.end(); ++it){
        std::cout << "[CP][PRefResult] VD GM:" << it->validation_data_summary.perf.at(Gmean) <<
                     ",TD GM:" << it->test_data_summary.perf.at(Gmean) <<
                     ", level:" << it->level << std::endl;
    }
}



