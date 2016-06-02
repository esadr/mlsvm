#ifndef CONFIG_PARAMS_H
#define CONFIG_PARAMS_H
#include <string>
#include <vector>
#include <map>
#include "OptionParser.h"

//remember to update measures which are defined in the model selection header file as enum (all supported metrics)
enum measures {TP, TN, FP, FN, Acc, Sens, Spec, Gmean, Fscore};

struct iter_summary{
    double C, gamma;
    std::map<measures, double> result;
};

class Config_params{
private:


    optparse::OptionParser parser_ = optparse::OptionParser() .description("just an example");

    optparse::Values options_;

    static Config_params* instance;
    //----- debug_export parameters -----
    int     debug_exp_CS;
    int     debug_exp_CS_level;
    int     debug_exp_MS;
    int     debug_exp_MS_level;
    int     debug_exp_MR;
    int     debug_exp_MR_level;


    std::string mlsvm_version;
    std::string p_indices_f_name, p_dist_f_name,
                n_indices_f_name, n_dist_f_name,
                p_norm_data_f_name, n_norm_data_f_name,
                test_ds_f_name;


    /// Descriptions are moved to param.xml
    //======= C++ parameters ========
    std::string cpp_srand_seed;
    int level;
    //======= main ========
    int main_num_repeat_exp;
    int main_num_kf_iter;
    //======= Loader ========
    std::string ds_path;
    std::string ds_name;
    std::string ds_version;
    int     pre_init_loader_matrix;
    bool    inverse_weight;
    int     ld_weight_type;
    double  ld_weight_param;
    //======= CS ========
    double  coarse_Eta;
    int     coarse_threshold;
    double  coarse_q ;
    int     coarse_r;
    int     cp_max_coarse_level;    // Control parameter to stop
    int     cs_use_real_points;
    double  cs_weak_edges_ft;       // filter threshold for weak edges
    //======= Model selection ========
    int     ms_status;
    int     ms_limit;
    int     ms_svm_id;
    int     ms_first_stage;
    int     ms_second_stage;
    int     ms_best_selection;
    double  ms_validation_part;
    int     ms_print_untouch_reuslts;
    double  ms_bs_gm_threshold;
    int     ms_save_final_model;
    //======= SVM ========
    int     svm_type;
    int     kernel_type;
    int     degree;
    double  gamma;
    double  coef0;
    double  nu;
    double  cache_size;
    double  C;
    double  eps;
    double  p;
    int     shrinking;
    int     probability;
    int     nr_weight;
    //========== Refinement ==========
    float   rf_add_fraction;
    int     rf_sel_all_min_limit;
    bool    rf_add_distant_point_status;
    int     rf_weight_vol;
    int     rf_start_partitioning;
    int     rf_partition_max_size;
    int     rf_num_opposite_partitions;
    std::vector<iter_summary> all_summary;
    //======= Exp ========
    double best_C;
    double best_gamma;
    bool best_params_are_set;
public:
    static Config_params* getInstance();



    void print_params();
//    void set_inputs(std::string const ds_name, std::string const ds_version );
    void set_inputs();
    void set_ds_path(std::string const new_ds_path);
    void set_ds_name(std::string const new_ds_name);

    void init_to_default();
    void read_params(std::string XML_FILE_PATH,int argc, char * argv[]);
    void update_srand_seed();
    void debug_only_set_srand_seed(std::string new_seed);

//    std::string get_ds_path() {return ds_path;}
//    std::string get_ds_name() {return ds_name;}
    std::string get_ds_path() {return options_["ds_path"];}
    std::string get_ds_name() {return options_["ds_name"];}

    std::string get_p_indices_f_name() {return p_indices_f_name;}
    std::string get_p_dist_f_name() {return p_dist_f_name;}
    std::string get_n_indices_f_name() {return n_indices_f_name;}
    std::string get_n_dist_f_name() {return n_dist_f_name;}
    std::string get_p_norm_data_f_name() {return p_norm_data_f_name;}
    std::string get_n_norm_data_f_name() {return n_norm_data_f_name;}
    std::string get_test_ds_f_name() {return test_ds_f_name;}

    void add_final_summary(iter_summary current_summary);
    void set_best_parameters(measures preferred_measure=Gmean);
    bool get_best_params_status(){ return this->best_params_are_set; }
    double get_best_C(){ return this->best_C; }
    double get_best_gamma(){ return this->best_gamma; }
    void print_final_results();

    // Debug parameters
    int get_debug_exp_CS() {return debug_exp_CS; }
    int get_debug_exp_CS_level() {return debug_exp_CS_level; }
    int get_debug_exp_MS() {return debug_exp_MS; }
    int get_debug_exp_MS_level() {return debug_exp_MS_level; }
    int get_debug_exp_MR() {return debug_exp_MR; }
    int get_debug_exp_MR_level() {return debug_exp_MR_level; }

    // Multilevel parameters
//    std::string get_cpp_srand_seed() {return cpp_srand_seed;}
//    int    get_main_num_repeat_exp(){ return main_num_repeat_exp;}
//    int    get_main_num_kf_iter(){ return main_num_kf_iter;}
    std::string get_cpp_srand_seed() {return options_["cpp_srand_seed"];}
    int    get_main_num_repeat_exp(){ return stoi(options_["main_num_repeat_exp"]);}
    int    get_main_num_kf_iter(){ return stoi(options_["main_num_kf_iter"]);}

//    int    get_pre_init_loader_matrix(){ return pre_init_loader_matrix;}
    bool   get_inverse_weight() { return inverse_weight; }
    int    get_ld_weight_type() { return ld_weight_type; }
    double get_ld_weight_param() { return ld_weight_param; }
    int    get_pre_init_loader_matrix(){ return stoi(options_["pre_init_loader_matrix"]);}

    // Coarsening
//    double  get_coarse_Eta() { return coarse_Eta;}
//    double  get_coarse_threshold() { return coarse_threshold; }
//    double  get_coarse_q() { return coarse_q; }
//    int     get_coarse_r() { return coarse_r; }
//    int     get_cp_max_coarse_level() { return cp_max_coarse_level; }
    double  get_coarse_Eta()        { return stod(options_["coarse_Eta"]); }
    double  get_coarse_threshold()  { return stod(options_["coarse_threshold"]); }
    double  get_coarse_q()          { return stod(options_["coarse_q"]); }
    int     get_coarse_r()          { return stoi(options_["coarse_r"]); }
    int     get_cp_max_coarse_level() { return stoi(options_["cp_max_coarse_level"]); }
    int     get_cs_use_real_points() { return cs_use_real_points; }
    double  get_cs_weak_edges_ft()          { return stod(options_["cs_weak_edges_ft"]); }

    // Model Selection
//    bool    get_ms_status() { return  ms_status; }
//    int     get_ms_limit() { return ms_limit;}
//    int     get_ms_svm_id() { return  ms_svm_id; }
//    int     get_ms_first_stage() { return  ms_first_stage; }
//    int     get_ms_second_stage() { return ms_second_stage; }
//    bool    get_ms_print_untouch_reuslts() { return  ms_print_untouch_reuslts; }
    int     get_ms_status()         { return  stoi(options_["ms_status"]); }
    int     get_ms_limit()          { return  stoi(options_["ms_limit"]);}
    int     get_ms_svm_id()         { return  stoi(options_["ms_svm_id"]); }
    int     get_ms_first_stage()    { return  stoi(options_["ms_first_stage"]); }
    int     get_ms_second_stage()   { return  stoi(options_["ms_second_stage"]); }
    double  get_ms_validation_part(){ return  stod(options_["ms_validation_part"]); }
    int     get_ms_print_untouch_reuslts() { return  stoi(options_["ms_print_untouch_reuslts"]); }
    int     get_ms_save_final_model(){ return ms_save_final_model; }

    double  get_ms_bs_gm_threshold() { return ms_bs_gm_threshold; }
    int    get_ms_best_selection() { return stoi(options_["ms_best_selection"]); }

    // SVM
    int     get_svm_svm_type() { return svm_type; }
//    int     get_svm_kernel_type() { return kernel_type; }
    int     get_svm_kernel_type() { return stoi(options_["kernel_type"]); }

    int     get_svm_degree() { return degree; }
//    double  get_svm_gamma() { return gamma; }
    double  get_svm_gamma() { return stod(options_["gamma"]); }

    double  get_svm_coef0() { return coef0; }
    double  get_svm_nu() { return nu; }
    double  get_svm_cache_size() { return cache_size; }
    double  get_svm_p() { return p; }
    int     get_svm_nr_weight() { return nr_weight; }
//    double  get_svm_C() { return C; }
//    double  get_svm_eps() { return eps; }
//    int     get_svm_shrinking() { return shrinking; }
//    int     get_svm_probability() { return probability; }
    double  get_svm_C() { return  stod(options_["C"]); }
    double  get_svm_eps() { return  stod(options_["eps"]); }
    int     get_svm_shrinking() { return  stoi(options_["shrinking"]); }
    int     get_svm_probability() { return  stoi(options_["probability"]); }


    // Refienment
    int     get_rf_sel_all_min_limit()  { return  rf_sel_all_min_limit; }
    bool    get_rf_add_distant_point_status()        { return  rf_add_distant_point_status; }

//    float   get_rf_add_fraction()       { return  rf_add_fraction; }
//    bool    get_rf_weight_vol()         { return  rf_weight_vol; }
//    int     get_rf_start_partitioning() { return  rf_start_partitioning; }
//    int     get_rf_partition_max_size() { return  rf_partition_max_size; }

    float   get_rf_add_fraction()       { return  stof(options_["rf_add_fraction"]); }
    int     get_rf_weight_vol()         { return  stoi(options_["rf_weight_vol"]); }
    int     get_rf_start_partitioning()   { return  stoi(options_["rf_start_partitioning"]); }
    int     get_rf_partition_max_size()   { return  stoi(options_["rf_partition_max_size"]); }
    int     get_rf_num_opposite_partitions() { return  rf_num_opposite_partitions; }


};

#endif // CONFIG_PARAMS_H


