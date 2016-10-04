#ifndef CONFIG_PARAMS_H
#define CONFIG_PARAMS_H
#include <string>
#include <vector>
#include <map>
#include "OptionParser.h"
#include "ds_global.h"

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
    int         main_num_repeat_exp;
    int         main_num_kf_iter;
    int         multi_level_status;
    std::string exp_info;
    //======= NN ========
    int         nn_number;
    int         nn_distance_type;
    //======= Loader ========
    std::string ds_path;
    std::string ds_name;
    std::string tmp_path;
    int         pre_init_loader_matrix;
    bool        inverse_weight;
    int         ld_weight_type;
    double      ld_weight_param;
    //======= CS ========
    double      coarse_Eta;
    int         coarse_threshold;
    double      coarse_q ;
    int         coarse_r;
    int         cp_max_coarse_level;    // Control parameter to stop
    int         cs_use_real_points;
    double      cs_weak_edges_ft;       // filter threshold for weak edges
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
    bool    rf_add_distant_point_status;
    int     rf_weight_vol;
    int     pr_start_partitioning;
    //========== Partitioning ==========
    int     pr_partition_max_size;
    int     pr_maj_voting_id;

//    std::vector<iter_summary> all_summary;
    std::vector<summary> all_summary;
    //======= Exp ========
    double best_C;
    double best_gamma;
    bool best_params_are_set;


    void check_input_parameters();
public:
    static Config_params* getInstance();



    void print_params();
//    void set_inputs(std::string const ds_name, std::string const ds_version );
    void set_inputs_file_names();

    void set_ds_path(std::string const new_ds_path);
    void set_ds_name(std::string const new_ds_name);
    void debug_only_set_p_norm_data_path_file_name(std::string const path_file_name);
    void debug_only_set_n_norm_data_path_file_name(std::string const path_file_name);

    void init_to_default();
    void read_params(std::string XML_FILE_PATH,int argc, char * argv[]);
    void update_srand_seed();
    void debug_only_set_srand_seed(std::string new_seed);

//    std::string get_ds_path() {return ds_path;}
//    std::string get_ds_name() {return ds_name;}
    std::string get_ds_path() const {return options_["ds_path"];}
    std::string get_ds_name() const {return options_["ds_name"];}
//    std::string get_p_fname() {return options_["p_norm_data_f_name"];}
//    std::string get_n_fname() {return options_["n_norm_data_f_name"];}
    std::string get_tmp_path() const ;
    std::string get_exp_info() const {return options_["exp_info"];}

    std::string get_p_indices_f_name() const {return p_indices_f_name;}
    std::string get_p_dist_f_name() const {return p_dist_f_name;}
    std::string get_n_indices_f_name() const {return n_indices_f_name;}
    std::string get_n_dist_f_name() const {return n_dist_f_name;}
    std::string get_p_norm_data_f_name() const {return p_norm_data_f_name;}
    std::string get_n_norm_data_f_name() const {return n_norm_data_f_name;}

    void set_test_ds_f_name(std::string new_test_f_name) { test_ds_f_name = new_test_f_name;}
    std::string get_test_ds_f_name() const {return test_ds_f_name;}

    void add_final_summary(summary current_summary, int selected_level);
    void print_summary(const summary& summary_in, std::string caller_method, int level=-1, int iter=-1, int stage=-1, int fold=-1) const;
    void set_best_parameters(measures preferred_measure=Gmean);
    bool get_best_params_status() const{ return this->best_params_are_set; }
    double get_best_C() const{ return this->best_C; }
    double get_best_gamma() const{ return this->best_gamma; }
    void print_final_results() const;
    void print_ref_result(const std::vector<ref_results>& v_ref_results) const;

    // Debug parameters
    int get_debug_exp_CS()          const {return debug_exp_CS; }
    int get_debug_exp_CS_level()    const {return debug_exp_CS_level; }
    int get_debug_exp_MS()          const {return debug_exp_MS; }
    int get_debug_exp_MS_level()    const {return debug_exp_MS_level; }
    int get_debug_exp_MR()          const {return debug_exp_MR; }
    int get_debug_exp_MR_level()    const {return debug_exp_MR_level; }
    // NN
    int   get_nn_number() const {return stoi(options_["nn_number"]);}
    int   get_nn_distance_type() const {return stoi(options_["nn_distance_type"]);}

    // Multilevel parameters
    std::string get_cpp_srand_seed()    const { return options_["cpp_srand_seed"];}
    int    get_main_num_repeat_exp()    const { return stoi(options_["main_num_repeat_exp"]);}
    int    get_main_num_kf_iter()       const { return stoi(options_["main_num_kf_iter"]);}
    int    get_multi_level_status()     const { return stoi(options_["multi_level_status"]); }

    bool   get_inverse_weight()     const { return inverse_weight; }
    int    get_ld_weight_type()     const { return ld_weight_type; }
    double get_ld_weight_param()    const { return ld_weight_param; }
    int    get_pre_init_loader_matrix() const{ return stoi(options_["pre_init_loader_matrix"]);}

    // Coarsening
    double  get_coarse_Eta() const        { return stod(options_["coarse_Eta"]); }
    double  get_coarse_threshold() const  { return stod(options_["coarse_threshold"]); }
    double  get_coarse_q() const          { return stod(options_["coarse_q"]); }
    int     get_coarse_r() const          { return stoi(options_["coarse_r"]); }
    int     get_cp_max_coarse_level() const { return stoi(options_["cp_max_coarse_level"]); }
    int     get_cs_use_real_points() const { return cs_use_real_points; }
    double  get_cs_weak_edges_ft() const          { return stod(options_["cs_weak_edges_ft"]); }

    // Model Selection
    int     get_ms_status() const         { return  stoi(options_["ms_status"]); }
    int     get_ms_limit() const          { return  stoi(options_["ms_limit"]);}
    int     get_ms_svm_id() const         { return  stoi(options_["ms_svm_id"]); }
    int     get_ms_first_stage() const    { return  stoi(options_["ms_first_stage"]); }
    int     get_ms_second_stage() const   { return  stoi(options_["ms_second_stage"]); }
    double  get_ms_validation_part() const{ return  stod(options_["ms_validation_part"]); }
    int     get_ms_print_untouch_reuslts() const { return  stoi(options_["ms_print_untouch_reuslts"]); }
    int     get_ms_save_final_model() const{ return ms_save_final_model; }

    double  get_ms_bs_gm_threshold() const { return ms_bs_gm_threshold; }
    int    get_ms_best_selection() const { return stoi(options_["ms_best_selection"]); }

    // SVM
    int     get_svm_svm_type() const { return svm_type; }
    int     get_svm_kernel_type() const { return stoi(options_["kernel_type"]); }

    int     get_svm_degree() const { return degree; }
    double  get_svm_gamma() const { return stod(options_["gamma"]); }

    double  get_svm_coef0() const { return coef0; }
    double  get_svm_nu() const { return nu; }
    double  get_svm_cache_size() const { return cache_size; }
    double  get_svm_p() const { return p; }
    int     get_svm_nr_weight() const { return nr_weight; }
    double  get_svm_C() const { return  stod(options_["C"]); }
    double  get_svm_eps() const { return  stod(options_["eps"]); }
    int     get_svm_shrinking() const { return  stoi(options_["shrinking"]); }
    int     get_svm_probability() const { return  stoi(options_["probability"]); }


    // Refienment
    bool    get_rf_add_distant_point_status() const        { return (bool) stoi(options_["rf_add_distant_point_status"]); }
    float   get_rf_add_fraction() const       { return  stof(options_["rf_add_fraction"]); }
    int     get_rf_weight_vol() const         { return  stoi(options_["rf_weight_vol"]); }
    int     get_pr_start_partitioning() const   { return  stoi(options_["pr_start_partitioning"]); }
    int     get_pr_partition_max_size() const   { return  stoi(options_["pr_partition_max_size"]); }

    // Partioning
    int     get_pr_maj_voting_id() const         { return  stoi(options_["pr_maj_voting_id"]); }
};

#endif // CONFIG_PARAMS_H


