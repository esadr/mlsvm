 #ifndef SOLVER_H
#define SOLVER_H

#define weight_instance 1

#if weight_instance == 1
    #include "svm_weighted.h"       //support instance weights, version 3.20 at 09/05/2016
#else
    #include "svm_unweighted.h"     //normal libsvm-3.21
#endif

#include "ds_global.h"
#include <petscmat.h>
#include <iostream>
#include <string>
#include <map>
#include <unordered_set>
#include "config_params.h"
#include <memory>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type)) //from svm_train


struct weight_info{
    PetscInt s_vol_p, s_vol_n;
};


class Solver{
private:

    struct weight_info{
        double s_vol_p, s_vol_n;
    };

    struct svm_parameter param;		// set by parse_command_line
    struct svm_problem prob;		// set by read_problem
    struct svm_model *local_model;
    struct svm_node *x_space;
    int p_num_node_=0, n_num_node_=0, p_num_elem_=0,n_num_elem_=0;
    int t_num_node_=0, t_num_elem_=0;
    int test_num_node_=0, test_num_elem_=0;
    int predict_probability=0;
    const char * test_dataset_f_name;

    void read_parameters();
    void print_parameters();
    void read_problem_index_base(Mat& p_train_data, Mat& n_train_data,
                                 std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
                                 PetscInt iter_p_end,PetscInt iter_n_end,
                                 Vec& v_vol_p, Vec& v_vol_n);

    void PD_read_problem_index_base(Mat& m_data, std::vector<int>& v_lbl, const PetscScalar * arr_index, int num_nnz); //personalized classifier

    void read_problem(Mat& m_train_data_p, Vec& v_vol_p, Mat& m_train_data_n, Vec& v_vol_n);

    void read_problem_without_instance_weight(Mat& m_train_data_p, Mat& m_train_data_n);

    void set_weights_num_points(svm_parameter& param_, PetscInt num_p_point, PetscInt num_n_point);

    void set_weights_sum_volume(svm_parameter& param_, Vec& v_vol_p, Vec& v_vol_n);

    void set_weights_sum_volume_index_base(svm_parameter& param_, Vec& v_vol_p, Vec& v_vol_n,
                        std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index, PetscInt iter_p_end, PetscInt iter_n_end);

    void PD_set_weights_sum_num_point_IB(svm_parameter& param_,std::vector<int>& v_lbl, const PetscScalar * arr_index, int num_nnz); //personalized classifier

    void alloc_memory_for_weights(svm_parameter& in_param, bool free_first);



public:
    Solver(){
        test_dataset_f_name = Config_params::getInstance()->get_test_ds_f_name().c_str();
        predict_probability = Config_params::getInstance()->get_svm_probability();
    }

    ~Solver(){
        std::cout << "[SV] solver goes out of scope\n";
    }
    void set_local_model(svm_model * in_model){
        local_model = in_model;
    }

    void free_solver(std::string caller_name);    //work as deconstructor

    svm_model * train_model(Mat& m_data_p, Vec& v_vol_p, Mat& m_data_n, Vec& v_vol_n,
                            bool inherit_params, double param_c, double param_gamma);


    svm_model * train_model_index_base(Mat& m_data_p, Vec& v_vol_p, Mat& m_data_n, Vec& v_vol_n,
                                               std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
                                               PetscInt iter_p_end, PetscInt iter_n_end,
                                               bool inherit_params, double param_c, double param_gamma);

    void stand_alone_train_without_instance_weight(Mat& m_data_p, Mat& m_data_n, std::string model_fname);

    void stand_alone_train_instance_weight(Mat& m_data_p , Vec& v_vol_p, Mat& m_data_n, Vec& v_vol_n, std::string model_fname);

    void PD_train_model_index_base(Mat& m_data, std::vector<int>& v_lbl,
                                            const PetscScalar * arr_train_index, PetscInt num_nnz_train,
                                            bool inherit_params, double param_c, double param_gamma);

    void partial_solver(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n, double last_c, double last_gamma,
                                    int level, std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
                                    std::unordered_set<PetscInt>& uset_SV_index_p, std::unordered_set<PetscInt>& uset_SV_index_n,
                                    Mat& m_VD_p, Mat& m_VD_n, Mat& m_VD_both, Mat& m_all_predict_VD, Mat& m_testdata, int classifier_id, Mat& m_all_predict_TD);

//    void cross_fold_data(const Mat data_p, const Mat data_n, Mat& train_, Mat& train_n_, Mat& test_total_);

    void combine_test_data(Mat& test_total, Mat& dt_test_p, PetscInt size_p, Mat& dt_test_n, PetscInt size_n, PetscInt max_num_col_);


    void predict_test_data_in_matrix_output(Mat& test_data, int target_row, Mat& m_predicted_label);

    void predict_VD_in_output_matrix(Mat& m_VD_p,Mat& m_VD_n, int target_row, Mat& m_predicted_label);

//    std::map<measures,double> evaluate_testdata(int level);
    void evaluate_testdata(int level, summary& final_summary);
    void evaluate_testdata(Mat& untouched_test_data, int level, summary& final_summary);

//    std::map<measures,double> test_predict(Mat& );
    void test_predict(Mat& test_data, summary& result_summary, int iteration=-1);
    
    void predict_validation_data(Mat& m_VD_p,Mat& m_VD_n, summary& result_summary, int iteration);

    svm_model * get_model(){return local_model;}

    svm_parameter get_model_params() {return local_model->param;}

//    std::map<measures,double> test_predict_index_base(Mat& m_data_p, Mat& m_data_n,
//                                                   std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
//                                                   PetscInt iter_p_end, PetscInt iter_n_end);
    void test_predict_index_base(Mat& m_data_p, Mat& m_data_n, std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
                                PetscInt iter_p_end, PetscInt iter_n_end, summary& result_summary, int iteration=-1);

    void test_predict_index_base_separate_validation(Mat& m_data_p, Mat& m_data_n,
                                                   std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
                                                   PetscInt iter_p_end, PetscInt iter_n_end, summary& result_summary, int iteration,
                                                   Mat& m_VD_p, Mat& m_VD_n);



    void PD_test_predict_index_base(Mat& m_data, std::vector<PetscInt> v_lbl, const PetscScalar * arr_train_index,
                                PetscInt idx_start_test, PetscInt idx_end_test, summary& result_summary, int iteration); //personalized classifier

    int PD_predict_a_label(Mat& m_data, int target_row);                //personalized classifier


    void prepare_solution_single_model(svm_model * model_, int num_point_p, solution& sol_single_model);


    void get_separate_matrices(Mat& m_data, std::vector<int>& v_target_lbl,
                                       const PetscScalar * arr_train_index, PetscInt num_nnz,
                                             Mat& m_p_data, Mat& m_n_data);
    
};

#endif // SOLVER_H
