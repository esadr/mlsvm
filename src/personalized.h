#ifndef PERSONALIZED_H
#define PERSONALIZED_H

#include "model_selection.h"

class personalized  : ModelSelection
{
public:
    personalized() {}
    void uniform_design_index_base(int id, Mat& m_data, std::vector<int>& v_lbl, bool inh_params, double last_c,
                            double last_gamma, const PetscScalar * arr_train_index, PetscInt num_nnz, std::vector<int>& v_predict_lbl);

    void Single_training_index_base(int id, Mat& m_data, std::vector<int>& v_lbl, bool inh_params, double last_c,
                            double last_gamma, const PetscScalar * arr_train_index, PetscInt num_nnz, std::vector<int>& v_predict_lbl);

    void evaluate_results(const std::vector<int>& v_target_lbl, const std::vector<int>& v_predict_lbl, summary& final_summary );

    void calc_statistics(int id, Mat& m_stats, std::vector<int>& v_target_lbl, std::vector<int>& v_predict_lbl,
                                       const PetscScalar * arr_train_index, PetscInt num_nnz);

    void filter_test_point(Mat& m_NN, Mat& m_filtered_NN);

    void multi_level_solver(int id, Mat& m_data, Mat& m_NN_filtered, std::vector<int>& v_lbl,
                            const PetscScalar * arr_train_index, PetscInt num_nnz, std::vector<int>& v_predict_lbl);
};

#endif // PERSONALIZED_H
