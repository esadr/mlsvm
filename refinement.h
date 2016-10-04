#ifndef REFINEMENT_H
#define REFINEMENT_H

#include "model_selection.h"



class Refinement{
private:
    std::string test_dataset_f_name;
//    struct svm_model* model;
    struct group{
        std::vector<PetscInt> v_part_same_class;
        std::vector<PetscInt> v_part_oppo_class;
    };

public:
    Refinement(){
        test_dataset_f_name = Config_params::getInstance()->get_test_ds_f_name();
    }

    solution main(Mat& m_data_p, Mat& m_P_p, Vec& v_vol_p, Mat&m_WA_p,
                Mat& m_data_n, Mat& m_P_n, Vec& v_vol_n, Mat&m_WA_n,
                solution& sol_coarser,int level, std::vector<ref_results>& v_ref_results);

    void find_SV_neighbors(Mat& m_data, Mat& m_P, std::vector<int>& seeds_ind, Mat& m_WA, Mat& m_neighbors,
                                                                        std::string cc_name, IS& IS_neigh_id);


    void process_coarsest_level(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n, int level,
                                solution& sol_coarsest, std::vector<ref_results>& v_ref_results);

    void prepare_single_solution(svm_model **svm_trained_model, int num_row_p, solution& result_solution);
    void add_best_model(std::vector<ref_results>& v_ref_results) const;
};


#endif // REFINEMENT_H
