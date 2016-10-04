#ifndef MODEL_SELECTION_H
#define MODEL_SELECTION_H

#include "solver.h"

struct ms_range{
    double min;
    double max;
};
struct ud_point{
    double C;
    double G;
};




class ModelSelection{
public:
    ModelSelection(){
        set_range();
    }

    void main(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n, int level, svm_model& best_trained_model);

    void uniform_design(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n, bool inh_params,
                        double param_C, double param_G, int level, solution & udc_sol);

    void uniform_design_separate_validation(Mat& m_train_data_p, Vec& v_train_vol_p, Mat& m_train_data_n, Vec& v_train_vol_n, bool inh_params,
                            double param_C, double param_G, int level, solution & udc_sol, std::vector<ref_results>& v_ref_results);
//    void uniform_design_coarsest(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n, bool inh_params,
//                                 double param_C, double param_G, int level, solution & udc_sol);




    void uniform_design_index_base(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n, bool inh_params, double last_c,
                            double last_gamma,int level, std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
                            std::unordered_set<PetscInt>& uset_SV_index_p, std::unordered_set<PetscInt>& uset_SV_index_n,
                            Mat& m_testdata, int classifier_id, Mat& m_all_predict);

    /*
     * p_data and n_data are the matrices for whole positive class and negative class respectively
     * v_vol_p and v_vol_n are the volume for all the points in p_data, n_data matrices respectively
     * inh_params is either 0, 1 : 0 means not to inherit the C, gamma parameters and 1 is vice versa
     * v_p_index is the vector of indices for the points in p_data (v_vol_p) which were Support Vector or their neighbors in coarser level
     * v_n_index is the same as v_p_index for negative class (majority class)
     * uset_SV_index_p is the set of indices for all the points which are selected as SV as this level, it stores all other SVs
     * which are selected in other calls to this method
     * uset_SV_index_n is the same as above for negative class
     * the m_VD_p and m_VD_n are validation data which comes from whole training data in the beginning of the v-cycle
     * the m_testdata is the real testdata from the beginning of the v-cycle
     * the classifier_id is the group id which is a set of partitions from both classes
     * m_all_predict stores the predicted lables for classifier i at row i
     */
    void uniform_design_index_base_separate_validation(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n,
                            bool inh_params, double last_c, double last_gamma,int level,
                            std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
                            std::unordered_set<PetscInt>& uset_SV_index_p, std::unordered_set<PetscInt>& uset_SV_index_n,
                            Mat& m_VD_p, Mat& m_VD_n, Mat& m_testdata, int classifier_id, Mat& m_all_predict);
private :

    ms_range range_c;     // range of C
    ms_range range_g;     // range of gamma
    ud_point point_center;      // center point

//    bool sortByGmean(const summary &lhs, const summary &rhs);
    summary summary_factory_update_iter(const summary& in_summary, const int iter);

protected:
    void set_range();
    std::vector<ud_point> ud_param_generator(int stage, bool inh_param, double param_C, double param_G);
    int select_best_model(std::vector<summary> map_summary, int level, int stage);

};
#endif // MODEL_SELECTION_H

