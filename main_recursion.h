#ifndef MAIN_RECURSION_H
#define MAIN_RECURSION_H

#include "coarsening.h"
#include "model_selection.h"
#include <unordered_set>

class MainRecursion {
private:
//    const char * test_dataset_f_name;
    std::string test_dataset_f_name;
    int last_result = 0;
    struct svm_model* model;
//    ModelSelection ms;            //moved to implementation code
//    float neighbor_fraction_ =0;    //percentage it should be between 0 and 1

    struct group{
        std::vector<PetscInt> v_part_same_class;
        std::vector<PetscInt> v_part_oppo_class;
    };


public:
//    MainRecursion(const char * test_f_name){
//    MainRecursion(const std::string test_f_name){
    MainRecursion(){
        test_dataset_f_name = Config_params::getInstance()->get_test_ds_f_name();
//        std::cout << "[MR][constructor] test_dataset_f_name: " << test_dataset_f_name << std::endl;
    }

//    solution main(Mat& p_data, Mat& p_WA, Vec& p_vol, Mat& n_data, Mat& n_WA, Vec& n_vol, int level);
    solution main(Mat& p_data, Mat& m_P_p, Mat& p_WA, Vec& p_vol, Mat& n_data, Mat& m_P_n, Mat& n_WA, Vec& n_vol, int level);

    /* @param WA
     *      weighted Adjancency matrix
     * @param vol
     *      the volume from the last level
     * @return
     *      Solution that includes SV, C, Gamma
     */

    bool set_neighbor_threshold(float fraction);

    solution refinement(Mat& m_data_p, Mat& m_P_p, Vec& v_vol_p, Mat&m_WA_p,
                        Mat& m_data_n, Mat& m_P_n, Vec& v_vol_n, Mat&m_WA_n,
                        solution& sol_coarser,int level );
    std::map<measures,double> calc_performance_measure(Mat& m_TD, Mat& m_predicted_labels);

    void calc_majority_voting(Mat& m_predicted_labels, int num_TD_points, std::vector<double>& v_final_predicted_label );

    bool create_group_index(int target_group, int num_part_p, int num_part_n, std::vector<std::vector<PetscInt> >& vv_groups,
                           Mat& m_parts_p, Mat& m_parts_n, std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index);

    //    void find_opposite_sets(int num_part_p, int num_part_n, int num_features,
//                                        std::vector<std::vector<std::pair<int, double>>>& dist_p);
//    void find_groups(int num_part_p, int num_part_n, Mat& m_dist, std::vector<group>& v_groups);
    void find_groups(int num_part_p, int num_part_n, Mat& m_dist, std::vector<std::vector<PetscInt> >& vv_groups);


    void calc_distances(int num_part_p, int num_part_n, Mat& m_centers_p,Mat& m_centers_n, Mat& m_dist);


    void calc_center(Mat& m_neigh_Data, Vec& v_vol, std::vector<std::vector<PetscInt>>& vv_parts, Mat& m_parts, Mat& m_centers);

    void get_parts(Mat& m_WA, Vec& v_vol, std::vector<std::vector<PetscInt>>& vv_parts, Mat& m_parts);

    void single_part_matrix(Mat& m_neigh_Data, Mat& m_parts);

    void calc_single_center(Mat& m_neigh_Data, Vec& v_vol, Mat& m_centers);

//    void find_SV_neighbors(Mat& m_data, Mat& m_P, std::vector<int>& seeds_ind,
//                           Mat& m_SV, Mat& m_neighbors, std::string cc_name,
//                           IS& IS_neigh_id);
    void find_SV_neighbors(Mat& m_data, Mat& m_P, std::vector<int>& seeds_ind,
                           Mat& m_WA, Mat& m_neighbors, std::string cc_name,
                           IS& IS_neigh_id);


    int get_parts_test();
};

#endif // MAIN_RECURSION_H
