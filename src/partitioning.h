#ifndef PARTITIONING_H
#define PARTITIONING_H

#include "petscmat.h"
#include <vector>
#include <unordered_set>
#include <algorithm>
//#include <map>
#include "solver.h"
#include "model_selection.h"
#include <metis.h>

class Partitioning{
    void scale_edge_weights(idx_t *adjwgt, PetscInt num_edge,
                            PetscScalar *arr_adjwgt);
public:

    int get_parts(Mat& m_WA, Vec& v_vol, int num_req_parts, int level,
                  std::vector<std::vector<PetscInt>>& vv_parts,
                  Mat& m_parts, unsigned int metis_seed=1);

    void single_part_matrix(Mat& m_neigh_Data, Mat& m_parts);

    void calc_single_center(Mat& m_neigh_Data, Vec& v_vol,
                            Mat& m_centers,
                            std::vector<PetscScalar>& v_sum_vol_parts);

    void calc_center(Mat& m_neigh_Data, Vec& v_neigh_vol,
                     std::vector<std::vector<PetscInt>>& vv_parts,
                     Mat& m_parts, Mat& m_centers,
                     std::vector<PetscScalar>& v_sum_vol_parts);

    void calc_distances(int num_part_p, int num_part_n, Mat& m_centers_p,
                        Mat& m_centers_n, Mat& m_dist);

    void find_groups(int num_part_p, int num_part_n, Mat& m_dist,
                     std::vector<std::pair<int, int> >& v_groups);

    void calc_avg_center(Mat& m_centers_p, Mat& m_centers_n,
                         std::vector<std::pair<int, int> >& v_groups,
                         std::vector<PetscScalar>& v_sum_vol_parts_p,
                         std::vector<PetscScalar>& v_sum_vol_parts_n,
                         Mat& m_avg_centers);

    void calc_center_volume(Mat& m_centers,
                            std::vector<PetscScalar>& v_sum_vol_parts,
                            Mat& m_center_vol);

    void create_group_index(int target_group,
                            std::vector<std::pair<int, int> > v_groups,
                            Mat& m_parts_p, Mat& m_parts_n,
                            std::vector<PetscInt>& v_p_index,
                            std::vector<PetscInt>& v_n_index);

    void calc_performance_measure(Mat& m_TD,
                            std::vector<Mat>& v_mat_avg_center,
                            std::vector<Mat>& v_mat_predicted_labels,
                            summary& result_summary);

    void calc_performance_measure(PetscInt num_VD_p, PetscInt num_VD_n,
                            std::vector<Mat>& v_mat_avg_center,
                            std::vector<Mat>& v_mat_predicted_labels,
                            summary& result_summary);

    void calc_majority_voting(std::vector<Mat>& v_mat_predicted_labels,
                            int num_TD_points,
                            std::vector<double>& v_final_predicted_label );

    void calc_majority_voting_distance_weight(
                            std::vector<Mat>& v_mat_predicted_labels,
                            int num_TD, Mat& m_TD,
                            std::vector<Mat>& v_mat_AC, int distance_id,
                            std::vector<double>& v_final_predicted_label );

    void calc_majority_voting_nearest(std::vector<Mat>& v_mat_predicted_labels,
                            int num_TD, Mat& m_TD, std::vector<Mat>& v_mat_AC,
                            int distance_id,
                            std::vector<double>& v_final_predicted_label );
};

#endif // PARTITIONING_H
