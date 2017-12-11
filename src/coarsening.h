#ifndef COARSENING_H
#define COARSENING_H

#include <petscmat.h>
#include <algorithm>            //for using std::sort
#include "ds_graph.h"
#include "ds_temps.h"

//#include "utility.h"

//#define boundary_points 1       //experiment Jan 9, 2017: 1 means, the boundary points are fractioned in multiple aggregates

struct cs_info{
    PetscInt num_point;
    PetscInt num_edge;
    PetscInt min_num_edge;
    PetscScalar avg_num_edge;
    PetscInt max_num_edge;
    PetscScalar median_num_edge;
};

class Coarsening {
private:
    PetscInt num_coarse_points = 0;
    std::string cc_name;        // classifier_class_name for printing

    struct pair_hash{
        inline std::size_t operator()(const std::pair<PetscInt,PetscInt> &v) const{
            return v.first * 31 + v.second;
        }
    };
public:
    Coarsening() {}

    Coarsening(std::string classfier_class_name){ cc_name = classfier_class_name; }

    /* @param WA
     *      weighted Adjancency matrix (graph)
     * @param vol
     *      vector of volumes for the nodes (current level)
     * @param ref_info
     *      graph information for printing during the refinement
     * @return
     *      P matrix
     */
    Mat calc_P(Mat& WA, Vec& vol,std::vector<NodeId>& v_seeds_indices, cs_info& ref_info, bool debug=0);

    /*
     * calculate the P matrix without coarsening, the minority class start using this instead of usual when its size is reduced to acceptable size
     */
    Mat calc_P_without_shrinking(Mat& WA, Vec& vol,std::vector<NodeId>& v_seeds_indices, cs_info& ref_info);

//    Mat calc_aggregate_data(Mat& P, Mat& data, Vec& v_vol);
    Mat calc_aggregate_data(Mat& P, Mat& data, Vec& v_vol, std::vector<NodeId>& seeds_indices);
    /* @param P
     *      P matrix
     * @param data
     *      data matrix (current level)
     * @return
     *      data matrix (next level)
     */
    int normalize_vector(Vec& v_raw, Vec& v_norm );
    int calc_inverse(Vec& v_raw, Vec& v_inv);

    Mat calc_WA_c(Mat&, Mat&);
    /* @param P
     *      P matrix
     * @param WA
     *      WA matrix (current level)
     * @return
     *      WA_c matrix (coarser level weighted adjacency)
     */

    Vec calc_coarse_volumes(Mat&, Vec&);
    /* @param P
     *      P matrix
     * @param vol
     *      vector of volumes for the nodes (current level)
     * @return
     *      vector of volumes for the nodes (next level)
     */
    void filter_weak_edges(Mat & A, double alfa, int level);

    Mat calc_real_weight(Mat& m_WA_c, Mat& m_data_c);

    double calc_stat_nnz(Mat& m_A, bool approximate=1);
};


#endif // COARSENING_H
