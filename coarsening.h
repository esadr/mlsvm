#ifndef COARSENING_H
#define COARSENING_H

#include <petscmat.h>
#include <algorithm>            //for using std::sort
#include "ds_graph.h"
#include "ds_temps.h"

//#include "utility.h"

struct cs_info{
    PetscInt num_point;
    PetscInt num_edge;
    PetscInt min_num_edge;
    PetscInt avg_num_edge;
    PetscInt max_num_edge;
};

class Coarsening {
private:
    PetscInt num_coarse_points = 0;
    std::string cc_name;        // classifier_class_name for printing
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
    Mat calc_P(Mat& WA, Vec& vol,std::vector<NodeId>& v_seeds_indices, cs_info& ref_info);


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
    void filter_weak_edges(Mat & A, double alfa);
};


#endif // COARSENING_H
