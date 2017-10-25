#ifndef MAIN_RECURSION_H
#define MAIN_RECURSION_H

#include "coarsening.h"
#include "refinement.h"

class MainRecursion {

public:
    solution main(Mat& p_data, Mat& m_P_p, Mat& p_WA, Vec& p_vol,
                  Mat& n_data, Mat& m_P_n, Mat& n_WA, Vec& n_vol,
                  Mat& m_VD_p, Mat& m_VD_n, int level, std::vector<ref_results>& v_ref_results);

    solution main_community_detection(Mat& m_data, Mat& m_WA, Vec& v_vol,
                                      int num_clusters, int level);
};

#endif // MAIN_RECURSION_H
