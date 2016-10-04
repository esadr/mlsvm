#ifndef MAIN_RECURSION_H
#define MAIN_RECURSION_H

#include "coarsening.h"
#include "refinement.h"


class MainRecursion {

public:
    solution main(Mat& p_data, Mat& m_P_p, Mat& p_WA, Vec& p_vol, Mat& n_data, Mat& m_P_n, Mat& n_WA, Vec& n_vol, int level,
                  std::vector<ref_results>& v_ref_results);

};

#endif // MAIN_RECURSION_H
