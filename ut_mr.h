#ifndef UT_MR_H
#define UT_MR_H

#include <petscmat.h>
#include <vector>

class ut_MR
{
public:
//    ut_MR();
    ut_MR(){}
    void test_calc_center();
    void test_calc_distances();
    void load_matrix(const char * f_name, Mat& m_data, bool print);
    void load_vec(const char * f_name, Vec& v_input, bool print);
    void read_parts(std::vector<std::vector<PetscInt>>& vv_parts, Mat& m_parts, bool print);


// - - - - model selection - - - -
    void test_ms_ud();
};

#endif // UT_MR_H
