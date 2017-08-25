#ifndef UT_CS_H
#define UT_CS_H

#include "coarsening.h"

class UT_CS : Coarsening
{
public:
//    UT_CS();
    void test_filtering_weak_edges();
    void test_calc_p();
    void test_calc_p(Mat& m_WA);
};

#endif // UT_CS_H
