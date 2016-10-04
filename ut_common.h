#ifndef UT_COMMON_H
#define UT_COMMON_H

#include <petscmat.h>
#include <iostream>

class UT_Common{
public:
    void load_matrix(const char * f_name, Mat& m_data, bool print);
    void load_vec(const char * f_name, Vec& v_input, bool print);
};

#endif // UT_COMMON_H
