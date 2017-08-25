#include "ut_cf.h"
#include "common_funcs.h"
//#include <cmath>
//#include <iostream>


//    template <class T>
//    T UT_CF::sum_array(const std::vector<T>& vec_In){
//        T total=0;
//        for(unsigned i=0; i < vec_In.size(); ++i){
//            total += vec_In[i];
//        }
//        std::cout << "total:" << total << std::endl;
//        return total;
//    }


//void UT_CF::t_mean_vector(){

//    std::vector<float> v_test;
//    for(int i=0; i < 100 ; i++){
//        v_test.push_back(i*.3);
//    }
//    float mean = 0;
//    std::cout << "size:" << v_test.size() << std::endl;
//    CommonFuncs cf;
//    mean = cf.sum_vector(v_test);

////    mean = sum_array(v_test);

//    printf("UT_CF:t_mean mean:%g\n", mean);

//}


void UT_CF::t_mean_array(){
    double * arr_test = (double *) malloc(sizeof(double) * 100);
    for(int i=0; i < 100 ; i++){
        arr_test[i] = i*.3;
    }
    float mean = 0;
    CommonFuncs cf;
    mean = cf.mean_array(arr_test, 100);
//    mean = cf.sum_array(arr_test, 100);


    printf("UT_CF:t_mean mean:%g\n", mean);

}


void UT_CF::t_STD_array(){
    double * arr_test = (double *) malloc(sizeof(double) * 100);
    for(int i=0; i < 100 ; i++){
        arr_test[i] = i;
    }
    double mean = 0, f_std=0;
    CommonFuncs cf;
    f_std = cf.STD_array(arr_test, 100, mean);

    printf("UT_CF:t_mean mean:%g\n", mean);
    printf("UT_CF:t_mean std:%g\n", f_std);

}

void UT_CF::t_zscore_array(){
    double * arr_test = (double *) malloc(sizeof(double) * 100);
    for(int i=0; i < 100 ; i++){
        arr_test[i] = i;
    }
    double d_mean = 0, d_std=0;
    CommonFuncs cf;
    cf.zscore_array(arr_test, 100);

    for(int i=0; i < 100 ; i++){
        printf("UT_CF::t_zscore i:%d zscore:%g\n", i,arr_test[i]);
    }


}
