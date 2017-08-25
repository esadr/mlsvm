#include "ut_ms.h"

#include "math.h"

void UT_MS::test_params(){
//    ms_range c_range, gamma_range;
////    ms_point center;
//    c_range.min = -6;
//    c_range.max = 10;
//    gamma_range.min = -12;
//    gamma_range.max = 1;
//    int fround = 9;

//    c_range.min = pow(2, c_range.min);
//    c_range.max = pow(2, c_range.max);
//    gamma_range.min = pow(2, gamma_range.min);

    //center.x = (c_range.max + c_range.min) / 2;
    //center.y = (gamma_range.max + gamma_range.min) / 2;
//    center.x = 6 ; //(c_range.max + c_range.min) / 2;
//    center.y = 0.4 ;// (gamma_range.max + gamma_range.min) / 2;


    set_range();
    std::vector<ud_point> v_points;
    v_points = ud_param_generator(1, 1, 1500, 0.005);
//    for(unsigned int i =0; i < v_points.size();++i){
//        std::cout << "[UT_MS] i: "<< i << ", C: "<< v_points[i].C << ", G: "<< v_points[i].G << std::endl;
//    }

//    ms.set_center(100, .5);
//    ms.ud_param_generator(1,false,0,0);
//    ms.ud_param_generator(1,true,100,0.5);        //the method is private now, so maybe we need to add a public tester method to ms class
//    ms.ud_param_generator(ms.get_range_c(),ms.get_range_gamma(),1, fround, ms.get_point_center());
//    ms.udpg_1st();
}



