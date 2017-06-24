#include "model_selection.h"
#include "algorithm"
#include "k_fold.h"
#include "config_logs.h"
#include <cmath>
#include "etimer.h"
#include "common_funcs.h"
#include "loader.h"     //only for testing the SNGM experiment Sep 21, 2016

struct BetterGmean
{
    bool operator () (const summary& a, const summary& b) const
    {
        if( (a.perf.at(Gmean) - b.perf.at(Gmean)) > 0.01 )         //a has completely better gmean than b
            return true;
        else{
            if( (b.perf.at(Gmean) - a.perf.at(Gmean)) > 0.01 )     //b has completely better gmean than a
                return false;
            else{                                                    //similar gmean
                if(Config_params::getInstance()->get_ms_best_selection() == 1)
                    return (a.num_SV_p + a.num_SV_n <  b.num_SV_p + b.num_SV_n );    // a has less nSV than b which is better
                else
                    return false;              // the gmeans  are similar and we don't care for nSV  ???
            }
        }
    }
};

struct BetterSN_Gmean
{
    bool operator () (const summary& a, const summary& b) const
    {
        if( (a.perf.at(Sens) - b.perf.at(Sens)) > 0.10 )         //a has completely better gmean than b
            return true;
        else{
            if( (b.perf.at(Sens) - a.perf.at(Sens)) > 0.10 )     //b has completely better gmean than a
                return false;
            else{                                                    //similar gmean
                if(Config_params::getInstance()->get_ms_best_selection() == 1)
                    return (a.perf.at(Gmean) >  b.perf.at(Gmean) );    // a has less nSV than b which is better
                else
                    return false;              // the gmeans  are similar and we don't care for nSV  ???
            }
        }
    }
};

struct Better_Gmean_SN
{
    bool operator () (const summary& a, const summary& b) const
    {
        float filter_range = 0.02;
        if( (a.perf.at(Gmean) - b.perf.at(Gmean)) > filter_range )         //a has completely better gmean than b
            return true;
        else{
            if( (b.perf.at(Gmean) - a.perf.at(Gmean)) > filter_range )     //b has completely better gmean than a
                return false;
            else{                                                    //similar gmean
                if(Config_params::getInstance()->get_ms_best_selection() == 1)
                    return (a.perf.at(Sens) >  b.perf.at(Sens) );    // a has less nSV than b which is better
                else
                    return false;              // the gmeans  are similar and we don't care for nSV  ???
            }
        }
    }
};


//bool sortByGmean(const summary &lhs, const summary &rhs)
//{
//    if( fabs((lhs.perf.at(Gmean) - rhs.perf.at(Gmean)) > 0.01 )     //a has better gmean
//        return true;
//    else{                                                    //similar gmean
//        if(Config_params::getInstance()->get_ms_best_selection() == 1)
//            return (lhs.num_SV_p + lhs.num_SV_n <  rhs.num_SV_p + rhs.num_SV_n );    // less nSV is better
//        else
//            return false;              // the gmean were similat and we don't care for nSV
//    }
//}


struct BetterNPV
{
    bool operator () (const summary& a, const summary& b) const
    {
        if( (a.perf.at(NPV) - b.perf.at(NPV)) > 0.01 )     //a has better gmean
            return true;
        else{                                                    //similar gmean
            if(Config_params::getInstance()->get_ms_best_selection() == 1)
                return (a.num_SV_p + a.num_SV_n <  b.num_SV_p + b.num_SV_n );    // less nSV is better
            else
                return false;
        }
    }
};

struct BetterPPV
{
    bool operator () (const summary& a, const summary& b) const
    {
        if( (a.perf.at(PPV) - b.perf.at(PPV)) > 0.01 )     //a has better gmean
            return true;
        else{                                                    //similar gmean
            if(Config_params::getInstance()->get_ms_best_selection() == 1)
                return (a.num_SV_p + a.num_SV_n <  b.num_SV_p + b.num_SV_n );    // less nSV is better
            else
                return false;
        }
    }
};


struct BetterAcc
{
    bool operator () (const summary& a, const summary& b) const
    {
        if( (a.perf.at(Acc) - b.perf.at(Acc)) > 0.01 )     //a has better gmean
            return true;
        else{                                                    //similar gmean
//            if(Config_params::getInstance()->get_ms_best_selection() == 1)
//                return (a.num_SV_p + a.num_SV_n <  b.num_SV_p + b.num_SV_n );    // less nSV is better
//            else
                return false;
        }
    }
};





/*
 * shift the points with respect to the last parameters
 * input values are large since they are best C, gamma
 * So, we use log base 2 to scale them to right dimension
 */
//void ModelSelection::set_center(double center_C, double center_G){
//    point_center.C = center_C;
//    point_center.G = center_G;
//    std::cout << "[MS][set_center] point_center.C:"<< point_center.C <<", point_center.G:" << point_center.G << std::endl;

////    float min = 0.5;
////    float max = 1.2;
//////    double scaled_C_ = log2(center_C ) ;
//////    double scaled_G_ = log2(center_G);


////    C_start = log2(center_C * min);
////    C_end   = log2(center_C * max);

////    if(C_end > 10)
////        C_end = 10;

////    G_start = log2(center_G * min) ;     // if it's negative, the larger value means smaller
////    G_end   = log2(center_G * max) ;

////    if(G_end > 1.5)
////        G_end = 1.5;
////    std::cout << "[MS][update_center] center_C:"<< center_C <<", center_G:" << center_G << std::endl;
////    std::cout << "[MS][update_center] C_start:"<< C_start <<", C_end:" << C_end << std::endl;
////    std::cout << "[MS][update_center] G_start:"<< G_start <<", G_end:" << G_end << std::endl;
//////    double Cstart = scaled_C_ - f_Ccross;
//////    if(Cstart <= 0 )
//////        Cstart = 1e-10;
//////    double Cend = best_C_1st + f_Ccross;
//////    double Gstart = best_G_1st - f_Gcross;
//////    if(Gstart <= 0 )
//////        Gstart = 1e-10;//
//////    double Gend = best_G_1st + f_Gcross;
//////    if(Gend >= 1 )
//////        Gend = 0.9999999999;
//}



void ModelSelection::set_range(){
    if(Config_params::getInstance()->get_ms_svm_id()==1){                   //normal SVM
        range_c.min = -6.64385619 ;
        range_c.max =  6.64385619 ;
    }else{                                                      //Weighted SVM (ms_svm_id == 2)
//        range_c.min = -6.64385619 ;
//        range_c.max =  10.5507468 ;
        range_c.min = -10 ;
        range_c.max =  10 ;

    }
//    range_g.min = -7.6439 ;
    range_g.min = -10 ;
//    range_g.max =  1.5850 ;
    range_g.max =  10 ;

//    // Assume no center is set, we set the center to (0,0) {notice that the pow2(0,0) is (1,1)}
//    point_center.C =1;  // log2(1) == 0, later inside the ud_param_generator, log2 applied to this first
//    point_center.G =1;

}




/* use UD table to determine the parameter values for model selection
 * @Input params:
 * svm_id :         type of SVM (svm_id=1: WSVM and SVM,     svm_id=2: WRSVM)
 * range_c :        range of C (minimum and maximum) (this are not the real values for C, these are log(c) )
 * range gamma :    range of Gamma OR range of epsilon if svm_id==1
 *                      In Regression, the searching range of episilon is equal to [0 , 0.5]
 * stage :          the stage of nested UDs (1 or 2)
 * pattern :        the UD pattern, 5-5runs,9-9runs,and 13-13runs
 * point center :   the UD center for current stage  (default= center of the searching box)
 * @Outputs :
 * Vector of ud_point  : the UD sampling points for current stage
 */
std::vector<ud_point> ModelSelection::ud_param_generator(int stage, bool inh_param, double param_C, double param_G){

    int UDTable[31][30][2] = {
        {},
        {},
        {},
        {{1, 3}, {2, 1}, {3, 2}}, //round3
        {{4, 3}, {1, 2}, {3, 1}, {2, 4}}, //round4
        {{1, 2}, {2, 5}, {4, 1}, {5, 4}, {3, 3}}, //round5
        {{5, 5}, {4, 1}, {2, 2}, {3, 6}, {1, 4}, {6, 3}}, //round6
        {{4, 4}, {3, 7}, {5, 1}, {2, 2}, {1, 5}, {7, 3}, {6, 6}}, //round7
        {{3, 4}, {5, 1}, {4, 8}, {8, 3}, {6, 5}, {1, 6}, {2, 2}, {7, 7}}, //round8
        {{5, 5}, {1, 4}, {7, 8}, {2, 7}, {3, 2}, {9, 6}, {8, 3}, {6, 1}, {4, 9}}, //round9
        {{2, 9}, {8, 10}, {10, 6}, {4, 7}, {1, 3}, {5, 1}, {9, 2}, {7, 4}, {6, 8}, {3, 5}}, //round10
        {{10, 10}, {9, 2}, {2, 9}, {6, 3}, {5, 11}, {1, 4}, {11, 5}, {8, 8}, {4, 1}, {7, 6}, {3, 7}}, //round11
        {{4, 8}, {10, 11}, {9, 5}, {8, 9}, {5, 4}, {3, 2}, {6, 12}, {7, 1}, {1, 6}, {11, 3}, {2, 10}, {12, 7}}, //round12
        {{5, 4}, {12, 3}, {2, 11}, {9, 10}, {7, 7}, {6, 13}, {3, 2}, {11, 12}, {13, 8}, {10, 5}, {1, 6}, {4, 9}, {8, 1}}, //round13
        {{5, 11}, {9, 14}, {14, 7}, {11, 9}, {7, 10}, {6, 1}, {12, 2}, {2, 3}, {8, 5}, {4, 6}, {3, 13}, {1, 8}, {13, 12}, {10, 4}}, //round14
        {{10, 1}, {15, 9}, {14, 3}, {9, 12}, {6, 15}, {2, 13}, {12, 6}, {13, 14}, {11, 11}, {5, 4}, {1, 7}, {8, 5}, {3, 2}, {4, 10}, {7, 8}}, //round15
        {{2, 3}, {9, 5}, {15, 14}, {1, 10}, {16, 7}, {6, 13}, {7, 1}, {13, 11}, {3, 15}, {10, 16}, {5, 8}, {14, 2}, {11, 4}, {8, 12}, {12, 9}, {4, 6}}, //round16
        {{10, 13}, {9, 9}, {5, 10}, {16, 3}, {13, 8}, {8, 5}, {15, 16}, {3, 2}, {7, 17}, {6, 4}, {1, 7}, {2, 15}, {17, 11}, {14, 6}, {11, 1}, {12, 14}, {4, 12}}, //round17
        {{1, 11}, {18, 8}, {12, 18}, {4, 7}, {6, 2}, {17, 16}, {7, 9}, {15, 13}, {11, 1}, {8, 15}, {3, 17}, {10, 12}, {16, 3}, {2, 4}, {14, 5}, {13, 10}, {9, 6}, {5, 14}}, //round18
        {{1, 9}, {5, 5}, {19, 6}, {15, 18}, {2, 16}, {18, 15}, {6, 19}, {8, 14}, {3, 3}, {7, 7}, {17, 11}, {16, 2}, {10, 10}, {14, 8}, {11, 17}, {13, 13}, {9, 1}, {4, 12}, {12, 4}}, //round19
        {{16, 15}, {18, 19}, {12, 1}, {19, 3}, {1, 9}, {10, 7}, {9, 20}, {4, 13}, {2, 18}, {14, 10}, {6, 16}, {15, 5}, {5, 6}, {20, 12}, {11, 14}, {13, 17}, {8, 4}, {7, 11}, {3, 2}, {17, 8}}, //round20
        {{17, 6}, {15, 12}, {16, 17}, {11, 11}, {3, 20}, {19, 2}, {6, 5}, {20, 19}, {4, 8}, {14, 4}, {8, 18}, {21, 9}, {13, 21}, {9, 1}, {2, 3}, {12, 7}, {18, 14}, {1, 13}, {7, 10}, {10, 15}, {5, 16}}, //round21
        {{13, 22}, {11, 15}, {15, 11}, {19, 8}, {1, 14}, {4, 10}, {9, 12}, {20, 20}, {12, 9}, {7, 5}, {21, 4}, {16, 18}, {3, 21}, {5, 7}, {18, 16}, {8, 19}, {17, 2}, {14, 6}, {22, 13}, {10, 1}, {2, 3}, {6, 17}}, //round22
        {{23, 10}, {7, 23}, {8, 14}, {13, 21}, {3, 2}, {14, 1}, {17, 19}, {21, 3}, {15, 15}, {4, 16}, {18, 5}, {2, 20}, {16, 8}, {9, 4}, {5, 12}, {11, 7}, {6, 6}, {10, 18}, {12, 11}, {20, 22}, {22, 17}, {1, 9}, {19, 13}}, //round23
        {{12, 9}, {20, 7}, {22, 3}, {8, 20}, {14, 1}, {16, 12}, {13, 16}, {5, 18}, {2, 4}, {11, 24}, {7, 2}, {6, 11}, {10, 6}, {24, 10}, {21, 17}, {9, 13}, {19, 14}, {1, 15}, {3, 22}, {18, 23}, {4, 8}, {15, 19}, {17, 5}, {23, 21}}, //round24
        {{13, 13}, {16, 17}, {7, 25}, {4, 2}, {5, 18}, {8, 4}, {14, 23}, {22, 24}, {18, 21}, {11, 20}, {1, 7}, {12, 6}, {20, 15}, {9, 16}, {3, 14}, {2, 22}, {25, 12}, {15, 1}, {6, 11}, {10, 9}, {17, 10}, {19, 5}, {24, 19}, {23, 3}, {21, 8}}, //round25
        {{23, 19}, {21, 16}, {1, 11}, {24, 2}, {26, 12}, {6, 25}, {20, 10}, {12, 24}, {11, 1}, {9, 20}, {25, 23}, {22, 7}, {13, 9}, {17, 14}, {5, 8}, {18, 4}, {15, 6}, {7, 15}, {19, 26}, {14, 18}, {8, 5}, {4, 17}, {10, 13}, {3, 3}, {16, 21}, {2, 22}}, //round26
        {{25, 2}, {26, 24}, {17, 9}, {22, 7}, {3, 3}, {24, 16}, {1, 12}, {18, 18}, {6, 26}, {21, 13}, {23, 20}, {5, 8}, {9, 21}, {2, 23}, {8, 5}, {12, 1}, {15, 6}, {16, 22}, {14, 14}, {27, 11}, {10, 10}, {13, 25}, {7, 15}, {19, 4}, {20, 27}, {4, 17}, {11, 19}}, //round27
        {{13, 13}, {20, 18}, {24, 8}, {16, 16}, {11, 4}, {5, 21}, {1, 17}, {18, 25}, {8, 24}, {9, 11}, {14, 7}, {27, 20}, {3, 26}, {6, 14}, {17, 1}, {12, 28}, {15, 22}, {28, 12}, {19, 10}, {22, 23}, {7, 6}, {21, 5}, {26, 3}, {10, 19}, {25, 27}, {23, 15}, {2, 9}, {4, 2}}, //round28
        {{1, 18}, {17, 1}, {26, 19}, {16, 24}, {27, 3}, {6, 2}, {21, 28}, {28, 26}, {20, 17}, {14, 7}, {11, 4}, {8, 25}, {29, 12}, {9, 9}, {22, 5}, {2, 6}, {18, 21}, {7, 14}, {3, 27}, {25, 8}, {19, 10}, {13, 29}, {4, 11}, {5, 22}, {23, 15}, {12, 16}, {15, 13}, {10, 20}, {24, 23}}, //round29
        {{24, 25}, {23, 6}, {1, 12}, {9, 18}, {5, 16}, {11, 7}, {19, 4}, {6, 9}, {21, 11}, {3, 3}, {12, 14}, {15, 20}, {20, 30}, {18, 15}, {17, 24}, {26, 2}, {7, 29}, {4, 21}, {28, 13}, {27, 28}, {25, 17}, {13, 27}, {14, 1}, {29, 8}, {22, 19}, {8, 5}, {2, 26}, {16, 10}, {30, 22}, {10, 23}} //round30
        };

    float lg_base = 2;
    int pattern;
    if(stage == 1){
        pattern = Config_params::getInstance()->get_ms_first_stage();
    }else{
        pattern = Config_params::getInstance()->get_ms_second_stage();
    }

    double p_center_c;
    double p_center_g;

    if(inh_param){
        p_center_c = log(param_C) / log(lg_base);       // the input is in real values, but we manipulate the log2 of them, until end
        p_center_g = log(param_G) / log(lg_base);
    }else{
        p_center_c = (range_c.min + range_c.max) / lg_base ;
        p_center_g = (range_g.min + range_g.max) / lg_base ;
    }

    double c_len = (range_c.max - range_c.min) ;                // length for C range
    double g_len = (range_g.max - range_g.min) ;                // length for gamma range

//    printf("[MS][UPG] DEBUG p_center_c:%.2f, p_center_g:%.2f\n",p_center_c, p_center_g);             //$$debug
//    printf("[MS][UPG] DEBUG c_len:%.2f, g_len:%.2f\n",c_len, g_len);             //$$debug


    std::vector<ud_point> params(pattern);
    double cen_c = p_center_c - (c_len /pow(lg_base,stage));
    double cen_g = p_center_g - (g_len /pow(lg_base,stage));
//    std::cout << " \n\n cen_c:" << cen_c  <<"\n";
    double pow_2_stage_minus_one = pow(lg_base,stage -1);
    double pattern_minus_one = (pattern -1);

    for(int i=0; i < pattern;i++){
//        params[i].C =  (UDTable[pattern][i][0] - 1) * c_len  / (pattern -1)  / pow_2_stage_minus_one + cen_c;
        params[i].C = (  ((UDTable[pattern][i][0] - 1) * c_len) / (pattern_minus_one * pow_2_stage_minus_one)  ) + cen_c;
        params[i].G = (  ((UDTable[pattern][i][1] - 1) * g_len) / (pattern_minus_one * pow_2_stage_minus_one)  ) + cen_g;
//        printf("[MS][UPG] DEBUG Before correction to edges: C: %.2f, G: %.2f\n", params[i].C, params[i].G);             //$$debug

    }

    for(int i=0; i < pattern;i++){              // make the output real
//        printf("[MS][UPG] LOG stage: %d, C:%.2f, G:%.4f\n",stage, params[i].C, params[i].G);        //$$debug
        params[i].C = pow(lg_base, params[i].C);
        params[i].G = pow(lg_base, params[i].G);
//        printf("[MS][UPG] stage: %d, C:%.2f, G:%.4f\n",stage, params[i].C, params[i].G);        //$$debug
    }


    float real_c_max = pow(lg_base, range_c.max );
    float real_c_min = pow(lg_base, range_c.min );
    float real_g_max = pow(lg_base, range_g.max );
    float real_g_min = pow(lg_base, range_g.min );
    long long random_seed = std::stoll(Config_params::getInstance()->get_cpp_srand_seed());

    for(int i=0; i < pattern;i++){
        srand(random_seed + i);
        if (params[i].C > real_c_max ){           //keep it inside the scope (set the outside value to max on that side)
            params[i].C = real_c_max + (rand() % 500) * (rand()%2? 1 :-1) ;
        }
        if(params[i].C < real_c_min ){
            params[i].C = real_c_min - (rand() % 100) * (rand()%2? 1 :-1);
            while(params[i].C < 0.001){
                params[i].C +=  (rand() % 500);
            }
        }

        if (params[i].G > real_g_max ){
            params[i].G = real_g_max + (rand() % 100 * 0.001) * (rand()%2? 1 :-1);
        }
        if(params[i].G < real_g_min ){
            params[i].G = real_g_min - (rand() % 100 * 0.00001) * (rand()%2? 1 :-1);
            while(params[i].G < 0.00001){
                params[i].G +=  (rand() % 100 * 0.00001);
            }
        }
    #if dbl_MS_ud_param_generator >= 3
        printf("[MS][UPG] stage: %d, C: %.2f, G: %.4f, After scale with random numbers \n",stage, params[i].C,params[i].G); //$$debug
    #endif
    }
    return params;
}








//======================================================================
int ModelSelection::select_best_model(std::vector<summary> v_summary, int level, int stage){

#if dbl_MS_SB1 >=5
    printf("\n[MS][BSM] Before gmean sort level:%d\n",level);
    for(auto it=v_summary.begin() ; it != v_summary.end(); ++it){
        Config_params::getInstance()->print_summary(*it, "[MS][BSM]",level);
    }
#endif

//    std::sort(v_summary.begin(),v_summary.end(),std::greater<summary>());
//    std::sort(v_summary.begin(),v_summary.end(),BetterGmean());   // the main method for a year before Sep 21, 2016 - 09:00
//    std::sort(v_summary.begin(),v_summary.end(),BetterSN_Gmean());  // experiment 092116_0940
    std::sort(v_summary.begin(),v_summary.end(),Better_Gmean_SN());  // experiment 092816_1600

//    std::sort(v_summary.begin(),v_summary.end(),sortByGmean);
//    std::sort(v_summary.begin(),v_summary.end(),BetterAcc());
//    std::sort(v_summary.begin(),v_summary.end(),BetterPPV());

#if dbl_MS_SB1 >=3
    printf("\n[MS][BSM] After sort at level:%d\n",level);
    for(auto it=v_summary.begin() ; it != v_summary.end(); ++it){
        Config_params::getInstance()->print_summary(*it, "[MS][BSM]",level);
    }
#endif
    // - - - - ignore the gmean zero - - - -
    for(unsigned int i =0; i != v_summary.size(); ++i){
//        std::cout << "v_summary[i].perf.at(Gmean): " << v_summary[i].perf.at(Gmean) << std::endl;
        if(v_summary[i].perf.at(Gmean) > 0.05)
            return v_summary[i].iter;

    }
    return v_summary[0].iter;   // in case there is model with gmean larger than zero, return the 1st one
}




void ModelSelection::uniform_design(Mat& m_data_p, Vec& v_vol_p, Mat& m_data_n, Vec& v_vol_n, bool inh_params,
                                    double param_C, double param_G, int level, solution & udc_sol){
    ETimer t_whole_UD;
    // - - - - - k-cross-fold data - - - - -
    k_fold kf;
    Mat m_train_data_p, m_train_data_n, m_test_data;
    Vec v_train_vol_p, v_train_vol_n;
    //0,5 means the first fold out of 5 fold is test and the rest are training
    int total_num_fold = Config_params::getInstance()->getInstance()->get_main_num_kf_iter();
    unsigned int num_iter_st1 = Config_params::getInstance()->get_ms_first_stage();
    unsigned int num_iter_st2 = Config_params::getInstance()->get_ms_second_stage();
    std::vector<summary> v_summary_folds;
    v_summary_folds.reserve(total_num_fold);

    for(int fold_id=0; fold_id < total_num_fold; ++fold_id){    // run the 2 stages on all k fold
        // cross fold the data and volumes for the fold_id
        kf.cross_validation_simple(m_data_p, m_data_n, v_vol_p, v_vol_n, fold_id, total_num_fold,
                                   m_train_data_p, m_train_data_n, m_test_data, v_train_vol_p, v_train_vol_n);

        /* DEBUG: export the matrices for further test and comparison
        CommonFuncs cf;
        cf.exp_matrix(m_train_data_p,   "./debug/" ,"m_train_data_p.dat",   "[MS][UD]");
        cf.exp_matrix(m_train_data_n,   "./debug/" ,"m_train_data_n.dat",   "[MS][UD]");
        cf.exp_vector(v_train_vol_p,    "./debug/" ,"v_train_vol_p.dat",    "[MS][UD]");
        cf.exp_vector(v_train_vol_n,    "./debug/" ,"v_train_vol_n.dat",    "[MS][UD]");
        cf.exp_matrix(m_test_data,      "./debug/" ,"m_test_data.dat",      "[MS][UD]");
        */

//        std::vector<Solver> v_solver;       // vector of solvers for current fold_id
//        v_solver.reserve(num_iter_st1 + num_iter_st2);
        unsigned int solver_id=0;
        std::vector<summary> v_summary;

        summary current_summary;     //map that contains all the measures
        int stage = 1;
        printf("[MS][UD] ------ stage:%d, level:%d, fold:%d ------ \n", stage, level, fold_id);

        std::vector<ud_point> ud_params_st_1;
        ud_params_st_1 = ud_param_generator(stage, inh_params, param_C, param_G);
        for(unsigned int i =0; i < num_iter_st1;++i){
            Solver sv;
            svm_model * curr_svm_model;
            curr_svm_model = sv.train_model(m_train_data_p, v_train_vol_p, m_train_data_n, v_train_vol_n, 1,
                                        ud_params_st_1[i].C, ud_params_st_1[i].G);
//            v_solver.push_back(sv);
            sv.test_predict(m_test_data, current_summary, solver_id);   //predict the validation data not the test data
            sv.free_solver("[MS][UD] ");   //free the solver
            v_summary.push_back(current_summary);
    #if dbl_MS_UD >= 1
            Config_params::getInstance()->print_summary(current_summary,"[MS][UD]", level, i, stage,fold_id);
    #endif
            ++solver_id;
        }
        int best_1st_stage = select_best_model(v_summary,level,1);
    #if dbl_MS_UD >= 3
        std::cout <<"[MS][UD] best_1st_stage:"<< best_1st_stage << " nSV+:" << v_summary[best_1st_stage].num_SV_p
                                                    <<", paramsC:"<< v_summary[best_1st_stage].C << std::endl;
    #endif
        stage = 2 ;
        printf("[MS][UD] ------ stage:%d, level:%d, fold:%d ------ \n", stage, level, fold_id);
        std::vector<ud_point> ud_params_st_2;
        printf("[MS][UD] 2nd stage model selection center C:%g, G:%g\n",ud_params_st_1[best_1st_stage].C , ud_params_st_1[best_1st_stage].G);
        ud_params_st_2 = ud_param_generator(stage,true, ud_params_st_1[best_1st_stage].C , ud_params_st_1[best_1st_stage].G);
        for(unsigned int i = 0; i < num_iter_st2 ;i++){
            //skip the center of second stage(duplicate)
            if(ud_params_st_2[i].C == ud_params_st_1[best_1st_stage].C && ud_params_st_2[i].G == ud_params_st_1[best_1st_stage].G)
                continue;
            Solver sv;
            svm_model * curr_svm_model;
            curr_svm_model = sv.train_model(m_train_data_p, v_train_vol_p, m_train_data_n, v_train_vol_n, 1,
                                        ud_params_st_2[i].C, ud_params_st_2[i].G);
            sv.test_predict(m_test_data, current_summary, solver_id);
//            v_solver.push_back(sv);
            sv.free_solver("[MS][UD] ");   //free the solver
            v_summary.push_back(current_summary);
    #if dbl_MS_UD >= 1
            Config_params::getInstance()->print_summary(current_summary,"[MS][UD]", level, i, stage,fold_id);
    #endif
            ++solver_id;
        }
        int best_of_all =  select_best_model(v_summary,level,2);
    //    printf("[MS][UD] best of all iter :%d\n", best_of_all);
        v_summary_folds.push_back( summary_factory_update_iter(v_summary[best_of_all], fold_id)  );
    #if dbl_MS_UD >= 1
        Config_params::getInstance()->print_summary(v_summary[best_of_all],"[MS][UD] Validation Data", level, -1, stage,fold_id);
    #endif

    } // end of for loop for fold_id


    /* - - - - - start experiment all the best values to find a better selection technique - - - - - */
    ETimer t_exp;
    printf("[MS][UD] expriment all the parameters for training and testing the test data(not validation data)\n");
    Loader test_loader;
    Mat untouched_test_data ;
    untouched_test_data = test_loader.load_norm_data_sep(Config_params::getInstance()->get_test_ds_f_name() );
    for(int fold_id=0; fold_id < total_num_fold; ++fold_id){
        // - - - - - train whole the data at this level - - - -
        Solver sv_whole_trd;
        svm_model * whole_trd_model;
        whole_trd_model = sv_whole_trd.train_model(m_data_p, v_vol_p, m_data_n, v_vol_n, 1,
                                            v_summary_folds[fold_id].C,v_summary_folds[fold_id].gamma);
        summary exp_summary;
        sv_whole_trd.evaluate_testdata(untouched_test_data, 4444000+level, exp_summary);
        Config_params::getInstance()->print_summary(exp_summary,"[MS][UD]TD [experiment only] ", level);
    }
    MatDestroy(&untouched_test_data);
    t_exp.stop_timer("[MS][UD][exp] evaluate the performance of all the parameters on the testdata at level", std::to_string(level) );
    /* - - - - - end   experiment all the best values to find a better selection technique - - - - - */

    // - - - - - select best of all folds - - - -
    int best_of_all_kfold = select_best_model(v_summary_folds, level,0);    // for now I pass stage 0 as the final stage for the k-fold
    printf("[MS][UD] best of all kfolds is:%d\n", best_of_all_kfold);

    // - - - - - train whole the data at this level - - - -
    Solver sv_whole_training_data;
    svm_model * whole_training_data_model;
    whole_training_data_model = sv_whole_training_data.train_model(m_data_p, v_vol_p, m_data_n, v_vol_n, 1,
                                        v_summary_folds[best_of_all_kfold].C,v_summary_folds[best_of_all_kfold].gamma);
    PetscInt num_point_p;
    MatGetSize(m_data_p, &num_point_p, NULL);
    sv_whole_training_data.prepare_solution_single_model(whole_training_data_model, num_point_p, udc_sol);

    summary final_summary;
    sv_whole_training_data.evaluate_testdata(level, final_summary);

    #if dbl_MS_UD >= 1
        final_summary.iter = -1;       //not print the iteration is summary
        Config_params::getInstance()->print_summary(final_summary,"[MS][UD]final TD ", level);
    #endif
    t_whole_UD.stop_timer("[MS][UD] whole UD including stage1,2 and preparing the solution at level",std::to_string(level));
}




void ModelSelection::add_debug_parameters(std::vector<ud_point>& v_initialized_params){
    ud_point extra_parameters;
    extra_parameters.C= Config_params::getInstance()->get_svm_C();
    extra_parameters.G= Config_params::getInstance()->get_svm_gamma();
    v_initialized_params.push_back(extra_parameters);

}


void ModelSelection::uniform_design_separate_validation(Mat& m_train_data_p, Vec& v_train_vol_p, Mat& m_train_data_n, Vec& v_train_vol_n,
                                                        bool inh_params, double param_C, double param_G, Mat& m_VD_p, Mat& m_VD_n, int level,
                                                        solution & udc_sol, std::vector<ref_results>& v_ref_results){
    // - - - -  Load validation data which is the training part of whole data in the beginning of the coarsening - - - -
    ETimer t_whole_UD;
    Loader ld;
    // - - - - set number of iterations in each stage - - - - -
//    unsigned int num_iter_st1 = Config_params::getInstance()->get_ms_first_stage();
    unsigned int num_iter_st1 = Config_params::getInstance()->get_ms_first_stage()  + 1; //+1 for one extra parameter to test
    unsigned int num_iter_st2 = Config_params::getInstance()->get_ms_second_stage();
    // - - - - define required variables - - - - -
    std::vector<Solver> v_solver;       // vector of solvers for current fold_id
    v_solver.reserve(num_iter_st1 + num_iter_st2);
    unsigned int solver_id=0;
    std::vector<summary> v_summary;
    summary current_summary;     //map that contains all the measures

    ETimer t_stage1;
    int stage = 1;
    printf("[MS][UDSepVal] ------ stage:%d, level:%d------ \n", stage, level);
    std::vector<ud_point> ud_params_st_1;
    ud_params_st_1 = ud_param_generator(stage, inh_params, param_C, param_G);
    add_debug_parameters(ud_params_st_1);
    for(unsigned int i =0; i < num_iter_st1;++i){
        Solver sv;
        svm_model * curr_svm_model;
        curr_svm_model = sv.train_model(m_train_data_p, v_train_vol_p, m_train_data_n, v_train_vol_n, 1,
                                    ud_params_st_1[i].C, ud_params_st_1[i].G);
        v_solver.push_back(sv);
        //predict the validation data not the test data
        sv.predict_validation_data(m_VD_p, m_VD_n, current_summary, solver_id);
//            sv.free_solver("[MS][UDSepVal] ");   //free the solver
        v_summary.push_back(current_summary);
#if dbl_MS_UDSepVal >= 1
        Config_params::getInstance()->print_summary(current_summary,"[MS][UDSepVal]", level, i, stage);
#endif
        ++solver_id;
    }
    int best_1st_stage = select_best_model(v_summary,level,1);
#if dbl_MS_UDSepVal >= 3
    std::cout <<"[MS][UDSepVal] best_1st_stage:"<< best_1st_stage << " nSV+:" << v_summary[best_1st_stage].num_SV_p
                                                <<", paramsC:"<< v_summary[best_1st_stage].C << std::endl;
#endif
    t_stage1.stop_timer("[MS][UDSepVal] stage 1 at level", std::to_string(level) );

    ETimer t_stage2;
    stage = 2 ;
    printf("[MS][UDSepVal] ------ stage:%d, level:%d------ \n", stage, level);
    std::vector<ud_point> ud_params_st_2;
    printf("[MS][UDSepVal] 2nd stage model selection center C:%g, G:%g\n",ud_params_st_1[best_1st_stage].C , ud_params_st_1[best_1st_stage].G);
    ud_params_st_2 = ud_param_generator(stage,true, ud_params_st_1[best_1st_stage].C , ud_params_st_1[best_1st_stage].G);
    for(unsigned int i = 0; i < num_iter_st2 ;i++){
        //skip the center of second stage(duplicate)
        if(ud_params_st_2[i].C == ud_params_st_1[best_1st_stage].C && ud_params_st_2[i].G == ud_params_st_1[best_1st_stage].G)
            continue;
        Solver sv;
        svm_model * curr_svm_model;
        curr_svm_model = sv.train_model(m_train_data_p, v_train_vol_p, m_train_data_n, v_train_vol_n, 1,
                                    ud_params_st_2[i].C, ud_params_st_2[i].G);
        sv.predict_validation_data(m_VD_p, m_VD_n, current_summary, solver_id);
        v_solver.push_back(sv);
//            sv.free_solver("[MS][UDSepVal] ");   //free the solver
        v_summary.push_back(current_summary);
#if dbl_MS_UDSepVal >= 1
        Config_params::getInstance()->print_summary(current_summary,"[MS][UDSepVal]", level, i, stage);
#endif
        ++solver_id;
    }
    int best_of_all =  select_best_model(v_summary,level,2);
    printf("[MS][UDSepVal] best of both stage of UD is: (iter :%d)\n", best_of_all);
#if dbl_MS_UDSepVal >= 1
    Config_params::getInstance()->print_summary(v_summary[best_of_all],"[MS][UDSepVal] Validation Data", level, -1, stage);
#endif
    t_stage2.stop_timer("[MS][UDSepVal] stage 2 at level", std::to_string(level) );

    // - - - - load the test data from file - - - -
    Mat m_TD;
    m_TD = ld.load_norm_data_sep(Config_params::getInstance()->get_test_ds_f_name() );

    /* - - - - - start experiment all the best values to find a better selection technique - - - - - */
//    ETimer t_exp;
//    printf("[MS][UDSepVal] expriment all the solvers of model selection stage on the test data(check the quality of the sort we use)\n");
//    for(auto it= v_solver.begin(); it!= v_solver.end(); ++it){
//        it->test_predict(m_TD, current_summary);
//        Config_params::getInstance()->print_summary(current_summary,"[MS][UDSepVal]TD [experiment only] ", 4444000+level);
//    }
//    t_exp.stop_timer("[MS][UDSepVal][exp] evaluate the performance of all the parameters on the testdata at level", std::to_string(level) );
    /* - - - - - end   experiment all the best values to find a better selection technique - - - - - */

    // - - - - - prepare solution for finer level - - - -
    ETimer t_prep_sol;
    PetscInt num_point_p;
    MatGetSize(m_train_data_p, &num_point_p, NULL);
    Solver best_solver = v_solver[best_of_all];

    svm_model * best_model = best_solver.get_model() ;
    best_solver.prepare_solution_single_model(best_model , num_point_p, udc_sol);
    summary final_summary;
    best_solver.evaluate_testdata(level, final_summary);
    ref_results refinement_results ;

    refinement_results.validation_data_summary = v_summary[best_of_all];
    refinement_results.test_data_summary = final_summary;
    refinement_results.level = level;
    v_ref_results.push_back(refinement_results);           //collect the final best model at each level
//    Config_params::getInstance()->print_ref_result(v_ref_results);
//    exit(1);
    #if dbl_MS_UDSepVal >= 1
        final_summary.iter = -1;       //not print the iteration is summary
        Config_params::getInstance()->print_summary(final_summary,"[MS][UDSepVal] final TD ", level);
    #endif
    
        
    // - - - - - free resources - - - - -
    for(auto it=v_solver.begin(); it!= v_solver.end(); ++it){
        it->free_solver("[MS][UDSepVal] ");   //free all solvers
    }
    MatDestroy(&m_TD);
//    MatDestroy(&m_VD_p);
//    MatDestroy(&m_VD_n);
    t_prep_sol.stop_timer("[MS][UDSepVal] Prepare solution at level", std::to_string(level) );
    t_whole_UD.stop_timer("[MS][UDSepVal] whole UDSepVal including stage1,2 and preparing the solution at level",std::to_string(level));
}





summary ModelSelection::summary_factory_update_iter(const summary& in_summary, const int iter){
    summary summary_result;
    summary_result = in_summary;
//    summary_result.C = in_summary.C;
//    summary_result.C = in_summary.C;

    summary_result.iter = iter;
    printf("[MS][SPUI] iter %d created successfully!\n", iter);
    return summary_result;
}










/*
 *
 *
 *  sum_all_vol_p, sum_all_vol_n is used to set the instance weights for each point in training
 */



void ModelSelection::uniform_design_index_base(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n, bool inh_params, double last_c,
                        double last_gamma,int level, std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
                        std::unordered_set<PetscInt>& uset_SV_index_p, std::unordered_set<PetscInt>& uset_SV_index_n,
                        Mat& m_testdata, int classifier_id, Mat& m_all_predict){
    ETimer t_sv_ps;
    srand(std::stoll(Config_params::getInstance()->get_cpp_srand_seed()));
    std::random_shuffle( v_p_index.begin(), v_p_index.end() ); //shuffle all nodes
    srand(std::stoll(Config_params::getInstance()->get_cpp_srand_seed()));
    std::random_shuffle( v_n_index.begin(), v_n_index.end() ); //shuffle all nodes

    double train_fraction = 1 - (1 / Config_params::getInstance()->get_main_num_kf_iter());

    PetscInt iter_train_p_end = ceil(v_p_index.size()* train_fraction);    //number of poitns in training data
    PetscInt iter_train_n_end = ceil(v_n_index.size()* train_fraction);

#if dbl_MS_UDIB >= 1
    std::cout << "[MS][UDIB] p index size:" << v_p_index.size() << ", iter_train_p_end:" << iter_train_p_end << std::endl;
    std::cout << "[MS][UDIB] n index size:" << v_n_index.size() << ", iter_train_n_end:" << iter_train_n_end << std::endl;
#endif

    unsigned int num_iter_st1 = Config_params::getInstance()->get_ms_first_stage();
    unsigned int num_iter_st2 = Config_params::getInstance()->get_ms_second_stage();
    std::vector<Solver> v_solver;
    v_solver.reserve(num_iter_st1 + num_iter_st2);
    unsigned int solver_id=0;

    int stage = 1;
    std::vector<ud_point> ud_params_st_1;
    summary current_summary;
    std::vector<summary> v_summary;
    ud_params_st_1 = ud_param_generator(1, inh_params, last_c, last_gamma);
    // - - - - 1st stage - - - -
    for(unsigned int i =0; i < num_iter_st1;i++){
        Solver sv;
        svm_model * curr_svm_model;
        curr_svm_model = sv.train_model_index_base(p_data, v_vol_p, n_data, v_vol_n, v_p_index, v_n_index,
                                        iter_train_p_end, iter_train_n_end,true, ud_params_st_1[i].C, ud_params_st_1[i].G);
        v_solver.push_back(sv);
//        sv.test_predict_index_base(p_data, n_data, v_p_index, v_n_index, iter_train_p_end, iter_train_n_end, current_summary,solver_id);

        v_summary.push_back(current_summary);

//        sv.free_solver("[MS][UDIB] stage 1");
        ++solver_id;
    }
    int best_1st_stage = select_best_model(v_summary,level,1);


    // - - - - 2nd stage - - - -
    stage = 2 ;
    std::vector<ud_point> ud_params_st_2;
    ud_params_st_2 = ud_param_generator(2,true, ud_params_st_1[best_1st_stage].C , ud_params_st_1[best_1st_stage].G);

    for(unsigned int i =0; i < num_iter_st2; i++){
        //skip the center of second stage(duplicate)
        if(ud_params_st_2[i].C == ud_params_st_1[best_1st_stage].C && ud_params_st_2[i].G == ud_params_st_1[best_1st_stage].G)
            continue;

        Solver sv;
        svm_model * curr_svm_model;
        curr_svm_model = sv.train_model_index_base(p_data, v_vol_p, n_data, v_vol_n, v_p_index, v_n_index,
                                        iter_train_p_end, iter_train_n_end,true, ud_params_st_2[i].C, ud_params_st_2[i].G);
        sv.test_predict_index_base(p_data, n_data, v_p_index, v_n_index, iter_train_p_end, iter_train_n_end, current_summary,solver_id);
        v_solver.push_back(sv);
        v_summary.push_back(current_summary);
        ++solver_id;
    }

    int best_of_all =  select_best_model(v_summary,level,2);
    t_sv_ps.stop_timer("[MS][UDIB] model training");

    // - - - - - - - - prepare the solution for refinement - - - - - - - - -
    Solver best_sv = v_solver[best_of_all];
    svm_model * best_model = best_sv.get_model() ;
    if(level > 1 ){     // at the finest level, we need to save the model (SV, C, gamma) for unseen points
        // ----- create the index of SVs in data points for each class seperately ----
        PetscInt i;
        for (i=0; i < best_model->nSV[0];i++){
            uset_SV_index_p.insert(v_p_index[best_model->sv_indices[i] - 1]);
        }
        for (int i=0; i < best_model->nSV[1];i++){
            uset_SV_index_n.insert(v_n_index[ best_model->sv_indices[best_model->nSV[0] + i] - 1 - iter_train_p_end]);
        }
    }else{
        if(Config_params::getInstance()->get_ms_save_final_model()){
            printf("\n\n\n[MS][UDIB] at the finest level the partial models needs to be saved!!! Exit is cancelled to continue the runs! \n\n\n");
//            exit(1);
        }
        //TODO: multiple hyperplain
    }

    best_sv.predict_test_data_in_matrix_output(m_testdata, classifier_id, m_all_predict);

    for(auto it=v_solver.begin(); it!= v_solver.end(); ++it){
        it->free_solver("[MS][UDIB] ");   //free all solvers
    }

    // no evaluation on test data //#TODO
}




/*
 *
 *
 *  sum_all_vol_p, sum_all_vol_n is used to set the instance weights for each point in training
 */
void ModelSelection::uniform_design_index_base_separate_validation(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n,
                        bool inh_params, double last_c, double last_gamma,int level,
                        std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
                        std::unordered_set<PetscInt>& uset_SV_index_p, std::unordered_set<PetscInt>& uset_SV_index_n,
                        Mat& m_VD_p, Mat& m_VD_n, Mat& m_VD_both, Mat& m_all_predict_VD, Mat& m_testdata, int classifier_id, Mat& m_all_predict_TD){
    ETimer t_sv_ps;
    srand(std::stoll(Config_params::getInstance()->get_cpp_srand_seed()));
    std::random_shuffle( v_p_index.begin(), v_p_index.end() ); //shuffle all nodes
    srand(std::stoll(Config_params::getInstance()->get_cpp_srand_seed()));
    std::random_shuffle( v_n_index.begin(), v_n_index.end() ); //shuffle all nodes

//    double train_fraction = 1 - (1 / Config_params::getInstance()->get_main_num_kf_iter());

    PetscInt iter_train_p_end = v_p_index.size();    //number of poitns in training data
    PetscInt iter_train_n_end = v_n_index.size();

#if dbl_MS_UDIB >= 0
    std::cout << "[MS][UDIBSepVal] p index size:" << v_p_index.size() << ", n index size:" << v_n_index.size() << std::endl;
#endif

    unsigned int num_iter_st1 = Config_params::getInstance()->get_ms_first_stage();
    unsigned int num_iter_st2 = Config_params::getInstance()->get_ms_second_stage();
    std::vector<Solver> v_solver;
    v_solver.reserve(num_iter_st1 + num_iter_st2);
    unsigned int solver_id=0;

    int stage = 1;
    std::vector<ud_point> ud_params_st_1;
    summary current_summary;
    std::vector<summary> v_summary;
    ud_params_st_1 = ud_param_generator(1, inh_params, last_c, last_gamma);
    // - - - - 1st stage - - - -
    for(unsigned int i =0; i < num_iter_st1;i++){
        Solver sv;
        svm_model * curr_svm_model;
        curr_svm_model = sv.train_model_index_base(p_data, v_vol_p, n_data, v_vol_n, v_p_index, v_n_index,
                                        iter_train_p_end, iter_train_n_end,true, ud_params_st_1[i].C, ud_params_st_1[i].G);
        v_solver.push_back(sv);
//        sv.test_predict_index_base(p_data, n_data, v_p_index, v_n_index, iter_train_p_end, iter_train_n_end, current_summary,solver_id);
//        std::cout << "[MS][UDIBSepVal] after push_back solver" << std::endl;
        sv.predict_validation_data(m_VD_p, m_VD_n, current_summary, solver_id);     // The normal predict method for full matrix is useful rather than index base methods
//        std::cout << "[MS][UDIBSepVal] after predicting the validation data" << std::endl;
        v_summary.push_back(current_summary);
        ++solver_id;
//        exit(1);
    }
    int best_1st_stage = select_best_model(v_summary,level,1);


    // - - - - 2nd stage - - - -
    stage = 2 ;
    std::vector<ud_point> ud_params_st_2;
    ud_params_st_2 = ud_param_generator(2,true, ud_params_st_1[best_1st_stage].C , ud_params_st_1[best_1st_stage].G);

    for(unsigned int i =0; i < num_iter_st2; i++){
        //skip the center of second stage(duplicate)
        if(ud_params_st_2[i].C == ud_params_st_1[best_1st_stage].C && ud_params_st_2[i].G == ud_params_st_1[best_1st_stage].G)
            continue;

        Solver sv;
        svm_model * curr_svm_model;
        curr_svm_model = sv.train_model_index_base(p_data, v_vol_p, n_data, v_vol_n, v_p_index, v_n_index,
                                        iter_train_p_end, iter_train_n_end,true, ud_params_st_2[i].C, ud_params_st_2[i].G);
//        sv.test_predict_index_base(p_data, n_data, v_p_index, v_n_index, iter_train_p_end, iter_train_n_end, current_summary,solver_id);
        sv.predict_validation_data(m_VD_p, m_VD_n, current_summary, solver_id);     // The normal predict method for full matrix is useful rather than index base methods
        v_solver.push_back(sv);
        v_summary.push_back(current_summary);
        ++solver_id;
    }

    int best_of_all =  select_best_model(v_summary,level,2);
    t_sv_ps.stop_timer("[MS][UDIBSepVal] model training");

    // - - - - - - - - prepare the solution for refinement - - - - - - - - -
    Solver best_sv = v_solver[best_of_all];
    svm_model * best_model = best_sv.get_model() ;
    if(level > 1 ){     // at the finest level, we need to save the model (SV, C, gamma) for unseen points
        // ----- create the index of SVs in data points for each class seperately ----
        PetscInt i;
        for (i=0; i < best_model->nSV[0];i++){
            uset_SV_index_p.insert(v_p_index[best_model->sv_indices[i] - 1]);
        }
        for (int i=0; i < best_model->nSV[1];i++){
            uset_SV_index_n.insert(v_n_index[ best_model->sv_indices[best_model->nSV[0] + i] - 1 - iter_train_p_end]);
        }
    }else{
        if(Config_params::getInstance()->get_ms_save_final_model()){
            printf("\n\n\n[MS][UDIBSepVal] at the finest level the partial models needs to be saved!!! Exit is cancelled to continue the runs! \n\n\n");
//            exit(1);
        }
        //TODO: multiple hyperplain
    }


    //@@ calculating the validation data for all the partition groups are missed, I need to save the results for picking the best level //#TODO 021317-1750
//    best_sv.predict_VD_in_output_matrix(m_VD_p, m_VD_n, classifier_id, m_all_predict_VD);  //added 021517-1328
    best_sv.predict_test_data_in_matrix_output(m_VD_both, classifier_id, m_all_predict_VD);  //added 021517-1456

    best_sv.predict_test_data_in_matrix_output(m_testdata, classifier_id, m_all_predict_TD);

    for(auto it=v_solver.begin(); it!= v_solver.end(); ++it){
        it->free_solver("[MS][UDIBSepVal] ");   //free all solvers
    }

    // no evaluation on test data //#TODO
}

