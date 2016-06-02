#ifndef MODEL_SELECTION_H
#define MODEL_SELECTION_H

#include "config_params.h"
#include <petscmat.h>
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include "svm.h"
#include <unordered_set>

#define Malloc(type,n) (type *)malloc((n)*sizeof(type)) //from svm_train
//#define log 3

struct ms_point{
    double x;
    double y;
    double z;
};
struct ms_range{
    double min;
    double max;
};
struct ud_point{
    double C;
    double G;
};

struct solution{
    std::vector<int> p_index;
    std::vector<int> n_index;
    double C, gamma;
    bool carry_SV;
    std::vector<int> p_SV_ind;
    std::vector<int> n_SV_ind;
    Mat     m_p_SV;         // Fake points data
    Mat     m_n_SV;         // Fake points data
};

struct summary{
    int iter;
    double gmean;
    int num_SV_p;
    int num_SV_n;
    int num_SV_t;
    double C;
    double gamma;
    bool operator > (const summary new_) const
        {
//            return (gmean > new_.gmean);
            if ( (gmean - new_.gmean) > 0.001 ) {
                return 1;
            }else{
                return 0;
            }
        }
};


class ModelSelection{
private :
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

        struct svm_parameter param;		// set by parse_command_line
        struct svm_problem prob;		// set by read_problem
        struct svm_model *model;
        struct svm_node *x_space;
        int p_num_node_=0, n_num_node_=0, p_num_elem_=0,n_num_elem_=0;
        int t_num_node_=0, t_num_elem_=0;
        int test_num_node_=0, test_num_elem_=0;

        ms_range range_c;     // range of C
        ms_range range_gamma;     // range of gamma
        ms_point point_center;      // center point

        int max_nr_attr = 64;



        //predict parameters
        int predict_probability=0;
//        static int (*info)(const char *fmt,...) = &printf;
//        Mat p_data_, n_data_, untouched_test_data_; //whole data
        const char * test_dataset_f_name;

public:
        ModelSelection(){
            test_dataset_f_name = Config_params::getInstance()->get_test_ds_f_name().c_str();
            predict_probability = Config_params::getInstance()->get_svm_probability();
        }

        void free_model_selection(std::string caller_name);    //work as deconstructor for me


        void partial_solver(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n, double last_c, double last_gamma,
                                        int level, std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index,
                                        std::unordered_set<PetscInt>& uset_SV_index_p, std::unordered_set<PetscInt>& uset_SV_index_n);

        void set_range();
        void set_center(bool inherit_last_params, double last_c, double last_gamma);
        solution UD(Mat& p_data, Vec& v_vol_p, Mat& n_data, Vec& v_vol_n,
                                    int inherit_last_params, double last_c, double last_gamma, int level);

        void set_weights_num_points(svm_parameter& param_, PetscInt num_p_point, PetscInt num_n_point);

        void alloc_memory_for_weights(svm_parameter& in_param, bool free_first);

        std::vector<ud_point> ud_param_generator(ms_range range_c,ms_range range_g,
                                                 int stage, int pattern, ms_point p_center);

        void cross_fold_data(const Mat data_p, const Mat data_n, Mat& train_, Mat& train_n_, Mat& test_total_);

        void combine_test_data(Mat& test_total, Mat& dt_test_p, PetscInt size_p, Mat& dt_test_n, PetscInt size_n, PetscInt max_num_col_);

        void read_problem_index_base(Mat& p_train_data, Mat& n_train_data,
                                                     std::vector<PetscInt>& v_p_index, std::vector<PetscInt>& v_n_index);
//        void read_problem(Mat& p_train_data, Mat& n_train_data, Mat& p_SVs, Mat& n_SVs);
        void read_problem(Mat& p_train_data, Mat& n_train_data);

        void read_parameters();

        void predict_label(Mat& test_data, int target_row, Mat& m_predicted_label);

        std::map<measures,double> test_predict(Mat& );

        void grid_explore_rsvm(Mat , Mat );

        svm_model * get_model(){return model;}

        summary make_summary(int iter, double Gmean, int num_SV_p, int num_SV_n, double C, double gamma);

        int select_best_1st(std::vector<summary>& map_summary, int level);

        int select_best_both_stage(std::vector<summary>& map_summary_2, int level);

//        int calc_sum_vol(Vec& v_vol, std::vector<int>& v_index, int limit );
//        int calc_sum_vol(Vec& v_vol, std::vector<int>& v_index);
};
#endif // MODEL_SELECTION_H
