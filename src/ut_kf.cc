#include "ut_kf.h"
#include "k_fold.h"


void UT_KF::cross_validation_simple(){
//    Mat& m_data_p, Mat& m_data_n, int current_iteration,int total_iterations,Mat& m_train_data_p, Mat& m_train_data_n, Mat& m_test_data
    Mat m_dt_p, m_dt_n, m_tr_p, m_tr_n, m_td;
    Vec v_vol_p, v_vol_n, v_train_vol_p, v_train_vol_n;
    UT_Common utc;
    utc.load_matrix("./datasets/ut_kfold_20_zsc_data.dat", m_dt_p,0);
    utc.load_matrix("./datasets/ut_kfold_40_zsc_data.dat", m_dt_n,0);
    utc.load_vec("./datasets/v_vol_100.dat", v_vol_p,0);
    utc.load_vec("./datasets/v_vol_100.dat", v_vol_n,0);
    std::cout << "[UT_KF][CVS] input matrices are read!" << std::endl;
    k_fold kf;
    kf.cross_validation_simple(m_dt_p, m_dt_n, v_vol_p, v_vol_n, 0    , 5, m_tr_p, m_tr_n, m_td,v_train_vol_p,v_train_vol_n);

}

void UT_KF::cross_validation(){

    k_fold kf;

    kf.read_in_data();
    kf.divide_data();
    kf.shuffle_data();
    kf.cross_validation(1, 5);

}


//void UT_KF::cross_validation_class(){
//    Mat m_dt_p, m_dt_n, m_tr_p, m_tr_n, m_td;
//    Vec v_vol_p, v_vol_n, v_train_vol_p, v_train_vol_n;
//    UT_Common utc;
//    utc.load_matrix("./datasets/ut_kfold_20_zsc_data.dat", m_dt_p,0);
////    utc.load_matrix("./datasets/ut_kfold_40_zsc_data.dat", m_dt_n,0);
//    utc.load_vec("./datasets/v_vol_100.dat", v_vol_p,0);
////    utc.load_vec("./datasets/v_vol_100.dat", v_vol_n,0);
//    std::cout << "[UT_KF][CVS] input matrices are read!" << std::endl;

//    std::unordered_set<PetscInt>& uset_test_idx;
//    std::string info = "Unit_Test";
//    std::vector<PetscInt> v_shuffled_indices;

//    k_fold kf;
//    kf.cross_validation_class(0,5,m_dt_p,m_tr_p,m_td,);

//}
