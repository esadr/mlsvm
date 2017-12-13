#include "k_fold.h"
#include "loader.h"
#include <algorithm>    /* random_shuffle*/
#include <cmath>
#include "config_logs.h"
#include "etimer.h"

//#include "model_selection.h"


/*
 * read the input data from file and load it into m_in_data_ matrix
 * read the input label from file and load it into v_in_label_ vector
 * the ds_path and ds_name are read from config_params
 * the _data.dat, _label.dat are used to postfix to the ds_name for data and label files
 */
void k_fold::read_in_data(std::string input_train_data, std::string input_train_label){
    Loader ld;
    PetscInt data_size=0, label_size=0;

    if(input_train_data.empty())
        input_train_data = Config_params::getInstance()->get_ds_path()+ "/" +
                        Config_params::getInstance()->get_ds_name()+ "_zsc_data.dat";

    if(input_train_label.empty())
        input_train_label = Config_params::getInstance()->get_ds_path()+ "/" +
                        Config_params::getInstance()->get_ds_name()+ "_label.dat";


    this->m_in_data_ = ld.read_input_matrix(input_train_data);
    this->v_in_label_ = ld.read_input_vector(input_train_label);


    MatGetSize(this->m_in_data_, &data_size, NULL);
    VecGetSize(this->v_in_label_, &label_size);
    //check the size of data and label
    if(data_size == label_size)
        this->num_data_points_ = data_size;
    else{
        std::cout << "[k_fold] data size is:<<"<< data_size <<" label size is:"<< label_size << std::endl;
        PetscPrintf(PETSC_COMM_WORLD,"[KF][read_in_data] the data and label size are not match, Exit!\n");
        exit(1);
    }
}


void k_fold::read_in_divided_data(Mat& m_min_data, Mat& m_maj_data){
    ETimer t_all;

    std::string prefix = Config_params::getInstance()->get_ds_path() + "/" + Config_params::getInstance()->get_ds_name();
//    std::cout << "[k_fold] prefix: " << prefix << std::endl;
    std::string min_full_data {prefix + "_min_norm_data.dat"}; // add the dataset name
    std::string maj_full_data {prefix + "_maj_norm_data.dat"};
#if dbl_KF_rdd >= 1
    std::cout << "[k_fold] min_full_data:<<"<< min_full_data << std::endl;
    std::cout << "[k_fold] maj_full_data:<<"<< maj_full_data << std::endl;
#endif

    Loader ld;
    m_min_data = ld.read_input_matrix(min_full_data);
    m_maj_data = ld.read_input_matrix(maj_full_data);
    MatGetSize(m_min_data, &this->num_min_points, NULL);
    MatGetSize(m_maj_data, &this->num_maj_points, NULL);
    t_all.stop_timer("[k_fold] read_in_divided_data");
}


void k_fold::read_in_full_NN(Mat& m_min_NN_indices,Mat& m_min_NN_dists,Mat& m_maj_NN_indices,Mat& m_maj_NN_dists){
    ETimer t_all;

    std::string prefix = Config_params::getInstance()->get_ds_path() + "/" + Config_params::getInstance()->get_ds_name();
//    std::cout << "[k_fold][Read_Full_NN] prefix: " << prefix << std::endl;
    std::string min_NN_indices {prefix + "_min_norm_data_indices.dat"};
    std::string min_NN_dists {prefix + "_min_norm_data_dists.dat"};
    std::string maj_NN_indices {prefix + "_maj_norm_data_indices.dat"};
    std::string maj_NN_dists {prefix + "_maj_norm_data_dists.dat"};
#if dbl_KF_rfn >= 1
    std::cout << "[k_fold] min_NN_indices:<<"<< min_NN_indices << std::endl;
    std::cout << "[k_fold] min_NN_dists:<<"<< min_NN_dists << std::endl;
    std::cout << "[k_fold] maj_NN_indices:<<"<< maj_NN_indices << std::endl;
    std::cout << "[k_fold] maj_NN_dists:<<"<< maj_NN_dists << std::endl;
#endif
    Loader ld;
    m_min_NN_indices = ld.read_input_matrix(min_NN_indices);
    m_min_NN_dists = ld.read_input_matrix(min_NN_dists);
    m_maj_NN_indices = ld.read_input_matrix(maj_NN_indices);
    m_maj_NN_dists = ld.read_input_matrix(maj_NN_dists);
    t_all.stop_timer("[k_fold] read_in_full_NN");
}


/*
 * Only devide the input data to 2 separate classes
 * I don't need the original data and labels after this
 * so, I release them and keep m_min_data_, m_maj_data_
 *
 * The label for minority or positive class is +1
 * and for majority or negative class is -1
 */
void k_fold::divide_data(bool export_file){
    IS              is_min_, is_maj_;
    PetscInt        i, min_cnt_=0, maj_cnt_=0;
    PetscInt        * ind_min_, * ind_maj_;
    PetscScalar     *arr_labels_;

    //reserve enough space in memory for both classes as I don't know the size of each yet
    PetscMalloc1(this->num_data_points_, &arr_labels_);
    PetscMalloc1(this->num_data_points_, &ind_min_);
    PetscMalloc1(this->num_data_points_, &ind_maj_);

    VecGetArray(this->v_in_label_,&arr_labels_);
    for(i = 0; i< this->num_data_points_ ;i++){
        if(arr_labels_[i] == 1){
            ind_min_[min_cnt_] = i;     // if the label at for data point in row i is 1, add its index(i) to minority indices
            min_cnt_++;                 // shift minority current position forward
        }else{
            ind_maj_[maj_cnt_] = i;     // majority label is -1
            maj_cnt_++;                 // shift majority current position forward
        }
    }
    if(maj_cnt_ < (min_cnt_ * 0.9)){
        PetscPrintf(PETSC_COMM_WORLD, "[k_fold][divide_data] labels are wrong the majority size is %d which is less than minority size :%d\n",
                    maj_cnt_,min_cnt_);     //logic of coarsening
        exit(1);
    }
    VecRestoreArray(this->v_in_label_,&arr_labels_);

    /*
     * - - - - - Select minority class - - - - -
     * For now I just use the label which it is modified to match the class sizes and the label 1 belongs to minority class
     * No further action is required
     */


    // keep the number of points in each class inside local variables
    this->num_min_points = min_cnt_;
    this->num_maj_points = maj_cnt_;

    // ind_min should sort
    std::sort(ind_min_,ind_min_ + min_cnt_);   //this is critical for MatGetSubMatrix method
    //I think the sort get the first parameter as the array and the second one as the length // http://www.cplusplus.com/forum/beginner/122086/#msg665485
    std::sort(ind_maj_,ind_maj_ + maj_cnt_);


    ISCreateGeneral(PETSC_COMM_SELF,min_cnt_,ind_min_,PETSC_COPY_VALUES,&is_min_);
    ISCreateGeneral(PETSC_COMM_SELF,maj_cnt_,ind_maj_,PETSC_COPY_VALUES,&is_maj_);

    PetscFree(arr_labels_);
    PetscFree(ind_min_);
    PetscFree(ind_maj_);

    MatGetSubMatrix(this->m_in_data_,is_min_, NULL,MAT_INITIAL_MATRIX,&this->m_min_data_);
    ISDestroy(&is_min_);

    MatGetSubMatrix(this->m_in_data_,is_maj_, NULL,MAT_INITIAL_MATRIX,&this->m_maj_data_);
    ISDestroy(&is_maj_);

    MatDestroy(&this->m_in_data_);      // not required anymore, transform to m_min_data and m_maj_data
    VecDestroy(&this->v_in_label_);     // not required anymore, they already are in different classes

    if(export_file){ //for save_flann we export the divided data into files, the X_norm_data_f_name is different
        write_output(Config_params::getInstance()->get_p_norm_data_f_name(), m_min_data_);
        write_output(Config_params::getInstance()->get_n_norm_data_f_name(), m_maj_data_);
    }

}







/*
 * Get the size of the data from local variables (num_mXX_points) which are set during the divide_data/read_in_divided_data
 * initialize two local vectors (min_shuffled_indices_ , maj_shuffled_indices_)
 * this happens only once for many Exp and many iterations since the size of data in each class will not change
 */
void k_fold::initialize_vectors(){
    ETimer t_all;
    // Random generator without duplicates
    PetscInt min_iter= 0, maj_iter= 0;

    // - - - - - - minority - - - - - -
    this->min_shuffled_indices_.reserve(this->num_min_points);
    //create a vector of all possible nodes for minority class
    for (min_iter=0; min_iter < this->num_min_points; ++min_iter){
        this->min_shuffled_indices_.push_back(min_iter);
    }

    // - - - - - - majority - - - - - -
    this->maj_shuffled_indices_.reserve(this->num_maj_points);
    //create a vector of all possible nodes for majority class
    for (maj_iter=0; maj_iter < this->num_maj_points; ++maj_iter){
        this->maj_shuffled_indices_.push_back(maj_iter);
    }
    t_all.stop_timer("[KF] initialize vectors");
}












/*
 * Get the size of the data from local variables (num_mXX_points) which are set during the divide_data
 * Create a random sequence of numbers for them and store them in local vectors (mXX_shuffled_indices_)
 * The seed for srand comes from the config params and it is recorded in output for debug
 */
void k_fold::shuffle_data(std::string preferred_srand,bool debug_status){
    ETimer t_all;
    // Random generator without duplicates
//    PetscInt min_iter= 0, maj_iter= 0;

    // - - - - - - minority - - - - - -
//    this->min_shuffled_indices_.clear();
//    this->min_shuffled_indices_.reserve(this->num_min_points);

//    //create a vector of all possible nodes for minority class
//    for (min_iter=0; min_iter < this->num_min_points; ++min_iter){
//        this->min_shuffled_indices_.push_back(min_iter);
//    }
//#if dbl_kf_shuffle_data >= 5
//    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][shuffle_data] after initialize vector min\n");
//    for(unsigned int i=0; i < this->min_shuffled_indices_.size(); i++){
//        PetscPrintf(PETSC_COMM_WORLD, "%d,", this->min_shuffled_indices_[i]);
//    }
//    PetscPrintf(PETSC_COMM_WORLD, "\n");
//#endif
    if(preferred_srand.empty())
        preferred_srand = Config_params::getInstance()->get_cpp_srand_seed();

    srand(std::stoll(preferred_srand));
    std::random_shuffle( this->min_shuffled_indices_.begin(), this->min_shuffled_indices_.end() ); //shuffle all nodes

//#if dbl_kf_shuffle_data >= 3
    if(debug_status){
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][shuffle_data] shuffled indices for min\n");
    for(unsigned int i=0; i < this->min_shuffled_indices_.size(); i++){
//        PetscPrintf(PETSC_COMM_WORLD, "%d,", this->min_shuffled_indices_[i]);
        PetscPrintf(PETSC_COMM_WORLD, "%d\n", this->min_shuffled_indices_[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
    }
//#endif
    
    
//    // - - - - - - majority - - - - - -
//    this->maj_shuffled_indices_.reserve(this->num_maj_points);
//    //create a vector of all possible nodes for majority class
//    for (maj_iter=0; maj_iter < this->num_maj_points; ++maj_iter){
//        this->maj_shuffled_indices_.push_back(maj_iter);
//    }
//#if dbl_kf_shuffle_data >= 5
//    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][shuffle_data] after initialize vector maj\n");
//    for(unsigned int i=0; i < this->maj_shuffled_indices_.size(); i++){
//        PetscPrintf(PETSC_COMM_WORLD, "%d,", this->maj_shuffled_indices_[i]);
//    }
//    PetscPrintf(PETSC_COMM_WORLD, "\n");
//#endif

    if(preferred_srand.empty())
        preferred_srand = Config_params::getInstance()->get_cpp_srand_seed();

    srand(std::stoll(preferred_srand));
    std::random_shuffle( this->maj_shuffled_indices_.begin(), this->maj_shuffled_indices_.end() ); //shuffle all nodes

#if dbl_kf_shuffle_data >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][shuffle_data] shuffled indices for maj\n");
    for(unsigned int i=0; i < this->maj_shuffled_indices_.size(); i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", this->maj_shuffled_indices_[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
#endif
    t_all.stop_timer("[k_fold] shuffle_data");
}







/*
 * Get the random sequence from local vectors for both class
 * At each iteration, take i-th subset of each vector and sort it
 * Then get submatrix for that as test part for that class
 * Take the rest as training for that class
 * Combine both test parts with labels as one test matrix
 * Note: iteration should start from ZERO
 */
void k_fold::cross_validation(int current_iteration,int total_iterations,
                    std::string p_data_fname, std::string n_data_fname, std::string test_data_fname){

    Mat m_min_train_data, m_maj_train_data, m_test_data; //Temporary for file output
    // - - - - - - minority - - - - - -
    PetscInt    min_test_start, min_test_end, min_subset_size, min_train_size;
    PetscInt    * ind_min_train, * ind_min_test;
    PetscInt    min_iter=0, min_test_curr=0, min_train_curr=0;

    // - - - - - - majority - - - - - -
    PetscInt    maj_test_start, maj_test_end, maj_subset_size, maj_train_size;
    PetscInt    * ind_maj_train, * ind_maj_test;
    PetscInt    maj_iter=0, maj_test_curr=0, maj_train_curr=0;

    PetscMalloc1(this->num_min_points, &ind_min_train);      //allocate memory for arrays largers than 1 M points
    PetscMalloc1(this->num_maj_points, &ind_maj_train);

    PetscMalloc1(ceil(this->num_min_points / total_iterations) + 1, &ind_min_test);
    PetscMalloc1(ceil(this->num_maj_points / total_iterations) + 1, &ind_maj_test);

#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] memory for arrays are allocatted \n");
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] number of points in min:%d, in maj:%d \n",this->num_min_points, this->num_maj_points);
#endif
    /// = = = = =  Prepare indices = = = = =
    // - - - - - A (TEST - Train)  - - - - - -
    if(current_iteration == 0){
        min_subset_size = floor(this->num_min_points / total_iterations);
        min_test_start  = 0;
        min_test_end    = min_subset_size ;
        min_train_size  = this->num_min_points - min_subset_size;
        
        maj_subset_size = floor(this->num_maj_points / total_iterations);
        maj_test_start  = 0;
        maj_test_end    = maj_subset_size ;
        maj_train_size  = this->num_maj_points - maj_subset_size;
#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] condition: A (TEST - Train)\n");
//    PetscPrintf(PETSC_COMM_WORLD, "min_subset_size:%d,min_test_end:%d,min_train_size:%d\n",min_subset_size,min_test_end,min_train_size);
//    PetscPrintf(PETSC_COMM_WORLD, "maj_subset_size:%d,maj_test_end:%d,maj_train_size:%d\n",maj_subset_size,maj_test_end,maj_train_size);
#endif
        // - - - - - Min Test - - - - -
        for(min_iter= 0; min_iter< min_test_end; min_iter++){  //min_test_start should be 0
            //at each iteration copy the index from shuffled vector to array
            ind_min_test[min_test_curr] = this->min_shuffled_indices_[min_iter];
            //move the pointer forward
            min_test_curr++;
        }
        // - - - - - Min Train - - - - -
        for(min_iter= min_test_end; min_iter< this->num_min_points+1; min_iter++){
            ind_min_train[min_train_curr] = this->min_shuffled_indices_[min_iter];
            min_train_curr++;

        }

        // - - - - - Maj Test - - - - -
        for(maj_iter= 0; maj_iter< maj_test_end; maj_iter++){  //maj_test_start should be 0
            //at each iteration copy the index from shuffled vector to array
            ind_maj_test[maj_test_curr] = this->maj_shuffled_indices_[maj_iter];
            //move the pointer forward
            maj_test_curr++;
        }

        // - - - - - Maj Train - - - - -
        for(maj_iter= maj_test_end; maj_iter< this->num_maj_points; maj_iter++){
            ind_maj_train[maj_train_curr] = this->maj_shuffled_indices_[maj_iter];
            maj_train_curr++;
        }
    }
    // - - - - - - B ( Train - TEST )  - - - - - -
    if(current_iteration == (total_iterations - 1)){
        // Train start = 0
        // Train end = Test_start - 1
        int min_remaining_part = (this->num_min_points % total_iterations);
        min_subset_size = floor(this->num_min_points / total_iterations);
        min_test_start  = (current_iteration * min_subset_size) + min_remaining_part ;
        min_test_end    = this->num_min_points;
        min_train_size  = this->num_min_points - min_subset_size;
        
        int maj_remaining_part = this->num_maj_points % total_iterations;
        maj_subset_size = floor(this->num_maj_points / total_iterations);
        maj_test_start  = (current_iteration * maj_subset_size) + maj_remaining_part ;
        maj_test_end    = this->num_maj_points;
        maj_train_size  = this->num_maj_points - maj_subset_size;
#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] condition: B (Train - TEST) \n");
#endif

        // - - - - - Min Train - - - - -
        min_train_curr = 0;
        for(min_iter= 0; min_iter< min_test_start; min_iter++){  //min_test_start should be 0
            ind_min_train[min_train_curr] = this->min_shuffled_indices_[min_iter];
            min_train_curr++;
        }
        // - - - - - Min Test - - - - -
        min_test_curr = 0;
        for(min_iter= min_test_start; min_iter< this->num_min_points; min_iter++){  //min_test_end should be num_min_point
            ind_min_test[min_test_curr] = this->min_shuffled_indices_[min_iter];
            min_test_curr++;
        }

        // - - - - - Maj Train - - - - -
        maj_train_curr = 0;
        for(maj_iter= 0; maj_iter< maj_test_start; maj_iter++){  //maj_test_start should be 0
            ind_maj_train[maj_train_curr] = this->maj_shuffled_indices_[maj_iter];
            maj_train_curr++;
        }
        // - - - - - Maj Test - - - - -
        maj_test_curr = 0;
        for(maj_iter= maj_test_start; maj_iter< this->num_maj_points; maj_iter++){  //maj_test_end should be num_maj_point
            ind_maj_test[maj_test_curr] = this->maj_shuffled_indices_[maj_iter];
            maj_test_curr++;
        }
    }
#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] condition: C (Train_part 1 - TEST - Train_part 2)\n");
#endif
    // - - - - - - C (Train_part 1 - TEST - Train_part 2) - - - - - -
    if(!(current_iteration == 0 || current_iteration == (total_iterations - 1))){
        // P1 start = 0
        // P1 end = Test_start - 1
        // P2 start = Test_end +1
        // P2 end = num_points
        // largest part is the last one which is P2 (0..k-1 larger than others)
        min_subset_size = floor(this->num_min_points / total_iterations);
        min_test_start  = current_iteration * min_subset_size;
        min_test_end    = min_test_start + min_subset_size ;
        min_train_size  = this->num_min_points - min_subset_size;

        maj_subset_size = floor(this->num_maj_points / total_iterations);
        maj_test_start  = current_iteration * maj_subset_size;
        maj_test_end    = maj_test_start + maj_subset_size ;
        maj_train_size  = this->num_maj_points - maj_subset_size;
        
        // - - - - - Min Train P1 (Left)- - - - -
        min_train_curr = 0;
        for(min_iter= 0; min_iter< min_test_start; min_iter++){
            ind_min_train[min_train_curr] = this->min_shuffled_indices_[min_iter];
            min_train_curr++;
        }
        // - - - - - Min Test - - - - -
        min_test_curr = 0;
        for(min_iter= min_test_start; min_iter< min_test_end; min_iter++){
            ind_min_test[min_test_curr] = this->min_shuffled_indices_[min_iter];
            min_test_curr++;
        }
        // - - - - - Min Train 2 (Right)- - - - -
        // don't touch min_train_curr because it needs to follow the Train 1 (Left)
        for(min_iter= min_test_end; min_iter< this->num_min_points; min_iter++){
            ind_min_train[min_train_curr] = this->min_shuffled_indices_[min_iter];
            min_train_curr++;
        }

        // - - - - - Maj Train 1 (Left)- - - - -
        maj_train_curr = 0;
        for(maj_iter= 0; maj_iter< maj_test_start; maj_iter++){
            ind_maj_train[maj_train_curr] = this->maj_shuffled_indices_[maj_iter];
            maj_train_curr++;
        }
        // - - - - - Maj Test - - - - -
        maj_test_curr = 0;
        for(maj_iter= maj_test_start; maj_iter< maj_test_end; maj_iter++){
            ind_maj_test[maj_test_curr] = this->maj_shuffled_indices_[maj_iter];
            maj_test_curr++;
        }
        // - - - - - Maj Train 2 (Right)- - - - -
        // don't touch maj_train_curr because it needs to follow the Train 1 (Left)
        for(maj_iter= maj_test_end; maj_iter< this->num_maj_points; maj_iter++){
            ind_maj_train[maj_train_curr] = this->maj_shuffled_indices_[maj_iter];
            maj_train_curr++;
        }
    }
#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] parts are set in the vectors \n");
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] Before sorting the arrays\n");
    int max_debug=0, max_loc=0;
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] indices in ind_min_train are:");
    for(int i=0; i< min_train_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", ind_min_train[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n[k_fold][cross_validation] indices in ind_min_test are:");
    for(int i=0; i< min_subset_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", ind_min_test[i]);
    }
    for(int i=0; i< min_train_size; i++){
        if(ind_min_train[i] > max_debug){
            max_debug= ind_min_train[i];
            max_loc = i;
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, " and Max index is:%d in location:%d\n", max_debug, max_loc);
    
    max_debug =0;
    max_loc = 0;
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] indices in ind_maj_train are:");
    for(int i=0; i< maj_train_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", ind_maj_train[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n[k_fold][cross_validation] indices in ind_maj_test are:");
    for(int i=0; i< maj_subset_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", ind_maj_test[i]);
    }
    for(int i=0; i< maj_train_size; i++){
        if(ind_maj_train[i] > max_debug){
            max_debug= ind_maj_train[i];
            max_loc = i;
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, " and Max index is:%d in location:%d\n", max_debug, max_loc);
#endif

    // Sort the array for MatGetSubMatrix method
    std::sort(ind_min_train, (ind_min_train + min_train_size));
    std::sort(ind_maj_train, (ind_maj_train + maj_train_size));
    std::sort(ind_min_test, (ind_min_test + min_subset_size));
    std::sort(ind_maj_test, (ind_maj_test + maj_subset_size));

#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] parts are sorted \n");
    max_debug=0, max_loc=0;
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] indices in ind_min_train are:");
    for(int i=0; i< min_train_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", ind_min_train[i]);
    }
    for(int i=0; i< min_train_size; i++){
        if(ind_min_train[i] > max_debug){
            max_debug= ind_min_train[i];
            max_loc = i;
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, " and Max index is:%d in location:%d\n", max_debug, max_loc);
    
    max_debug =0;
    max_loc = 0;
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] indices in ind_maj_train are:");
    for(int i=0; i< maj_train_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", ind_maj_train[i]);
    }
    for(int i=0; i< maj_train_size; i++){
        if(ind_maj_train[i] > max_debug){
            max_debug= ind_maj_train[i];
            max_loc = i;
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, " and Max index is:%d in location:%d\n", max_debug, max_loc);
#endif

    IS      is_min_train_, is_maj_train_;
    IS      is_min_test_, is_maj_test_;
    Mat     m_min_test_data, m_maj_test_data;
    ISCreateGeneral(PETSC_COMM_SELF,min_train_size,ind_min_train,PETSC_COPY_VALUES,&is_min_train_);
    ISCreateGeneral(PETSC_COMM_SELF,maj_train_size,ind_maj_train,PETSC_COPY_VALUES,&is_maj_train_);
    ISCreateGeneral(PETSC_COMM_SELF,min_subset_size,ind_min_test,PETSC_COPY_VALUES,&is_min_test_);
    ISCreateGeneral(PETSC_COMM_SELF,maj_subset_size,ind_maj_test,PETSC_COPY_VALUES,&is_maj_test_);
    
#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] ISs are created \n");
//    PetscInt num_row_debug =0 ;
    
//    MatGetSize(this->m_min_data_, &num_row_debug, NULL);
//    if(num_row_debug == 0){
//        PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] {debug} empty this->m_min_data_ matrix \n");
//    }else{
//        PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] {debug} max number of rows in this->m_min_data_ matrix is:%d, ", num_row_debug);
//        PetscPrintf(PETSC_COMM_WORLD, "last min_train_curr:%d, min_test_curr:%d, ", min_train_curr,min_test_curr);
//        PetscPrintf(PETSC_COMM_WORLD, "ind_min_train[min_train_curr-1]:%d, ind_min_test[min_test_curr-1]:%d\n", ind_min_train[min_train_curr-1],ind_min_test[min_test_curr-1]);
//    }
    
//    MatGetSize(this->m_maj_data_, &num_row_debug, NULL);
//    if(num_row_debug == 0){
//        PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] {debug} empty this->m_maj_data_ matrix \n");
//    }else{
//        PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] {debug} max number of rows in this->m_maj_data_ matrix is:%d, ", num_row_debug);
//        PetscPrintf(PETSC_COMM_WORLD, "last maj_train_curr:%d, maj_test_curr:%d, ", maj_train_curr, maj_test_curr);
//        PetscPrintf(PETSC_COMM_WORLD, "ind_maj_train[maj_train_curr-1]:%d, ind_maj_test[maj_test_curr-1]:%d\n", ind_maj_train[maj_train_curr-1],ind_maj_test[maj_test_curr-1]);
//    }
#endif

    PetscFree(ind_min_train);      //release memory for arrays
    PetscFree(ind_maj_train);
    PetscFree(ind_min_test);
    PetscFree(ind_maj_test);

#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] arrays are freed \n");
    
    
#endif

    MatGetSubMatrix(this->m_min_data_,is_min_train_, NULL,MAT_INITIAL_MATRIX,&m_min_train_data);
    MatGetSubMatrix(this->m_maj_data_,is_maj_train_, NULL,MAT_INITIAL_MATRIX,&m_maj_train_data);
    MatGetSubMatrix(this->m_min_data_,is_min_test_, NULL,MAT_INITIAL_MATRIX,&m_min_test_data);
    MatGetSubMatrix(this->m_maj_data_,is_maj_test_, NULL,MAT_INITIAL_MATRIX,&m_maj_test_data);

#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] sub matrices are created \n");
#endif

    ISDestroy(&is_min_train_);
    ISDestroy(&is_maj_train_);
    ISDestroy(&is_min_test_);
    ISDestroy(&is_maj_test_);
    /* - - - - - - outcome of above part - - - - -
     *  The data of each class is divided to 2 part and the test part
     * is going to add together. I need to rethink about this but for now,
     * I just do this
     */

    /// = = = = =  Combine Test Data = = = = =
    combine_two_classes_in_one(m_test_data, m_min_test_data, m_maj_test_data );

#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] test data is combined \n");
#endif

    if(p_data_fname.empty())
        p_data_fname = Config_params::getInstance()->get_p_norm_data_f_name();

    if(n_data_fname.empty())
        n_data_fname = Config_params::getInstance()->get_n_norm_data_f_name();

    if(test_data_fname.empty())
        test_data_fname = Config_params::getInstance()->get_test_ds_f_name();


    write_output(p_data_fname, m_min_train_data);
    write_output(n_data_fname, m_maj_train_data);
    write_output(test_data_fname, m_test_data);

    MatDestroy(&m_min_train_data);      //added 012617
    MatDestroy(&m_maj_train_data);      //added 012617
    MatDestroy(&m_test_data);           //added 012617
   
#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] write all files successfully. \n");
#endif 
}





//Jan 23, 2017  --- 19:28
/*
 * Get the random sequence in input vectors a class
 * At each iteration, take i-th subset of each vector and sort it
 * Then get submatrix for that as test part for that class
 * Take the rest as training
 * Note: iteration should start from ZERO
 */

/*
 * I need to create a hash table (set) for test indices and then query the neighbor of points in each row of the training data
 * inside the filter_NN to find out which nodes are not in the training data.
 * The test data is used to track them because it is at most 1/k of whole data where k is the number of folds in Cross validation
 *
 * flann's indices are sorted ascending by their distance (first one is closer than second one)
 * Checked Feb 1, 2017 using dists results
 *
 * Structure:       **** for each class separately **** (different from earlier approach which is longer coding)
 * First:   A function to give me the indices for training and test data
 *          Create the training_data_matrix, test_data_matrix for a class not both
 *
 * Second: I need another function to load the flann results and give me 2 matrix for training data (indices & dists)(2nd function)
 *
 * Warning: the indices in the Flann full indices are not valid for train data, since some records are removed for test data
 *          so v_full_idx_to_train_idx will be used to keep the mapping information
 *
 */
void k_fold::cross_validation_class(int curr_iter,int total_iter, Mat& m_full_data, Mat& m_train_data, Mat& m_test_data,
                    PetscInt * arr_idx_train, PetscInt& train_size, std::unordered_set<PetscInt>& uset_test_indices,
                    std::vector<PetscInt>& v_full_idx_to_train_idx, const std::string& info,
                    const std::vector<PetscInt>& v_shuffled_indices,bool debug_status){

//    Mat m_min_train_data, m_maj_train_data, m_test_data; //Temporary for file output
    // - - - - - - minority - - - - - -
    PetscInt    idx_test_start, idx_test_end, subset_size;

    //the arr_idx_train is used for filter_NN function, so I need it later and I need to keep it,
    //don't forget to destroy it in that function
    PetscInt    * arr_idx_test;
    PetscInt    idx_iter=0, idx_test_curr=0, idx_train_curr=0;

    PetscInt    size_full_data;
    MatGetSize(m_full_data, &size_full_data, NULL);
//    PetscMalloc1(size_full_data, &arr_idx_train); // I should not malloc inside the function since it will change the address

    PetscMalloc1(ceil(size_full_data / total_iter) + 1, &arr_idx_test);

#if dbl_KF_CVC >= 3
    std::cout << "[KF][CV-C] class: " << info << std::endl;
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] memory for arrays are allocatted \n");
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] number of points in are %d \n",size_full_data);
#endif
    /// = = = = =  Prepare indices = = = = =
    subset_size = floor(size_full_data / total_iter);       //it is also the test size
    train_size  = size_full_data - subset_size;

    // - - - - - A (TEST - Train)  - - - - - -
    if(curr_iter == 0){
        idx_test_start  = 0;
        idx_test_end    = subset_size ;
#if dbl_KF_CVC >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] condition: A (TEST - Train)\n");
    std::cout << "[KF][CV-C] A idx_test_end: " << idx_test_end << std::endl;
#endif

        // - - - - - Test - - - - -
        for(idx_iter= 0; idx_iter< idx_test_end; idx_iter++){  //idx_test_start should be 0
            //at each iteration copy the index from shuffled vector to array
            arr_idx_test[idx_test_curr] = v_shuffled_indices[idx_iter];
            //move the pointer forward
            idx_test_curr++;
        }
        // - - - - - Train - - - - -
        for(idx_iter= idx_test_end; idx_iter< size_full_data+1; idx_iter++){
            arr_idx_train[ idx_train_curr] = v_shuffled_indices[idx_iter];
            idx_train_curr++;

        }

    }
    // - - - - - - B ( Train - TEST )  - - - - - -
    if(curr_iter == (total_iter - 1)){
        // Train start = 0
        // Train end = Test_start - 1
        int min_remaining_part = (size_full_data % total_iter);
        idx_test_start  = (curr_iter * subset_size) + min_remaining_part ;
        idx_test_end    = size_full_data;
#if dbl_KF_CVC >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] condition: B (Train - TEST) \n");
#endif

        // - - - - - Train - - - - -
        idx_train_curr = 0;
        for(idx_iter= 0; idx_iter< idx_test_start; idx_iter++){  //idx_test_start should be 0
            arr_idx_train[ idx_train_curr] = v_shuffled_indices[idx_iter];
            idx_train_curr++;
        }
        // - - - - - Test - - - - -
        idx_test_curr = 0;
        for(idx_iter= idx_test_start; idx_iter< size_full_data; idx_iter++){  //idx_test_end should be num_min_point
            arr_idx_test[ idx_test_curr] = v_shuffled_indices[idx_iter];
            idx_test_curr++;
        }

    }

    // - - - - - - C (Train_part 1 - TEST - Train_part 2) - - - - - -
    if(!(curr_iter == 0 || curr_iter == (total_iter - 1))){
        // P1 start = 0
        // P1 end = Test_start - 1
        // P2 start = Test_end +1
        // P2 end = num_points
        // largest part is the last one which is P2 (0..k-1 larger than others)
        idx_test_start  = curr_iter * subset_size;
        idx_test_end    = idx_test_start + subset_size ;
#if dbl_KF_CVC >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] condition: C (Train_part 1 - TEST - Train_part 2)\n");
#endif

        // - - - - - Train P1 (Left)- - - - -
        idx_train_curr = 0;
        for(idx_iter= 0; idx_iter< idx_test_start; idx_iter++){
            arr_idx_train[ idx_train_curr] = v_shuffled_indices[idx_iter];
            idx_train_curr++;
        }
        // - - - - - Test - - - - -
        idx_test_curr = 0;
        for(idx_iter= idx_test_start; idx_iter< idx_test_end; idx_iter++){
            arr_idx_test[ idx_test_curr] = v_shuffled_indices[idx_iter];
            idx_test_curr++;
        }
        // - - - - - Min Train 2 (Right)- - - - -
        // don't touch  idx_train_curr because it needs to follow the Train 1 (Left)
        for(idx_iter= idx_test_end; idx_iter< size_full_data; idx_iter++){
            arr_idx_train[ idx_train_curr] = v_shuffled_indices[idx_iter];
            idx_train_curr++;
        }

    }
#if dbl_KF_CVC >= 7
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] parts are set in the vectors (Before sorting the arrays) \n");
    int max_debug=0, max_loc=0;
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] indices in arr_idx_train are:");
    for(int i=0; i< train_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", arr_idx_train[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n[KF][CV-C] indices in arr_idx_test are:");
    for(int i=0; i< subset_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", arr_idx_test[i]);
    }
    for(int i=0; i< train_size; i++){
        if(arr_idx_train[i] > max_debug){
            max_debug= arr_idx_train[i];
            max_loc = i;
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, " and Max index is:%d in location:%d\n", max_debug, max_loc);
#endif

    // Sort the array for MatGetSubMatrix method
    std::sort(arr_idx_train, (arr_idx_train + train_size));
    std::sort(arr_idx_test, (arr_idx_test + subset_size));

    //fill the sorted train indices into hash set
    uset_test_indices.reserve(subset_size * 2);
    for(int i=0; i< subset_size; i++){              //this should be the test size
        uset_test_indices.insert(arr_idx_test[i]);

    }

    //store the indices of train data as values in right indices from full data,
    //so there are about 10- 20% garbage info in the vector which I don't care,
    //since I only use the train_indices later in filter_NN
    // the test part has not used at all or in the future
    v_full_idx_to_train_idx.reserve(size_full_data);
    std::fill(v_full_idx_to_train_idx.begin(), v_full_idx_to_train_idx.end(), -1);
    for(int i=0; i< train_size; i++){              //this should be the test size
        // i.e. value of index 31 is equal to 19 (since some of 30 first points might be selected for test data)
        v_full_idx_to_train_idx[arr_idx_train[i]]=i;
//        if(info == "minority" && i < 150 && debug_status)

//        std::cout << "[KF][CV-C] i:" <<i << ", arr_idx_train[i]: " << arr_idx_train[i] <<
//                     ", v_full_idx_to_train_idx[" << arr_idx_train[i]<< "]: "
//                      << v_full_idx_to_train_idx[arr_idx_train[i]]<< std::endl;
    }


#if dbl_KF_CVC >= 7
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] parts are sorted \n");
    max_debug=0, max_loc=0;
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] indices in arr_idx_train are:");
    for(int i=0; i< train_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", arr_idx_train[i]);
    }
    for(int i=0; i< train_size; i++){
        if(arr_idx_train[i] > max_debug){
            max_debug= arr_idx_train[i];
            max_loc = i;
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, " and Max index is:%d in location:%d\n", max_debug, max_loc);
#endif

    IS      is_train, is_test;
    ISCreateGeneral(PETSC_COMM_SELF,train_size,arr_idx_train,PETSC_COPY_VALUES,&is_train);
    ISCreateGeneral(PETSC_COMM_SELF,subset_size,arr_idx_test,PETSC_COPY_VALUES,&is_test);

#if dbl_KF_CVC >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] ISs are created \n");
#endif

//    PetscFree(arr_idx_train);     //this one should not be released because I need it later in  filter_NN function
    PetscFree(arr_idx_test);    //release memory for arrays

#if dbl_KF_CVC >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] arrays are freed \n");
#endif

    MatGetSubMatrix(m_full_data,is_train, NULL,MAT_INITIAL_MATRIX,&m_train_data);       //deprecated after version 3.7
    MatGetSubMatrix(m_full_data,is_test, NULL,MAT_INITIAL_MATRIX,&m_test_data);

//    MatCreateSubMatrix(m_full_data,is_train, NULL,MAT_INITIAL_MATRIX,&m_train_data);    //version 3.8 and beyond
//    MatCreateSubMatrix(m_full_data,is_test, NULL,MAT_INITIAL_MATRIX,&m_test_data);

#if dbl_KF_CVC >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CV-C] sub matrices are created \n");
#endif

    ISDestroy(&is_train);
    ISDestroy(&is_test);
}











void k_fold::prepare_traindata_for_flann(){
    std::string p_data_fname = Config_params::getInstance()->get_p_norm_data_f_name();
    std::string n_data_fname = Config_params::getInstance()->get_n_norm_data_f_name();

    write_output(p_data_fname, this->m_min_data_);
    write_output(n_data_fname, this->m_maj_data_);
}





void k_fold::cross_validation_simple(Mat& m_data_p, Mat& m_data_n, Vec& v_vol_p, Vec& v_vol_n,
                                     int current_iteration, int total_iterations,
                                     Mat& m_train_data_p, Mat& m_train_data_n, Mat& m_test_data,
                                     Vec& v_train_vol_p, Vec& v_train_vol_n){
#if dbl_KF_CVS >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] start!\n");
#endif
    PetscInt num_point_p, num_point_n;
    MatGetSize(m_data_p, &num_point_p, NULL);
    MatGetSize(m_data_n, &num_point_n, NULL);
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] num_point_p:%d, num_point_n:%d, curr_iter:%d, total_iter:%d \n",
                                             num_point_p,num_point_n, current_iteration, total_iterations);
    // - - - - - - minority - - - - - -
    PetscInt    min_test_start, min_test_end, min_subset_size, min_train_size;
    PetscInt    * ind_min_train, * ind_min_test;
    PetscInt    min_iter=0; //, min_test_curr=0, min_train_curr=0;
    // - - - - - - majority - - - - - -
    PetscInt    maj_test_start, maj_test_end, maj_subset_size, maj_train_size;
    PetscInt    * ind_maj_train, * ind_maj_test;
    PetscInt    maj_iter=0; //, maj_test_curr=0, maj_train_curr=0;

    PetscMalloc1(num_point_p, &ind_min_train);      //allocate memory for arrays largers than 1 M points
    PetscMalloc1(num_point_n, &ind_maj_train);

    PetscMalloc1(ceil(num_point_p / total_iterations) + 1, &ind_min_test);
    PetscMalloc1(ceil(num_point_n / total_iterations) + 1, &ind_maj_test);

#if dbl_KF_CVS >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] memory for arrays are allocatted \n");
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] number of points in min:%d, in maj:%d \n",num_point_p, num_point_n);
#endif
    /// = = = = =  Prepare indices = = = = =
    // - - - - - A (TEST - Train)  - - - - - -
    if(current_iteration == 0){
        min_subset_size = floor(num_point_p / total_iterations);
        min_test_start  = 0;
        min_test_end    = min_subset_size ;
        min_train_size  = num_point_p - min_subset_size;

        maj_subset_size = floor(num_point_n / total_iterations);
        maj_test_start  = 0;
        maj_test_end    = maj_subset_size ;
        maj_train_size  = num_point_n - maj_subset_size;
#if dbl_KF_CVS >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] condition: A (TEST - Train)\n");
    PetscPrintf(PETSC_COMM_WORLD, "min_subset_size:%d,min_test_end:%d,min_train_size:%d\n",min_subset_size,min_test_end,min_train_size);
    PetscPrintf(PETSC_COMM_WORLD, "maj_subset_size:%d,maj_test_end:%d,maj_train_size:%d\n",maj_subset_size,maj_test_end,maj_train_size);
#endif
        // - - - - - Min Test - - - - -
        for(min_iter= 0; min_iter< min_test_end; min_iter++){  //min_test_start should be 0
            ind_min_test[min_iter] = min_iter;
// 	std::cout << "A test min_iter:" << min_iter << std::endl;
        }
        // - - - - - Min Train - - - - -
        for(min_iter= min_test_end; min_iter< num_point_p+1; min_iter++){
            // Notice: Each array should fill from zero index, the values are different 
            ind_min_train[min_iter - min_test_end] = min_iter ;         
//	std::cout << "A Train min_iter:" << min_iter << std::endl;
        }

        // - - - - - Maj Test - - - - -
        for(maj_iter= 0; maj_iter< maj_test_end; maj_iter++){  //maj_test_start should be 0
            ind_maj_test[maj_iter] = maj_iter;
        }
        // - - - - - Maj Train - - - - -
        for(maj_iter= maj_test_end; maj_iter< num_point_n; maj_iter++){
            ind_maj_train[maj_iter - maj_test_end] = maj_iter;
        }
    }
    // - - - - - - B ( Train - TEST )  - - - - - -
    if(current_iteration == (total_iterations - 1)){
        // Train start = 0
        // Train end = Test_start - 1
        int min_remaining_part = (num_point_p % total_iterations);
        min_subset_size = floor(num_point_p / total_iterations);
        min_test_start  = (current_iteration * min_subset_size) + min_remaining_part ;
        min_test_end    = num_point_p;
        min_train_size  = num_point_p - min_subset_size;

        int maj_remaining_part = num_point_n % total_iterations;
        maj_subset_size = floor(num_point_n / total_iterations);
        maj_test_start  = (current_iteration * maj_subset_size) + maj_remaining_part ;
        maj_test_end    = num_point_n;
        maj_train_size  = num_point_n - maj_subset_size;

#if dbl_KF_CVS >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] condition: B (Train - TEST) \n");
#endif
        // - - - - - Min Train - - - - -
        for(min_iter= 0; min_iter< min_test_start; min_iter++){  //min_test_start should be 0
            ind_min_train[min_iter] = min_iter;
        }
        // - - - - - Min Test - - - - -
        for(min_iter= min_test_start; min_iter< num_point_p; min_iter++){  //min_test_end should be num_min_point
            ind_min_test[min_iter - min_test_start] = min_iter;
        }

        // - - - - - Maj Train - - - - -
        for(maj_iter= 0; maj_iter< maj_test_start; maj_iter++){  //maj_test_start should be 0
            ind_maj_train[maj_iter] = maj_iter;
        }
        // - - - - - Maj Test - - - - -
        for(maj_iter= maj_test_start; maj_iter< num_point_n; maj_iter++){  //maj_test_end should be num_maj_point
            ind_maj_test[maj_iter - maj_test_start] = maj_iter;
        }
    }
        
    // - - - - - - C (Train_part 1 - TEST - Train_part 2) - - - - - -
    if(!(current_iteration == 0 || current_iteration == (total_iterations - 1))){
#if dbl_KF_CVS >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] condition: C (Train_part 1 - TEST - Train_part 2)\n");
#endif
//exit(1);
        // P1 start = 0
        // P1 end = Test_start - 1
        // P2 start = Test_end +1
        // P2 end = num_points
        // largest part is the last one which is P2 (0..k-1 larger than others)
        min_subset_size = floor(num_point_p / total_iterations);
        min_test_start  = current_iteration * min_subset_size;
        min_test_end    = min_test_start + min_subset_size ;
        min_train_size  = num_point_p - min_subset_size;
#if dbl_KF_CVS >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] min_subset_size:%d, min_test_start:%d, min_test_end:%d  \n",
                    min_subset_size, min_test_start, min_test_end);
#endif
        maj_subset_size = floor(num_point_n / total_iterations);
        maj_test_start  = current_iteration * maj_subset_size;
        maj_test_end    = maj_test_start + maj_subset_size ;
        maj_train_size  = num_point_n - maj_subset_size;

        // - - - - - Min Train P1 (Left)- - - - -
        for(min_iter= 0; min_iter< min_test_start; min_iter++){
            ind_min_train[min_iter] = min_iter;
        }
        // - - - - - Min Test - - - - -
        for(min_iter= min_test_start; min_iter< min_test_end; min_iter++){
            ind_min_test[min_iter-min_test_start] = min_iter ;
        }
        // - - - - - Min Train 2 (Right)- - - - -
        for(min_iter= min_test_end; min_iter< num_point_p; min_iter++){
            ind_min_train[min_iter - min_test_end + min_test_start] = min_iter;
        }

        // - - - - - Maj Train 1 (Left)- - - - -
        for(maj_iter= 0; maj_iter< maj_test_start; maj_iter++){
            ind_maj_train[maj_iter] = maj_iter;
        }
        // - - - - - Maj Test - - - - -
        for(maj_iter= maj_test_start; maj_iter< maj_test_end; maj_iter++){
            ind_maj_test[maj_iter - maj_test_start] = maj_iter ;
        }
        // - - - - - Maj Train 2 (Right)- - - - -
        for(maj_iter= maj_test_end; maj_iter< num_point_n; maj_iter++){
            ind_maj_train[maj_iter - maj_test_end + maj_test_start] = maj_iter;
        }
    }
#if dbl_KF_CVS >= 5 //5 default
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] parts are set in the vectors \n");
    int max_debug=0, max_loc=0;
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] indices in ind_min_train are:");
    for(int i=0; i< min_train_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", ind_min_train[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n[KF][CVS] indices in ind_min_test are:");
    for(int i=0; i< min_subset_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", ind_min_test[i]);
    }
    for(int i=0; i< min_train_size; i++){
        if(ind_min_train[i] > max_debug){
            max_debug= ind_min_train[i];
            max_loc = i;
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, " and Max index is:%d in location:%d\n", max_debug, max_loc);

    max_debug =0;
    max_loc = 0;
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] indices in ind_maj_train are:");
    for(int i=0; i< maj_train_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", ind_maj_train[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n[KF][CVS] indices in ind_maj_test are:");
    for(int i=0; i< maj_subset_size; i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", ind_maj_test[i]);
    }
    for(int i=0; i< maj_train_size; i++){
        if(ind_maj_train[i] > max_debug){
            max_debug= ind_maj_train[i];
            max_loc = i;
        }
    }
    PetscPrintf(PETSC_COMM_WORLD, " and Max index is:%d in location:%d\n", max_debug, max_loc);
#endif

    IS      is_min_train_, is_maj_train_;
    IS      is_min_test_, is_maj_test_;
    ISCreateGeneral(PETSC_COMM_SELF,min_train_size,ind_min_train,PETSC_COPY_VALUES,&is_min_train_);
    ISCreateGeneral(PETSC_COMM_SELF,maj_train_size,ind_maj_train,PETSC_COPY_VALUES,&is_maj_train_);
    ISCreateGeneral(PETSC_COMM_SELF,min_subset_size,ind_min_test,PETSC_COPY_VALUES,&is_min_test_);
    ISCreateGeneral(PETSC_COMM_SELF,maj_subset_size,ind_maj_test,PETSC_COPY_VALUES,&is_maj_test_);

#if dbl_KF_CVS >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] ISs are created \n");
#endif

    PetscFree(ind_min_train);      //release memory for arrays
    PetscFree(ind_maj_train);
    PetscFree(ind_min_test);
    PetscFree(ind_maj_test);

    Mat m_test_data_p_, m_test_data_n_;
    MatGetSubMatrix(m_data_p,is_min_train_, NULL,MAT_INITIAL_MATRIX,&m_train_data_p);
    MatGetSubMatrix(m_data_n,is_maj_train_, NULL,MAT_INITIAL_MATRIX,&m_train_data_n);
    MatGetSubMatrix(m_data_p,is_min_test_, NULL,MAT_INITIAL_MATRIX,&m_test_data_p_);
    MatGetSubMatrix(m_data_n,is_maj_test_, NULL,MAT_INITIAL_MATRIX,&m_test_data_n_);

    VecGetSubVector(v_vol_p,is_min_train_, &v_train_vol_p);
    VecGetSubVector(v_vol_n,is_maj_train_, &v_train_vol_n);
#if dbl_KF_CVS >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CVS] sub matrices are created \n");
//    VecView(v_train_vol_p, PETSC_VIEWER_STDOUT_WORLD);
//    VecView(v_train_vol_n, PETSC_VIEWER_STDOUT_WORLD);
#endif

    ISDestroy(&is_min_train_);
    ISDestroy(&is_maj_train_);
    ISDestroy(&is_min_test_);
    ISDestroy(&is_maj_test_);
    /* - - - - - - outcome of above part - - - - -
     *  The data of each class is divided to 2 part and the test part
     * is going to add together. I need to rethink about this but for now,
     * I just do this
     */

    /// = = = = =  Combine Test Data = = = = =
//    printf("[k_fold] [cross validated] min test data before combine_test_data Matrix:\n");     //$$debug
//    MatView(m_min_test_data ,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
//    printf("[k_fold] [cross validated] maj test data before combine_test_data Matrix:\n");     //$$debug
//    MatView(m_maj_test_data ,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

    combine_two_classes_in_one(m_test_data, m_test_data_p_, m_test_data_n_ );
}

/*
 * I guess I should skip this in future, but for now it is very similar to what I have in the model selection
 * It gets 2 test matrices for both classes and add the labels to the first column of a new matrix
 * which contains both of them
 * Note: the dt_test_p, dt_test_n will destroy in the end of this function because they are not needed anymore
 * changed: 021517-1605 the name of matrices are changed since they are not only used for testdata.
 * Now I use this function for validation data as well
 */
void k_fold::combine_two_classes_in_one(Mat& m_output, Mat& m_positive_class, Mat& m_negative_class, bool destroy_input_matrices){
    ETimer t_all;
    PetscInt max_num_col_;
    PetscInt num_row_min, num_row_maj, num_row;
    PetscInt i, ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    MatGetSize(m_positive_class, &num_row_min, &max_num_col_);   //set the number of columns
    MatGetSize(m_negative_class, &num_row_maj, NULL);

    num_row = num_row_min + num_row_maj;
#if dbl_KF_CTC >= 1
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CTC] num_row_min: %d, num_row_maj:%d\n",num_row_min,num_row_maj);     //$$debug
    PetscPrintf(PETSC_COMM_WORLD, "[KF][CTC] num_row: %d, num_col:%d, nz:%d\n",num_row,max_num_col_ + 1 , max_num_col_ + 1);     //$$debug
#endif
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row ,max_num_col_ + 1 ,(max_num_col_ + 1 ),PETSC_NULL, &m_output); //+1 is for label
    for(i =0; i < num_row_min ; i++){
        MatSetValue(m_output, i, 0, +1,INSERT_VALUES);        //Insert positive lable
        MatGetRow(m_positive_class,i,&ncols,&cols,&vals);
        for(int j=0; j < ncols ; j++){
            MatSetValue(m_output,i,cols[j]+1, vals[j],INSERT_VALUES) ;    //+1 shifts the columns 1 to the right
        }
        MatRestoreRow(m_positive_class,i,&ncols,&cols,&vals);
    }

    for(i =0; i < num_row_maj ; i++){
        MatSetValue(m_output, i + num_row_min, 0, -1, INSERT_VALUES);        //Insert negative lable
        MatGetRow(m_negative_class,i,&ncols,&cols,&vals);
        for(int j=0; j < ncols ; j++){
            MatSetValue(m_output, i + num_row_min, cols[j]+1, vals[j],INSERT_VALUES) ;    //+1 shifts the columns 1 to the right
        }
        MatRestoreRow(m_negative_class,i,&ncols,&cols,&vals);
    }
    MatAssemblyBegin(m_output, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_output, MAT_FINAL_ASSEMBLY);
    if(destroy_input_matrices){
        MatDestroy(&m_positive_class);                             //release the separated class of test data
        MatDestroy(&m_negative_class);
    }

#if dbl_KF_CTD > 7
    PetscViewer     viewer_testdata;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Coarsest_test_data.dat",FILE_MODE_WRITE,&viewer_testdata);
    MatView(m_output,viewer_testdata);
    PetscViewerDestroy(&viewer_testdata);

    printf("[KF][CTC] total output Matrix:\n");                                               //$$debug
    MatView(m_output ,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif
    t_all.stop_timer("[KF][CTC]");
}





void k_fold::write_output(std::string f_name, Mat m_Out, std::string desc){    //write the output to file
    PetscViewer     viewer_data_;
//    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_name.c_str(),FILE_MODE_WRITE,&viewer_data_);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_name.c_str(), FILE_MODE_WRITE,&viewer_data_);
    MatView(m_Out,viewer_data_);
//    PetscPrintf(PETSC_COMM_WORLD,"\nOutput matrix is written to file %s\n\n",f_name);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer
#if dbl_KF_WOUT >= 1
    std::cout << "[KF][WOUT] "<< desc <<" matrix is successfully written to " << f_name << std::endl;
#endif
}







/*
 * the input matrices are loaded m_min_full_data,m_maj_full_data
 * the test data is created and save to file and load later, so we destroy the matrix
 * the full NN indices and dists are read and filtered,
 * the filtered NN is used to create the WA matrices, and the NN data could be removed #memory optimization
 *
 */
void k_fold::prepare_data_for_iteration(int current_iteration,int total_iterations,
                        Mat& m_min_full_data,Mat& m_min_train_data,Mat& m_min_full_NN_indices,Mat& m_min_full_NN_dists,Mat& m_min_WA,Vec& v_min_vol,
                        Mat& m_maj_full_data,Mat& m_maj_train_data,Mat& m_maj_full_NN_indices,Mat& m_maj_full_NN_dists,Mat& m_maj_WA,Vec& v_maj_vol,
                        bool debug_status){
    // - - - - - cross fold the data for positive class - - - - -
    Mat m_min_test_data;
    std::unordered_set<PetscInt> uset_min_test_idx;
    PetscInt    size_min_full_data;
    MatGetSize(m_min_full_data, &size_min_full_data, NULL); //get the size of minority class
    PetscInt * arr_min_idx_train;
    PetscMalloc1(size_min_full_data, &arr_min_idx_train);   //it should be allocate now, not inside the function https://goo.gl/SbGk0y
    PetscInt min_train_size=0;                              //pass by reference
    std::vector<PetscInt> v_min_full_idx_train_dix;
    bool debug_flg_CVC_min=false;
    if(debug_status) debug_flg_CVC_min=true;

    cross_validation_class(current_iteration, total_iterations, m_min_full_data,
                           m_min_train_data, m_min_test_data, arr_min_idx_train, min_train_size,
                           uset_min_test_idx,v_min_full_idx_train_dix, "minority", this->min_shuffled_indices_,debug_flg_CVC_min);
#if dbl_exp_train_data ==1      //only for comparison with other solvers, not part of normal process
    write_output(Config_params::getInstance()->get_p_e_k_train_data_f_name() , m_min_train_data, "minority data");
#endif
#if dbl_KF_PDFI >= 1
    std::cout << "[KF][PDFI] min, train_size: " << min_train_size << std::endl;
#endif

    // - - - - - cross fold the data for negative class - - - - -
    Mat m_maj_test_data;
    std::unordered_set<PetscInt> uset_maj_test_idx;
    PetscInt    size_maj_full_data;
    MatGetSize(m_maj_full_data, &size_maj_full_data, NULL); //get the size of majority class
    PetscInt * arr_maj_idx_train;
    PetscMalloc1(size_maj_full_data, &arr_maj_idx_train);   //it should be allocate now, not inside the function https://goo.gl/SbGk0y
    PetscInt maj_train_size=0;
    std::vector<PetscInt> v_maj_full_idx_train_dix;
    cross_validation_class(current_iteration, total_iterations, m_maj_full_data,
                           m_maj_train_data, m_maj_test_data, arr_maj_idx_train, maj_train_size,
                           uset_maj_test_idx,v_maj_full_idx_train_dix, "majority", this->maj_shuffled_indices_);
#if dbl_exp_train_data ==1      //only for comparison with other solvers, not part of normal process
    write_output(Config_params::getInstance()->get_n_e_k_train_data_f_name(), m_maj_train_data, "majority data");
#endif
#if dbl_KF_PDFI >= 1
    std::cout << "[KF][PDFI] maj, train_size: " << maj_train_size << std::endl;
#endif

    // - - - - -  Combine Test Data - - - - -
    Mat m_test_data;
    combine_two_classes_in_one(m_test_data, m_min_test_data, m_maj_test_data );
    write_output(Config_params::getInstance()->get_test_ds_f_name(), m_test_data, "test data");
    MatDestroy(&m_test_data);


    //for comparison with other methods, I need to export the training data and test data
    //I am not going to clean them in the end of the program, since I need them
    // #comment_on_release_version
//    write_output(Config_params::getInstance()->get_p_e_k_train_data_f_name(), m_min_train_data);
//    write_output(Config_params::getInstance()->get_n_e_k_train_data_f_name(), m_maj_train_data);



    // - - - - -  Filter flann regard to training data points - - - - -
    // -------- filter training NN data from total NN data ---------
    // load the NN data, get the training indices, filter the points related to training indices with enough neighbors
    Mat m_min_filtered_indices, m_min_filtered_dists, m_maj_filtered_indices, m_maj_filtered_dists;
    filter_NN(m_min_full_NN_indices,m_min_full_NN_dists,uset_min_test_idx,arr_min_idx_train,min_train_size,
              v_min_full_idx_train_dix,m_min_filtered_indices, m_min_filtered_dists,"minority",true);
//if(current_iteration == 1) exit(1);
    filter_NN(m_maj_full_NN_indices,m_maj_full_NN_dists,uset_maj_test_idx,arr_maj_idx_train,maj_train_size,
              v_maj_full_idx_train_dix,m_maj_filtered_indices, m_maj_filtered_dists,"majority");

    Loader ld;
    bool debug_flg=false;
    if(debug_status) debug_flg=true;
    ld.create_WA_matrix(m_min_filtered_indices,m_min_filtered_dists,m_min_WA,"minority",debug_flg);
    if(debug_status){
        std::cout << "[KF][PDFI] debug is on and exit!: " << std::endl;
        exit(1);
    }

    ld.create_WA_matrix(m_maj_filtered_indices,m_maj_filtered_dists,m_maj_WA,"majority");

    v_min_vol = ld.init_volume(1,min_train_size);
    v_maj_vol = ld.init_volume(1,maj_train_size);
}



void k_fold::filter_NN(Mat& m_full_NN_indices, Mat& m_full_NN_dists, std::unordered_set<PetscInt>& uset_test_indices,
                       PetscInt * arr_train_indices, PetscInt& train_size, std::vector<PetscInt>& v_full_idx_to_train_idx,
                       Mat& m_filtered_NN_indices, Mat& m_filtered_NN_dists, const std::string& info,bool debug_status){
    ETimer t_all;
    int count_index, count_col_id = 0;
    const int required_num_NN = Config_params::getInstance()->get_nn_number_of_neighbors();

#if dbl_KF_FN >= 3
    PetscInt num_row_debug=0, num_col_debug=0;
    std::cout << "[KF][FilterNN] class:" << info << std::endl;
    MatGetSize(m_full_NN_indices,&num_row_debug,&num_col_debug);
    printf("[KF][FilterNN] number of rows in full NN indices matrix: %d, num_col_debug: %d \n",num_row_debug,num_col_debug);  //$$debug
    MatGetSize(m_full_NN_dists,&num_row_debug,&num_col_debug);
    printf("[KF][FilterNN] number of rows in full NN dists matrix: %d, num_col_debug: %d \n",num_row_debug,num_col_debug);  //$$debug
    printf("[KF][FilterNN] train_size: %d, required_num_NN:%d \n",train_size,required_num_NN);  //$$debug
    printf("[KF][FilterNN] arr_train_indices[1]: %d\n",arr_train_indices[1]);  //$$debug
#endif

    PetscInt i=0, idx_ncols, dis_ncols;
    const PetscInt    *idx_cols, *dis_cols;
    const PetscScalar *idx_vals, *dis_vals;
    //WARNING: the indices are changed as the data is divided to train and test,
    //the indices in the m_full_NN_indices are not valid and needs to be mapped
    //MatCreate row: train_size, col: required_num_NN  ,nnz: required_num_NN
    MatCreateSeqAIJ(PETSC_COMM_SELF,train_size , required_num_NN , required_num_NN, PETSC_NULL, &m_filtered_NN_indices);
    MatCreateSeqAIJ(PETSC_COMM_SELF,train_size , required_num_NN , required_num_NN, PETSC_NULL, &m_filtered_NN_dists);

    std::cout << "[KF][FilterNN] filtering"<< std::endl;
    for(i=0; i< train_size; i++){
        MatGetRow(m_full_NN_indices,arr_train_indices[i], &idx_ncols,&idx_cols,&idx_vals);
        MatGetRow(m_full_NN_dists,arr_train_indices[i], &dis_ncols,&dis_cols,&dis_vals);
        count_index = 0;
        count_col_id =0;


//        full row --> filtered row
        int full_row_index= arr_train_indices[i];
#if dbl_KF_FN >= 0
        std::cout << "full_row_index:"<<full_row_index << " ------>  filtered_row_index:" << i <<std::endl;
#endif
        for(int index_col=0; index_col < idx_ncols ; index_col++){
            if(uset_test_indices.find(idx_vals[index_col]) == uset_test_indices.end()){
                if(dis_cols[0] != 0 && count_col_id == 0)            //check if the first dist is zero, then jump to the right index
                    count_col_id = dis_cols[0];
        //it is not in test data(equal to end of unordered_set means not found)
        //therefore, it is in training data
        //filtered row index:i, col:count, value: index of full data converted to index in train data using
        // i is the filtered (train) row idx
        // arr_train_indices[i]  is the full idx
        //full row --> filtered row,
        //idx:(i,j)        (i,j)
        //dis:(i,d)        (i,d)

#if dbl_KF_FN >= 0
                double full_NN_idx = idx_vals[index_col];                                     //debug
                double filtered_NN_idx = v_full_idx_to_train_idx[idx_vals[index_col]];

                std::cout << "index:(" << idx_cols[index_col] << "," <<full_NN_idx <<") \t "<<
                             index_col<< "(" << count_index << "," << filtered_NN_idx <<")\n";

                if(count_col_id < dis_ncols) //the distance zero reduces the size of distance row
                    std::cout << "dist:(" << dis_cols[index_col] << "," << dis_vals[index_col] <<") \t "<<
                             "(" << count_col_id << "," << dis_vals[index_col] <<")\n";
#endif

                MatSetValue(m_filtered_NN_indices, i, count_index, v_full_idx_to_train_idx[idx_vals[index_col]], INSERT_VALUES);
                // -row, col are same as above, the value is corresponding distance at same location which is dis_vals[index_col]

                // -The distance between a point to itself is zero
                //      the sparse format in the filter_NN ignores it and the first distance is not for count 0
                //      we use the col index of indices for the distances and stop before reaching the end of distances
                if(count_col_id < dis_ncols) //the distance zero reduces the size of distance row
                    MatSetValue(m_filtered_NN_dists, i, count_col_id, dis_vals[index_col], INSERT_VALUES);
                ++count_index;
                ++count_col_id;

                if(count_index == (required_num_NN )) break;     //stop adding more NN for this row(data point)
            }
        }

        MatRestoreRow(m_full_NN_dists,arr_train_indices[i], &dis_ncols,&dis_cols,&dis_vals);
        MatRestoreRow(m_full_NN_indices,arr_train_indices[i], &idx_ncols,&idx_cols,&idx_vals);
    }
    MatAssemblyBegin(m_filtered_NN_indices, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_filtered_NN_indices, MAT_FINAL_ASSEMBLY);

    MatAssemblyBegin(m_filtered_NN_dists, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_filtered_NN_dists, MAT_FINAL_ASSEMBLY);
//    MatDestroy(&m_full_NN_indices);                             //do not release them since they will release in the end of main.cc
//    MatDestroy(&m_full_NN_dists);
    PetscFree(arr_train_indices);     //memory allocated in cross_validation_class
#if dbl_KF_FN >= 0
    printf("[KF][FN] Matrix of filtered NN indices:\n");                                               //$$debug
    MatView(m_filtered_NN_indices,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    printf("[KF][FN] Matrix of filtered NN dists:\n");                                               //$$debug
    MatView(m_filtered_NN_dists,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif

    t_all.stop_timer("[KF][FN] filtering NN for the class of",info);
}



void k_fold::free_resources(){
    MatDestroy(&this->m_min_data_);
    MatDestroy(&this->m_maj_data_);

}
