#include "k_fold.h"
#include "loader.h"
#include <algorithm>    /* random_shuffle*/
#include <cmath>
#include "config_logs.h"
//#include "model_selection.h"

k_fold::k_fold(){
    read_in_data();
    divide_data();
    shuffle_data();
    PetscPrintf(PETSC_COMM_WORLD,"[K_fold] Data is read, divided and shuffled\n");
}

/*
 * read the input data from file and load it into m_in_data_ matrix
 * read the input label from file and load it into v_in_label_ vector
 * the ds_path and ds_name are read from config_params
 * the _data.dat, _label.dat are used to postfix to the ds_name for data and label files
 */
bool k_fold::read_in_data(){
    Loader ld;
    PetscInt data_size=0, label_size=0;
    this->m_in_data_ = ld.read_input_matrix(Config_params::getInstance()->get_ds_path()+
                                            Config_params::getInstance()->get_ds_name()+
                                            "_zsc_data.dat");
    this->v_in_label_ = ld.read_input_vector(Config_params::getInstance()->get_ds_path()+
                                             Config_params::getInstance()->get_ds_name()+
                                             "_label.dat");
    MatGetSize(this->m_in_data_, &data_size, NULL);
    VecGetSize(this->v_in_label_, &label_size);
    if(data_size == label_size){
        this->num_data_points_ = data_size;
        return true;
    }else{
        std::cout << "[k_fold] the data and label size are not match" << std::endl;
        std::cout << "[k_fold] data size is:<<"<< data_size <<" label size is:"<< label_size << std::endl;
        std::cout << "[k_fold] Exit(1)" << std::endl;
        exit(1);
        return false;
    }
}

/*
 * Only devide the input data to 2 separate classes
 * I don't need the original data and labels after this
 * so, I release them and keep m_min_data_, m_maj_data_
 *
 * The label for minority or positive class is +1
 * and for majority or negative class is -1
 */
void k_fold::divide_data(){
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

}

/*
 * Get the size of the data from local variables (num_mXX_points) which are set during the divide_data
 * Create a random sequence of numbers for them and store them in local vectors (mXX_shuffled_indices_)
 * The seed for srand comes from the config params and it is recorded in output for debug
 */
void k_fold::shuffle_data(){
    // Random generator without duplicates
    PetscInt min_iter= 0, maj_iter= 0;

    // - - - - - - minority - - - - - -
    this->min_shuffled_indices_.reserve(this->num_min_points);

    //create a vector of all possible nodes for minority class
    for (min_iter=0; min_iter < this->num_min_points; ++min_iter){
        this->min_shuffled_indices_.push_back(min_iter);
    }
#if dbl_kf_shuffle_data >= 5
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][shuffle_data] after initialize vector min\n");
    for(unsigned int i=0; i < this->min_shuffled_indices_.size(); i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", this->min_shuffled_indices_[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");    
#endif
    
    srand(std::stoll(Config_params::getInstance()->get_cpp_srand_seed()));
    std::random_shuffle( this->min_shuffled_indices_.begin(), this->min_shuffled_indices_.end() ); //shuffle all nodes

#if dbl_kf_shuffle_data >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][shuffle_data] shuffled indices for min\n");
    for(unsigned int i=0; i < this->min_shuffled_indices_.size(); i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", this->min_shuffled_indices_[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
#endif
    
    
    // - - - - - - majority - - - - - -
    this->maj_shuffled_indices_.reserve(this->num_maj_points);
    //create a vector of all possible nodes for majority class
    for (maj_iter=0; maj_iter < this->num_maj_points; ++maj_iter){
        this->maj_shuffled_indices_.push_back(maj_iter);
    }
#if dbl_kf_shuffle_data >= 5
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][shuffle_data] after initialize vector maj\n");
    for(unsigned int i=0; i < this->maj_shuffled_indices_.size(); i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", this->maj_shuffled_indices_[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");    
#endif

    
    srand(std::stoll(Config_params::getInstance()->get_cpp_srand_seed()));
    std::random_shuffle( this->maj_shuffled_indices_.begin(), this->maj_shuffled_indices_.end() ); //shuffle all nodes

#if dbl_kf_shuffle_data >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][shuffle_data] shuffled indices for maj\n");
    for(unsigned int i=0; i < this->maj_shuffled_indices_.size(); i++){
        PetscPrintf(PETSC_COMM_WORLD, "%d,", this->maj_shuffled_indices_[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "\n");
#endif
    
}

/*
 * Get the random sequence from local vectors for both class
 * At each iteration, take i-th subset of each vector and sort it
 * Then get submatrix for that as test part for that class
 * Take the rest as training for that class
 * Combine both test parts with labels as one test matrix
 * Note: iteration should start from ZERO
 */
//bool k_fold::cross_validation(int current_iteration,int total_iterations, Mat& m_min_train_data, Mat& m_maj_train_data, Mat& m_test_data){
void k_fold::cross_validation(int current_iteration,int total_iterations){
#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] starts\n");
#endif
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
        
#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] condition: C (Train_part 1 - TEST - Train_part 2)\n");
#endif
        // - - - - - Min Train P1 (Left)- - - - -
        min_train_curr = 0;
        for(min_iter= 0; min_iter< min_test_start; min_iter++){
            ind_min_train[min_train_curr] = this->min_shuffled_indices_[min_iter];
            min_train_curr++;
        }
        // - - - - - Min Test - - - - -
        min_test_curr = 0;
        for(min_iter= min_test_start; min_iter< min_test_end; min_iter++){  //min_test_end should be num_min_point
            ind_min_test[min_test_curr] = this->min_shuffled_indices_[min_iter];
            min_test_curr++;
        }
        // - - - - - Min Train 2 (Right)- - - - -
        // don't touch min_train_curr because it needs to follow the Train 1 (Left)
        for(min_iter= min_test_end; min_iter< this->num_min_points; min_iter++){  //min_test_start should be 0
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
        for(maj_iter= maj_test_start; maj_iter< maj_test_end; maj_iter++){  //maj_test_end should be num_maj_point
            ind_maj_test[maj_test_curr] = this->maj_shuffled_indices_[maj_iter];
            maj_test_curr++;
        }
        // - - - - - Maj Train 2 (Right)- - - - -
        // don't touch maj_train_curr because it needs to follow the Train 1 (Left)
        for(maj_iter= maj_test_end; maj_iter< this->num_maj_points; maj_iter++){  //maj_test_start should be 0
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
//    printf("[k_fold] [cross validated] min test data before combine_test_data Matrix:\n");     //$$debug
//    MatView(m_min_test_data ,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
//    printf("[k_fold] [cross validated] maj test data before combine_test_data Matrix:\n");     //$$debug
//    MatView(m_maj_test_data ,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

    combine_test_data(m_test_data, m_min_test_data, m_maj_test_data );

#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] test data is combined \n");
#endif

    char min_train_file_name_[PETSC_MAX_PATH_LEN]="./data/kfold_min_train.dat";
    char maj_train_file_name_[PETSC_MAX_PATH_LEN]="./data/kfold_maj_train.dat";
    char test_file_name_[PETSC_MAX_PATH_LEN]="./data/kfold_test_data.dat";
    write_output(min_train_file_name_, m_min_train_data);
    write_output(maj_train_file_name_, m_maj_train_data);
    write_output(test_file_name_, m_test_data);
   
#if dbl_kf_cross_validation >= 3
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][cross_validation] write all files successfully. \n");
#endif 
}

/*
 * I guess I should skip this in future, but for now it is very similar to what I have in the model selection
 * It gets 2 test matrices for both classes and add the labels to the first column of a new matrix
 * which contains both of them
 * Note: the dt_test_p, dt_test_n will destroy in the end of this function because they are not needed anymore
 */
void k_fold::combine_test_data(Mat& test_total, Mat& dt_test_p, Mat& dt_test_n){
    PetscInt max_num_col_;
    PetscInt num_row_min, num_row_maj, num_row;
    PetscInt i, ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    MatGetSize(dt_test_p, &num_row_min, &max_num_col_);   //set the number of columns
    MatGetSize(dt_test_n, &num_row_maj, NULL);

    num_row = num_row_min + num_row_maj;

    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][combine_test_data] num_row_min: %d, num_row_maj:%d\n",num_row_min,num_row_maj);     //$$debug
    PetscPrintf(PETSC_COMM_WORLD, "[k_fold][combine_test_data] num_row: %d, num_col:%d, nz:%d\n",num_row,max_num_col_ + 1 , max_num_col_ + 1);     //$$debug
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row ,max_num_col_ + 1 ,(max_num_col_ + 1 ),PETSC_NULL, &test_total); //+1 is for label
    for(i =0; i < num_row_min ; i++){
        MatSetValue(test_total, i, 0, +1,INSERT_VALUES);        //Insert positive lable
        MatGetRow(dt_test_p,i,&ncols,&cols,&vals);
        for(int j=0; j < ncols ; j++){
            MatSetValue(test_total,i,cols[j]+1, vals[j],INSERT_VALUES) ;    //+1 shifts the columns 1 to the right
        }
        MatRestoreRow(dt_test_p,i,&ncols,&cols,&vals);
    }

    for(i =0; i < num_row_maj ; i++){
        MatSetValue(test_total, i + num_row_min, 0, -1, INSERT_VALUES);        //Insert negative lable
        MatGetRow(dt_test_n,i,&ncols,&cols,&vals);
        for(int j=0; j < ncols ; j++){
            MatSetValue(test_total, i + num_row_min, cols[j]+1, vals[j],INSERT_VALUES) ;    //+1 shifts the columns 1 to the right
        }
        MatRestoreRow(dt_test_n,i,&ncols,&cols,&vals);
    }
    MatAssemblyBegin(test_total, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(test_total, MAT_FINAL_ASSEMBLY);
    MatDestroy(&dt_test_p);                             //release the separated class of test data
    MatDestroy(&dt_test_n);

#if dbl_k_fold_combine_test_data > 7
    PetscViewer     viewer_testdata;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,"Coarsest_test_data.dat",FILE_MODE_WRITE,&viewer_testdata);
    MatView(test_total,viewer_testdata);
    PetscViewerDestroy(&viewer_testdata);

    printf("[combine_test_data]total test Matrix:\n");                                               //$$debug
    MatView(test_total ,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
#endif

}

//void write_output(const std::string f_name, Mat m_Out){    //write the output to file
void k_fold::write_output(char f_name[PETSC_MAX_PATH_LEN], Mat m_Out){    //write the output to file
    PetscViewer     viewer_data_;


//    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_name.c_str(),FILE_MODE_WRITE,&viewer_data_);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_name, FILE_MODE_WRITE,&viewer_data_);
    MatView(m_Out,viewer_data_);
//    PetscPrintf(PETSC_COMM_WORLD,"\nOutput matrix is written to file %s\n\n",f_name);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer
}












