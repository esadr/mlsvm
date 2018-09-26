#include "preprocessor.h"
#include "common_funcs.h"
#include <cassert>

Mat& Preprocessor::readData(const char * f_name){
    PetscViewer     viewer;               /* viewer */

//    Open binary file.  Note that we use FILE_MODE_READ to indicate reading from this file.
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_name,FILE_MODE_READ,&viewer);

    MatCreate(PETSC_COMM_WORLD, &data_mat_);        //Create Matrix WA : Weighted Adjancency
    MatSetType(data_mat_,MATSEQAIJ);                 // Set the type of matrix to sparse sequential
    MatLoad(data_mat_, viewer);         //load the matrix with viewer that have the file information
    PetscViewerDestroy(&viewer);        //destroy the viewer

//    printf("data_mat Matrix:\n");                                               //$$debug
//    MatView(data_mat_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

    return data_mat_;
}

/*
 * Notice 1: the data should not be saved in CSR format for since it would be hard to use optimized methods on each column
 * Notice 2: if data is separated, both (multiple) parts whould be considered in min, max, std.
 * the above have not fixed 050717-1330
 */
Mat Preprocessor::normalizeDataZscore(Mat& raw_mat){

    Mat norm_data_;
    PetscInt i, j, num_row, num_col, ncols;
    PetscInt num_nnz=0;           //number of non zero components
    PetscScalar sigma_val=0, mean=0;
    PetscScalar std=0;                  //standard deviation
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;
    float variance_= 0;

//Calculate the Mean
    MatGetSize(raw_mat,&num_row,&num_col);
    for (i=0;i<num_row;++i){
        MatGetRow(raw_mat,i,&ncols,&cols,&vals); //ncols : number if non-zeros in the row
            for (j=0; j<ncols; j++) {
                ++num_nnz;
                sigma_val += vals[j];
            }
        MatRestoreRow(raw_mat,i,&ncols,&cols,&vals);
    }
    mean = sigma_val / num_nnz;                     //calculate the Mean for all non-zero values
    printf("sigma : %g, num_nnz: %d\n",sigma_val,num_nnz);
    printf("mean is :%g\n",mean);       //$$debug

//Calculate the Standard Deviation
    sigma_val = 0;
    for (i=0;i<num_row;++i){
        MatGetRow(raw_mat,i,&ncols,&cols,&vals); //ncols : number if non-zeros in the row
            for (j=0; j<ncols; j++) {
                variance_ = vals[j] - mean;
                sigma_val +=  variance_ * variance_ ;
            }
        MatRestoreRow(raw_mat,i,&ncols,&cols,&vals);
    }
    std = sqrt(sigma_val / num_nnz);
    printf("std is :%g\n",std);       //$$debug

//Calculate the Z score

    MatDuplicate(raw_mat,MAT_SHARE_NONZERO_PATTERN,&norm_data_); //duplicate the structure, not copy the values

    for (i=0;i<num_row;++i){
        MatGetRow(raw_mat,i,&ncols,&cols,&vals); //ncols : number if non-zeros in the row
            for (j=0; j<ncols; j++) {
//                (vals[j] - mean) / std
                MatSetValue(norm_data_,i,cols[j], ( (vals[j]-mean) / std  ) ,INSERT_VALUES);
            }
        MatRestoreRow(raw_mat,i,&ncols,&cols,&vals);
    }
    //Assembly Matrices
    MatAssemblyBegin(norm_data_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(norm_data_,MAT_FINAL_ASSEMBLY);
//    printf("norm_data Matrix:\n");                                               //$$debug
//    MatView(norm_data_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    return norm_data_;
}





/*
 * 080317-1312
 * Input matrix should be traspose of real data which means each row should
 * Each column is normalized separately
 * return matrix is in CSC format
 */
Mat Preprocessor::normalizeDataZscore_Transposed(Mat& m_raw_data){

    Mat m_norm_data_;
    PetscInt i, j, num_row, num_col, ncols;
//    PetscInt num_nnz=0;           //number of non zero components
    const PetscInt    *cols;                        //if not NULL, the column numbers
    const PetscScalar *vals;

//    printf("m_raw_data transpose:\n");                                        //$$debug
//    MatView(m_raw_data,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug

    MatGetSize(m_raw_data,&num_row,&num_col);

    Vec v_sum_;
    VecCreateSeq(PETSC_COMM_SELF, num_row, &v_sum_);
    MatGetRowSum(m_raw_data, v_sum_);
//    printf("vector sum:\n");                                               //$$debug
//    VecView(v_sum_,PETSC_VIEWER_STDOUT_WORLD);                                  //$$debug

//Calculate the Mean
    std::vector<double> v_mean(num_row,0);

    PetscScalar     *arr_sum_;
    PetscMalloc1(num_row, &arr_sum_);

    VecGetArray(v_sum_,&arr_sum_);
    for(i = 0; i< num_row; i++){
        v_mean[i] = arr_sum_[i] / double(num_col);
//        std::cout << "mean[i]:"    << v_mean[i] << std::endl;
    }
    VecRestoreArray(v_sum_,&arr_sum_);
    PetscFree(arr_sum_);
    VecDestroy(&v_sum_);


//Calculate the Standard Deviation
    std::vector<double> v_std(num_row,0);
    std::cout << "num_col:" << num_col << std::endl;
    for (i=0;i<num_row;++i){
        double variance_ = 0;
        MatGetRow(m_raw_data,i,&ncols,&cols,&vals); //ncols : number if non-zeros in the row
        for (j=0; j<ncols; j++) {
                variance_ += pow((vals[j] - v_mean[i]), 2);
            }
        // #bug  the zero values might be far away from mean, but are not considered in the internal loop
        // Fix: add the variance for zero values in raw data:  sum( Z*(0-mean)^2)
        variance_ += (num_col - ncols) * pow(v_mean[i], 2);
        v_std[i] = sqrt(variance_ / (double) (num_col-1));
//        std::cout << "ncols:" << ncols <<",std[i]:"    << v_std[i] << std::endl;
        MatRestoreRow(m_raw_data,i,&ncols,&cols,&vals);
    }

//Calculate the Z score

//    MatDuplicate(m_raw_data,MAT_SHARE_NONZERO_PATTERN,&m_norm_data_); //duplicate the structure, not copy the values

    //the normalized data could be dense while the raw data is sparse
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row ,num_col ,num_col,PETSC_NULL, &m_norm_data_);
    for (i=0;i<num_row;++i){
        MatGetRow(m_raw_data,i,&ncols,&cols,&vals); //ncols : number if non-zeros in the row
//            for (j=0; j<ncols; j++) {
        int tmp_nnz_indices=0;
        double norm_value = 0;
        for (j=0; j<num_col; j++) {
            if(j < ncols && j==cols[tmp_nnz_indices]){
                norm_value = (vals[j] - v_mean[i]) / v_std[i];
                tmp_nnz_indices++;
            }else{
                norm_value = -(v_mean[i]) / v_std[i];
            }
            MatSetValue(m_norm_data_,i,j, norm_value,INSERT_VALUES);
        }
        MatRestoreRow(m_raw_data,i,&ncols,&cols,&vals);
    }
    //Assembly Matrices
    MatAssemblyBegin(m_norm_data_,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_norm_data_,MAT_FINAL_ASSEMBLY);
//    printf("norm_data Matrix:\n");                                               //$$debug
//    MatView(m_norm_data_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    return m_norm_data_;
}


/*
 * Notice the end is not part of return,
 * example:
 *  start=3, end=7  -> the subMatrix includes row 3, 4, 5, 6.
 *  The 7th row is not included.
 *
 */
void Preprocessor::getSubMatrixByRange(Mat& in_mat, Mat& subMatrix,
                                       PetscInt start, PetscInt end){
    PetscInt        i=0, num_row=0, num_col=0, range= end-start;
    PetscInt        *arr_ind;
    IS              isrow;

    MatGetSize(in_mat,&num_row, &num_col);
    printf("[PP][getSubMatrixByRange] input matrix dim: (%d,%d) \n",
           num_row, num_col);

    PetscMalloc1(range, &arr_ind);
    for (i=start; i < end; i++){
        arr_ind[i - start] = i;
    }

    ISCreateGeneral(PETSC_COMM_SELF,range,arr_ind,PETSC_COPY_VALUES,&isrow);
    PetscFree(arr_ind);


    MatGetSubMatrix(in_mat,isrow, NULL,MAT_INITIAL_MATRIX,&subMatrix);
    MatGetSize(subMatrix,&num_row, &num_col);
    assert(num_row == range && "submatrix size doesn't match with request");

//    MatView(subMatrix,PETSC_VIEWER_STDOUT_WORLD);               //$$debug
    ISDestroy(&isrow);
}

/*
 * Notice the end is not part of return,
 * example:
 *  start=3, end=7  -> the subVector includes row 3, 4, 5, 6.
 *  The 7th row is not included.
 *
 */
void Preprocessor::getSubVectorByRange(Vec& in_vec, Vec& subVector,
                                       PetscInt start, PetscInt end){
    PetscInt        i=0, num_row=0, range= end-start;
    PetscInt        *arr_ind;
    IS              isrow;

    VecGetSize(in_vec,&num_row);
    printf("[PP][getSubVectorByRange] input vector length: %d \n",
           num_row);

    PetscMalloc1(range, &arr_ind);
    for (i=start; i < end; i++){
        arr_ind[i - start] = i;
    }

    ISCreateGeneral(PETSC_COMM_SELF,range,arr_ind,PETSC_COPY_VALUES,&isrow);
    PetscFree(arr_ind);


    VecGetSubVector(in_vec, isrow, &subVector);
    VecGetSize(subVector,&num_row);
    assert(num_row == range && "subvector size doesn't match with request");

//    MatView(subMatrix,PETSC_VIEWER_STDOUT_WORLD);               //$$debug
    ISDestroy(&isrow);
}


//void Preprocessor::separateData(Mat& data, const char * f_name){
//    PetscViewer     viewer;               /* viewer */
//    PetscInt        i=0, lbl_size=0, lbl_rstart=0 ,lbl_rend=0;
//    PetscScalar     * lbl_val;              // array of scalar to read the local vector into it
//    PetscInt        *ind_min, *ind_maj;
//    IS              isrow_min_, isrow_maj_;         //minority is coded as P
//    PetscInt        min_index_ =0 , maj_index_=0;

//    PetscMPIInt     rank,size;
//    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
//    MPI_Comm_size(PETSC_COMM_WORLD,&size);

////    Open binary file.  Note that we use FILE_MODE_READ to indicate reading from this file.

//    printf("fname : %s\n", f_name);

////    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_name,FILE_MODE_READ,&viewer);
//    PetscViewerHDF5Open(PETSC_COMM_WORLD,f_name,FILE_MODE_READ,&viewer);

//    VecCreate(PETSC_COMM_WORLD, &label_);                       //Create vector of labels
//    PetscObjectSetName((PetscObject) label_,"/lbl");
//    VecLoad(label_, viewer);         //load the matrix with viewer that have the file information
//    PetscViewerDestroy(&viewer);        //destroy the viewer

////    printf("label vector:\n");                                               //$$debug
////    VecView(label_,PETSC_VIEWER_STDOUT_WORLD);                                  //$$debug

//    VecGetSize(label_,&lbl_size);           //get the size of vector of labels
//    VecGetOwnershipRange(label_,&lbl_rstart,&lbl_rend);    //get the local range of the vector
//    VecGetArray(label_,&lbl_val);


//    PetscMalloc1(lbl_size, &ind_maj);
//    PetscMalloc1(lbl_size, &ind_min);


//    for (i=0; i < lbl_size; ++i){
//        if(lbl_val[i] == 1){         //if the label is 1, it means it's minority
//            ind_min[min_index_] = i;
//            ++min_index_;
//        }
//        else
//        {
//            ind_maj[maj_index_] = i;
//            ++maj_index_;
//        }
//    }

//    ISCreateGeneral(PETSC_COMM_SELF,min_index_,ind_min,PETSC_COPY_VALUES,&isrow_min_);
//    ISCreateGeneral(PETSC_COMM_SELF,maj_index_,ind_maj,PETSC_COPY_VALUES,&isrow_maj_);
//    PetscFree(ind_min);
//    PetscFree(ind_maj);


//    MatGetSubMatrix(data,isrow_min_, NULL,MAT_INITIAL_MATRIX,&minority_);
//    MatGetSubMatrix(data,isrow_maj_, NULL,MAT_INITIAL_MATRIX,&majority_);
////    printf("minority Matrix:\n");                                               //$$debug
////    MatView(minority_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug


//    ISDestroy(&isrow_min_);
//    ISDestroy(&isrow_maj_);
//}


//void Preprocessor::export_data(){
//    /*
//     * write the minority and majority matrices to files with hdf5 format
//     * that later flann can read them
//     */
//    PetscViewer     viewer_min_,viewer_maj_;               /* viewer */
//    PetscObjectSetName((PetscObject) minority_,"/dataset");
////    MatLoad(minority_,viewer_min_);
//    PetscViewerHDF5Open(PETSC_COMM_WORLD,"dt_min.h5",FILE_MODE_WRITE,&viewer_min_);



//    PetscViewerHDF5Open(PETSC_COMM_WORLD,"dt_maj.h5",FILE_MODE_WRITE,&viewer_maj_);
//    PetscObjectSetName((PetscObject) majority_,"/dataset");
////    MatLoad(minority_,viewer_min_);

//}
































