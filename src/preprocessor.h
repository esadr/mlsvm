#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include "ds_global.h"
#include "petscmat.h"
#include "petscvec.h"
#include <petscviewerhdf5.h>    //for hdf5

class Preprocessor{

private:
    double  mean_= 0;
    double  num_nnz = 0;
    Vec     label_;
    Mat     data_mat_;
    Mat     majority_, minority_;
public:
    Mat& readData(const char *);
    Mat normalizeDataZscore(Mat&);
    Mat normalizeDataZscore_Transposed(Mat& raw_mat);
//    void separateData(Mat&, const char *);
//    void export_data();
    void getSubMatrixByRange(Mat& in_mat, Mat& subMatrix,
                              PetscInt start, PetscInt end);
    void getSubVectorByRange(Vec& in_vec, Vec& subVector,
                              PetscInt start, PetscInt end);

};


#endif // PREPROCESSOR_H
