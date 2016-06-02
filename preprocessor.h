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
    void separateData(Mat&, const char *);
    void export_data();
};


#endif // PREPROCESSOR_H
