#!/bin/bash 

make mlsvm_csv_petsc
make mlsvm_libsvm_petsc
make mlsvm_knn
make mlsvm_zscore
make mlsvm_classifier
make mlsvm_predict
echo "All required modules are compiled successfully!"
