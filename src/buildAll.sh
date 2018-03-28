#!/bin/bash

echo "Make all the tools and main application from source code"
make clean
make mlsvm_save_knn 
make mlsvm_csv_petsc 
make mlsvm_knn 
make mlsvm_zscore 
make mlsvm_save_knn 
make mlsvm_predict 
make mlsvm_csv_petsc 
make mlsvm_libsvm_petsc 
make

echo "All targets are made successfully!"
