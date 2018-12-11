#!/bin/bash


echo "Make all the tools and main application from source code"
make clean
make mlsvm_save_knn 
make mlsvm_csv_petsc

# This is required for FLANN installation path which are different on HPC
out=`hostname | grep palmetto | wc -l` 
if [ $out -eq 1 ] 
then
  #echo "This is on the Palmetto Cluster"
  make mlsvm_knn -f Makefile_palmetto 
else
  make mlsvm_knn 
fi

make mlsvm_zscore 
make mlsvm_save_knn 
make mlsvm_predict 
make mlsvm_csv_petsc 
make mlsvm_libsvm_petsc 
make

echo "All targets are made successfully!"
