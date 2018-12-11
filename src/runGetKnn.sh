#!/bin/bash

# Author: Ehsan Sadrfaridpour
# Date: Aug 30, 2018
# Purpose: run knn on multiple smaller training data for testing Susy dataset for journal paper


#ds_path=/scratch2/esadrfa/mlsvm_data/
log_path=/scratch2/esadrfa/mlsvm_logs/
mkdir -pv $log_path
tmp_path="/local_scratch/"`ls  /local_scratch | grep pbs`"/data/"
echo "Path: "$ds_path
echo "File name: "$ds_name

#input_params=" --ds_p "$ds_path"  --tmp_p "$tmp_path
input_params=" --ds_p "$tmp_path"  --tmp_p "$tmp_path
echo $input_params


scale=400000
#largeFile=/scratch2/esadrfa/mlsvm_data/susy_4M_train_zsc_data.dat
for i in `seq 1 1 10`
do  
  size=$((i*$scale))
  echo $size  
  #./petsc_utility/getSubMatrix -i $largeFile -s 0 -e $size -p $tmp_path -w "susy_"$size"_train_zsc_data.dat"
  ds_name="susy_"$size
  echo " * * * * * "" Calculating the k-nearest neighbors"" * * * * * "
  ./mlsvm_knn $input_params -f  "susy_"$size"_train"  >  $log_path"/"$ds_name"_knn.log" 2>&1
done

