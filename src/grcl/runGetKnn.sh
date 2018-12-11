#!/bin/bash

# Author: Ehsan Sadrfaridpour
# Date: Oct 4, 2018
# Purpose: run knn on multiple training data for evaluating 
#          the SO features for graph classification

#ds_path=/scratch3/esadrfa/graphEmbData/net2017data/
ds_name=full_dr100_
log_path=/scratch2/esadrfa/mlsvm_logs/grcl/
mkdir -pv $log_path
tmp_path="/local_scratch/"`ls  /local_scratch | grep pbs`"/data/"

#echo "Path: "$ds_path
echo "Temp Path: "$tmp_path
echo "File name: "$ds_name

#input_params=" --ds_p "$ds_path"  --tmp_p "$tmp_path
input_params=" --ds_p "$tmp_path"  --tmp_p "$tmp_path
echo $input_params

classLargeMed=( )

for cl in ${classLargeMed[@]};
do  
  echo "class id: "$cl
  echo " * * * * * "" Calculating the k-nearest neighbors"" * * * * * "
  ./mlsvm_knn $input_params -f  $ds_name$cl  >  $log_path"/"$ds_name$cl"_knn.log" 2>&1
done

