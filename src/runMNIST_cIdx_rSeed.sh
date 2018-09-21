#!/bin/bash
: '
Author: Ehsan Sadrfaridpour
Date: Sep 7, 2018
Purpose: run experiment on different classes of mnist8m dataset 
which are sampled down by 50% using matlab code earlier.

Steps:
copy the data file which is common between all the classes in local harddisk 
    to a name for that class passed as $1 variable
copy the corresponding label file for the class
generate the approximate k nearest neighbors for this class using the FLANN 
run the mlsvm classifier and store the results

params:
1- integer for the class id as the first input variable in $1
2- random seed
3- extra input parameter for the custom run
"${@:3}"   this generate the rest of parameters as one parameter instead of $3
Ref: https://stackoverflow.com/a/9057392/2674061
'

#dataset_id=$1
cp -v /scratch2/esadrfa/mlsvm_data/mnist8m_p50_zsc_data.dat $TMPDIR/mnist8m_p50_"$1"_zsc_data.dat
find /scratch2/esadrfa/mlsvm_data/mnist/ -name "*_p50_"$1"*" -exec cp -v '{}' $TMPDIR \;
cp -v "/scratch2/esadrfa/mlsvm_data/mnist8m_p50_"$1"_label.dat" $TMPDIR

input_params=" --ds_p "$TMPDIR" -f  mnist8m_p50_"$1"   --tmp_p "$TMPDIR
echo "input_params are: "$input_params
#./mlsvm_knn $input_params
#./mlsvm_save_knn $input_params

# copy the calculated kNN results back to the scrach drive
#find $TMPDIR -name "*_p50_"$1"*" -exec cp -v '{}' /scratch2/esadrfa/mlsvm_data/mnist/  \;



case $1 in
  [0-2])
    input_params=$input_params" -x 1 -k 5  -r 1 --cs_we 0.01 --rf_2nd 0 -v .1 -e 0.1 -s "$2
    ;;                          
  3)                            
    input_params=$input_params" -x 1 -k 10 -r 1 --cs_we 0.05 --rf_2nd 0 -v .1 -e 0.1 -s "$2
    ;;                          
  4)                            
    input_params=$input_params" -x 1 -k 5  -r 1 --cs_we 0.05 --rf_2nd 0 -v .1 -e 0.1 -s "$2
    ;;                          
  [5-6])                        
    input_params=$input_params" -x 1 -k 10 -r 1 --cs_we 0.05 --rf_2nd 0 -v .1 -e 0.1 -s "$2
    ;;                          
  7)                            
    input_params=$input_params" -x 1 -k 5  -r 1 --cs_we 0.01 --rf_2nd 0 -v .1 -e 0.1 -s "$2
    ;;                          
  8)                            
    input_params=$input_params" -x 1 -k 5  -r 1 --cs_we 0.05 --rf_2nd 0 -v .1 -e 0.1 -s "$2
    ;;                          
  9)                            
    input_params=$input_params" -x 1 -k 10 -r 1 --cs_we 0.05 --rf_2nd 0 -v .1 -e 0.1 -s "$2
    ;;
  *)
    echo "input class is not in the range, Exit!"
    exit 1
    ;;
esac

echo "input_params are: "$input_params "${@:3}"

./mlsvm_classifier $input_params "${@:3}"  > "/home/esadrfa/es/results/v1.1.0_rerun_3rd_submissionAug2018/mnist8m_p50_"$1"_s"$2"_"`/bin/date +%m%d%y_%H%M`".log"

