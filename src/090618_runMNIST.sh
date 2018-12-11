#!/bin/bash
: '
Author: Ehsan Sadrfaridpour
Date: Sep 6, 2018
Purpose: run experiment on different classes of mnist8m dataset 
which are sampled down by 50% using matlab code earlier.

Steps:
copy the data file which is common between all the classes in local harddisk 
    to a name for that class passed as $1 variable
copy the corresponding label file for the class
generate the approximate k nearest neighbors for this class using the FLANN 
run the mlsvm classifier and store the results

params:
	integer for the class id as the first input variable in $1
'

#dataset_id=$1
cp -v /scratch2/esadrfa/mlsvm_data/mnist8m_p50_zsc_data.dat $TMPDIR/mnist8m_p50_"$1"_zsc_data.dat
cp -v "/scratch2/esadrfa/mlsvm_data/mnist8m_p50_"$1"_label.dat" $TMPDIR

input_params=" --ds_p "$TMPDIR" -f  mnist8m_p50_"$1"   --tmp_p "$TMPDIR
echo "input_params are: "$input_params
#./mlsvm_knn $input_params
./mlsvm_save_knn $input_params

pushd $TMPDIR
for i in `ls | grep -v zsc | grep mnist8m `
do
  cp -v $i	/scratch2/esadrfa/mlsvm_data/mnist/
done  


exit 1
./mlsvm_classifier -x 1 $input_params -r 1 --cs_we 0.05 --rf_2nd 0 -v .1 -s 15 -e 0.001 > "~/es/results/v1.1.0_rerun_3rd_submissionAug2018/mnist8m_p50_"$1"_x1_k5_eps0.001_r1_we0.05_mv2_csbps_rf2nd0_prs5000_prm1000_v0.1_s15_"`/bin/date +%m%d%y_%H%M`".log"

