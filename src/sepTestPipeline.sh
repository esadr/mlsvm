: '
Author: Ehsan Sadrfaridpour
Date: Aug 31, 2018
Purpose: benchmark performance


This loads a Libsvm data file, process it and generate final predictions for mlsvm
It uses a separate testdata to evaluate the performance of the model.

Parameters:
1- random seed
2- path to all files involve in the process
3- LibSVM file format as input raw data
4- training size 			# e.g. 4M
5- total number of raw data points 	# e.g. 5M
6- validation ratio 			# e.g. 0.1
'

#!/bin/bash

# - - - set the log file - - -
log_file=$2/"pipeline_"$3"_s"$1"_"`/bin/date +%m%d%y_%H%M`".log"
date > $log_file
echo "seed:"$1 >> $log_file
#==============================================================================
# - - - shuffle the data - - -
get_seeded_random()
{
  seed="$1"
  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
    </dev/zero 2>/dev/null
}
shuf --random-source=<(get_seeded_random $1) < $2"/"$3".libsvm"\
       	> $2/raw_data.libsvm
#==============================================================================
echo "*** convert libsvm 2 petsc ***" >> $log_file
./mlsvm_libsvm_petsc --ds_p $2 -f $3  >> $log_file 2>&1
#==============================================================================
echo "*** z-score ***" >> $log_file
./mlsvm_zscore --ds_p $2 -f $3  >> $log_file 2>&1 
#==============================================================================
echo "*** get the test data ***" >> $log_file
./petsc_utility/getSubMatrix -i $2/$3"_zsc_data.dat" -s $4 -e $5 -p $2 \
	-w "susy_"s$4_e$5"_test_zsc_data.dat" >> $log_file 2>&1
./petsc_utility/getSubVector -i $2/$3"_label.dat" -s $4 -e $5 -p $2 \
	-w "susy_"s$4_e$5"_test_label.dat" >> $log_file 2>&1
#==============================================================================
echo "*** merge test label and data to one file ***" >> $log_file
./petsc_utility/mergeLabel2Data -p $2 -d "susy_"s$4_e$5"_test_zsc_data.dat" \
	-l "susy_"s$4_e$5"_test_label.dat" \
	-w "susy_"s$4_e$5"_label_data_test.dat"

#==============================================================================
echo "*** get the train data ***" >> $log_file
scale=$(($4 / 10))
for i in `seq 1 1 10` 
do  
  size=$((i*$scale)); echo $size
  ./petsc_utility/getSubMatrix -i $2/$3"_zsc_data.dat" -s 0 -e $size -p $2 \
	  -w "susy_"$size"_train_zsc_data.dat" >> $log_file 2>&1
  ./petsc_utility/getSubVector -i $2/$3"_label.dat" -s 0 -e $size -p $2 \
	  -w "susy_"$size"_train_label.dat" >> $log_file 2>&1
done  
#==============================================================================
echo "*** get the knn ***" >> $log_file
scale=$(($4 / 10))
for i in `seq 1 1 10` 
do  
  size=$((i*$scale)); echo $size
  input_params=" --ds_p "$2"  --tmp_p "$2"  -f susy_"$size"_train"
  echo "input params:"$input_params >> $log_file 2>&1
  ./mlsvm_knn $input_params >> $log_file 2>&1
done
#==============================================================================
echo "*** get the mlsvm ***" >> $log_file
scale=$(($4 / 10))
for i in `seq 1 1 10` 
do  
  size=$((i*$scale)); echo $size
  input_params=" --ds_p "$2"  --tmp_p "$2"  -f susy_"$size"_train  -s "$1
  echo "input params:"$input_params >> $log_file 2>&1
  (time ./mlsvmSepTestClassifier $input_params \
	  --test_data "susy_"s$4_e$5 \
	  -r 1 \
	  --cs_we 0.001 \
	  --mv_id 2 \
	  -v $6 ) \
	  >  $2"/results_susy"$size"_s"$1"_"`/bin/date +%m%d%y_%H%M` 2>&1

done


echo "*** Everything is finished successfully ***" >> $log_file

pushd $TMPDIR/data/
result_path=~/es/results/v1.1.1susySepTest/

echo "*** Copy the final results to $result_path ***" >> $log_file
for i in `ls | grep result`; do echo $i && cp $i $result_path ; done
cp $log_file $result_path 

date >> $log_file
