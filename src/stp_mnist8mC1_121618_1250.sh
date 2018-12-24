: '
Author: Ehsan Sadrfaridpour
Date: Dec 16, 2018
Purpose: benchmark performance on MNIST8m class 1


This loads a Libsvm data file, process it and generate final predictions for mlsvm
It uses a separate testdata to evaluate the performance of the model.

Parameters:
1- random seed
2- path to all files involve in the process
3- LibSVM file format as input raw data
4- training size 			# e.g. 4M
5- total number of raw data points 	# e.g. 5M
6- validation ratio 			# e.g. 0.1
7- class id                             # e.g. 1
'

#!/bin/bash

s=$1 dsp=$2 fn=$3 trn=$4 tes=$5 v=$6 c=$7
i# e.g. s=42 dsp=/local_scratch/pbs.4769468.pbs02/ fn=mnist8m.scale trn=4000000 tes=5000000 v=0.1 c=1


log_path=/scratch4/esadrfa/mlsvm_logs/
# the 32-bit PETSc doesn't handle this size of matrix, I need to siwtch to 64-bit

#export export PETSC_ARCH="arch-linux-pkgs-64idx"
#make clean && make 
#rm ./mlsvm_libsvm_petsc mlsvmSepTestClassifier
#make mlsvm_libsvm_petsc
#make  mlsvmSepTestClassifier
#make mlsvm_knn -f Makefile_palmetto
#pushd petsc_utility 
#rm getSubMatrix getSubVector 
#make clean_make && make 
#popd 



# - - - set the log file - - -
log_file=$log_path"/pipeline_"$fn"_s"$trn"_class"$c"_"`/bin/date +%m%d%y_%H%M`".log"
date > $log_file
echo "seed:"$1 >> $log_file
echo "class id:"$7 >> $log_file
##==============================================================================
## - - - shuffle the data - - -
#get_seeded_random()
#{
#  seed="$1"
#  openssl enc -aes-256-ctr -pass pass:"$seed" -nosalt \
#    </dev/zero 2>/dev/null
#}
#shuf --random-source=<(get_seeded_random $1) < $2/$3 > $2/raw_data.libsvm
#head -n $5 $2/raw_data.libsvm > $2/filt.libsvm
##==============================================================================
#echo "*** convert libsvm 2 petsc ***" >> $log_file
#./mlsvm_libsvm_petsc --ds_p $2 -f filt >> $log_file 2>&1
##==============================================================================
##echbo "*** z-score ***" >> $log_file
##./mlsvm_zscore --ds_p $2 -f $3  >> $log_file 2>&1 
#==============================================================================
#echo "*** get the test data ***" >> $log_file
#./petsc_utility/getSubMatrix -i $dsp/"filt_data.dat" -s $trn -e $tes -p $dsp \
#     -w $fn"_"s$trn_e$tes"_test_zsc_data.dat" >> $log_file 2>&1

## ----------- Extra steps to convert multi class labels to  binary ------------
## I called the 
## python ConvertLabelsMnist.py -d /local_scratch/pbs.4769468.pbs02/ -f filt_mult_class_label.csv -l $7

#cp $dsp/"mnist4m_c"$7"_label.dat" $dsp/$fn"_label.dat"

./petsc_utility/getSubMatrix -i $dsp/"filt_data.dat" -s $trn -e $tes -p $dsp \
   -w $fn"_"s$trn"_e"$tes"_test_zsc_data.dat"
./petsc_utility/getSubVector -i $dsp/$fn"_label.dat" -s $trn -e $tes -p $dsp \
   -w $fn"_"s$trn"_e"$tes"_test_label.dat"
#==============================================================================
echo "*** merge test label and data to one file ***" >> $log_file
./petsc_utility/mergeLabel2Data -p $dsp  \
  -d $fn"_"s$trn"_e"$tes"_test_zsc_data.dat" \
  -l $fn"_"s$trn"_e"$tes"_test_label.dat"   \
  -w $fn"_"s$trn"_e"$tes"_label_data_test.dat"

#==============================================================================
echo "*** get the train data ***" >> $log_file
scale=$(($trn / 10))
for i in `seq 1 1 10` 
do  
  size=$((i*$scale)); echo $size
  ./petsc_utility/getSubMatrix -i $dsp"/filt_data.dat" -s 0 -e $size -p $dsp \
        ¦ -w $fn"_"$size"_train_zsc_data.dat" >> $log_file 2>&1
  ./petsc_utility/getSubVector -i $dsp/"mnist4m_c1_label.dat" -s 0 -e $size -p $dsp \
        ¦ -w $fn"_"$size"_train_label.dat" >> $log_file 2>&1
done  
#==============================================================================
echo "*** get the knn ***" >> $log_file
scale=$(($trn / 10))
for i in `seq 2 1 10`
do
  size=$((i*$scale)); echo $size
  input_params=" --ds_p "$dsp"  --tmp_p "$dsp"  -f"$fn"_"$size"_train"
  echo "input params:"$input_params  >> $log_file 2>&1
  ./mlsvm_save_knn $input_params >> $log_file 2>&1
done
#==============================================================================
echo "*** get the mlsvm ***" >> $log_file
scale=$(($4 / 10))
for i in `seq 1 1 1` 
do  
  size=$((i*$scale)); echo $size
  input_params=" --ds_p "$2"  --tmp_p "$2"  -f $3"_"$size"_train  -s "$1
  echo "input params:"$input_params >> $log_file 2>&1
  (time ./mlsvmSepTestClassifier $input_params \
	  --test_data $3"_"s$4_e$5 \
	  -r 1 \
	  --cs_we 0.001 \
	  --mv_id 2 \
	  -v $6 ) \
	  >  $2"/results_"$3"$size"_s"$1"_"`/bin/date +%m%d%y_%H%M` 2>&1

done


echo "*** Everything is finished successfully ***" >> $log_file

pushd $TMPDIR/data/
result_path="/home/esadrfa/es/results/v1.1.1/stp_"$3"/"
mkdir -pv $result_path

echo "*** Copy the final results to $result_path ***" >> $log_file
for i in `ls | grep result`; do echo $i && cp $i $result_path ; done
cp $log_file $result_path 

date >> $log_file
