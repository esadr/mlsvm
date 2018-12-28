: '
Author: Ehsan Sadrfaridpour
Date: Dec 22, 2018
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
8- low idx                              # e.g. 1
9- high idx                             # e.g. 10
'

#!/bin/bash

#check number of parameters 
if [ "$#" -ne 9 ]; then
  echo "Usage: seed_num data_path file_unique_name train_end_idx tes_end_idx valid_ratio class_id low_idx high_idx"
  exit 1
fi  


s=$1 dsp=$2/ fn=$3 trn=$4 tes=$5 v=$6 c=$7 lidx=$8 hidx=$9
# e.g. s=42 dsp=/local_scratch/pbs.4769468.pbs02/ fn=mnist8m.scale trn=4000000 tes=5000000 v=0.1 c=1


log_path=/scratch4/esadrfa/mlsvm_logs/
# the 32-bit PETSc doesn't handle this size of matrix, I need to siwtch to 64-bit

export export PETSC_ARCH="arch-linux-pkgs-64idx"
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
log_file=$log_path"/pipeline_"$fn"_s"$s"_class"$c"_"`/bin/date +%m%d%y_%H%M`".log"
echo "logfile:"
echo $log_file
date > $log_file
echo "seed:"$s >> $log_file
echo "class id:"$c >> $log_file
echo "low id:"$lidx >> $log_file
echo "high id:"$hidx >> $log_file
#==============================================================================
# - - - shuffle the data - - -
if [ ! -f $dsp/filt_s$s.libsvm ]; then
  echo "Shuffle data" >> $log_file
  get_seeded_random()
  {
    openssl enc -aes-256-ctr -pass pass:"$s" -nosalt \
      </dev/zero 2>/dev/null
  }
  shuf --random-source=<(get_seeded_random $s) < $dsp/$fn > $dsp/raw_data_s$s.libsvm
  head -n $tes $dsp/raw_data_s$s.libsvm > $dsp/filt_s$s.libsvm
  echo "Shuffle finished successfully" >> $log_file
else
  echo "Shuffled libsvm file for "$s" random seed exist" >> $log_file
fi
#==============================================================================
if [ ! -f $dsp/"filt_s"$s"_data.dat" ]; then
  echo "convert libsvm to petsc " >> $log_file
  ./mlsvm_libsvm_petsc --ds_p $dsp -f filt_s$s >> $log_file 2>&1
else
  echo "the petsc format exist, nothing to convert!" >> $log_file
fi
#==============================================================================
if [ ! -f $dsp/$fn"_s"$trn"_e"$tes"_test_zsc_data.dat" ]; then
  echo "*** get the test data ***" >> $log_file
  ./petsc_utility/getSubMatrix -i $dsp/"filt_s"$s"_data.dat" -s $trn -e $tes -p $dsp \
     -w $fn"_s"$trn"_e"$tes"_test_zsc_data.dat" >> $log_file 2>&1
fi
## ----------- Extra steps to convert multi class labels to  binary ------------
if [ ! -f $dsp"/filt_mult_class_label_s"$s".csv" ]; then
  cat $dsp/filt_s$s.libsvm | awk -F" " '{print $1}' \
    > $dsp"/filt_mult_class_label_s"$s".csv"
fi

if [ ! -f $dsp"/mnist_label_s"$s"_c"$c".dat" ]; then
## I called the  the filt_mult_class_label_sX.csv is the file with all the classes
  python ./datasets/ConvertLabelsMnist.py -d $dsp \
   -f "filt_mult_class_label_s"$s".csv" -l $c -s $s >> $log_file 2>&1
fi

if [ ! -f $dsp/$fn"_s"$trn"_e"$tes"_s"$s"_test_zsc_data.dat" ]; then
  ./petsc_utility/getSubMatrix -i $dsp/"filt_data.dat" -s $trn -e $tes -p $dsp \
     -w $fn"_s"$trn"_e"$tes"_s"$s"_test_zsc_data.dat"
fi

if [ ! -f $dsp/$fn"_s"$trn"_e"$tes"_s"$s"_test_label.dat" ]; then
  ./petsc_utility/getSubVector -i $dsp"/mnist_label_s"$s"_c"$c".dat" \
    -s $trn -e $tes -p $dsp -w $fn"_s"$trn"_e"$tes"_s"$s"_c"$c"_test_label.dat"
fi  

#==============================================================================
echo "*** merge test label and data to one file ***" >> $log_file
python ./scripts/mergeLabelData.py -d $dsp  -f $fn"_s"$trn"_e"$tes"_s"$s"_c"$c >> $log_file 2>&1
#==============================================================================
echo "*** get the train data ***" >> $log_file
scale=$(($trn / 10))
for i in `seq $lidx 1 $hidx` 
do  
  size=$((i*$scale)); echo $size
  ./petsc_utility/getSubMatrix -i $dsp"/filt_s"$s"_data.dat" -s 0 -e $size -p $dsp \
        ¦ -w $fn"_s"$s"_"$size"_s"$s"_c"$c"_train_zsc_data.dat" >> $log_file 2>&1
  ./petsc_utility/getSubVector -i $dsp"/mnist_label_s"$s"_c"$c".dat" -s 0 -e $size -p $dsp \
        ¦ -w $fn"_s"$s"_"$size"_s"$s"_c"$c"_train_label.dat" >> $log_file 2>&1
done  
#==============================================================================
echo "*** get the knn ***" >> $log_file
scale=$(($trn / 10))
for i in `seq $lidx 1 $hidx`
do
  size=$((i*$scale)); echo $size
  input_params=" --ds_p "$dsp"  --tmp_p "$dsp"  -f"$fn"_s"$s"_"$size"_s"$s"_c"$c"_train"
  echo "input params:"$input_params  >> $log_file 2>&1
  ./mlsvm_save_knn $input_params >> $log_file 2>&1
done
#==============================================================================
echo "*** get the mlsvm ***" >> $log_file
scale=$(($4 / 10))
for i in `seq $lidx 1 $hidx` 
do  
  size=$((i*$scale)); echo $size
  input_params=" --ds_p "$dsp"  --tmp_p "$dsp"  -f "$fn"_"$size"_s"$s"_c"$c"_train  -s "$s 
  echo "input params:"$input_params >> $log_file 2>&1
  (time ./mlsvmSepTestClassifier $input_params \
	  --test_data $fn"_s"$trn"_e"$tes"_s"$s"_c"$c \
	  -r 1 \
	  --cs_we 0.001 \
	  --mv_id 2 \
	  -v $v ) \
	  >  $dsp"/results_"$fn"_"$size"_s"$s"_c"$c"_"`/bin/date +%m%d%y_%H%M` 2>&1

done


echo "*** Everything is finished successfully ***" >> $log_file

date >> $log_file
