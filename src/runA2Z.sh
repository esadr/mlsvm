ds_path=/scratch2/esadrfa/mlsvm_data/
log_path=/scratch2/esadrfa/mlsvm_logs/
mkdir -pv $log_path
tmp_path="/local_scratch/"`ls  /local_scratch | grep pbs`"/"
ds_name=$1
echo "Path: "$ds_path
echo "File name: "$ds_name

input_params=" --ds_p "$ds_path" -f  "$ds_name"   --tmp_p "$tmp_path
echo $input_params
#exit 1

#echo " * * * * * ""1 Converting csv file to petsc format"" * * * * * "
#./mlsvm_csv_petsc $input_params  >  $log_path"/"$ds_name"_csv2petsc.log" 2>&1          

echo " * * * * * ""1 Converting LibSVM file to petsc format"" * * * * * "
./mlsvm_libsvm_petsc $input_params  >  $log_path"/"$ds_name"_libsvm2petsc.log" 2>&1        

echo " * * * * * ""#2 Normalizing the data using z-score"" * * * * * "
./mlsvm_zscore $input_params >  $log_path"/"$ds_name"_zscore.log" 2>&1               

echo " * * * * * ""#3 Calculating the k-nearest neighbors"" * * * * * "
./mlsvm_knn $input_params  >  $log_path"/"$ds_name"_knn.log" 2>&1                 

echo " * * * * * ""#4 Running the MLSVM"" * * * * * "
./mlsvm_classifier $input_params  -r 1 -s 10 -v .1    >  $log_path"/"$ds_name"_training_r1_s10_v.1.log" 2>&1


