ds_path=/scratch2/esadrfa/mlsvm_data/
log_path=/scratch2/esadrfa/mlsvm_logs/
mkdir -pv $log_path
tmp_path="/local_scratch/"`ls  /local_scratch | grep pbs`"/"
ds_name=$1
echo "Path: "$ds_path
echo "File name: "$ds_name

input_params=" --ds_p "$ds_path" -f  "$ds_name"   --tmp_p "$tmp_path
echo $input_params




scale=400000
largeDataFile=/scratch2/esadrfa/mlsvm_data/susy_4M_train_zsc_data.dat
largeLabelFile=/scratch2/esadrfa/mlsvm_data/susy5M_label.dat
for i in `seq 1 1 10`
do  
  size=$((i*$scale))
  echo $size  
  #./petsc_utility/getSubMatrix -i $largeDataFile -s 0 -e $size -p /scratch2/esadrfa/mlsvm_data/ -w "susy_"$size"_train_zsc_data.dat"
  ./petsc_utility/getSubVector -i $largeLabelFile -s 0 -e $size -p /scratch2/esadrfa/mlsvm_data/ -w "susy_"$size"_train_label_data.dat"

  #echo " * * * * * ""#3 Calculating the k-nearest neighbors"" * * * * * "
  #./mlsvm_knn $input_params  >  $log_path"/"$ds_name"_knn.log" 2>&1
done
