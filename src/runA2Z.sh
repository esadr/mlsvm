#make sure it has the "/" in the end 
ds_path=/scratch2/esadrfa/mlsvm_data/  
tmp_path="/local_scratch/"`ls  /local_scratch | grep pbs`"/"
#ds_name=susy_fixed_label
ds_name=$1
echo "Path: "$ds_path
echo "File name: "$ds_name
#exit 1
echo " * * * * * ""1 Converting csv file to petsc format"" * * * * * "
./mlsvm_csv_petsc --ds_p $ds_path -f  $ds_name   --tmp_p $tmp_path          #1 convert csv to petsc format
exit 1
echo " * * * * * ""#2 Normalizing the data using z-score"" * * * * * "
./mlsvm_zscore -f $ds_name                                       #2 normalized the data 
echo " * * * * * ""#3 Calculating the k-nearest neighbors"" * * * * * "
./mlsvm_save_knn -f $ds_name                                     #3 calculate the k-nearest neighbors
echo " * * * * * ""#4 Running the MLSVM"" * * * * * "
./mlsvm_classifier -f $ds_name  --tmp_p $tmp_path                #4 run the multilevel SVM 

