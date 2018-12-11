scale=400000
for i in `seq 3 1 10`
do  
    size=$((i*$scale))
    echo $size  
    bash runA2Z.sh "train_"$size; 
done

