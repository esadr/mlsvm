out=`hostname | grep paliametto | wc -l`
if [ $out -eq 1 ]
then
  echo "This is on the Palmetto Cluster"
fi
