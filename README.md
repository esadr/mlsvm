# MLSVM
Multi level Support Vector Machines

Installation
-------------
1- Install PETSc [here](https://www.mcs.anl.gov/petsc/documentation/installation.html) ,
Flann [here](http://www.cs.ubc.ca/research/flann), Anaconda [here](https://www.continuum.io/downloads), and pyflann [here](https://github.com/primetang/pyflann).

2- Configure the coresponding environment variables based on your installation. 
Examples are as follow:

`PY_PATH=/anaconda3/bin/`,  `PYTHONPATH=/lib/flann/usr/local/share/flann/python/:/lib/petsc/bin/`, `PETSC_ARCH=linux-cxx`, `PETSC_DIR=/lib/petsc`

3- Go to mlsvm folder

4- Run "make mxml"
If the program compiled correctly, it is ready to use.

To use the program, you can run it using below command and parameters. 
`./main -f twonorm -x 1 -k 5 -q 0.4 -r 4`
The parameters could be set inside the `param.xml` file. You can find the details for parameters in the document.
