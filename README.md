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

4- Run "make"
If the program compiled correctly, it is ready to use.

To use the program, you can run it using below command and parameters. 
`./main -f twonorm -x 1 -k 5 -q 0.4 -r 4`

The parameters could be set inside the `param.xml` file. You can find the details for parameters in the document.

Input files (Training data set)
-------------
The dataset files should be in datasets folder inside the mlsvm. You can change the path and name in the param.xml file.

The name format for dataset files are as follows.

X_zsc_data.dat is the data file for dataset X which includes both test and training data in normalized form. It is a matrix in PETSc binary format which rows are data points.

X_label.dat is the label file for all the data. It is a vector in PETSc binary format which has +1 for minority class and -1 for majority class.

To run the mlsvm on dataset X, you create the files in datasets folder regards to name format explained above and call `./main -f X ` or you can configure the name inside the param.xml file and just call `./main`

The rest of parameters are explained in the param.xml file, however the shortcuts for overriding them through the command prompt will explain later.

Contact
-------------
For question or suggestion email me at esadrfa@g.clemson.edu


