# MLSVM
Multi level Support Vector Machines

Download
-------------
You can use the recursive parameter to download the required libraries as part of this reposetory.

`git clone --recursive https://github.com/esadr/mlsvm.git`

It will download both PETSc and FLANN libraries which are needed.

Installation
-------------
1- We need to install the libraries which have been downloaded in the petsc and flann folders. 
The installation guides for them are provided by their developers and you can easily use them to install the libraries.
For PETSc please refer to [here](https://www.mcs.anl.gov/petsc/documentation/installation.html) and for Flann [here](http://www.cs.ubc.ca/research/flann).
In case of using python for calculating the k-nearest neighbors you need to install the Anaconda [here](https://www.continuum.io/downloads), and pyflann [here](https://github.com/primetang/pyflann).

2- Configure the coresponding environment variables based on your installation. 
Examples are as follow:

`PETSC_ARCH=linux-cxx`, `PETSC_DIR=/lib/petsc`

Below paths are required for python version of k-nearest neighbors.
`PY_PATH=/anaconda3/bin/`,  `PYTHONPATH=/lib/flann/usr/local/share/flann/python/:/lib/petsc/bin/`.

3- Please go to mlsvm directory (top level) and run `make` command.
If the program compiled correctly, it is ready to use. However, you need to prepare the data in the right format for mlsvm library.
The tools are provided for converting the CSV and libSVM file format, normalization, and calculating k-nearest neighbors.
They are explained in the tools sections.

4- The overall steps are presented in the following figure.
<p align="center">
  <img src="https://user-images.githubusercontent.com/9002689/29581890-cbd7cc9a-8748-11e7-8d6f-df8b7d3096d0.png" width="600"/>
</p>


To use the MLSVM program, you can run it using below command and parameters. 
`./main -f twonorm -x 1 -k 5 -q 0.4 -r 4`

The parameters could be set inside the `param.xml` file. You can find the details for parameters in the user guide document.

Input files (Training data set)
-------------
The dataset files should be in datasets folder inside the mlsvm. You can change the path and name in the param.xml file.

The name format for dataset files are as follows.

X_zsc_data.dat is the data file for dataset X which includes both test and training data in normalized form. It is a matrix in PETSc binary format which rows are data points.

X_label.dat is the label file for all the data. It is a vector in PETSc binary format which has +1 for minority class and -1 for majority class.

To run the mlsvm on dataset X, you create the files in datasets folder regards to name format explained above and call `./main -f X ` or you can configure the name inside the param.xml file and just call `./main`

The rest of parameters are explained in the param.xml file, however the shortcuts for overriding them through the command prompt will explain later.


List of Tools:
-------------
Save_flann which divides the data to two classes and calculate the k-NN using FLANN library. It saves the output for each class in indices and dists files. The mlsvm use these files to create a adjacency matrix for the multi level framework.
You can create it using 
`make save_flann`
The parameters are filename, number of nearest neighbors and type of distances which are passed by -f, --nn_n, --nn_d respectively.
The default parameter for --nn_n is 10 and for --nn_n is 10 which are set in the param.xml file.
The filename is the name of dataset without any extension like filename_zsc_data.dat.
For example for twonorm dataset you can use `./save_flann -f twonorm --nn_n 10 --nn_d 1`



Contact
-------------
For questions or suggestions please email me at esadrfa@g.clemson.edu 




