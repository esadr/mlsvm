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
For calculating the k-nearest neighbors you need to install the Anaconda [here](https://www.continuum.io/downloads), and pyflann [here](https://github.com/primetang/pyflann).

2- Configure the coresponding environment variables based on your installation. 
Examples are as follow:

`PETSC_ARCH=linux-cxx`, `PETSC_DIR=/lib/petsc`

Below paths are required for python version of k-nearest neighbors.
`PY_PATH=~/anaconda3/bin/`,  `PYTHONPATH=/lib/flann/usr/local/share/flann/python/:/lib/petsc/bin/`.

3- Please go to mlsvm directory (top level) and run `make` command.
If the program compiled correctly, it is ready to use. However, you need to prepare the data in the right format for mlsvm library.
The tools are provided for converting the CSV and libSVM file format, normalization, and calculating k-nearest neighbors.
They are explained in the tools sections.

4- The overall steps are presented in the following figure.
<p align="center">
  <img src="https://user-images.githubusercontent.com/9002689/29581890-cbd7cc9a-8748-11e7-8d6f-df8b7d3096d0.png" width="600"/>
</p>

In the below sections, we conver all the tasks from the top to the bottom.


Read Input
-------------
The input data might be in variety of formats such CSV, Text, and images. The mlsvm needs a PETSc binary format as input. We developed tools to convert the CSV file format or LibSVM file format to PETSc Binary format.
The tools are listed in the tools folder and more details on how to use them are provided in the User Guide in the docs folder.

The parameters to use the tools and the main library can be configured through XML config file or command line.
The XML config file is params.xml. Each parameter has a description and you can change the default values.
Using command line parameters you can override the parameter's values over the XML values temporarily for a specific run.
The list of command line parameters are in the User Guide.

The default path for input or data set files is the datasets folder. You can change that by setting the ds_path.
The name of the data set file is defined as ds_name in the XML config file. 

The name format for dataset files after convert are as follows.

X_data.dat is the data file for dataset X which includes both test and training data. It is a matrix in PETSc binary format which rows are data points.
X_label.dat is the label file for all the data. It is a vector in PETSc binary format which has +1 for minority class and -1 for majority class.

There is a sample data set in the datasets folder. You can download the rest of the data sets from UCI.
Another data set in prepared format is accessible using this [link](https://clemson.box.com/v/MLSVM-Datasets)
Please cite the original data provider in case of using these data sets which are listed in the license.txt file.

Preprocess Data
-------------
The converted data needs to be normalized. For normalization, mlsvm_zscore uses z-score normalization on the whole data including training and test parts.
The results of normalization is stored in X_zsc_data.dat file and the lable are not changed.

The k nearest neighbors are calculated by calling the mlsvm_save_flann over the normalized data. The mlsvm_save_flann calls the python scripts in the back to call flann.
It needs the right parameters for PY_PATH and PYTHONPATH to work.

The results are saved in two files for both minority and majority classes. The indices of neighbor nodes in the minority class are stored in X_min_norm_data_indices.dat inside the dataset folder.
The distances to neighbor nodes are saved in X_min_norm_data_dists.dat for the minority class.

For the majority class, the \_min_ is changed to \_maj_.

Classification
-------------
The classification use cross validation to make separate parts for validation and test from the training data. You can set the number of k-fold using -k parameter.
For running the same experiment multiple times, you can set the number of experiments using -x. Each experiments shuffles the data in the beginning.
The rest of parameters are explained in params.xml file and User Guide.

To run the mlsvm on dataset X, you create the files in datasets folder regards to name format explained above and call `./mlsvm_classifier -f X ` or you can configure the name inside the param.xml file and just call `./mlsvm_classifier`

The rest of parameters are explained in the param.xml file, however the shortcuts for overriding them through the command prompt will explain later.

For instance, the MLSVM program can run by calling below command and parameters. 
`./mlsvm_classifier -f twonorm -x 1 -k 5 -q 0.4 -r 4`



List of Tools:
-------------
mlsvm_save_knn which divides the data to two classes and calculate the k-NN. It saves the output for each class in indices and dists files. 
You can compile it by 
`make mlsvm_save_knn`
The parameters are filename, number of nearest neighbors and type of distances which are passed by -f, --nn_n, --nn_d respectively. 
The default parameter for --nn_n is 10 and for --nn_n is 10 which are set in the param.xml file.
The filename is the name of dataset without any extension like filename_zsc_data.dat.
For example for twonorm dataset you can use `./mlsvm_save_knn -f twonorm --nn_n 10 --nn_d 1`



Contact
-------------
For questions or suggestions please email me at esadrfa@g.clemson.edu 







The parameters could be set inside the `param.xml` file. You can find the details for parameters in the user guide document.
