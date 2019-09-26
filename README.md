[![License](https://img.shields.io/badge/License-BSD%202--Clause-orange.svg)](https://opensource.org/licenses/BSD-2-Clause)

# MLSVM
Multilevel Support Vector Machines


The correspoding papers with experimental results are as follows:
    
  * Sadrfaridpour, Ehsan, Talayeh Razzaghi, and Ilya Safro. "Engineering fast multilevel support vector machines." Machine Learning (2019): 1-39.
  [(PDF)](https://link.springer.com/article/10.1007/s10994-019-05800-7) [(ArXiv)](https://arxiv.org/pdf/1707.07657.pdf) [(BibTex)](https://raw.githubusercontent.com/esadr/mlsvm/master/Bibliography.txt)
  
  * Sadrfaridpour, Ehsan, Sandeep Jeereddy , Ken Kennedy, Andre Luckow, Talayeh Razzaghi, and Ilya Safro. "Algebraic multigrid support vector machines." ESANN 2017 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning.
  [(PDF)](https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2017-37.pdf)  [(BibTex)](https://raw.githubusercontent.com/esadr/mlsvm/master/Bibliography.txt)



Download Virtual Machine
-------------
There is a Ubuntu Virtual Machine with all required packages ready to use.
You can download it as a one file (3.9GB) or eight small file of 500MB each.
[Download Link](https://goo.gl/6MhLKq)

For running it on your machine (Laptop, Desktop), you can download and install a Virtual Machine application such as [VirtualBox](https://www.virtualbox.org/wiki/Downloads).
This allows you to run this Virtual Machine as a guest operating system on your current (host) operating system such Linux, Windows or macOS.
Next, you can import the mlsvm.ovf file which is an Open Virtualization Format.


How to use Virtual Machine
-------------
Both username and password are:demo (all lowercase)

The library is installed in /home/demo/mlsvm/src/ and you can run it using 
`./mlsvm_classifier [list of parameters]`

The list of parameters are optional and you can set them using param.xml file as well.

  
Download Source
-------------
You can use the recursive parameter to download the required libraries as part of this reposetory.

`git clone --recursive https://github.com/esadr/mlsvm.git`

It will download both PETSc and FLANN libraries which are needed.

Installation
-------------
1- We need to install the libraries which have been downloaded in the petsc, flann and pyflann folders. 
The installation guides for them are provided by their developers and you can easily use them to install the libraries.
For PETSc please refer to [link](https://www.mcs.anl.gov/petsc/documentation/installation.html) and for Flann [link](http://www.cs.ubc.ca/research/flann). 
For calculating the k-nearest neighbors you need to install the Anaconda [link](https://www.continuum.io/downloads), and pyflann [link](https://github.com/primetang/pyflann).


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

For the majority class, the `_min_` is changed to `_maj_`.

Classification
-------------
The classification use cross validation to make separate parts for validation and test from the training data. You can set the number of k-fold using -k parameter.
For running the same experiment multiple times, you can set the number of experiments using -x. Each experiments shuffles the data in the beginning.
The rest of parameters are explained in params.xml file and User Guide.

To run the mlsvm on dataset X, you create the files in datasets folder regards to name format explained above and call `./mlsvm_classifier -f X ` or you can configure the name inside the param.xml file and just call `./mlsvm_classifier`

The rest of parameters are explained in the param.xml file, however the shortcuts for overriding them through the command prompt will explain later.

For instance, the MLSVM program can run by calling below command and parameters. 
`./mlsvm_classifier -f twonorm -x 1 -k 5 -q 0.4 -r 4`

Results
-------------
* Summary Logs

The resutls are presented in the standard output. You can redirect the output to a file using two below approaches.
1- Send all the output to a file and not printing in the terminal.
`./mlsvm_classifier > ./summary.log`
2- The outputs print in the terminal as well.
`./mlsvm_classifier | tee ./summary.log`

In case of errors, please forward both error and standard output to a file using 
`./mlsvm_classifier > ./summary.log 2>&1`

* Train Models

The trained models are stored in the `./svm_models`. Each model has information about the experiment id and k-fold id which are used to train it. The test data for each of them are saved in the ./temp folder. 


List of Tools:
-------------
* mlsvm_save_knn

It divides the data to two classes and calculate the k-NN. It saves the output for each class in indices and dists files. 
You can compile it by 
`make mlsvm_save_knn`
The parameters are filename, number of nearest neighbors and type of distances which are passed by -f, `--nn_n`, `--nn_d` respectively. 
The default parameter for `--nn_n` is 10 and for `--nn_n` is 10 which are set in the param.xml file.
The filename is the name of dataset without any extension like filename_zsc_data.dat.
For example for twonorm dataset you can use `./mlsvm_save_knn -f twonorm --nn_n 10 --nn_d 1`

* mlsvm_csv_petsc

It converts the csv file format to the suitable format for MLSVM which is PETSc Binary format.

* mlsvm_libsvm_petsc

It converts the LibSVM file format to the suitable format for MLSVM.

* mlsvm_zscore

It normalizes the input data using z-scrore. For example, your data set should be stored in file X_data.dat and you run `./mlsvm_zscore -f X `. The output file is stored as `X_zsc_data.dat` in the same path. The path is set using `--ds_p` parameter.

Parameters:
-------------
Parameters are explained in briefly in the params.xml file. More information for the command lines arguments and parameters are covered in the User Guide.

Contact
-------------
For questions or suggestions please email me at esadrfa@g.clemson.edu 
