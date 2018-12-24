'''
Created on Feb 3, 2016
Version 2: Dec 16, 2018
    only remove the last suffix after the right most dot
Version 1: March 8, 2016
    The script reads 2 parameters from command line which are path and filename
    The script reads 2 parameters from command line which are path and filename
    It calculates the Flann and save the result, in the same path with similar file names
    Drawback: convert from csr to ndarray and vise versa 
@author: Ehsan Sadr
@note: read the data and calculate the flann
@input: PetscBinary format
@output: PetscBinary format
'''

import timeit
# import pickle
# import numpy as np
from sys import argv,exit,path
from os import environ
import scipy.sparse
from pyflann import *
from numpy import *
from numpy.random import *



dirs = environ['PETSC_DIR']
sys.path.insert(0, dirs+'/bin/')
#sys.path.insert(0, '/home/esfp/tools/libraries/petsc/bin/')
import PetscBinaryIO



def run_flann(file_path, file_name, num_NN, dist_type):

    # read input in PetscBinary format
    try:
        start = timeit.default_timer()
        print('Start reading input: '+file_path  + file_name)
        # mfile = open(outputfile,'r')
        io = PetscBinaryIO.PetscBinaryIO()
        #"scipy.sparse" is tested and works fine except, I need to convert it to numpy.ndarray and vise verse for flann and writing to petsc binary in the end
        sparse_data= io.readBinaryFile(file_path + file_name, mattype='scipy.sparse')
        # sparse_data= io.readBinaryFile(file_path + file_name, mattype='sparse')   # not useful
        #convert the "scipy.sparse.csr.csr_matrix" to "numpy.ndarray"
        data = sparse_data[0].toarray()
        print("data shape before transpose:"+ str(data.shape))
        #data_T = transpose(data)
        #print("data_T shape after transpose:"+ str(data_T.shape))
        stop = timeit.default_timer()
        print("Data is read from PETSc binary format: %.1f seconds" % (stop - start))
    except e:
        print("\n\nException in reading the input file:%s"% (file_path + file_name))
        exit(1)
    # print("data is:")
    # print(data)
    # print("data[0] is:")
    # print(data[0])


    try:
        start = timeit.default_timer()
        flann = FLANN()

        if(dist_type == 3):
            set_distance_type(3,100)	#this is for flann Minkowski
        else:
            set_distance_type(int(dist_type))

        m_result,m_dists = flann.nn(data, data, int(num_NN),algorithm="kdtree",branching=32, iterations=3, trees=1, checks=64);
        
        stop = timeit.default_timer()
        print("Flann calculation takes: %.1f seconds" % (stop - start))
    except:
        print("\n\nException in running Flann, error: %s" % sys.exc_info()[0])
        exit(1)

    #convert the "numpy.ndarray" to "scipy.sparse.csr.csr_matrix"
    start = timeit.default_timer()
    m_result_sparse=  scipy.sparse.csr.csr_matrix(m_result);
    m_dists_sparse=  scipy.sparse.csr.csr_matrix(m_dists);
    stop = timeit.default_timer()
    print("Convert the ndarray to sparse takes: %.1f seconds" % (stop - start))

    last_suffix = file_name.rfind('.')
    out_fname = file_name[:last_suffix] 
    # save the result
    try:
        start = timeit.default_timer()
        result_outputfile = file_path+''+ out_fname+'_indices.dat'
        print('Outputing Data: '+result_outputfile)
        result_file = open(result_outputfile,'w')
        PetscBinaryIO.PetscBinaryIO().writeMatSciPy(result_file, m_result_sparse)
        stop = timeit.default_timer()
        print("result saved in PETSc format: %.1f seconds" % (stop - start))
    except:
        print("\n\nException in saving the result, error: %s"% sys.exc_info()[0])
        exit(1)

    try:
        start = timeit.default_timer()
        dists_outputfile = file_path+''+ out_fname +'_dists.dat'
        print('Outputing Data: '+dists_outputfile)
        dists_file = open(dists_outputfile,'w')
        PetscBinaryIO.PetscBinaryIO().writeMatSciPy(dists_file, m_dists_sparse)
        stop = timeit.default_timer()
        print("dists saved in PETSc format: %.1f seconds" % (stop - start))
    except:
        print("\n\nException in saving the dists, error: %s" % sys.exc_info()[0])
        exit(1)

if __name__ == '__main__':
    debug = 0
    file_path=""
    file_name=""

    if debug:       # ----------------------- get parameters for debug -------------------------------
        file_path = './'
        file_name = 'kfold_min_train.dat'
    else:           # ----------------------- get parameters for production -------------------------------
        if (len(argv)) < 4:
            print ("please enter path and file_name")
            print("Example: /data/ twonorm_min_data.dat int_num_NN int_dist_type  \nExit due to lack of parameters!")
            exit(1)
        else:
            try:
                last_slash = argv[1].rfind('/')
                file_path = argv[1][0:last_slash+1]
                file_name = argv[1][last_slash+1:]
                num_NN = argv[2]
                dist_type = argv[3]		# 1: squared euclidean distance
            except:
                print("\n\nException on input path and file name, try again!")
                exit(1)

    # --------------------- After reading parameters, run the script  ------------------------
    print("Input filename is: "+file_path+ file_name)
    run_flann(file_path, file_name, num_NN, dist_type)



#serial 4
