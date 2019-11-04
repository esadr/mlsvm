'''
Version 1: Nov 4, 2019
    The script reads 2 parameters from command line which are path and filename
    This calls the FAISS and save the result, in the same path with similar file names

@author: Ehsan Sadr
@note: read the data and calculate the Approximate Nearest Neighbors using different
    metrics
@input: PetscBinary format
@output: PetscBinary format
'''

import argparse
import timeit
# import pickle
import numpy as np
import sys
from os import environ
import scipy.sparse
import faiss                   # make faiss available

#from numpy import *
#from numpy.random import *



dirs = environ['PETSC_DIR']
sys.path.insert(0, dirs+'/bin/')
import PetscBinaryIO



def load_data(file_path, file_name):
    """Load the data from PETSc Binary format

    Parameters
    ----------
    file_path: the path to file
    file_name: the filename with full extension

    Return
    ------
    numpy array
    """

    try:
        start = timeit.default_timer()
        print('Start reading input: '+file_path  + file_name)
        print(f'Start reading {args.path}/{args.f_name}')
        io = PetscBinaryIO.PetscBinaryIO()
        sparse_data= io.readBinaryFile(f'{file_path}/{file_name}'
                                        , mattype='scipy.sparse')
        data = sparse_data[0].toarray()
        # faiss requirement ->  convert all values to 32 bit
        data = data.astype('float32')
        print(f'data shape:{data.shape}')
        stop = timeit.default_timer()
        print("load data takes:{(stop - start):.1f} seconds")
    except e:
        raise Exception(f'Exception in loading file:{file_path}/{file_name}'+\
                        f', error:{e}')

    return data

#def run_faiss(data, num_NN, dist_type, verbose=False):
def run_faiss(data, num_NN, verbose=False):
    try:
        start = timeit.default_timer()

        nb, d = data.shape              # database size, dimension
        nq = nb                         # num queries
        np.random.seed(42)              # make reproducible

        # ========== Train =============
        index = faiss.IndexFlatL2(d)   # build the index
        if verbose:
            print(f'[run_faiss]index is trained:{index.is_trained}')
        index.add(xb)                  # add vectors to the index

        if verbose:
            print(f'[run_faiss]index ntotal:{index.ntotal}')
        
        # ========== Search =============
        k = num_NN                          # we want to see 4 nearest neighbors
        D, I = index.search(data, k)      # get k NN for all the points

        stop = timeit.default_timer()
        print("faiss calculation takes:{(stop - start):.1f} seconds")
    except e:
        raise Exception(f'Exception in running faiss, error:{e}')

    return D, I

def export_to_petsc(file_path, file_name, D, I):
    #convert the "numpy.ndarray" to "scipy.sparse.csr.csr_matrix"
    start = timeit.default_timer()
    m_result_sparse=  scipy.sparse.csr.csr_matrix(I);
    m_dists_sparse=  scipy.sparse.csr.csr_matrix(D);
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


def main(file_path, file_name, num_NN): #, dist_type):
#    dist_type has not used yet
    data = load_data(file_path, file_name)
    D, I = run_faiss(data, num_NN) # , dist_type)
    export_to_petsc(file_path, file_name, D, I)


if __name__ == '__main__':

    parser= argparse.ArgumentParser(description='post process the logs into MongoDB')
    parser.add_argument('-p', action='store', dest='path',
                        help='path is required, use -p',
                        required=True)
    parser.add_argument('-f', action='store', dest='f_name',
                        help='file name is required, use -f',
                        required=True)

    parser.add_argument('-n', action='store', dest='num_NN',
                        help='number of nearest neighbors, use -n',
                        required=True)

#    parser.add_argument('-d', action='store', dest='dist_type',
#                        help='distance type, use -d',
#                        required=True)

    args = parser.parse_args()

    # --------------------- After reading parameters, run the script  ------------------------
    print(f'Input path:{args.path} and filename:{args.f_name}')
#    main(file_path, file_name, num_NN, dist_type)
    main(file_path, file_name, num_NN)
