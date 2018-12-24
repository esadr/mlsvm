__version__ = '0.1'
__author__ = 'Ehsan Sadrfaridpour'
__date__ = '122418_1820'

'''
Read data matrix and label matrix in PETSc format

Create a matrix with the labels in the first column, 
Export it to PETSc binary format 

'''


import sys
import argparse
from scipy.sparse import csr_matrix, csc_matrix, vstack, hstack
from os import environ

dirs = environ['PETSC_DIR']
sys.path.insert(0, dirs+'/bin/')
import PetscBinaryIO

io = PetscBinaryIO.PetscBinaryIO()

def process(data_path, uniq_name):
    # data_path = '/scratch4/esadrfa/data/mnist/'
    # uniq_name= 'mnist8m.scale_s4000000_e5000000'
    test_data = f'{uniq_name}_test_zsc_data.dat'
    test_label = f'{uniq_name}_test_label.dat'
    output_fname = f'{uniq_name}_label_data_test.dat'

    td = io.readBinaryFile(data_path + test_data, mattype='scipy.sparse')
    tl = io.readBinaryFile(data_path + test_label)

    td_csc = td[0].tocsc()
    mat_l = csc_matrix(tl[0])

    new_mat= hstack( [mat_l.T, td_csc])
    new_mat_csr = new_mat.tocsr()

    io.writeBinaryFile(data_path + output_fname, [new_mat_csr,])
    print(f'export the results to {data_path + output_fname}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path",
                        action="store",
                        dest="data_path",
                        help="Path to data files")
    parser.add_argument("-f", "--uniq_name",
                        action="store",
                        dest="uniq_name",
                        help="Unique file name")

    args = parser.parse_args()
    assert(args.data_path), \
        "Path to data files  is required! -d"
    assert(args.uniq_name), "Unique file name is required! -f"
    process(args.data_path, args.uniq_name)

if __name__ == "__main__":
    main()
