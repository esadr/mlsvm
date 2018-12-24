__author__ = 'esfp'


import sys
from scipy.sparse import csr_matrix, csc_matrix, vstack, hstack
from os import environ

dirs = environ['PETSC_DIR']
sys.path.insert(0, dirs+'/bin/')
import PetscBinaryIO

io = PetscBinaryIO.PetscBinaryIO()


data_path = '/scratch4/esadrfa/data/mnist/'
uniq_name= 'mnist8m.scale_s4000000_e5000000'
test_data = f'{uniq_name}_test_zsc_data.dat'
test_label = f'{uniq_name}_test_label.dat'

output_fname = f'{uniq_name}_label_data_test.dat'

td = io.readBinaryFile(data_path + test_data, mattype='scipy.sparse')
tl = io.readBinaryFile(data_path + test_label)

td_csc = td[0].tocsc()
mat_l = csc_matrix(tl[0])

new_mat= hstack( [mat_l.T, td_csc])

new_mat_csr = new_mat.tocsr()

tl = io.writeBinaryFile(data_path + output_fname, [new_mat_csr,])
print(data_path + output_fname)