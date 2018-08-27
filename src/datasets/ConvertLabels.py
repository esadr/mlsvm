"""
Author: Ehsan Sadrfaridpour
Date: Aug 24, 2018
Purpose: map other labels to -1 and 1 labels, 
	make sure the number of 1 labels are smaller than the number of -1 labels for MLSVM framework
	

Usage: define the preferred mapping in the label_map which is a dictionary.
	The key is the old/current label(s) in the file which needs to change and the 
	value(s) are the new labels. For the labels which are ok, you can skip them from
	adding them to this dictionary and they will be ignored from conversion.

"""



import pandas as pd
import os

ds_path = '/scratch2/esadrfa/mlsvm_data'
in_ds_fname = 'susy.csv'
out_ds_fname = 'susy_fixed_label.csv'
df = pd.read_csv(os.path.join(ds_path, ds_fname), 
                 header=None, sep=' ', error_bad_lines=False, engine='c')

sep = ' '
label_map = {'0': '-1'}

out_file = open(os.path.join(ds_path, out_ds_fname), 'w')

with open(os.path.join(ds_path, in_ds_fname),'r') as in_file:
    for idx, line in enumerate(in_file):
        if not idx % 100000: print(idx, end=',')
        curr_data = line.split(sep)
        if(curr_data[0] in label_map):
            curr_data[0] = label_map[curr_data[0]]
        for item in curr_data:
            out_file.write(item + sep)
#         out_file.write('\n')  # it has the \n already, this cause empty lines
        
out_file.close()
print('convert is finished successfully!')
