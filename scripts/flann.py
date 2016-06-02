'''
Created on Feb 3, 2016
Version 1: March 8, 2016
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



#dirs = environ['PETSC_DIR']
#sys.path.insert(0, dirs+'/bin/')
sys.path.insert(0, '/home/esfp/tools/libraries/petsc/bin/')
import PetscBinaryIO



def run_flann(file_path, file_name):

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
        print("Exception in reading the input file:"+file_path + file_name)
        exit(1)
    # print("data is:")
    # print(data)
    # print("data[0] is:")
    # print(data[0])


    try:
        start = timeit.default_timer()
        flann = FLANN()
        m_result,m_dists = flann.nn(data, data, 10,algorithm="kdtree",branching=32, iterations=3, trees=1, checks=64);
        #This is not needed, as I transpose them back again in loader class in mlsvm
        #but for now that I don't want to change the loader, I transpose them for test
        m_result_T= transpose(m_result) 
        m_dists_T= transpose(m_dists)
        print("\n\nFlann indices Transposed shape:"+ str(m_result_T.shape))
        stop = timeit.default_timer()
        print("Flann calculation takes: %.1f seconds" % (stop - start))
    except:
        print("Exception in running Flann, error: "+ sys.exc_info()[0])
        exit(1)

    #convert the "numpy.ndarray" to "scipy.sparse.csr.csr_matrix"
    start = timeit.default_timer()
    m_result_sparse=  scipy.sparse.csr.csr_matrix(m_result_T);
    m_dists_sparse=  scipy.sparse.csr.csr_matrix(m_dists_T);
    stop = timeit.default_timer()
    print("Convert the ndarray to sparse takes: %.1f seconds" % (stop - start))


    # save the result
    try:
        no_suffix_name = file_name.split(".")[0]
        start = timeit.default_timer()
        result_outputfile = file_path+''+ no_suffix_name+'_indices.dat'
        print('Outputing Data: '+result_outputfile)
        result_file = open(result_outputfile,'w')
        PetscBinaryIO.PetscBinaryIO().writeMatSciPy(result_file, m_result_sparse)
        stop = timeit.default_timer()
        print("result saved in PETSc format: %.1f seconds" % (stop - start))
    except:
        print("Exception in saving the result, error: "+ sys.exc_info()[0])
        exit(1)

    try:
        start = timeit.default_timer()
        dists_outputfile = file_path+''+ no_suffix_name+'_dists.dat'
        print('Outputing Data: '+dists_outputfile)
        dists_file = open(dists_outputfile,'w')
        PetscBinaryIO.PetscBinaryIO().writeMatSciPy(dists_file, m_dists_sparse)
        stop = timeit.default_timer()
        print("dists saved in PETSc format: %.1f seconds" % (stop - start))
    except:
        print("Exception in saving the dists, error: "+ sys.exc_info()[0])
        exit(1)

if __name__ == '__main__':
    debug = 0
    file_path=""
    file_name=""

    if debug:       # ----------------------- get parameters for debug -------------------------------
        file_path = './'
        file_name = 'kfold_min_train.dat'
    else:           # ----------------------- get parameters for production -------------------------------
        if (len(argv)) < 3:
            print ("please enter path and file_name")
            print("Example: /data/ twonorm_min_data.dat  \nExit due to lack of parameters!")
            exit(1)
        else:
            try:
                file_path = argv[1]
                file_name = argv[2]
            except:
                print("Exception on input path and file name, try again!")
                exit(1)

    # --------------------- After reading parameters, run the script  ------------------------
    print("Input filename is: "+file_path+ file_name)
    run_flann(file_path, file_name)



#serial 1
























# def read_data(fname):
#     """
#     Get the file name and read it to a numpy matrix
#     @return: numpy matrix
#     """
#     try:

#     print("data is read from file"+ fname)
#     return training_data


#     def spell_check(self, word):
#         """
#         Given a word, characters are deleted, transposed, replaced, and inserted to
#         try and find the correct spelling. Out of the potential candidates, the one
#         that has been seen the most frequently is then chosen.

#         @type  word: string
#         @param word: The word that is to be spell-checked.
#         @return:     The potential correct spelling for the word.
#         """
#         try:
#             if word not in self.englishDict:
#                 known = self.spell_edits(word)
#                 if len(known) > 0:
#                     return max(known, key=self.englishDict.get)
#         except:
#             print("Exception at spell_check for word: "+ word)
#             # exit()
#         return word

#     def spell_edits(self, word):
#         """
#         Given a word, characters are deleted, transposed, replaced, and inserted to
#         try and find the correct spelling.

#         Adapted from Peter Norvig's Python spell checker.

#         @type  word: string
#         @param word: The word that is to be spell-checked.
#         @type:       set
#         @return:     The potential candidates for the correct spelling of the word.
#         """

#         # if word is already in the dictionary, no need to go further
#         if word in self.englishDict:
#             known = set([word])
#         else:
#             # modify the word to see if a change is in the dictionary
#             s = [(word[:i], word[i:]) for i in range(len(word) + 1)]
#             deletes    = [a + b[1:] for a, b in s if b]
#             transposes = [a + b[1] + b[0] + b[2:] for a, b in s if len(b)>1]
#             replaces   = [a + c + b[1:] for a, b in s for c in self.alphabet if b]
#             inserts    = [a + c + b     for a, b in s for c in self.alphabet]
#             # this is a list of the potential candidates
#             candidates = set(deletes + transposes + replaces + inserts)
#             # this is the set of words from the candidates that are in the dictionary
#             known = set()
#             for w in candidates:
#                 if w in self.englishDict:
#                     known.add(w)
#         return known


#     def filter_text(self,text):
#         # remove symbols
#         text = text.replace("#", " ").replace(".", " ").replace("?", " ").replace("!", " ").replace("-", " ").replace("quot;", " ").replace("(", " ").replace(")", " ").replace('"', " ").replace("\\", " ").replace("/", " ").replace("+", " ").replace("&amp;", " ").replace("&gt;", " ").replace("&lt;", " ").replace(":", " ").replace("*", " ").replace('&', " ").lower()
#         # substitute multiple whitespace with a single space (http://stackoverflow.com/a/2077944/2674061)
#         text = ' '.join(text.split())
#         return text


# # =========================================== Main part for clearing the data =======================================
#     def get_training_data(self, lines, text_col_num, lbl_col_num):
#         """
#         Creates the training data sets.

#         @type  lines: [string]
#         @param lines: The lines of the dataset read into memory.
#         @type  text_col_num: int
#         @param text_col_num: index of the field which contains the text
#         @type  lbl_col_num: int
#         @param lbl_col_num: index of the field which contains the label of the class
#         @return:  The training data and labels.
#         """

#         empty_text = "empty"
#         comment =""
#         # the training and labels as lists to be returned
#         training_data = []
#         training_labels = []

#         # get a stemmer
#         stemmer = nltk.stem.porter.PorterStemmer()

#         # examine the entries
#         start = timeit.default_timer()

#         #controls the header status
#         start_loop = 0
#         if self.ignore_header_status:
#             start_loop = 1
#             print("Header is ignored!")

#         for i in range(start_loop, len(lines)):
#         # for i in range(start_loop, 3):
#             line = lines[i]
#             try:
#                 # comments are the "question 11" that is part five (index starts from 0)
#                 parts = line.split(",")
# #                 comment = parts[5]    # the col_number is read from argv parameters
# #                 comment = parts[text_col_num].replace(".", " ").replace("?", " ").replace("!", " ").replace("-", " ").replace("quot;", " ").replace("(", " ").replace(")", " ").replace('"', " ").replace("\\", " ").replace("/", " ").replace("+", " ").replace("&amp;", " ").replace("&gt;", " ").replace("&lt;", " ").replace(":", " ").replace("*", " ").replace('&', " ").replace('-', " ").lower()
#                 comment = self.filter_text(parts[text_col_num])
#                 # print("raw comment :"+ comment)
#                 label = parts[lbl_col_num]
#                 # print("label :"+ label)
#                 training_labels.append(label)
#             except:
#                 # e = sys.exc_info()[0]
#                 print("Exception in splitting the input data")

#             try:
#                 # get the words of the sentences, spell-check, and stem
#                 # print("before tokenize:"+ comment)
#                 t = nltk.word_tokenize(comment)
#                 if len(t) == 0 :
#                     print ("nothing to tokenize")
#                 # print("After tokenize:   "+ ' '.join(t))
#             except:
#                 e = sys.exc_info()[0]
#                 print("Exception in Word_tokenize at line: "+str(i)+" error is: "+str(e))
#                 print("data is: "+comment)
#                 print("result of tokenize: "+ ' '.join(t))

#             try:
#                 u = []
#                 for w in t:
#                   if w.isalpha() and len(w) > 3:
#                     w = self.spell_check(w)
#                   u.append(stemmer.stem(w))
#                 t = u

#                 # print("After spell check:"+ ' '.join(t))

#                 g=None
#                 # t=comment.split(" ")

#                 if len(t) == 0 :
#                     print ("nothing to tokenize")
#                 else :                            #there are some terms
#                     training_data.append(' '.join(t))

#             except:
#                 e = sys.exc_info()[0]
#                 print("Exception in Spell check and stemmer at line: "+str(i)+" error is: "+str(e))
#                 print(comment)
#                 # # if exception happens, I just add an empty comment
#                 # print("final comment for :"+ comment +"is set to:"+empty_text)
#                 # training_data.append(empty_text)
#                 exit()



#         stop = timeit.default_timer()
#         print("Testing set created: %.1f seconds" % (stop - start))

#         return training_data, training_labels
# #---------------------------------------------- calc_TFIDF -------------------------------------------------------
#     def calc_TFIDF(self, filename,t_min,t_ng_max,t_norm,t_idf,text_col_num,lbl_col_num):
#         """
#         create tf-idf matrix

#         @type  filename: string
#         @param filename: The name of the filename.
#         """
#         lines = self.read_data(filename)
#         training_data, training_labels = self.get_training_data(lines,text_col_num,lbl_col_num)
#         # convert to numpy arrays
#         start = timeit.default_timer()
#         training_data = np.array([''.join(x) for x in training_data])
#         training_labels = np.array([''.join(x) for x in training_labels])
#         print("[DEBUG][calc_TFIDF] train data and label numpy arrays are created! ")

#         # Main function that creates the TF-IDF matrix
#         # vectorizer = TfidfVectorizer(min_df=t_min, ngram_range=(1,t_ng_max), strip_accents='unicode', norm=t_norm, use_idf=t_idf)
#         vectorizer = TfidfVectorizer(min_df=t_min, ngram_range=(1,t_ng_max), strip_accents='unicode', norm=t_norm,
#                                      use_idf=t_idf,decode_error='ignore')
#         # print("[DEBUG]--------------- training data before [vectorizer.fit_transform(training_data)]--------------------")
#         # for i in training_data:
#         #     print i
#         training_data = vectorizer.fit_transform(training_data)
#         print("[DEBUG][calc_TFIDF] Vectorize is finished! ")

#         # save the vectorizer - vocabulary (terms)
#         f_prefix = filename.split(".")[0]       # input data file name
#         col_name = ["DEALER_CODE", "PROGRAM_TYPE", "SURVEY_RECEIVED_DATE", "VIN",  "Carline", "comment"]
#         # f_name_prefix = "../data/results/"+f_prefix + "_m"+ str(t_min)+"_ng"+str(t_ng_max)+"_nor"+str(t_norm)+"_idf"+\
#         #                                     str(t_idf)+"_"+ col_name[text_col_num]
#         f_name_prefix = "../data/results/"+f_prefix + "_m"+ str(t_min)+"_ng"+str(t_ng_max)+"_nor"+str(t_norm)+"_idf"+ str(t_idf)

#         # start = timeit.default_timer()
#         # f = open( f_name_prefix+ ".terms", "w")
#         # #http://stackoverflow.com/questions/613183/sort-a-python-dictionary-by-value
#         # terms_sorted = sorted(vectorizer.vocabulary_.items(), key=operator.itemgetter(1))
#         # for i in terms_sorted :
#         #     f.write(str(i[0])+"\n")
#         # f.close()
#         # stop = timeit.default_timer()
#         # print("Vectorizer Vocabulary saved: %.1f seconds" % (stop - start))

#         self.write_TFIDF(training_data,training_labels,"","_test","petsc")



# #---------------------------------------------- normalize_matrix -------------------------------------------------------
#     def normalize_matrix(self, raw_matrix):
#         normalized_matrix = preprocessing.normalize(raw_matrix)
#         return normalized_matrix


# #---------------------------------------------- write_TFIDF -------------------------------------------------------
#     def write_TFIDF(self, t_data, t_label, f_path, f_name, f_format):
#         if f_format is "mat":
#             # save the vectorizer
#             start = timeit.default_timer()
#             outputfile = f_path+ '/tfidf/data_'+f_name+'_'+str(t_data.shape[0])+'x'+str(t_data.shape[1])+'.mat'
#             print('Outputing Data: '+outputfile)
#             savemat(outputfile , {'training_data':training_data})
#             stop = timeit.default_timer()
#             print("data saved in Matlab format: %.1f seconds" % (stop - start))

#         if f_format is "petsc":
#             # save the vectorizer
#             start = timeit.default_timer()
#             outputfile = f_path+ '../data/results/data_'+f_name+'_'+str(t_data.shape[0])+'x'+str(t_data.shape[1])+'.dat'
#             print('Outputing Data: '+outputfile)
#             mfile = open(outputfile,'w')
#             PetscBinaryIO.PetscBinaryIO().writeMatSciPy(mfile, t_data)
#             stop = timeit.default_timer()
#             print("data saved in PETSc format: %.1f seconds" % (stop - start))

#         #save the labels (serialized)
#         # outputfile = f_path+ '../data/results/label_'+f_name+'_'+str(t_label.shape[0])+'.dump'
#         # outputfile = "./labels.dump"
#         # print('Dumping Label: '+outputfile)
#         # pickle.dump(t_label, outputfile)





