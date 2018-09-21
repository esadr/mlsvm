All: 
	make puGetMatInfo puGetVecInfo puGetSubMatrix puGetSubVector puMergeLabel2Data
CC 	 = g++ -L. 
CFLAGS 	 = -I.	
CPPFLAGS = -std=c++11 -O3 

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

puGetMatInfo: get_mat_info.o  chkopts
	-${CLINKER} -o getMatInfo get_mat_info.o ${PETSC_MAT_LIB}
	${RM} -f get_mat_info.o 

puGetVecInfo: get_vec_info.o  chkopts
	-${CLINKER} -o getVecInfo get_vec_info.o ${PETSC_VEC_LIB}
	${RM} -f get_vec_info.o 

puGetSubMatrix: ../preprocessor.o ../common_funcs.o getSubMatrix.o  chkopts
	-${CLINKER} -o getSubMatrix  ../common_funcs.o ../preprocessor.o getSubMatrix.o ${PETSC_MAT_LIB}
	${RM} -f getSubMatrix.o 

puGetSubVector: ../preprocessor.o ../common_funcs.o getSubVector.o  chkopts
	-${CLINKER} -o getSubVector  ../common_funcs.o ../preprocessor.o getSubVector.o ${PETSC_MAT_LIB} ${PETSC_VEC_LIB}
	${RM} -f getSubVector.o 

puMergeLabel2Data: ../preprocessor.o ../common_funcs.o mergeLabel2Data.o  chkopts
	-${CLINKER} -o mergeLabel2Data  ../preprocessor.o ../common_funcs.o mergeLabel2Data.o ${PETSC_MAT_LIB} 
	${RM} -f mergeLabel2Data.o 

clean_make: 
	make clean
	rm getMatInfo getVecInfo getSubMatrix getSubVector mergeLabel2Data
	echo "all executable files are removed"