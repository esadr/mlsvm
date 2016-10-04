All: get_mat_info
#LOCDIR   = src/mat/examples/
#DIRS     = tests tutorials
CC 	 = g++ -L. 
CFLAGS 	 = -I.	
CPPFLAGS = -std=c++11 -O3 

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test

get_mat_info: get_mat_info.o  chkopts
	-${CLINKER} -o get_mat_info get_mat_info.o ${PETSC_MAT_LIB}
	${RM} -f get_mat_info.o 

get_vec_info: get_vec_info.o  chkopts
	-${CLINKER} -o get_vec_info get_vec_info.o ${PETSC_VEC_LIB}
	${RM} -f get_vec_info.o 
