ALL: main
CC 	 = g++ -L. 
CFLAGS 	 = -I.	
CPPFLAGS = -std=c++11 -g -O3 #-W -Wall -Weffc++ -Wextra -pedantic -O3
LOCDIR   = .
MAIN 	 = main.cc
MANSEC   = Mat


include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

#pugixml: g++ -c pugixml.cpp pugixml.hpp
pugixml: mpicxx -c pugixml.cpp pugixml.hpp


main_libs_inst:  etimer.o common_funcs.o OptionParser.o k_fold.o svm.o config_params.o model_selection.o main_recursion.o coarsening.o loader.o ds_node.o ds_graph.o main.o chkopts
	-${CLINKER}  etimer.o common_funcs.o OptionParser.o k_fold.o svm.o config_params.o model_selection.o main_recursion.o coarsening.o loader.o ds_node.o ds_graph.o main.o  ${PETSC_MAT_LIB} -o main -lpugixml
	${RM} main.o 

main: pugixml.o etimer.o common_funcs.o OptionParser.o k_fold.o svm.o config_params.o model_selection.o main_recursion.o coarsening.o loader.o ds_node.o ds_graph.o main.o chkopts
	-${CLINKER} pugixml.o etimer.o common_funcs.o OptionParser.o k_fold.o svm.o config_params.o model_selection.o main_recursion.o coarsening.o loader.o ds_node.o ds_graph.o main.o  ${PETSC_MAT_LIB} -o main 
	${RM} main.o 
	
prepare_labels:  prepare_labels.o chkopts
	-${CLINKER}  prepare_labels.o  ${PETSC_VEC_LIB} -o prepare_labels 
	${RM} prepare_labels.o 
	
ut_main_libs_inst:  ut_mr.o main_recursion.o etimer.o model_selection.o svm.o loader.o ds_node.o ds_graph.o coarsening.o common_funcs.o config_params.o OptionParser.o ut_main.o chkopts
	-${CLINKER}  ut_mr.o  main_recursion.o etimer.o model_selection.o svm.o loader.o ds_node.o ds_graph.o coarsening.o common_funcs.o config_params.o OptionParser.o ut_main.o ${PETSC_MAT_LIB} -o ut_main -lpugixml
	${RM} ut_main.o 
	
ut_main: pugixml.o etimer.o common_funcs.o OptionParser.o k_fold.o svm.o config_params.o model_selection.o ut_mr.o main_recursion.o coarsening.o loader.o ds_node.o ds_graph.o main.o chkopts
	-${CLINKER} pugixml.o etimer.o common_funcs.o OptionParser.o k_fold.o svm.o config_params.o model_selection.o ut_mr.o main_recursion.o coarsening.o loader.o ds_node.o ds_graph.o main.o  ${PETSC_MAT_LIB} -o main 
	${RM} main.o 