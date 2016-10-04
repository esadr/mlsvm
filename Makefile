ALL: main
CC 	 = g++ -L. 
CFLAGS 	 = -I.	
CPPFLAGS = -std=c++11 -g -O3 #-W -Wall -Weffc++ -Wextra -pedantic -O3
LOCDIR   = .
MAIN 	 = main.cc
MANSEC   = Mat

LIBS= -lpugixml -lm

MLSVM_SRCS = pugixml.cc etimer.cc common_funcs.cc OptionParser.cc k_fold.cc svm_weighted.cc config_params.cc model_selection.cc solver.cc partitioning.cc refinement.cc main_recursion.cc coarsening.cc loader.cc ds_node.cc ds_graph.cc main.cc
MLSVM_OBJS = $(MLSVM_SRCS:.cc=.o)

SLSVM_SRCS = pugixml.cc etimer.cc common_funcs.cc OptionParser.cc k_fold.cc svm.cc config_params.cc model_selection.cc solver.cc loader.cc ds_node.cc ds_graph.cc main_sl.cc
SLSVM_OBJS = $(SLSVM_SRCS:.cc=.o)

UT_SRCS= svm_weighted.cc solver.cc model_selection.cc ut_ms.cc ut_common.cc ut_kf.cc ut_partitioning.cc ds_node.cc ds_graph.cc coarsening.cc partitioning.cc ut_mr.cc pugixml.cc config_params.cc etimer.cc ut_cf.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc ut_main.cc
#svm.cc  model_selection.cc main_recursion.cc coarsening.cc  ds_node.cc ds_graph.cc 
UT_OBJS = $(UT_SRCS:.cc=.o)

CV_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc ./tools/cross_validation.cc
CV_OBJS = $(CV_SRCS:.cc=.o)

SAT_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc svm_unweighted.cc solver.cc ./tools/single_svm_train.cc
SAT_OBJS = $(SAT_SRCS:.cc=.o)

SATIW_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc svm_weighted.cc solver.cc ./tools/single_svm_train_instance_weight.cc
SATIW_OBJS = $(SATIW_SRCS:.cc=.o)


SAP_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc svm_weighted.cc solver.cc ./tools/single_svm_predict.cc
SAP_OBJS = $(SAP_SRCS:.cc=.o)

PERS_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc k_fold.cc svm_unweighted.cc solver.cc model_selection.cc personalized.cc personalized_main.cc
PERS_OBJS = $(PERS_SRCS:.cc=.o)

TestMatrix_SRCS= pugixml.cc config_params.cc etimer.cc common_funcs.cc OptionParser.cc loader.cc ./tools/test_matrix.cc
TestMatrix_OBJS = $(TestMatrix_SRCS:.cc=.o)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

info: 
	-@echo $(LIBS)
	-@echo $(MLSVM_OBJS)
#pugixml: g++ -c pugixml.cpp pugixml.hpp
pugixml: mpicxx -c pugixml.cpp pugixml.hpp


main_libs_inst:  etimer.o common_funcs.o OptionParser.o k_fold.o svm.o config_params.o model_selection.o solver.o partitioning.o refinement.o main_recursion.o coarsening.o loader.o ds_node.o ds_graph.o main.o chkopts
	-${CLINKER}  etimer.o common_funcs.o OptionParser.o k_fold.o svm.o config_params.o model_selection.o solver.o partitioning.o refinement.o main_recursion.o coarsening.o loader.o ds_node.o ds_graph.o main.o  ${PETSC_MAT_LIB} -o main $(LIBS)
	${RM} main.o 

main: $(MLSVM_OBJS) chkopts
	-${CLINKER} $(MLSVM_OBJS)  ${PETSC_MAT_LIB} -o main
	${RM} main.o 
	
main_sl: $(SLSVM_OBJS) chkopts			# single level (no multi level which means no v-cycle)
	-${CLINKER} $(SLSVM_OBJS)  ${PETSC_MAT_LIB} -o slsvm 
	${RM} main_sl.o 

prepare_labels:  prepare_labels.o chkopts
	-${CLINKER}  prepare_labels.o  ${PETSC_VEC_LIB} -o prepare_labels 
	${RM} prepare_labels.o 
	
ut_main_libs_inst:  ut_mr.o main_recursion.o etimer.o model_selection.o svm.o loader.o ds_node.o ds_graph.o coarsening.o common_funcs.o config_params.o OptionParser.o ut_main.o chkopts
	-${CLINKER}  ut_mr.o  main_recursion.o etimer.o model_selection.o svm.o loader.o ds_node.o ds_graph.o coarsening.o common_funcs.o config_params.o OptionParser.o ut_main.o ${PETSC_MAT_LIB} -o ut_main -lpugixml
	${RM} ut_main.o 
	
ut_main: $(UT_OBJS) chkopts
	-${CLINKER} $(UT_OBJS)  ${PETSC_MAT_LIB} -o ut_main 
	${RM} ut_main.o 
	
	
	
main_test: $(MLSVM_OBJS) chkopts
	-${CLINKER}  $(MLSVM_OBJS)  ${PETSC_MAT_LIB} -o main $(LIBS)
	${RM} main.o 


cv: $(CV_OBJS) chkopts
	-${CLINKER} $(CV_OBJS)  ${PETSC_MAT_LIB} -o cv 
	${RM} cv.o 

sat_normal: $(SAT_OBJS) chkopts
	-${CLINKER} $(SAT_OBJS)  ${PETSC_MAT_LIB} -o sat_normal 
	${RM} sat_normal.o 
	
sat_weighted: $(SATIW_OBJS) chkopts
	-${CLINKER} $(SATIW_OBJS)  ${PETSC_MAT_LIB} -o sat_weighted 
	${RM} sat_weighted.o 


sap: $(SAP_OBJS) chkopts
	-${CLINKER} $(SAP_OBJS)  ${PETSC_MAT_LIB} -o sap 
	${RM} sap.o 
	
pers: $(PERS_OBJS) chkopts
	-${CLINKER} $(PERS_OBJS)  ${PETSC_MAT_LIB} -o pers
	${RM} pers.o 

testmatrix: $(TestMatrix_OBJS) chkopts
	-${CLINKER} $(TestMatrix_OBJS)  ${PETSC_MAT_LIB} -o testmatrix
	${RM} testmatrix.o 
