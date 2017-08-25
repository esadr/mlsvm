#include "ut_common.h"
#include <iostream>

void UT_Common::load_matrix(const char * f_name, Mat& m_data, bool print){
    PetscViewer     viewer_data_;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, f_name,FILE_MODE_READ,&viewer_data_);
    MatCreate(PETSC_COMM_WORLD,&m_data);
    MatLoad(m_data,viewer_data_);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer
    if(print){
        std::cout  << "matrix in file {" << f_name <<"} is:\n";
        MatView(m_data,PETSC_VIEWER_STDOUT_WORLD);
    }
}

void UT_Common::load_vec(const char * f_name, Vec& v_input, bool print){
    PetscViewer     viewer_input_;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, f_name,FILE_MODE_READ,&viewer_input_);
    VecCreate(PETSC_COMM_WORLD,&v_input);
    VecLoad(v_input,viewer_input_);
    PetscViewerDestroy(&viewer_input_);        //destroy the viewer
    if(print){
        std::cout  << "vector in file {" << f_name <<"} is:\n";
        VecView(v_input,PETSC_VIEWER_STDOUT_WORLD);
    }
}


