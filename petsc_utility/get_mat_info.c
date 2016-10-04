static char help[] = "Load the input matrix from a file.\n"
  "Print it and print its dimension.\n"
  "The command line options are:\n"
  "  -in_file <filename>, where <filename> = input matrix file in PETSc binary form.\n\n";

#include <petscmat.h>

int main(int argc,char **args)
{
    PetscInitialize(&argc,&args,(char*)0,help);
    
    
//    Mat Loader::load_sample_matrix_data(const char * f_name){
    Mat             m_input;
    char            in_filename[PETSC_MAX_PATH_LEN];
    PetscViewer     viewer_data_;
    PetscInt        num_row_data=0,num_col_data=0;
    PetscBool       flg;
    PetscErrorCode  ierr;
    

    ierr = PetscOptionsGetString(NULL,"-in_file",in_filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate an input file name with the -in_file option");
    
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, in_filename,FILE_MODE_READ,&viewer_data_);
    MatCreate(PETSC_COMM_WORLD,&m_input);
    MatLoad(m_input,viewer_data_);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer


    

    printf("Input Matrix:\n");                                               //$$debug
    MatView(m_input,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    
    MatGetSize(m_input, &num_row_data, &num_col_data);
    PetscPrintf(PETSC_COMM_WORLD," Input matrix dim is:%dx%d\n",num_row_data,num_col_data);
    
    MatDestroy(&m_input);

    PetscFinalize();
    return 0;
}
