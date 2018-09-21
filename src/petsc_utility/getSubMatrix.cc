static char help[] = "Get a sub matrix based on range.\n"
  "  -i <filename>, where <filename> = input matrix file in PETSc binary form.\n\n";

#include "../preprocessor.h"
#include "../common_funcs.h"
//#include <petscmat.h>

int main(int argc,char **args)
{
    PetscInitialize(&argc,&args,(char*)0,help);
    
    
//    Mat Loader::load_sample_matrix_data(const char * f_name){
    Mat             m_input;
    char            in_filename[PETSC_MAX_PATH_LEN];
    char            out_path[PETSC_MAX_PATH_LEN];
    char            out_filename[PETSC_MAX_PATH_LEN];
    PetscViewer     viewer_data_;
    PetscInt        num_row_data=0,num_col_data=0, start_idx=0, end_idx=0;
    PetscBool       flg;
    PetscErrorCode  ierr;
    

    ierr = PetscOptionsGetString(NULL,NULL,"-i",in_filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate an input file name with the -i option");

    ierr = PetscOptionsGetInt(NULL,NULL,"-s", &start_idx,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a start index with the -s option");

    ierr = PetscOptionsGetInt(NULL,NULL,"-e", &end_idx,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate a end index with the -e option");

    ierr = PetscOptionsGetString(NULL,NULL,"-p",out_path,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate an output path with the -p option");

    ierr = PetscOptionsGetString(NULL,NULL,"-w",out_filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate an output file name with the -w option");


    if (start_idx > end_idx){
        printf("start index is larger than end index!\nExit\n");
        exit(1);
    }
    if (start_idx == end_idx){
        printf("start index is equal to end index!\nExit\n");
        exit(1);
    }
    
    PetscViewerBinaryOpen(PETSC_COMM_WORLD, in_filename,FILE_MODE_READ,&viewer_data_);
    MatCreate(PETSC_COMM_WORLD,&m_input);
    MatLoad(m_input,viewer_data_);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer

//    printf("Input Matrix:\n");                                               //$$debug
//    MatView(m_input,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    
    MatGetSize(m_input, &num_row_data, &num_col_data);
    PetscPrintf(PETSC_COMM_WORLD," Input matrix dim is:%dx%d\n",num_row_data,num_col_data);

    Mat sub_matrix;
    Preprocessor pp;
    pp.getSubMatrixByRange(m_input, sub_matrix, start_idx, end_idx);

//    printf("Sub Matrix:\n");                                               //$$debug
//    MatView(sub_matrix,PETSC_VIEWER_STDOUT_WORLD);                         //$$debug

    CommonFuncs cf;
    cf.exp_matrix(sub_matrix, out_path, out_filename, "getSubMatrix");

    MatDestroy(&m_input);
    MatDestroy(&sub_matrix);

    PetscFinalize();
    return 0;
}
