static char help[] = "Get a sub vector based on range.\n"
  "  -i <filename>, where <filename> = input vector file in PETSc binary form.\n\n";

#include "../preprocessor.h"
#include "../common_funcs.h"

int main(int argc,char **args)
{
    PetscInitialize(&argc,&args,(char*)0,help);
    
    Vec             in_vector;
    char            in_filename[PETSC_MAX_PATH_LEN];
    char            out_path[PETSC_MAX_PATH_LEN];
    char            out_filename[PETSC_MAX_PATH_LEN];
    PetscViewer     viewer_data_;
    PetscInt        num_row_data=0, start_idx=0, end_idx=0;
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
    VecCreate(PETSC_COMM_WORLD,&in_vector);
    VecLoad(in_vector,viewer_data_);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer

//    printf("Input Vector:\n");                                               //$$debug
//    VecView(in_vector,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    
    VecGetSize(in_vector, &num_row_data);
    PetscPrintf(PETSC_COMM_WORLD," Input vector lenght is:%d\n", num_row_data);

    Vec sub_vector;
    Preprocessor pp;
    pp.getSubVectorByRange(in_vector, sub_vector, start_idx, end_idx);

//    printf("Sub Vector:\n");                                               //$$debug
//    VecView(sub_vector,PETSC_VIEWER_STDOUT_WORLD);                         //$$debug

    CommonFuncs cf;
    cf.exp_vector(sub_vector, out_path, out_filename, "getSubVector");

    VecDestroy(&in_vector);
    VecDestroy(&sub_vector);

    PetscFinalize();
    return 0;
}
