static char help[] = "Get a sub vector based on range.\n"
  "  -i <filename>, where <filename> = input vector file in PETSc binary form.\n\n";

#include "../preprocessor.h"
#include "../common_funcs.h"
#include <cassert>

void load_matrix(const char * file_path, const char * file_name,
                 Mat& m_data, bool print=0);
void load_vec(const char * file_path, const char * file_name,
              Vec& v_input, bool print=0);



int main(int argc,char **args)
{
    PetscInitialize(&argc,&args,(char*)0,help);
    
    Vec             v_label;
    Mat             m_data, m_label_data;
    char            data_filename[PETSC_MAX_PATH_LEN];
    char            label_filename[PETSC_MAX_PATH_LEN];
    char            in_out_path[PETSC_MAX_PATH_LEN];
    char            label_data_filename[PETSC_MAX_PATH_LEN];
    PetscInt        num_row_data=0, num_row_label=0, num_col=0;
    PetscBool       flg;
    PetscErrorCode  ierr;
    
    ierr = PetscOptionsGetString(NULL,NULL,"-p",
                        in_out_path,PETSC_MAX_PATH_LEN,&flg); CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,
                      "Must indicate an both input and output path with the -p option");

    ierr = PetscOptionsGetString(NULL,NULL,"-d",
                        data_filename,PETSC_MAX_PATH_LEN,&flg); CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,
                      "Must indicate a data file name with the -d option");

    ierr = PetscOptionsGetString(NULL,NULL,"-l",
                        label_filename,PETSC_MAX_PATH_LEN,&flg); CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,
                      "Must indicate a label file name with the -l option");

    ierr = PetscOptionsGetString(NULL,NULL,"-w",
                        label_data_filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,
                      "Must indicate a label-data file name with the -w option");
    // Load inputs
    load_matrix(in_out_path, data_filename, m_data);
    load_vec(in_out_path, label_filename, v_label);

    // Check inputs
    VecGetSize(v_label, &num_row_label);
    printf("Input vector lenght is:%d\n", num_row_label);

    MatGetSize(m_data, &num_row_data, &num_col);
    printf("Input data dim is:%dx%d\n", num_row_data, num_col);

    assert(num_row_data == num_row_label &&
           "length of data and label are not match!");

    CommonFuncs cf;
    cf.addLabel2Data(m_data, v_label, m_label_data);
    cf.exp_matrix(m_label_data, in_out_path, label_data_filename, "mergeLabel2Data");

    VecDestroy(&v_label);
    MatDestroy(&m_data);
    MatDestroy(&m_label_data);

    PetscFinalize();
    return 0;
}


void load_matrix(const char * file_path,const char * file_name,
                 Mat& m_data, bool print){

    PetscViewer     viewer_data_;

    std::string     full_file;
    if (strlen(file_path)){
        full_file = file_path;
        full_file += "/";
        full_file += file_name;
    }else
        full_file = file_name;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, full_file.c_str(),
                          FILE_MODE_READ,&viewer_data_);
    MatCreate(PETSC_COMM_WORLD,&m_data);
    MatLoad(m_data,viewer_data_);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer
    if(print){
        std::cout  << "matrix in file {" << file_name <<"} is:\n";
        MatView(m_data,PETSC_VIEWER_STDOUT_WORLD);
    }
}

void load_vec(const char * file_path, const char * file_name,
              Vec& v_input, bool print){

    PetscViewer     viewer_input_;

    std::string     full_file;
    if (strlen(file_path)){
        full_file = file_path;
        full_file += "/";
        full_file += file_name;
    }else
        full_file = file_name;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, full_file.c_str(),
                          FILE_MODE_READ,&viewer_input_);
    VecCreate(PETSC_COMM_WORLD,&v_input);
    VecLoad(v_input,viewer_input_);
    PetscViewerDestroy(&viewer_input_);        //destroy the viewer
    if(print){
        std::cout  << "vector in file {" << file_name <<"} is:\n";
        VecView(v_input,PETSC_VIEWER_STDOUT_WORLD);
    }
}

