
#include <iostream>
//#include <vector>
#include <algorithm>            //for using std::sort
#include "petscvec.h"
#include <fstream>

Vec read_input_vector(const std::string f_name);
void make_binary_labels(Vec& v_label,int minority_label, PetscInt inverse_status=0);
void write_output(char f_name[PETSC_MAX_PATH_LEN], Vec& v_label);    //write the output to file
Vec read_input_vector_from_text(char f_name[PETSC_MAX_PATH_LEN], PetscInt num_lines);

int main(int argc, char *argv[])
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    
    PetscPrintf(PETSC_COMM_WORLD,"correct usage is:\n -in_file (input file name) -out_file (output file name) -n (#lines in file) -m (minority_label) -inverse (0 minority class is smaller than majority, 1 reverse(e.g. class 2 in BMW)) ");
    
    char           in_filename[PETSC_MAX_PATH_LEN];
    char           out_filename[PETSC_MAX_PATH_LEN];
    PetscInt       num_lines=0, minority_label, inverse;
    PetscBool      flg;
    PetscErrorCode ierr;

    ierr = PetscOptionsGetString(NULL,NULL,"-in_file",in_filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate an input file name with the -in_file option");

    ierr = PetscOptionsGetString(NULL,NULL,"-out_file",out_filename,PETSC_MAX_PATH_LEN,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate an output file name with the -out_file option");

    ierr = PetscOptionsGetInt(NULL,NULL,"-n",&num_lines,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate number of lines in file with the -n option");
    
    ierr = PetscOptionsGetInt(NULL,NULL,"-m",&minority_label,&flg);CHKERRQ(ierr);
    if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate minority_label with the -m option");
    
    ierr = PetscOptionsGetInt(NULL,NULL,"-inverse",&inverse,&flg);CHKERRQ(ierr);
//     if (!flg) SETERRQ(PETSC_COMM_WORLD,1,"Must indicate minority_label with the -m option");
    
    
    Vec v_labels;
    v_labels = read_input_vector_from_text(in_filename,num_lines);
    make_binary_labels(v_labels, minority_label,inverse);
    write_output(out_filename, v_labels);
    PetscFinalize();
    return 0;
}

void make_binary_labels(Vec& v_labels, PetscInt minority_label, PetscInt inverse_status){
    PetscInt        i,label_size=0;
    PetscScalar     *arr_labels_;

    VecGetSize(v_labels, &label_size);

    PetscMalloc1(label_size, &arr_labels_);
    VecGetArray(v_labels,&arr_labels_);
    if(inverse_status){                         // special case when the size of expected class is larger than sum of the rest of the classes
        for(i=0;i<label_size;i++){
            if(arr_labels_[i] == minority_label){
                arr_labels_[i] = -1;
            }else{
                arr_labels_[i] = +1;    //majority (other classes)
            }
        }
    }else{
        for(i=0;i<label_size;i++){
            if(arr_labels_[i] == minority_label){
                arr_labels_[i] = +1;
            }else{
                arr_labels_[i] = -1;    //majority (other classes)
            }
        }
    }
    VecRestoreArray(v_labels, &arr_labels_);
    PetscFree(arr_labels_);
//     printf("labels after modification:\n");                                               //$$debug
//     VecView(v_labels,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    
//     return v_labels;   //here is contains binary values
}

Vec read_input_vector_from_text(char f_name[PETSC_MAX_PATH_LEN], PetscInt num_lines){
    Vec v_labels;
//    VecCreateSeq(PETSC_COMM_SELF,num_lines,&v_labels);
    PetscScalar * arr_labels;
    PetscInt curr_line=0;
    PetscMalloc1(num_lines, &arr_labels);

    std::string line;
    std::ifstream myfile(f_name);
    if (myfile.is_open())  {
        while ( std::getline(myfile,line) )
        {
            arr_labels[curr_line] = std::stoi(line);
            curr_line++;
        }
        myfile.close();
        
        if(curr_line != num_lines){
            PetscPrintf(PETSC_COMM_WORLD,"[Error] number of lines in file are:%d\n",curr_line);
            exit(1);
        }
        VecCreateSeqWithArray(MPI_COMM_SELF,1,num_lines,arr_labels,&v_labels);
    }else{
        std::cout << "Unable to open file\n";
    }
//    VecDestroy(&v_labels);
//    PetscFree(arr_labels);

//     printf("Sample Vector:\n");                                               //$$debug
//     VecView(v_labels,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    return v_labels;
}

Vec read_input_vector(const std::string f_name){
    Vec             v_data_;
    PetscViewer     viewer_data_;

    PetscViewerBinaryOpen(PETSC_COMM_WORLD, f_name.c_str(),FILE_MODE_READ,&viewer_data_);
    VecCreate(PETSC_COMM_WORLD,&v_data_);
    VecLoad(v_data_,viewer_data_);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer


    PetscPrintf(PETSC_COMM_WORLD, "input Vector:\n");                                               //$$debug
    VecView(v_data_,PETSC_VIEWER_STDOUT_WORLD);                                //$$debug
    return v_data_;
}

//void write_output(const std::string f_name, Mat m_Out){    //write the output to file
void write_output(char f_name[PETSC_MAX_PATH_LEN], Vec& v_label){    //write the output to file
    PetscViewer     viewer_data_;


//    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_name.c_str(),FILE_MODE_WRITE,&viewer_data_);
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,f_name, FILE_MODE_WRITE,&viewer_data_);
    VecView(v_label,viewer_data_);
   PetscPrintf(PETSC_COMM_WORLD,"\nOutput matrix is written to file %s\n\n",f_name);
    PetscViewerDestroy(&viewer_data_);        //destroy the viewer
}
