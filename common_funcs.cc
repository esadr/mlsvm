#include "common_funcs.h"
#include "config_logs.h"
/*
 * Export matrix to a file
 */
void CommonFuncs::exp_matrix(Mat& A, std::string file_path, std::string file_name){
    exp_matrix(A,file_path,file_name,"N/A");
}

void CommonFuncs::exp_matrix(Mat& A, std::string file_path, std::string file_name, std::string sender_func){
#if debug_export == 1
    PetscViewer     viewer;
    std::string     full_file = file_path +"/" +file_name;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,full_file.c_str(),FILE_MODE_WRITE,&viewer);
    MatView(A,viewer);  // petsc binary format

    PetscViewerDestroy(&viewer);        //destroy the viewer
    std::cout<< "[CF][exp_matrix] {"<< sender_func <<"} Matrix exported to file: "<< full_file << std::endl;
#endif
#if debug_export == 0
    std::cout<< "The export functionality is disabled in the config_log.h. You can modify and recompile the code in case you need it." << std::endl;
#endif
}

void CommonFuncs::exp_vector(Vec& A, std::string file_path, std::string file_name){
    exp_vector(A,file_path,file_name,"N/A");
}

void CommonFuncs::exp_vector(Vec& A, std::string file_path, std::string file_name, std::string sender_func){
#if debug_export == 1
    PetscViewer     viewer;
    std::string     full_file = file_path +"/" +file_name;
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,full_file.c_str(),FILE_MODE_WRITE,&viewer);
    VecView(A,viewer);  // petsc binary format

    PetscViewerDestroy(&viewer);        //destroy the viewer
    std::cout<< "[CF][exp_vector] {"<< sender_func <<"} Vector exported to file: "<< full_file << std::endl;
#endif
#if debug_export == 0
    std::cout<< "The export functionality is disabled in the config_log.h. You can modify and recompile the code in case you need it." << std::endl;
#endif
}

/*
 * Gets 2 vector and calculate the Eucidean distance
 */
double CommonFuncs::calc_euclidean_dist(const PetscInt ncols_A, const PetscInt ncols_B,
                                        const PetscInt *cols_A, const PetscInt *cols_B,
                                        const PetscScalar *vals_A, const PetscScalar *vals_B){
    PetscInt            it_p=0, it_n=0;
    double              current_distance=0;

    while( it_p < ncols_A || it_n < ncols_B ){
        if(it_p == ncols_A){                                    // we reached to end of A
            while(it_n < ncols_B){                              // add the rest from B
                current_distance += pow(vals_B[it_n], 2 );
                it_n++;
            }                                                   // B is finished too
            continue;                       //  skip the rest since we are done completely
        }

        if(it_n == ncols_B){                                    // we reached to end of B
            while(it_p < ncols_A){                              // add the rest from A
                current_distance += pow(vals_A[it_p], 2 );
                it_p++;
            }                                                   // A is finished too
            continue;                       //  skip the rest since we are done completely
        }

        // - - - - - neither P or N have not finished yet - - - - -
        if(cols_A[it_p] == cols_B[it_n]){                       // Common index on both A, B
            current_distance += pow(  (vals_A[it_p] - vals_B[it_n]) , 2 );  // calc their difference
            it_p++; it_n++;
            continue;
        }
        if(cols_A[it_p] < cols_B[it_n] ){                       // A has some values while B has nothing
            current_distance += pow(vals_A[it_p], 2 );
            it_p++;
            continue;
        }else{                                                  // B has some values while A has nothing
            current_distance += pow(vals_B[it_n], 2 );
            it_n++;
            continue;
        }
    }
    return sqrt(current_distance);
}





















//void make_full_vector(PetscInt num_features, PetscInt ncols, const PetscInt *cols, const PetscScalar *vals, PetscScalar * arr_output ){
//    int i=0, j=0;
//    for(j=0; j < ncols; j++){
//        while(i < cols[j]){   //fill zeros
//            printf("inside make_full_vector: 1st while i:%d\n",i);
//            arr_output[i]=0;
//            i++;
//        }
//        if(i == cols[j]){     //fill a value
//            printf("inside make_full_vector: i:%d, j:%d, vals[j]:%g\n",i,j, vals[j]);
//            arr_output[i]= vals[j];
//            i++;
//        }
//    }
//    while(i < num_features){    //fill the rest of zeros
//        printf("inside make_full_vector: 2nd while i:%d\n",i);
//        arr_output[i]=0;
//        i++;
//    }
//}
