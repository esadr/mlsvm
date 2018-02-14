#include "../loader.h"


Config_params* Config_params::instance = NULL;

int main(int argc, char **argv)
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    Loader ld ;
    Mat NN;
    NN = ld.load_norm_data_sep("./data/m_filteredNN.dat");



    PetscInt i, ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;
    PetscInt num_col, num_row;

    MatGetSize(NN, &num_row, &num_col);   //set the number of columns

    for(i =0; i < num_row ; i++){
        MatGetRow(NN,i,&ncols,&cols,&vals);
        for(int j=0; j < ncols ; j++){
            if(vals[j] == i)
                std::cout << "problem in row: " << i << " ,j:"<<j << " ,cols[j] " << cols[j] << std::endl;
        }
        MatRestoreRow(NN,i,&ncols,&cols,&vals);
    }

    std::cout << "finished successfully!" << std::endl;
    PetscFinalize();
    return 0;
}
