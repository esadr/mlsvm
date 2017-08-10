#include "knn.h"
#include <flann/flann.hpp>
#include "loader.h"


knn::knn()
{

}


flann::Matrix& knn::load_data(Mat m_data){
    PetscInt num_row, num_col;
    PetscInt i, j, ncols;
    const PetscInt    *cols;
    const PetscScalar *vals;

    MatGetSize(m_data, &num_row, &num_col);
    std::vector<float>vecData;
//    std::vector<std::string> labels;

    for(i =0; i <num_row; i++){
        int tmp_nnz_indices=0;
        std::vector<float> vals;
        MatGetRow(m_data,i,&ncols,&cols,&vals);
        for (j=0; j<ncols; j++) {
            if(j < ncols && j==cols[tmp_nnz_indices]){
                vals.push_back(vals[j]);
            }else{
                vals.push_back(0);
            }
        }
        MatRestoreRow(m_data,i,&ncols,&cols,&vals);
        vecData.insert(vecData.end(), vals.begin(), vals.end());
    }
    flann::Matrix<float> data(vecData.data(),labels.size(),vecSize);
    std::cout << "[KNN][LD] "<< num_row << " data points are loaded successfully!\n";
    return data;
}





void knn::run_flann(Mat m_data){
    //suppose the data is loaded into matrix

    //load the flann::matrix using m_data
    Mat m_data;
    Loader ld;
    std::string raw_file_path_name = Config_params::getInstance()->get_ds_path() +
                    Config_params::getInstance()->get_ds_name() + "_data.dat";
    m_raw_data = ld.load_norm_data_sep(raw_file_path_name);

    //call flann
    flann::Index<flann::L2<float> > index(load_data(m_data), flann::KDTreeIndexParams(16));
//    index.buildIndex();

//    vector<vector<int> > indicies;
//    vector<vector<float> > dists;

//    flann::SearchParams params(128);
//    params.cores = 0; //automatic core selection
//    index.knnSearch(data, indicies, dists, numNN,  params);


    //store the indices, dists to 2 separate matrices

    //end!
}
