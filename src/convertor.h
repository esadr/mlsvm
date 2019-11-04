#ifndef CONVERTOR_H
#define CONVERTOR_H

#include "ds_global.h"
#include <fstream>
#include <string>
#include "petscmat.h"

class Convertor
{
    void print_a_line(std::fstream& fs_out, NodeId first_node, NodeId second_node, EdgeWeight edge_weight);
    PetscInt get_num_rows(std::string in_file_name);
    PetscInt get_num_columns(std::string in_file_name);
public:
    void Adj_matrix_to_edgelist(std::string in_fname, std::string out_fname);

    void Libsvm_file_to_PETSc_format(std::string in_file_name, Mat& m_data, Vec& v_lbl, PetscInt num_row =-1, PetscInt num_col=-1);

    void CSV_file_to_PETSc_format();
    void CSV_file_to_PETSc_Vector();
};

#endif // CONVERTOR_H
