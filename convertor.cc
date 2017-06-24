#include "convertor.h"
#include <iostream>
#include <sstream>
#include "common_funcs.h"
#include "ut_common.h"
#include <fstream>      //read and write from/to files


void Convertor::Adj_matrix_to_edgelist(std::string in_fname, std::string out_fname)
{
    UT_Common ut_com;
    Mat m_data;
    ut_com.load_matrix(in_fname.c_str(), m_data, 0);


    PetscInt i, ncols, num_row, num_col;
    const PetscInt    *cols;
    const PetscScalar *vals;

    MatGetSize(m_data, &num_row, &num_col);
    std::fstream fs_output;
    fs_output.open(out_fname, std::fstream::out);

    // count number of edges
    PetscInt num_edge=0;
    for(i=0; i< num_row; i++){
        MatGetRow(m_data,i,&ncols,&cols,&vals);
        num_edge += ncols;
        MatRestoreRow(m_data,i,&ncols,&cols,&vals);
    }
    //write the header line (#nodes, #edges)
    fs_output << "#" << std::to_string(num_row) << ",#" << std::to_string(num_edge) << std::endl;
    std::cout << "number of nodes:" << std::to_string(num_row) << ", number of edges:" << std::to_string(num_edge) << std::endl;

    //write the edges
    int start_index = 1;
    for(i=0; i< num_row; i++){
        MatGetRow(m_data,i,&ncols,&cols,&vals);
        for(int j=0; j < ncols ; j++){
            print_a_line(fs_output, (NodeId)i + start_index, (NodeId)cols[j] + start_index, (EdgeWeight)vals[j]);    //
        }
        MatRestoreRow(m_data,i,&ncols,&cols,&vals);
    }
    fs_output.close();

}

void Convertor::print_a_line(std::fstream& fs_out, NodeId first_node, NodeId second_node, EdgeWeight edge_weight){
    fs_out << std::to_string(first_node) << "," << std::to_string(second_node) << "," << std::to_string(edge_weight) << std::endl;
}

PetscInt Convertor::get_num_rows(std::string in_file_name){
    CommonFuncs cf;
    std::string res = cf.run_ext_command_single_output("wc -l " + in_file_name);
    return stoi(res);
}

PetscInt Convertor::get_num_columns(std::string in_file_name){
    CommonFuncs cf;
    std::string res = cf.run_ext_command_single_output("cat "+ in_file_name +" | tr ' ' '\n' | awk -F: '{print $1}' | grep -Eo '[0-9]+'  | sort -rn  | head -n 1");
    return stoi(res);
}

void Convertor::Libsvm_file_to_PETSc_format(std::string in_file_name, Mat& m_data, Vec& v_lbl, PetscInt num_row, PetscInt num_col){
    if(num_row == -1)
        num_row = get_num_rows(in_file_name);
    if(num_col == -1)
        num_col = get_num_columns(in_file_name);

    std::cout << "row: " << num_row << ", col: " << num_col << std::endl;

    VecCreateSeq(PETSC_COMM_SELF,num_row,&v_lbl);
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row ,num_col + 1 ,(num_col + 1 ),PETSC_NULL, &m_data); //+1 is for label
    //read the file withlibsvm format
    std::ifstream file(in_file_name);
    std::string str;
    PetscInt curr_row=0;
    while (std::getline(file, str))
    {
        std::stringstream ss(str);
        std::string item;
        std::vector<std::string> tokens;
        while (getline(ss, item, ' ')) {    //split by space, tokens[0] is label, and the rest of them are index:value
            tokens.push_back(item);
        }

        VecSetValue(v_lbl, curr_row, stoi(tokens[0]), INSERT_VALUES);
        for(int i=0; i< tokens.size(); i++){
            PetscInt idx;
            PetscScalar val;
            auto pos = tokens[i].find(":");
            if (pos != std::string::npos) {
//                std::cout << "pos: " << pos << std::endl;
                idx= stoi(tokens[i].substr(0, pos)) - 1;                        //real index from file and -1 is for PETSc index that is from zero not one
                val = stof(tokens[i].substr(pos+1,tokens[i].size()-1));       //value corresponding to the idx index
//                std::cout << "idx: " << idx << ", val: " << val << std::endl;
            }
            MatSetValue(m_data,curr_row,idx, val,INSERT_VALUES);
        }
        curr_row++;
    }
    VecAssemblyBegin(v_lbl);
    VecAssemblyEnd(v_lbl);
    MatAssemblyBegin(m_data,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_data,MAT_FINAL_ASSEMBLY);

    CommonFuncs cf;
    cf.exp_matrix(m_data, "", in_file_name + "_data.dat", "Libsvm_file_to_PETSc_format" );
    cf.exp_vector(v_lbl, "", in_file_name + "_label.dat", "Libsvm_file_to_PETSc_format" );
    MatDestroy(&m_data);
    VecDestroy(&v_lbl);
}


