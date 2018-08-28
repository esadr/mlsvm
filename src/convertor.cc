#include "convertor.h"
#include <iostream>
#include <sstream>
#include "common_funcs.h"
#include "ut_common.h"
#include <fstream>      //read and write from/to files
#include "loader.h"

void Convertor::Adj_matrix_to_edgelist(std::string in_fname, std::string out_fname)
{
    //depreacted
    // @080517-1135 modified to use loader class instead during developing the CSV_PETSc
//    UT_Common ut_com;
//    ut_com.load_matrix(in_fname.c_str(), m_data, 0);

    Loader ld;
    ld.load_norm_data_sep(in_fname);

    Mat m_data;
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
        num_row = get_num_rows(in_file_name + ".libsvm");
    if(num_col == -1)
        num_col = get_num_columns(in_file_name + ".libsvm");

    std::cout << "row: " << num_row << ", col: " << num_col << std::endl;

    VecCreateSeq(PETSC_COMM_SELF,num_row,&v_lbl);
    MatCreateSeqAIJ(PETSC_COMM_SELF,num_row ,num_col ,num_col ,PETSC_NULL, &m_data);
    //read the file withlibsvm format
    // std::ifstream file(in_file_name);
    std::ifstream file(in_file_name + ".libsvm");
    std::string str;
    PetscInt curr_row=0;
    while (std::getline(file, str))
    {
        std::stringstream ss(str);
        std::string item;
        std::vector<std::string> tokens;
        while (getline(ss, item, ' ')) {    //split by space, tokens[0] is label, and the rest of them are index:value
            if(item!="")
		tokens.push_back(item);
        }

        VecSetValue(v_lbl, curr_row, stoi(tokens[0]), INSERT_VALUES);
        for(int i=1; i< tokens.size(); i++){
            unsigned int idx;
            PetscScalar val;
            auto pos = tokens[i].find(":");
            if (pos != std::string::npos && pos <= num_col) {
                try{
                //real index from file and -1 is for PETSc index that is from zero not one
                idx= stoi(tokens[i].substr(0, pos)) - 1;

                //value corresponding to the idx index
                val = stof(tokens[i].substr(pos+1,tokens[i].size()-1));
                }catch (const std::runtime_error& e) {
                    std::cout << e.what();
                    std::cout << "pos: " << pos << std::endl;
                    std::cout << "curr_row:"<<curr_row << ",i:" <<i<< ",idx: " << idx << ", val: " << val << std::endl;
                }
                MatSetValue(m_data,curr_row,idx, val,INSERT_VALUES);
            }
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






void Convertor::CSV_file_to_PETSc_format(){
    //find the level in the summary
    std::fstream in_file;
    std::string fname = Config_params::getInstance()->get_ds_path()
                        + Config_params::getInstance()->get_ds_name() + ".csv";
//    const std::string fname = "./datasets/Wimbledon-men-2013_NoMissing.csv";
//    const std::string fname = "./datasets/b.csv";
    //    std::cout << fname << std::endl;
    in_file.open(fname);

    if(!in_file.is_open()){
        std::cout << "[RCF] failed to open " << fname <<" file! \nExit";
        exit(1);
    }
    //    std::vector<std::pair<int,int>> v_lines;
    std::string line;


    int line_num = 0;
    std::vector<std::vector<double>> vv_data;
    while(std::getline(in_file, line)){
        std::stringstream sep(line);
        std::string item;
        vv_data.push_back(std::vector<double>());
        while (std::getline(sep, item, ',')) {
            try{
                vv_data.back().push_back(stod(item));
            }
            catch(const std::exception e){
                std::cout << e.what() ;
                std::cout << "A good guess:\n" << "data: " <<
                             item << " at line " << line_num <<
                             " is not convertable to double!" << std::endl;
                std::cout << "Please check the file, \nExit!\n";
                exit(1);
            }
        }
        line_num++;
    }
    in_file.close();


    Mat m_data;
    PetscInt num_row= vv_data.size();
    PetscInt num_col= vv_data[0].size();    //number of items in the first row
    std::cout << "num_rows: "<<num_row<< ", num_cols:" << num_col << std::endl;

    Vec v_lbl;
    VecCreateSeq(PETSC_COMM_SELF,num_row,&v_lbl);
    // Notice the Dense matrix is column major order. I need to fill the columns
    // For performance, I insert each row of a file to a column in matrix
    // In the end, I transpose the matrix before save to file

    MatCreateSeqDense(PETSC_COMM_SELF, num_col-1, num_row, NULL, &m_data);

    for(int i=0; i< num_row; i++ ){
        if(abs(vv_data[i][0]) != 1){
            std::cout << "[RCF] Wrong label is provided at row " << i <<" with value "<< vv_data[i][0] <<" \nExit";
            exit(1);
        }

        VecSetValue(v_lbl, i, vv_data[i][0], INSERT_VALUES);
        for(int j=0; j< num_col-1; j++){
            MatSetValue(m_data,j ,i , vv_data[i][j+1], INSERT_VALUES);
        }
    }
    VecAssemblyBegin(v_lbl);
    VecAssemblyEnd(v_lbl);
    MatAssemblyBegin(m_data, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(m_data, MAT_FINAL_ASSEMBLY);

    Mat m_data_t;
    MatTranspose(m_data,MAT_INITIAL_MATRIX,&m_data_t);
    MatDestroy(&m_data);

//    std::cout  << "[MCP][RDF] m_data_t matrix:\n";                       //$$debug
//    MatView(m_data_t, PETSC_VIEWER_STDOUT_WORLD);

    std::string out_fname = Config_params::getInstance()->get_ds_path()
                            + Config_params::getInstance()->get_ds_name();
    CommonFuncs cf;
    cf.exp_matrix(m_data_t, "", out_fname + "_data.dat", "CSV_PETSc" );
    cf.exp_vector(v_lbl, "", out_fname + "_label.dat", "CSV_PETSc" );
    MatDestroy(&m_data_t);
    VecDestroy(&v_lbl);


//TODO list: handle the empty lines Aug 24, 2018

}

