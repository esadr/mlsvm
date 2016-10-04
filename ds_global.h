#ifndef DS_GLOBAL_H
#define DS_GLOBAL_H

#include <map>
#include <vector>
#include <cstdint>

            /* This file would be useful in case I need to
             * use PETSc data types instead of stdint or normal data types in future
             */

typedef uint_fast64_t Index;        // more expressive name for an Index into an array
typedef uint_fast64_t Count;        // more expressive name for an integer quantity
typedef Index NodeId;               // node indices are 0-based
typedef double EdgeWeight;          // edge weight type
typedef Index EdgeId;               // edge id
typedef double Volume;              // Volume, future volume have decimal places
typedef int SmallSize;              // I don't think I need a very large number
typedef uint_fast64_t LargeSize;    // For large volume of objects, I can use this one


//remember to update measures which are defined in the model selection header file as enum (all supported metrics)
enum measures {TP, TN, FP, FN, Acc, Sens, Spec, Gmean, F1, PPV, NPV};

//struct iter_summary{
//    double C, gamma;
//    std::map<measures, double> result;
//};



struct solution{
    std::vector<int> p_index;
    std::vector<int> n_index;
    double C, gamma;
};



struct summary{
    int iter;
//    double gmean;
    std::map<measures,double> perf;
    int num_SV_p;
    int num_SV_n;
//    int num_SV_t;
    double C;
    double gamma;
    bool operator > (const summary new_) const{
//        return ((this->perf.at(Gmean) - new_.perf.at(Gmean)) > 0.001);
        return (this->perf.at(Gmean) > new_.perf.at(Gmean));
    }
    int selected_level = -1;
};

struct ref_results{
    summary validation_data_summary;
    summary test_data_summary;
    int level;
};



#endif // DS_GLOBAL_H



            /* I defined some of the types from below url (Jun 12, 2015 - 12:00)
             * http://en.cppreference.com/w/cpp/types/integer
             */
