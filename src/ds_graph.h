#ifndef DS_GRAPH_H
#define DS_GRAPH_H

//#include <petscmat.h>
#include <map>
#include "ds_node.h"


class Graph{

private:
    int level_;
    Volume avg_future_volume_;
    std::vector<Node> g_nodes_;

public:
    //Constructor
    Graph();

    //Deconstrutor
    ~Graph() { g_nodes_.clear();}
        //Destroy the Nodes
        //TODO: Destroy the P Matrix

    //Accessors
    int getSize() const {return g_nodes_.size();}

    int getLevel() const {return level_; }

    Node& getNode(NodeId)  ;

    int getNumSeeds() const;
        //returns the number of seeds

    void getSeedsIndices(std::map<int,int>&, int&);

    //Mutators
//    void setSize(int);
        // improve the performance by preallocating the memory for Vector of Nodes correctly in advance

    void addNode(Node&);


    void setLevel(const int level) {level_ = level; }

    void setAvgFutureVolume(Volume);

    //Operators

//    void calcAllFutureVolume();
//        //calculate the future volume for all nodes belong to this graph

//    void calcFutureVolume(NodeId&);
//        //calculate the future volume only for the input node

//    Volume calculateAverageFutureVolume();


    int selectSeeds(); //NOTE: Always notice that set avg-future-volume in advance

    void printFutureVolumes() const;

    void printSeeds() const;

//    bool calcualtePMat();
//        //only calculate the P matrix

//    std::vector<NodeId> find_seed_indices(int num_seeds) const;
    void find_seed_indices(std::vector<NodeId>& seeds_indices) const;
};




#endif // DS_GRAPH_H
