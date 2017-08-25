#ifndef DS_NODE_H
#define DS_NODE_H

#include <vector>
#include <iostream>
#include "ds_global.h"


class Node {
private:
    NodeId      node_id_;
    Volume      volume_;
    Volume      future_volume_;
    bool        is_it_seed_;
    Volume      sum_neighbors_weight_;

public:
    //Default Constructor
    Node();

    //Overload constructure
    Node(NodeId);

    Node(NodeId, Volume);

    //Deconstructure
    ~Node(){}

    //Accessors
    NodeId getIndex() const {return node_id_; }
//        @return Index - the pointer to index of this node

    Volume getVolume() const{ return volume_;}

    double getFutureVolume() const {return future_volume_;}

    bool getIsSeed() const;
        //IsSeed
        //@return bool - True if it is a seed and false if it is not

    Volume getSumNeighborsWeight();

    //Mutators
    void setVolume(double);


    void setFutureVolume(double);

    void setSeed(bool);
        //setSeed
        //initialize the _IsSeed with 1

    void calcFutureVolume();

    void setSumNeighborsWeight(Volume);

};

#endif // DS_NODE_H
