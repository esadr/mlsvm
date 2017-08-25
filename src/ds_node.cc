#include "ds_node.h"



Node::Node(){
    node_id_ = 0;
    volume_ = 1.0;
    future_volume_ = 1.0;
    is_it_seed_ = 0;
    sum_neighbors_weight_ = 0;
}

Node::Node(NodeId id){
    node_id_ = id;
    volume_ = 1.0;
    future_volume_ = 1.0;
    is_it_seed_ = 0;
    sum_neighbors_weight_ = 0;
}

Node::Node(NodeId id, Volume vol){
    node_id_ = id;
    volume_ = vol;
    future_volume_ = 1.0;
    is_it_seed_ = 0;
    sum_neighbors_weight_ = 0;
}




bool Node::getIsSeed() const{
    return is_it_seed_;
}

Volume Node::getSumNeighborsWeight(){
    return sum_neighbors_weight_;
}

void Node::setSeed(bool status){
    is_it_seed_ = status;
}

void Node::setVolume(double volume){
    volume_ = volume;
}

void Node::setFutureVolume(double future_volume){
    future_volume_ = future_volume;
}

/*

//TODO: DIRTY (learn how to cast a pointer and redesign below function)
void Node::calcFutureVolume(){
    Volume tmp=0,vol_tmp=0;
    EdgeWeight sum_w_tmp =0, j_weight =0 ;
    tmp += volume_;
//    for (auto it = neighbors_.begin(); it != neighbors_.end(); ++it) {
////        tmp += (it->getVolume() * it->weight) / it->calcSumNeighborsWeight();  //TODO: check if it works normal

//        Node *n_temp = it->N;
//        sum_w_tmp = n_temp->calcSumNeighborsWeight();
//        vol_tmp = n_temp->getVolume();
//        j_weight = it->weight;
////        std::cout << "vol_tmp :" << vol_tmp<< "\n";
////        std::cout << "sum_w_tmp :" << sum_w_tmp<< "\n";
////        std::cout << "j-weight :" << j_weight << "\n";
////        tmp += (n_temp->getVolume() * it->weight) / w_tmp;  //TODO: check if it works normal
//        tmp += vol_tmp * (j_weight /sum_w_tmp );

////        printf("neighbors : %d \n", n_temp->getIndex());
//    }
    future_volume_= tmp;
//    std::cout << "Future Volume :" << future_volume_ << "\n";
//    return future_volume_;
}
*/

/*                          calcSumNeighborsWeight()
 * this function should calculate the sum of weights from each neighber of this node
 * to it's neighber. It will only sum up the weights and it will not include any volume.
 * In another word, it is the denuminator of the 2 sum formula Wjk
 */

void Node::setSumNeighborsWeight(Volume sum_neigh_weight){
    sum_neighbors_weight_ = sum_neigh_weight;
}
