#include "ds_graph.h"
#include "config_params.h"

Graph::Graph(){
    level_ = 0;
//    size_ = 0;
}

//Graph::~Graph(){
//    std::cout <<"!!! graph deconstructor Not done!!!"<<std::endl;
//}

Node& Graph::getNode(NodeId node_id) {
    return g_nodes_[node_id];
}

int Graph::getNumSeeds() const{
    int num_seeds = 0;
    for (auto it = g_nodes_.begin(); it != g_nodes_.end(); ++it) {
        if (it-> getIsSeed()) {
            ++num_seeds;
        }
    }
    return num_seeds;
}

void Graph::getSeedsIndices(std::map<int,int>& c_indices, int& c_index){
//    int v_index = 0;
    c_index = 0;        //it also shows the number of seeds
    for (auto it = g_nodes_.begin(); it != g_nodes_.end(); ++it) {
        if (it->getIsSeed()){
            c_indices[it->getIndex()] = c_index;
            ++c_index;
        }
//        ++v_index;
    }
}

void Graph::addNode(Node& node){
    g_nodes_.push_back(node);
//    ++size_;
}

void Graph::setAvgFutureVolume(Volume avg_fv){
    avg_future_volume_ = avg_fv;
}

//void Graph::calcAllFutureVolume(){
//    std::cout << "TODO: calc fv for all, not done" << std::endl;
//}

//void Graph::calcFutureVolume(NodeId& node_id){
//    Volume new_fv = 0;
//    new_fv += g_nodes_[node_id].getVolume();


////    g_nodes_[node_id].setFutureVolume()
//}


void Graph::printFutureVolumes() const{
    std::cout<< "Print Future volume for graph\n" ;
    for (auto it = g_nodes_.begin(); it != g_nodes_.end(); ++it) {
        std::cout<< "Node["<<it->getIndex() <<"] future volume : "<< it->getFutureVolume() << "\n";
    }
}
void Graph::printSeeds() const{
    std::cout<< "Print Seeds for graph\n" ;
    for (auto it = g_nodes_.begin(); it != g_nodes_.end(); ++it) {
        if (it->getIsSeed()){
            std::cout << it->getIndex() << "\n";
        }
    }
}

// selectSeeds goes through all nodes and set their seed parameter if they choice to be seed
int Graph::selectSeeds(){
//    std::cout << "coarse_Eta is :" << coarse_Eta <<" avg_future_volume_:"<<avg_future_volume_<< "\n";
    int num_seeds=0;
    for (auto it = g_nodes_.begin(); it != g_nodes_.end(); ++it) {  //for nodes that have future volume higher than average * coarse_Eta
//        std::cout <<"Node:" << it->getIndex() <<" FV:"<< it->getFutureVolume() << " AVG*coarse_Eta :"<< (avg_future_volume_ * coarse_Eta) <<"\n";
//        if( it->getFutureVolume() > (avg_future_volume_ * coarse_Eta) ){
        if( it->getFutureVolume() > (avg_future_volume_ * Config_params::getInstance()->get_coarse_Eta() )){
//            std::cout <<"Node:" << it->getIndex() <<" FV:"<< it->getFutureVolume() << "AVG*coarse_Eta :"<< (avg_future_volume_ * coarse_Eta) <<"\n";
            it->setSeed(1);
            num_seeds++;
        }
    }
    return num_seeds;
}

void Graph::find_seed_indices(std::vector<NodeId>& seeds_indices) const{
    int cnt=0;
//    std::vector<NodeId> seeds_indices(num_seeds);
    for (auto it = g_nodes_.begin(); it != g_nodes_.end(); ++it) {
        if (it->getIsSeed()){
            seeds_indices.push_back(it->getIndex());
            cnt++;
        }
    }
//    std::cout <<"[DG][FSI] total number of seeds for next level are " << cnt << std::endl;
//    return seeds_indices;
}
