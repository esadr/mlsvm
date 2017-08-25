#ifndef DS_TEMPS_H
#define DS_TEMPS_H

#include "ds_global.h"
struct tmp_future_volume
{
    Index node_index;
    Volume future_volume;

    tmp_future_volume(Index node_id, Volume fv) : node_index(node_id), future_volume(fv) {}

    bool operator > (const tmp_future_volume tmp_fv) const
        {
            return (future_volume > tmp_fv.future_volume);
        }
};

struct tmp_filter_p
{
    Index seed_index;
    Volume p_value;

    tmp_filter_p(Index seed_id, Volume p_val) : seed_index(seed_id), p_value(p_val) {}

    bool operator > (const tmp_filter_p tmp_f_p) const
        {
            return (p_value > tmp_f_p.p_value);
        }
};

struct tmp_degree
{
    int id_;
    int degree_;

    tmp_degree(int node_id, int node_degree): id_(node_id), degree_(node_degree) {}

    bool operator > (const tmp_degree tmp_dg_) const
        {
            return (degree_ > tmp_dg_.degree_);
        }
};
#endif // DS_TEMPS_H
