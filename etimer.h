#ifndef ETIMER_H
#define ETIMER_H

#include <chrono>
#include <iostream>
#include "config_logs.h"

class ETimer{
private:
    std::chrono::high_resolution_clock::time_point t1,t2;
public:
    ETimer(){t1 = std::chrono::high_resolution_clock::now(); }

    void stop_timer(const std::string desc);
    void stop_timer(const std::string desc1, const std::string desc2);
};

#endif // ETIMER_H
