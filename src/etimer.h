#ifndef ETIMER_H
#define ETIMER_H

#include <ctime>
//#include <chrono>
#include <iostream>
#include "config_logs.h"

class ETimer{
private:
//    std::chrono::high_resolution_clock::time_point t1,t2;
    std::clock_t start_cpu_time;
public:
//    ETimer(){t1 = std::chrono::high_resolution_clock::now(); }

    ETimer(){start_cpu_time = std::clock(); }

    void stop_timer(const std::string desc);
    void stop_timer(const std::string desc1, const std::string desc2);
};

#endif // ETIMER_H
