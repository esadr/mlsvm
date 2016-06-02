#include "etimer.h"
#include <string>

void ETimer::stop_timer(const std::string desc){
    t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();

#if timer_print
    std::cout <<"[Time] "<< desc <<" takes " << duration << " seconds " << std::endl;
#endif
}

void ETimer::stop_timer(const std::string desc1, const std::string desc2){
    stop_timer(desc1+" "+desc2);
}
