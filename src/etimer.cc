#include "etimer.h"
#include <string>

//void ETimer::stop_timer(const std::string desc){
//    t2 = std::chrono::high_resolution_clock::now();
//    auto duration = std::chrono::duration_cast<std::chrono::seconds>( t2 - t1 ).count();

//#if timer_print
//    std::cout <<"[Time] "<< desc <<" takes " << duration << " seconds " << std::endl;
//#endif
//}

//http://stackoverflow.com/questions/17432502/how-can-i-measure-cpu-time-and-wall-clock-time-on-both-linux-windows


void ETimer::stop_timer(const std::string desc){
//    t2 = std::clock();
    double cpu_duration = (std::clock() - start_cpu_time) / (double)CLOCKS_PER_SEC;

#if timer_print
    std::cout <<"[CPU Time] "<< desc <<" takes " << cpu_duration << " seconds " << std::endl;
#endif
}


void ETimer::stop_timer(const std::string desc1, const std::string desc2){
    stop_timer(desc1+" "+desc2);
}
