#!/bin/bash
num_tail_lines=10
for i in `ls | grep _2 ` 
    do 
        if [ $(grep ERR $i | wc -l ) == 0 ]; then 
            echo
            echo
            echo $i
            tail -n $num_tail_lines $i 
        fi
    done