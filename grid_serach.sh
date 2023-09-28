#!/bin/bash

# Define a semaphore to limit the number of concurrent processes
max_concurrent=10
current_jobs=0

for a in $(seq 0.0 0.2 0.8)
do
    for b in $(seq 0.0 0.2 0.8)
    do
        for c in $(seq 0.0 0.2 0.8)
        do
            # Check the number of current jobs and wait if it's at the limit
            while [ $current_jobs -ge $max_concurrent ]; do
                sleep 1
                jobs_running=$(jobs -p | wc -l)
                current_jobs=$((jobs_running))
            done

            # Run the Python command in a new xterm window
            xterm -e "python api/search_cable_pull_ratio.py --config config_touch_with_obstacles.yaml --alpha $a $b $c" &
            sleep 5
            current_jobs=$((current_jobs + 1))
        done
    done
done

# Wait for all background jobs to finish
wait
