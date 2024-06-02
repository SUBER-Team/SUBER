#!/bin/bash
for folder in ./reports/*; do 
    echo $folder
    python3 ./src/sampling_analysis_ci.py  --name=$folder
done