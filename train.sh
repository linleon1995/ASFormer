#!/bin/bash

python run.py --action=train --dataset=${1} --split=${2} \
                --num_epochs=${3} \
                --num_layers_PG=11 \
                --num_layers_R=10 \
                --num_R=4 \
                --lr=0.0001
