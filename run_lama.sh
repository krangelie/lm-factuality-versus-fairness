#!/bin/bash

model_type=luke
log_dir=/export/home/kraft/data/lama/$model_type
log_file=$log_dir/lama_log.out

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

mkdir -p $log_dir
touch $log_file

nohup python run_experiment.py benchmark=lama > $log_file 2>&1 &
